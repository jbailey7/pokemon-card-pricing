"""FastAPI app: live camera card identification + pricing.

Run:  uv run uvicorn server:app --reload
Then open http://localhost:8000 (camera requires localhost or HTTPS).

Single-user local server by design: scan state is module-level, and the
frontend's response-driven loop guarantees requests never overlap.
Models are provided via lazily-initialised dependencies so tests can
override them (tests/test_server.py) and the suite passes with no
model artifacts present.
"""

import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model.card_detector import CardDetector, guide_rect, pad_bbox
from pricing.display import CONDITION_ORDER, condition_idx, price_order_violations
from pricing.price_lookup import get_variants, tcgplayer_url

log = logging.getLogger("server")

ROOT             = Path(__file__).parent
FRONTEND         = ROOT / "frontend" / "index.html"
CHECKPOINT       = ROOT / "checkpoints" / "best_model.pt"
INDEX_DIR        = ROOT / "index"
IMAGES_DIR       = ROOT / "data" / "images"
DETECTOR_WEIGHTS = ROOT / "model" / "card_detector.pt"

# ── Tunables ────────────────────────────────────────────────────────────────
# TODO(Phase 4): STABILITY_THRESHOLD is a placeholder. It must be set from
# calibration (eval/calibrate_stability.py): 95th percentile of the stable-
# phase diff distribution across varied lighting/distance/background.
# Do not tune this by feel.
STABILITY_THRESHOLD = 8.0    # max mean abs pixel diff (0-255 grey) on guide interior
STABILITY_FRAMES    = 5      # consecutive stable frames to trigger identification
CROP_PADDING        = 0.10   # bbox margin before identification — the index was
                             # built from full card scans, so a tight crop shifts
                             # the input distribution (see CLAUDE.local.md)
STABILITY_REGION    = (72, 100)  # (w, h) the guide interior is resized to for diffing


class ScanState:
    """Stability tracking for the current scan cycle. Single-user by design."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.prev_region: np.ndarray | None = None
        self.stable_count = 0
        self.identified = False
        self.last_matches: list[dict] = []

    def reset_stability(self) -> None:
        """Any non-detecting status resets the counter — a card removed and
        reinserted must always start a fresh hold."""
        self.prev_region = None
        self.stable_count = 0


state = ScanState()

# ── Model providers (lazy, override-able in tests) ──────────────────────────
_detector: CardDetector | None = None
_identifier = None
_identifier_loaded = False


def get_detector() -> CardDetector:
    global _detector
    if _detector is None:
        if DETECTOR_WEIGHTS.exists():
            _detector = CardDetector(weights=DETECTOR_WEIGHTS)
            log.info("CardDetector loaded from %s", DETECTOR_WEIGHTS)
        else:
            _detector = CardDetector(stub=True)
            log.warning("%s not found — running with the STUB detector "
                        "(guide rectangle as bbox). Train it in Phase 3.",
                        DETECTOR_WEIGHTS)
    return _detector


def get_identifier():
    global _identifier, _identifier_loaded
    if not _identifier_loaded:
        _identifier_loaded = True
        try:
            from model.inference import CardIdentifier
            _identifier = CardIdentifier(CHECKPOINT, INDEX_DIR)
            log.info("CardIdentifier loaded")
        except Exception as e:
            _identifier = None
            log.warning("CardIdentifier unavailable (%s) — /frame will return "
                        "empty matches. Provide checkpoints/ and index/.", e)
    return _identifier


def get_price_fn():
    return get_variants


app = FastAPI(title="Pokémon Card Pricer")

if IMAGES_DIR.exists():
    app.mount("/card-images", StaticFiles(directory=IMAGES_DIR), name="card-images")


def image_url(img_path: str) -> str | None:
    """Convert an index-metadata filesystem path to a /card-images URL."""
    parts = Path(img_path).parts
    if "images" in parts:
        rel = Path(*parts[parts.index("images") + 1:])
        return f"/card-images/{rel.as_posix()}"
    return None


def guide_region(frame: np.ndarray) -> np.ndarray:
    """Guide-rectangle interior as a small grayscale array for stability diffs.

    Fixed coordinates — deliberately NOT the YOLO crop, which jitters a few
    pixels frame-to-frame even when the card is perfectly still."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = guide_rect(w, h)
    region = Image.fromarray(frame[y1:y2, x1:x2]).convert("L").resize(STABILITY_REGION)
    return np.asarray(region, dtype=np.float32)


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.post("/frame")
async def frame(
    file: UploadFile = File(...),
    detector: CardDetector = Depends(get_detector),
    identifier=Depends(get_identifier),
):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image")
    frame_arr = np.asarray(img)

    det = detector.detect(frame_arr)
    if det.status != "detecting":
        state.reset_stability()
        return {"status": det.status}

    # Stability check on the fixed guide interior
    region = guide_region(frame_arr)
    if state.prev_region is None:
        diff = None
        state.stable_count = 0
    else:
        diff = float(np.abs(region - state.prev_region).mean())
        state.stable_count = state.stable_count + 1 if diff <= STABILITY_THRESHOLD else 0
    state.prev_region = region

    if state.stable_count < STABILITY_FRAMES:
        return {
            "status": "stable" if state.stable_count > 0 else "detecting",
            "stable_frames": state.stable_count,
            "diff": diff,
        }

    # Stable → identify (once per scan cycle; frontend pauses on identified)
    if not state.identified:
        h, w = frame_arr.shape[:2]
        x1, y1, x2, y2 = pad_bbox(det.bbox, CROP_PADDING, w, h)
        crop = Image.fromarray(frame_arr[y1:y2, x1:x2])
        if identifier is not None:
            matches = identifier.predict(crop, k=3)
            for m in matches:
                m["image_url"] = image_url(m.get("img_path", ""))
                m.pop("img_path", None)
        else:
            matches = []
        state.identified = True
        state.last_matches = matches

    return {
        "status": "identified",
        "matches": state.last_matches,
        "warning": None if identifier is not None else
                   "Identifier unavailable — checkpoints/ or index/ missing",
    }


@app.post("/price")
def price(card: dict = Body(...), price_fn=Depends(get_price_fn)):
    required = {"id", "name", "set_name", "number"}
    missing = required - card.keys()
    if missing:
        raise HTTPException(status_code=422,
                            detail=f"Missing card fields: {sorted(missing)}")

    url = tcgplayer_url(card)
    result = price_fn(card)
    if result is None:
        return {"available": False, "tcgplayer_url": url}

    # Price-order violations per printing (ported trust feature)
    violations_by_printing: dict[str, list] = {}
    variants = result["variants"]
    for printing in dict.fromkeys(v["printing"] for v in variants):
        pv = [v for v in variants if v["printing"] == printing]
        conditions = sorted(dict.fromkeys(v["condition"] for v in pv), key=condition_idx)
        vio = price_order_violations(pv, conditions)
        if vio:
            violations_by_printing[printing] = vio

    return {
        "available": True,
        **result,
        "tcgplayer_url": url,
        "violations_by_printing": violations_by_printing,
        "condition_order": CONDITION_ORDER,
    }


@app.post("/reset")
def reset(detector: CardDetector = Depends(get_detector)):
    """Clear all server-side scan state — called by 'Scan another'."""
    state.reset()
    detector.reset()
    return {"status": "reset"}
