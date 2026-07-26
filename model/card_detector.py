"""YOLOv8n card detector wrapper with frame-skipping and validity checks.

Detection contract (see CLAUDE.local.md):
- The guide-rectangle IoU check is the PRIMARY validity check. Aspect ratio
  is only a loose sanity bound: YOLO boxes are axis-aligned, so a card
  rotated ~15° legitimately deviates from the 0.716 card ratio.
- YOLO runs every `frame_skip` frames. On skipped frames the cached bbox
  from the last YOLO run is reused (flagged cached=True) so the caller can
  still crop and run the stability check at full frame rate. The cache is
  cleared whenever a YOLO run produces anything other than a valid detection.
- Stub mode (no trained weights) returns the guide rectangle as the bbox on
  every frame, so the full app runs end-to-end before Phase 3 training.

ultralytics is imported lazily — tests and stub mode never need it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pokémon cards are 63×88mm. The guide overlay and all geometry derive from this.
CARD_ASPECT = 63 / 88  # ≈ 0.716 (width / height)

# Guide rectangle geometry, as fractions of the frame. frontend/index.html
# mirrors these values — keep them in sync.
GUIDE_HEIGHT_FRAC = 0.70

# Tunables (defaults; see CLAUDE.local.md tunables table)
YOLO_FRAME_SKIP = 3       # run YOLO every N frames
GUIDE_IOU_MIN   = 0.5     # primary validity check
AR_TOLERANCE    = 0.35    # loose sanity bound: reject beyond ±35% of CARD_ASPECT
MIN_AREA_FRAC   = 0.30    # bbox area below this fraction of guide area → too_far
EDGE_MARGIN_PX  = 8       # bbox side within this of the frame edge → out_of_frame

# TODO(Phase 3): set from the F1-maximising point on the val P/R curve —
# do NOT ship the default. Placeholder until the detector is trained.
CONF_THRESHOLD = 0.25


@dataclass
class DetectionResult:
    status: str                      # no_card | too_far | out_of_frame | detecting
    bbox: tuple | None = None        # (x1, y1, x2, y2) ints
    crop: np.ndarray | None = None   # HWC RGB crop of bbox
    cached: bool = False             # True when bbox reused from a skipped frame
    confidence: float | None = None


def guide_rect(frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Fixed guide rectangle (x1, y1, x2, y2), centred, card aspect ratio."""
    gh = frame_h * GUIDE_HEIGHT_FRAC
    gw = gh * CARD_ASPECT
    x1 = (frame_w - gw) / 2
    y1 = (frame_h - gh) / 2
    return (round(x1), round(y1), round(x1 + gw), round(y1 + gh))


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def pad_bbox(bbox: tuple, frac: float, frame_w: int, frame_h: int) -> tuple:
    """Expand bbox by `frac` on every side, clamped to frame bounds."""
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * frac, (y2 - y1) * frac
    return (
        max(0, round(x1 - pw)), max(0, round(y1 - ph)),
        min(frame_w, round(x2 + pw)), min(frame_h, round(y2 + ph)),
    )


class CardDetector:
    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        stub: bool = False,
        conf_threshold: float = CONF_THRESHOLD,
        frame_skip: int = YOLO_FRAME_SKIP,
    ) -> None:
        self.stub = stub
        self.conf_threshold = conf_threshold
        self.frame_skip = max(1, frame_skip)
        self._counter = 0
        self._cached_bbox: tuple | None = None
        self._cached_conf: float | None = None

        if not stub:
            if weights is None:
                raise ValueError("weights path required unless stub=True")
            from ultralytics import YOLO  # lazy: stub mode / tests don't need it
            import torch
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = device
            self.model = YOLO(str(weights))

    def reset(self) -> None:
        """Clear frame counter and cached bbox (new scan cycle)."""
        self._counter = 0
        self._cached_bbox = None
        self._cached_conf = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        self._counter += 1

        if self.stub:
            # Guide rectangle inset slightly, always valid — Phase 2 stand-in
            bbox = pad_bbox(guide_rect(w, h), -0.02, w, h)
            return self._valid(frame, bbox, confidence=1.0)

        is_yolo_frame = (self._counter - 1) % self.frame_skip == 0
        if not is_yolo_frame:
            if self._cached_bbox is None:
                return DetectionResult(status="no_card", cached=True)
            return self._valid(frame, self._cached_bbox,
                               confidence=self._cached_conf, cached=True)

        result = self.model.predict(frame, conf=self.conf_threshold,
                                    device=self.device, verbose=False)[0]
        status = self._validate(result, w, h)
        if isinstance(status, str):
            self._cached_bbox = None
            self._cached_conf = None
            return DetectionResult(status=status)

        bbox, conf = status
        self._cached_bbox = bbox
        self._cached_conf = conf
        return self._valid(frame, bbox, confidence=conf)

    def _validate(self, result, w: int, h: int):
        """Apply the detection contract in order. Returns a failure status
        string, or (bbox, conf) for a valid detection."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return "no_card"  # no detection (or all below conf threshold)

        best = int(boxes.conf.argmax())
        conf = float(boxes.conf[best])
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[best])
        bbox = (round(x1), round(y1), round(x2), round(y2))
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw <= 0 or bh <= 0:
            return "no_card"

        # Aspect ratio: loose sanity bound only (rotation skews axis-aligned boxes)
        ar = bw / bh
        if not (CARD_ASPECT * (1 - AR_TOLERANCE) <= ar <= CARD_ASPECT * (1 + AR_TOLERANCE)):
            return "no_card"

        # Size: card too far from the camera. Checked BEFORE guide IoU — a
        # small bbox necessarily fails IoU too (IoU >= 0.5 implies bbox area
        # >= 0.5x guide area), so with the IoU check first this status was
        # unreachable and the user would get "centre the card" instead of the
        # actionable "move closer".
        guide = guide_rect(w, h)
        guide_area = (guide[2] - guide[0]) * (guide[3] - guide[1])
        if bw * bh < MIN_AREA_FRAC * guide_area:
            return "too_far"

        # Guide overlap: PRIMARY validity check
        if iou(bbox, guide) < GUIDE_IOU_MIN:
            return "out_of_frame"

        # Edge clipping
        if (bbox[0] < EDGE_MARGIN_PX or bbox[1] < EDGE_MARGIN_PX
                or bbox[2] > w - EDGE_MARGIN_PX or bbox[3] > h - EDGE_MARGIN_PX):
            return "out_of_frame"

        return bbox, conf

    def _valid(self, frame: np.ndarray, bbox: tuple,
               confidence: float | None, cached: bool = False) -> DetectionResult:
        x1, y1, x2, y2 = bbox
        return DetectionResult(
            status="detecting",
            bbox=bbox,
            crop=frame[y1:y2, x1:x2],
            cached=cached,
            confidence=confidence,
        )
