"""API unit tests — all models mocked via FastAPI dependency overrides.

Must pass on a fresh clone with NO model artifacts (checkpoints/, index/,
model/card_detector.pt are gitignored). Real-weights integration tests live
separately and skip when weights are missing (Phase 3).
"""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from model.card_detector import DetectionResult
from server import app, get_detector, get_identifier, get_price_fn

BBOX = (200, 50, 440, 430)  # roughly the guide rectangle for 640×480


class FakeDetector:
    """Returns queued DetectionResults, then repeats the last one."""

    def __init__(self, *results):
        self.queue = list(results) or [detecting()]
        self.reset_calls = 0

    def detect(self, frame):
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]

    def reset(self):
        self.reset_calls += 1


def detecting():
    return DetectionResult(status="detecting", bbox=BBOX,
                           crop=np.zeros((380, 240, 3), np.uint8), confidence=0.9)


def failure(status):
    return DetectionResult(status=status)


class FakeIdentifier:
    def predict(self, image, k=3):
        return [
            {"rank": i + 1, "score": 0.9 - i * 0.1, "id": f"base1-{i}",
             "name": f"Card {i}", "set_name": "Base Set", "number": str(i),
             "rarity": "Rare Holo", "img_path": f"data/images/base1/{i}_hires.png"}
            for i in range(k)
        ]


def jpeg_bytes(color=(120, 120, 120), size=(640, 480)):
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def post_frame(client, color=(120, 120, 120)):
    return client.post("/frame", files={"file": ("f.jpg", jpeg_bytes(color), "image/jpeg")})


@pytest.fixture
def client():
    """TestClient with fake models; server scan state reset around each test."""
    overrides = {
        get_detector: lambda: FakeDetector(),
        get_identifier: lambda: FakeIdentifier(),
        get_price_fn: lambda: lambda card: None,
    }
    app.dependency_overrides.update(overrides)
    server.state.reset()
    yield TestClient(app)
    app.dependency_overrides.clear()
    server.state.reset()


def use_detector(det):
    app.dependency_overrides[get_detector] = lambda: det


def use_price_fn(fn):
    app.dependency_overrides[get_price_fn] = lambda: fn


# ── /frame: status mapping ──────────────────────────────────────────────────

class TestFrameStatuses:
    @pytest.mark.parametrize("status", ["no_card", "too_far", "out_of_frame"])
    def test_failure_statuses_pass_through(self, client, status):
        use_detector(FakeDetector(failure(status)))
        resp = post_frame(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == status

    def test_first_detecting_frame_returns_detecting(self, client):
        resp = post_frame(client)
        assert resp.json()["status"] == "detecting"

    def test_missing_file_returns_422(self, client):
        assert client.post("/frame").status_code == 422

    def test_garbage_bytes_return_422(self, client):
        resp = client.post("/frame", files={"file": ("f.jpg", b"not a jpeg", "image/jpeg")})
        assert resp.status_code == 422


# ── /frame: stability logic ─────────────────────────────────────────────────

class TestStability:
    def test_identical_frames_trigger_identification(self, client):
        # Frame 1 primes prev_region; frames 2..M+1 count stability
        for _ in range(server.STABILITY_FRAMES):
            resp = post_frame(client)
        assert resp.json()["status"] in ("stable", "detecting")
        resp = post_frame(client)
        body = resp.json()
        assert body["status"] == "identified"
        assert len(body["matches"]) == 3
        assert body["warning"] is None

    def test_matches_have_image_url_not_img_path(self, client):
        for _ in range(server.STABILITY_FRAMES + 1):
            resp = post_frame(client)
        m = resp.json()["matches"][0]
        assert m["image_url"] == "/card-images/base1/0_hires.png"
        assert "img_path" not in m

    def test_changed_frame_resets_counter(self, client):
        for _ in range(server.STABILITY_FRAMES):
            post_frame(client, color=(120, 120, 120))
        # A very different frame breaks the stable run…
        resp = post_frame(client, color=(255, 255, 255))
        assert resp.json()["status"] == "detecting"
        assert resp.json()["stable_frames"] == 0
        # …and it takes a full fresh run to identify again
        for i in range(server.STABILITY_FRAMES):
            resp = post_frame(client, color=(255, 255, 255))
        assert resp.json()["status"] == "identified"

    def test_non_detecting_status_resets_counter(self, client):
        det = FakeDetector()
        use_detector(det)
        for _ in range(server.STABILITY_FRAMES - 1):
            post_frame(client)
        # Card removed mid-hold
        det.queue = [failure("no_card"), detecting()]
        post_frame(client)
        assert server.state.stable_count == 0
        # Reinsertion starts a fresh hold, not an instant trigger
        resp = post_frame(client)
        assert resp.json()["status"] == "detecting"

    def test_identifier_unavailable_still_identifies_with_warning(self, client):
        app.dependency_overrides[get_identifier] = lambda: None
        for _ in range(server.STABILITY_FRAMES + 1):
            resp = post_frame(client)
        body = resp.json()
        assert body["status"] == "identified"
        assert body["matches"] == []
        assert "unavailable" in body["warning"]


# ── /price ──────────────────────────────────────────────────────────────────

CARD = {"id": "base1-4", "name": "Charizard", "set_name": "Base Set",
        "number": "4", "rarity": "Rare Holo"}


def price_result(**overrides):
    result = {
        "variants": [
            {"label": "Unlimited Holofoil", "market": 100.0, "low_90d": 80.0,
             "high_90d": 130.0, "change_90d": 2.5, "condition": "Near Mint",
             "printing": "Unlimited Holofoil", "updated_at": 1780000000},
            {"label": "Unlimited Holofoil", "market": 120.0, "low_90d": None,
             "high_90d": None, "change_90d": None, "condition": "Lightly Played",
             "printing": "Unlimited Holofoil", "updated_at": 1780000000},
        ],
        "matched_set_name": "Base Set",
        "matched_number": "4",
        "set_number_mismatch": False,
        "printing_filter_fallback": False,
    }
    result.update(overrides)
    return result


class TestPrice:
    def test_unavailable_when_lookup_returns_none(self, client):
        resp = client.post("/price", json=CARD)
        body = resp.json()
        assert body["available"] is False
        assert "tcgplayer.com" in body["tcgplayer_url"]

    def test_full_result_structure(self, client):
        use_price_fn(lambda card: price_result())
        body = client.post("/price", json=CARD).json()
        assert body["available"] is True
        assert len(body["variants"]) == 2
        assert body["set_number_mismatch"] is False
        assert body["condition_order"][0] == "Near Mint"
        assert "tcgplayer.com" in body["tcgplayer_url"]

    def test_price_order_violation_surfaced(self, client):
        # LP ($120) > NM ($100) in the fixture → must be flagged
        use_price_fn(lambda card: price_result())
        body = client.post("/price", json=CARD).json()
        vio = body["violations_by_printing"]["Unlimited Holofoil"]
        assert len(vio) == 1
        assert vio[0][0] == "Near Mint"

    def test_mismatch_flag_passes_through(self, client):
        use_price_fn(lambda card: price_result(set_number_mismatch=True,
                                               matched_set_name="Jungle"))
        body = client.post("/price", json=CARD).json()
        assert body["set_number_mismatch"] is True
        assert body["matched_set_name"] == "Jungle"

    def test_missing_fields_return_422(self, client):
        resp = client.post("/price", json={"name": "Charizard"})
        assert resp.status_code == 422

    def test_non_json_body_returns_422(self, client):
        resp = client.post("/price", content=b"nonsense",
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# ── /reset ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_scan_state_and_detector(self, client):
        det = FakeDetector()
        use_detector(det)
        for _ in range(server.STABILITY_FRAMES - 1):
            post_frame(client)
        assert server.state.stable_count > 0

        resp = client.post("/reset")
        assert resp.json() == {"status": "reset"}
        assert server.state.stable_count == 0
        assert server.state.identified is False
        assert det.reset_calls == 1

        # Post-reset frame starts a fresh cycle, no instant identification
        resp = post_frame(client)
        assert resp.json()["status"] == "detecting"

    def test_full_scan_cycle(self, client):
        """identify → reset → identify again (the 'Scan another' flow)."""
        for _ in range(server.STABILITY_FRAMES + 1):
            resp = post_frame(client)
        assert resp.json()["status"] == "identified"

        client.post("/reset")
        for _ in range(server.STABILITY_FRAMES + 1):
            resp = post_frame(client)
        assert resp.json()["status"] == "identified"


# ── GET / ───────────────────────────────────────────────────────────────────

def test_index_serves_frontend(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "getUserMedia" in resp.text
