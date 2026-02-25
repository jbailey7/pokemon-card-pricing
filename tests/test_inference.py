import numpy as np
import torch
from PIL import Image

from model.dataset import build_val_transform
from model.embedding_model import EmbeddingModel
from model.inference import CardIdentifier


def make_identifier(num_cards=50):
    # Build a CardIdentifier with a random untrained model — no checkpoint file needed
    identifier = CardIdentifier.__new__(CardIdentifier)
    identifier.device = torch.device("cpu")
    identifier.model = EmbeddingModel(pretrained=False)
    identifier.model.eval()
    identifier.transform = build_val_transform()

    rng = np.random.default_rng(0)
    embs = rng.standard_normal((num_cards, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    identifier.embeddings = embs

    identifier.metadata = [
        {"id": f"set1-{i}", "name": f"Card {i}", "set_name": "Test Set",
         "number": str(i), "rarity": "Common"}
        for i in range(num_cards)
    ]
    return identifier


class TestCardIdentifierPredict:
    def test_returns_k_results(self):
        identifier = make_identifier()
        img = Image.new("RGB", (224, 224))
        assert len(identifier.predict(img, k=3)) == 3
        assert len(identifier.predict(img, k=1)) == 1

    def test_results_have_required_fields(self):
        identifier = make_identifier()
        img = Image.new("RGB", (224, 224))
        for r in identifier.predict(img, k=3):
            assert "rank" in r
            assert "score" in r
            assert "id" in r
            assert "name" in r

    def test_ranks_are_sequential_from_one(self):
        identifier = make_identifier()
        img = Image.new("RGB", (224, 224))
        ranks = [r["rank"] for r in identifier.predict(img, k=3)]
        assert ranks == [1, 2, 3]

    def test_scores_are_in_valid_range(self):
        identifier = make_identifier()
        img = Image.new("RGB", (224, 224))
        for r in identifier.predict(img, k=5):
            assert 0 < r["score"] <= 1.0

    def test_scores_are_in_descending_order(self):
        identifier = make_identifier()
        img = Image.new("RGB", (224, 224))
        scores = [r["score"] for r in identifier.predict(img, k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_handles_rgba_image(self):
        identifier = make_identifier()
        img = Image.new("RGBA", (300, 400), color=(255, 0, 0, 128))
        results = identifier.predict(img, k=1)
        assert len(results) == 1
