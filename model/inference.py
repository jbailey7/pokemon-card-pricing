import pickle
from pathlib import Path

import faiss
import torch
from PIL import Image

from model.constants import EMBEDDING_DIM
from model.dataset import build_val_transform
from model.embedding_model import EmbeddingModel


class CardIdentifier:
    def __init__(
        self,
        checkpoint: str | Path,
        index_dir: str | Path = "index/",
        device: str | None = None,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = EmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=False)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_val_transform()

        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "card_index.faiss"))
        with open(index_dir / "card_metadata.pkl", "rb") as f:
            self.metadata: list[dict] = pickle.load(f)

        assert self.index.ntotal == len(self.metadata), (
            f"Index/metadata mismatch: {self.index.ntotal} vs {len(self.metadata)}"
        )

    def predict(self, image: Image.Image, k: int = 3) -> list[dict]:
        img_rgb = image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        distances, indices = self.index.search(embedding.cpu().numpy(), k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            entry = dict(self.metadata[idx])  # copy so we don't mutate the metadata list
            entry["rank"] = rank
            entry["score"] = round(float(1.0 / (1.0 + dist)), 4)
            results.append(entry)

        return results
