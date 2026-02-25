import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from model.constants import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def build_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.8),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.6, 0.85)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CardDataset(Dataset):
    def __init__(
        self,
        cards_csv: str | Path,
        data_dir: str | Path,
        transform: transforms.Compose | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform or build_val_transform()

        df = pd.read_csv(cards_csv, dtype=str).fillna("")

        df["_img_path"] = df.apply(
            lambda r: self.data_dir / "images" / r["set_id"] / f"{r['number'].replace('/', '-')}_hires.png",
            axis=1,
        )
        df = df[df["_img_path"].apply(lambda p: p.exists())].copy()
        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise FileNotFoundError(
                f"No card images found under {self.data_dir / 'images'}. "
                "Run data/download_cards.py first."
            )

        # Sort then shuffle for a deterministic but randomised train/val split
        all_ids = sorted(df["id"].unique())
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        n_val = max(1, int(len(all_ids) * val_fraction))
        val_ids = set(all_ids[:n_val])
        train_ids = set(all_ids[n_val:])

        if split == "val":
            df = df[df["id"].isin(val_ids)].copy()
        else:
            df = df[df["id"].isin(train_ids)].copy()

        df = df.reset_index(drop=True)

        split_ids = sorted(df["id"].unique())
        self.id_to_label: dict[str, int] = {cid: i for i, cid in enumerate(split_ids)}
        self.label_to_id: dict[int, str] = {i: cid for cid, i in self.id_to_label.items()}
        self.num_classes = len(split_ids)

        self.df = df
        self.labels: list[int] = [self.id_to_label[cid] for cid in df["id"]]
        self.img_paths: list[Path] = list(df["_img_path"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, int]]:
        return [self.__getitem__(i) for i in indices]

    def get_metadata(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "id":       row["id"],
            "name":     row["name"],
            "set_id":   row["set_id"],
            "set_name": row["set_name"],
            "number":   row["number"],
            "rarity":   row["rarity"],
        }


class PKSampler(Sampler):
    def __init__(
        self,
        labels: list[int],
        p: int = 16,
        k: int = 4,
        num_batches: int | None = None,
    ) -> None:
        self.p = p
        self.k = k

        label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)
        self.label_to_indices = label_to_indices
        self.classes = list(label_to_indices.keys())

        batch_size = p * k
        self.num_batches = num_batches or (len(labels) // batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            # P random classes, K samples each (with replacement for small classes)
            chosen_classes = random.sample(self.classes, min(self.p, len(self.classes)))
            batch = []
            for cls in chosen_classes:
                batch.extend(random.choices(self.label_to_indices[cls], k=self.k))
            yield batch
