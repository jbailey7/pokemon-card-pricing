"""
Full-index validation eval (2026-07-26 review finding).

train.py's evaluate() ranks val queries against a gallery of val cards only
(~2k). The deployed app ranks against the full index (~20k) — 10x more
distractors, including same-artwork reprints of val cards that live in the
train split. This script measures the same query protocol against BOTH
galleries, isolating the distractor effect, so the documented accuracy
matches the deployed task.

Protocol (matches train.py evaluate()):
  - queries: train-transform (augmented) views of the val-split cards, seeded
  - gallery: the prebuilt index/ embeddings (clean images, val transform)
  - correct iff the retrieved card id == the query card id

Run locally (needs checkpoints/, index/, data/):
    uv run python eval/evaluate_full_index.py

Results are printed and appended to results/full_index_eval.txt.
"""

import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import CardDataset, build_train_transform
from model.inference import CardIdentifier

ROOT = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
INDEX_DIR = ROOT / "index"
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "results" / "full_index_eval.txt"
SEED = 42
K = 3


def topk_accuracy(dots: np.ndarray, gallery_ids: list[str],
                  query_ids: list[str], k: int) -> tuple[float, float]:
    """Top-1/top-k retrieval accuracy given a (num_queries, gallery) score matrix."""
    gallery_ids = np.asarray(gallery_ids)
    top1 = topk = 0
    for i, qid in enumerate(query_ids):
        idx = np.argpartition(-dots[i], k)[:k]
        idx = idx[np.argsort(-dots[i][idx])]
        retrieved = gallery_ids[idx]
        top1 += retrieved[0] == qid
        topk += qid in retrieved
    n = len(query_ids)
    return top1 / n, topk / n


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    # CardIdentifier loads the model + index and verifies the checkpoint/index
    # hash coupling — reuse it rather than reimplementing the load.
    identifier = CardIdentifier(CHECKPOINT, INDEX_DIR)
    index_emb = identifier.embeddings                      # (N, 512), L2-normalised
    index_ids = [m["id"] for m in identifier.metadata]

    val_ds = CardDataset(
        cards_csv=DATA_DIR / "cards.csv",
        data_dir=DATA_DIR,
        transform=build_train_transform(),  # augmented queries, as in train.py
        split="val",
    )
    query_ids = list(val_ds.df["id"])
    print(f"Embedding {len(val_ds):,} augmented val queries "
          f"(gallery: {len(index_ids):,}-card index) on {identifier.device}...")

    loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    chunks = []
    with torch.no_grad():
        for imgs, _ in loader:
            chunks.append(identifier.model(imgs.to(identifier.device)).cpu())
    query_emb = torch.cat(chunks).numpy()                  # (Q, 512)

    dots = query_emb @ index_emb.T                         # (Q, N)

    # Full-index gallery: the deployed task
    full_top1, full_topk = topk_accuracy(dots, index_ids, query_ids, K)

    # Val-only gallery: what train.py reports (same queries, fewer distractors)
    val_id_set = set(query_ids)
    val_mask = np.array([cid in val_id_set for cid in index_ids])
    val_top1, val_topk = topk_accuracy(
        dots[:, val_mask], list(np.asarray(index_ids)[val_mask]), query_ids, K)

    lines = [
        f"Full-index val eval — {date.today()} "
        f"(seed {SEED}, {len(query_ids):,} augmented queries)",
        f"  val-only gallery ({int(val_mask.sum()):,} cards):  "
        f"top-1 {val_top1:.1%}  top-{K} {val_topk:.1%}   <- protocol train.py reports",
        f"  full index      ({len(index_ids):,} cards):  "
        f"top-1 {full_top1:.1%}  top-{K} {full_topk:.1%}   <- the deployed task",
        f"  distractor cost: top-1 {val_top1 - full_top1:+.1%}, "
        f"top-{K} {val_topk - full_topk:+.1%}",
    ]
    print("\n" + "\n".join(lines))

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"\nAppended to {OUT_FILE}")


if __name__ == "__main__":
    main()
