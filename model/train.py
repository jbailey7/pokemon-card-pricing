import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model.constants import EMBEDDING_DIM
from model.dataset import CardDataset, PKSampler, build_train_transform, build_val_transform
from model.embedding_model import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Device selection
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")
    return device


# Triplet loss — online batch-hard mining
def pairwise_l2_distance(embeddings: torch.Tensor) -> torch.Tensor:
    # For L2-normalised vectors: ||a-b||^2 = 2 - 2*dot(a,b)
    dot = embeddings @ embeddings.t()
    dot = dot.clamp(-1, 1)
    sq_dist = 2.0 - 2.0 * dot
    sq_dist = sq_dist.clamp(min=0)  # avoid tiny negatives from float precision
    return sq_dist.sqrt()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, float]:
    dist = pairwise_l2_distance(embeddings)
    n = embeddings.size(0)

    labels = labels.unsqueeze(1) 
    same_label = labels == labels.t()
    diff_label = ~same_label

    # Exclude self-comparisons
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same_label = same_label & ~eye

    # Hardest positive: max distance among same-label
    dist_pos = dist.clone()
    dist_pos[~same_label] = 0.0
    hardest_pos = dist_pos.max(dim=1).values

    # Hardest negative: min distance among different-label
    dist_neg = dist.clone()
    dist_neg[~diff_label] = float("inf")
    hardest_neg = dist_neg.min(dim=1).values

    # Skip anchors with no positives in batch (can happen at edges of PKSampler)
    has_positive = same_label.any(dim=1)

    triplet_loss = F.relu(hardest_pos - hardest_neg + margin)
    triplet_loss = triplet_loss[has_positive]

    active_fraction = (triplet_loss > 0).float().mean().item() if len(triplet_loss) > 0 else 0.0
    loss = triplet_loss.mean() if len(triplet_loss) > 0 else torch.tensor(0.0, device=embeddings.device)

    return loss, active_fraction


# Validation — retrieval top-k accuracy
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    k: int = 3,
) -> tuple[float, float]:
    model.eval()
    all_embeddings = []
    all_labels = []

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        emb = model(imgs)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)       

    n = all_embeddings.size(0)

    # Pairwise distances on CPU (avoid OOM on large val sets)
    dist = pairwise_l2_distance(all_embeddings)  # (N, N)

    # Exclude self (set diagonal to inf)
    dist.fill_diagonal_(float("inf"))

    top1_correct = 0
    topk_correct = 0

    for i in range(n):
        query_label = all_labels[i].item()
        sorted_idx = dist[i].argsort()

        top1_label = all_labels[sorted_idx[0]].item()
        topk_labels = [all_labels[sorted_idx[j]].item() for j in range(min(k, n - 1))]

        if top1_label == query_label:
            top1_correct += 1
        if query_label in topk_labels:
            topk_correct += 1

    top1_acc = top1_correct / n
    topk_acc = topk_correct / n
    return top1_acc, topk_acc


# Training loop
def train(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    log.info("Loading datasets...")
    train_ds = CardDataset(
        cards_csv=Path(args.data_dir) / "cards.csv",
        data_dir=args.data_dir,
        transform=build_train_transform(),
        split="train",
    )
    val_ds = CardDataset(
        cards_csv=Path(args.data_dir) / "cards.csv",
        data_dir=args.data_dir,
        transform=build_val_transform(),
        split="val",
    )

    # Optional: limit to N cards for smoke testing
    if args.limit:
        # Take first `limit` unique classes from train
        class_counts: dict[int, int] = {}
        limited_indices = []
        for idx in range(len(train_ds)):
            lbl = train_ds.labels[idx]
            if lbl not in class_counts:
                class_counts[lbl] = 0
            if len(class_counts) <= args.limit:
                limited_indices.append(idx)
                class_counts[lbl] += 1
        train_ds = Subset(train_ds, limited_indices)
        # Rebuild labels list for PKSampler
        train_labels = [train_ds.dataset.labels[i] for i in limited_indices]
        log.info(f"Smoke test: limited to {len(class_counts)} classes, {len(limited_indices)} samples")
    else:
        train_labels = train_ds.labels

    log.info(f"Train: {len(train_ds):,} images | Val: {len(val_ds):,} images")
    log.info(f"Train classes: {len(set(train_labels)):,} | Val classes: {val_ds.num_classes:,}")

    # Samplers & Loaders 
    train_sampler = PKSampler(train_labels, p=args.p, k=args.k)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = EmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)
    model.freeze_backbone()
    log.info(f"Model: EfficientNet-B0 + projection head | Backbone frozen")

    # Phase 1 optimizer (head only)
    optimizer = torch.optim.Adam(model.trainable_params(), lr=args.lr)
    scheduler = None

    best_val_top1 = 0.0

    for epoch in range(1, args.epochs_total + 1):
        # Phase transition
        if epoch == args.frozen_epochs + 1:
            log.info(f"Epoch {epoch}: Unfreezing last 2 backbone blocks, lr → {args.lr / 10:.1e}")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.trainable_params(), lr=args.lr / 10)
            remaining_epochs = args.epochs_total - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=1e-6
            )

        # Train epoch
        model.train()
        epoch_loss = 0.0
        epoch_active = 0.0
        num_batches = 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(imgs)
            loss, active_frac = batch_hard_triplet_loss(embeddings, labels, margin=args.margin)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_active += active_frac
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_active = epoch_active / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Validate
        val_top1, val_top3 = evaluate(model, val_loader, device, k=3)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs_total} | "
            f"loss={avg_loss:.4f} | "
            f"active={avg_active:.1%} | "
            f"val_top1={val_top1:.3f} | "
            f"val_top3={val_top3:.3f} | "
            f"lr={current_lr:.1e} | "
            f"time={elapsed:.0f}s"
        )

        if avg_active < 0.05 and num_batches > 5:
            log.warning(
                "Active triplet fraction < 5% — embeddings may be collapsing. "
                "Consider increasing margin or checking data diversity."
            )

        # Checkpoints
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_top1": val_top1},
            checkpoint_dir / "last_model.pt",
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_top1": val_top1},
                checkpoint_dir / "best_model.pt",
            )
            log.info(f"  ✓ New best model saved (val_top1={val_top1:.3f})")

    log.info(f"Training complete. Best val_top1={best_val_top1:.3f}")
    log.info(f"Checkpoints saved to {checkpoint_dir}")


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pokémon card embedding model")
    parser.add_argument("--data-dir",        default="data/",        help="Root data directory (contains cards.csv and images/)")
    parser.add_argument("--checkpoint-dir",  default="checkpoints/", help="Directory to save model checkpoints")
    parser.add_argument("--epochs-total",    type=int, default=30,   help="Total number of training epochs")
    parser.add_argument("--frozen-epochs",   type=int, default=5,    help="Epochs to train with backbone frozen")
    parser.add_argument("--lr",              type=float, default=1e-3, help="Initial learning rate (phase 1)")
    parser.add_argument("--margin",          type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--p",               type=int, default=16,   help="Classes per batch (PK sampler)")
    parser.add_argument("--k",               type=int, default=4,    help="Instances per class per batch (PK sampler)")
    parser.add_argument("--num-workers",     type=int, default=2,    help="DataLoader num_workers")
    parser.add_argument("--limit",           type=int, default=None, help="Limit to N classes for smoke testing")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
