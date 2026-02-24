import argparse
import logging
import pickle
from pathlib import Path

import faiss
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from model.constants import EMBEDDING_DIM
from model.dataset import build_val_transform
from model.embedding_model import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_index(args: argparse.Namespace) -> None:
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Load model 
    model = EmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    log.info(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load card metadata
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "cards.csv", dtype=str).fillna("")

    # Reconstruct image paths, filter to existing
    def img_path(row: pd.Series) -> Path:
        safe_num = str(row["number"]).replace("/", "-")
        return data_dir / "images" / row["set_id"] / f"{safe_num}_hires.png"

    df["_img_path"] = df.apply(img_path, axis=1)
    df = df[df["_img_path"].apply(lambda p: p.exists())].copy()
    df = df.reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    log.info(f"Building index over {len(df):,} cards")

    # Transform
    transform = build_val_transform()

    # Embed all cards
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata: list[dict] = []

    batch_size = 64
    embeddings_list = []

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="Embedding cards", unit="batch"):
            batch_rows = df.iloc[start : start + batch_size]
            imgs = []
            valid_rows = []

            for _, row in batch_rows.iterrows():
                try:
                    img = Image.open(row["_img_path"]).convert("RGB")
                    imgs.append(transform(img))
                    valid_rows.append(row)
                except Exception as exc:
                    log.warning(f"Skipping {row['id']}: {exc}")

            if not imgs:
                continue

            batch_tensor = torch.stack(imgs).to(device) 
            emb = model(batch_tensor) 
            embeddings_list.append(emb.cpu())

            for row in valid_rows:
                metadata.append({
                    "id":       row["id"],
                    "name":     row["name"],
                    "set_id":   row["set_id"],
                    "set_name": row["set_name"],
                    "number":   row["number"],
                    "rarity":   row["rarity"],
                    "img_path": str(row["_img_path"]),
                })

    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()  # (N, 512)
    index.add(all_embeddings)
    log.info(f"Index contains {index.ntotal:,} vectors")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = output_dir / "card_index.faiss"
    meta_path  = output_dir / "card_metadata.pkl"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    log.info(f"FAISS index saved to  {faiss_path}")
    log.info(f"Metadata saved to     {meta_path}")
    log.info(f"Index size: {faiss_path.stat().st_size / 1e6:.1f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS card embedding index")
    parser.add_argument("--checkpoint",  required=True,     help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data-dir",    default="data/",   help="Root data directory (contains cards.csv and images/)")
    parser.add_argument("--output-dir",  default="index/",  help="Directory to write index files")
    parser.add_argument("--limit",       type=int, default=None, help="Only index first N cards (for testing)")
    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
