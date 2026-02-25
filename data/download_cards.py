import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
REPO_DIR = DATA_DIR / "pokemon-tcg-data"
CARDS_EN_DIR = REPO_DIR / "cards" / "en"
IMAGES_DIR = DATA_DIR / "images"
CARDS_CSV = DATA_DIR / "cards.csv"
FAILED_LOG = DATA_DIR / "failed_images.txt"

TCG_DATA_REPO = "https://github.com/PokemonTCG/pokemon-tcg-data.git"


def download_json_data() -> None:
    if REPO_DIR.exists():
        log.info("pokemon-tcg-data repo already exists — pulling latest changes")
        result = subprocess.run(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
            capture_output=True, text=True
        )
        log.info(result.stdout.strip() or "Already up to date")
    else:
        log.info("Cloning pokemon-tcg-data repo (JSON only, ~5MB)...")
        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--filter=blob:none",
                "--sparse",
                TCG_DATA_REPO,
                str(REPO_DIR),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "sparse-checkout", "set", "cards/en", "sets"],
            check=True,
        )
        log.info("Clone complete")


def parse_all_sets() -> pd.DataFrame:
    set_files = sorted(CARDS_EN_DIR.glob("*.json"))
    if not set_files:
        raise FileNotFoundError(f"No JSON files found in {CARDS_EN_DIR}")

    sets_file = REPO_DIR / "sets" / "en.json"
    set_name_map: dict[str, str] = {}
    if sets_file.exists():
        with open(sets_file, encoding="utf-8") as f:
            for s in json.load(f):
                set_name_map[s["id"]] = s["name"]
        log.info(f"Loaded {len(set_name_map)} set names from sets/en.json")
    else:
        log.warning("sets/en.json not found — set_name column will be empty")

    log.info(f"Parsing {len(set_files)} set files...")
    rows = []

    for set_file in set_files:
        with open(set_file, encoding="utf-8") as f:
            cards = json.load(f)

        set_id = set_file.stem
        set_name = set_name_map.get(set_id, "")

        for card in cards:
            images = card.get("images", {})
            rows.append({
                "id":          card.get("id", ""),
                "name":        card.get("name", ""),
                "set_id":      set_id,
                "set_name":    set_name,
                "number":      card.get("number", ""),
                "rarity":      card.get("rarity", ""),
                "supertype":   card.get("supertype", ""),
                "image_small": images.get("small", ""),
                "image_large": images.get("large", ""),
            })

    df = pd.DataFrame(rows)
    log.info(f"Parsed {len(df):,} cards across {len(set_files)} sets")
    return df


def download_images(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    image_paths = []

    rows_to_download = df if limit is None else df.head(limit)
    skipped = downloaded = errors = 0

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for _, row in tqdm(rows_to_download.iterrows(), total=len(rows_to_download), desc="Downloading images", unit="card"):
            set_dir = IMAGES_DIR / row["set_id"]
            set_dir.mkdir(parents=True, exist_ok=True)

            # Sanitise slashes in promo card numbers (e.g. "SWSH001")
            safe_number = str(row["number"]).replace("/", "-")
            img_path = set_dir / f"{safe_number}_hires.png"
            rel_path = str(img_path.relative_to(DATA_DIR))

            if img_path.exists():
                skipped += 1
                image_paths.append(rel_path)
                continue

            url = row["image_large"]
            if not url:
                errors += 1
                image_paths.append("")
                failed.append(f"NO_URL\t{row['id']}")
                continue

            try:
                response = client.get(url)
                response.raise_for_status()
                img_path.write_bytes(response.content)
                image_paths.append(rel_path)
                downloaded += 1
            except Exception as exc:
                errors += 1
                image_paths.append("")
                failed.append(f"{exc}\t{row['id']}\t{url}")

            time.sleep(0.1)

    df = df.copy()
    df["image_path"] = image_paths + [""] * (len(df) - len(rows_to_download))

    log.info(f"Images — downloaded: {downloaded:,} | skipped: {skipped:,} | failed: {errors:,}")

    if failed:
        FAILED_LOG.write_text("\n".join(failed))
        log.warning(f"Failed images logged to {FAILED_LOG}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Pokémon TCG card data and images")
    parser.add_argument("--sets-only", action="store_true", help="Download metadata only, skip image downloads")
    parser.add_argument("--limit", type=int, default=None, help="Only download this many images (useful for testing)")
    args = parser.parse_args()

    download_json_data()
    df = parse_all_sets()

    if args.sets_only:
        df["image_path"] = ""
    else:
        df = download_images(df, limit=args.limit)

    df.to_csv(CARDS_CSV, index=False)
    log.info(f"Saved {len(df):,} rows to {CARDS_CSV}")

    print("DOWNLOAD SUMMARY")
    print(f"Total cards:     {len(df):,}")
    print(f"Total sets:      {df['set_id'].nunique():,}")
    print(f"Supertypes:      {df['supertype'].value_counts().to_dict()}")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"CSV saved to:    {CARDS_CSV}")
    if not args.sets_only:
        downloaded = (df["image_path"] != "").sum()
        print(f"Images on disk:  {downloaded:,} / {len(df):,}")


if __name__ == "__main__":
    main()
