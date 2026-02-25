"""
Real-photo evaluation — measures how well the model handles actual phone photos.

Setup:
    1. Place your card photos in eval/photos/  (jpg, jpeg, png, or webp)
    2. Fill in eval/ground_truth.csv:
           filename,card_id
           charizard_front.jpg,base1-4
           pikachu_sleeve.jpg,base1-58
       The card_id is {set_id}-{number}.  Look yours up in data/cards.csv.
    3. uv run python eval/evaluate.py

Output:
    Prints a summary to the terminal and saves per-photo results to
    eval/real_photo_results.csv.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.inference import CardIdentifier

CHECKPOINT   = "checkpoints/best_model.pt"
INDEX_DIR    = "index/"
PHOTOS_DIR   = Path("eval/photos")
GT_FILE      = Path("eval/ground_truth.csv")
RESULTS_FILE = Path("eval/real_photo_results.csv")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def classify_failure(preds: list, ground_truth_id: str, cards_df: pd.DataFrame) -> str:
    top1_id   = preds[0]["id"]
    gt_rows   = cards_df[cards_df["id"] == ground_truth_id]
    top1_rows = cards_df[cards_df["id"] == top1_id]

    if not gt_rows.empty and not top1_rows.empty:
        if gt_rows.iloc[0]["name"] == top1_rows.iloc[0]["name"]:
            return "reprint_confusion"

    if preds[0]["score"] < 0.4:
        return "low_confidence"

    return "wrong_card"


def main():
    if not GT_FILE.exists():
        sys.exit(f"Ground truth file not found: {GT_FILE}\n"
                 "Create it with columns: filename,card_id")

    photos = sorted(
        p for p in PHOTOS_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not photos:
        sys.exit(f"No image files found in {PHOTOS_DIR}\n"
                 "Add your card photos there and re-run.")

    with open(GT_FILE) as f:
        ground_truth = {row["filename"]: row["card_id"] for row in csv.DictReader(f)}

    if not ground_truth:
        sys.exit("ground_truth.csv is empty — add rows before running.")

    print("Loading model...")
    identifier = CardIdentifier(CHECKPOINT, INDEX_DIR)
    cards_df   = pd.read_csv("data/cards.csv", dtype=str).fillna("")

    rows       = []
    top1_hits  = 0
    top3_hits  = 0

    print(f"\nEvaluating {len(photos)} photo(s)...\n")

    for photo_path in photos:
        filename = photo_path.name

        if filename not in ground_truth:
            print(f"  [skip]  {filename} — not in ground_truth.csv")
            continue

        gt_id = ground_truth[filename]

        try:
            preds = identifier.predict(Image.open(photo_path), k=3)
        except Exception as e:
            print(f"  [error] {filename}: {e}")
            continue

        pred_ids    = [r["id"] for r in preds]
        top1_ok     = pred_ids[0] == gt_id
        top3_ok     = gt_id in pred_ids
        failure     = "" if top1_ok else classify_failure(preds, gt_id, cards_df)

        top1_hits  += top1_ok
        top3_hits  += top3_ok

        marker = "✓" if top1_ok else ("~" if top3_ok else "✗")
        print(f"  {marker}  {filename}  ->  {pred_ids[0]}  (score={preds[0]['score']:.3f})"
              + (f"  [{failure}]" if failure else ""))

        rows.append({
            "filename":        filename,
            "ground_truth_id": gt_id,
            "top1_id":         pred_ids[0],
            "top2_id":         pred_ids[1] if len(pred_ids) > 1 else "",
            "top3_id":         pred_ids[2] if len(pred_ids) > 2 else "",
            "top1_score":      round(preds[0]["score"], 4),
            "top1_correct":    top1_ok,
            "top3_correct":    top3_ok,
            "failure_reason":  failure,
        })

    n = len(rows)
    if n == 0:
        sys.exit("No photos matched entries in ground_truth.csv.")

    # Save results
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"\n{'─' * 50}")
    print(f"Real-photo evaluation  ({n} photos)")
    print(f"  Top-1:  {top1_hits / n:.1%}  ({top1_hits}/{n})")
    print(f"  Top-3:  {top3_hits / n:.1%}  ({top3_hits}/{n})")

    failures = [r for r in rows if not r["top1_correct"]]
    if failures:
        print(f"\nTop-1 failure breakdown ({len(failures)} misses):")
        for reason, count in Counter(r["failure_reason"] for r in failures).most_common():
            print(f"  {reason}: {count}")

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
