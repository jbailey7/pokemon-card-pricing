"""
Look up a card's ID in data/cards.csv by name, set, or card number.

Usage:
    uv run python eval/find_card.py "Charizard"
    uv run python eval/find_card.py "Charizard" --set "Base Set"
    uv run python eval/find_card.py "Charizard" --number 4
    uv run python eval/find_card.py --number 4 --set "Base Set"

All text matches are case-insensitive partial matches.
The card_id you get back (e.g. base1-4) is what goes in ground_truth.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CARDS_CSV  = Path("data/cards.csv")
MAX_ROWS   = 25


def main():
    parser = argparse.ArgumentParser(
        description="Find a card ID in data/cards.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("name",       nargs="?",       help="Card name (partial, case-insensitive)")
    parser.add_argument("--set", "-s", dest="set_name", help="Set name  (partial, case-insensitive)")
    parser.add_argument("--number", "-n",               help="Card number, e.g. 4 or GX45")
    args = parser.parse_args()

    if not args.name and not args.set_name and not args.number:
        parser.print_help()
        sys.exit(1)

    if not CARDS_CSV.exists():
        sys.exit(f"cards.csv not found at {CARDS_CSV}\nRun data/download_cards.py first.")

    df = pd.read_csv(CARDS_CSV, dtype=str).fillna("")

    if args.name:
        df = df[df["name"].str.contains(args.name, case=False, na=False)]
    if args.set_name:
        df = df[df["set_name"].str.contains(args.set_name, case=False, na=False)]
    if args.number:
        df = df[df["number"].str.lower() == args.number.lower()]

    if df.empty:
        print("No cards found. Try broadening your search.")
        sys.exit(0)

    total = len(df)
    display = df[["id", "name", "set_name", "number", "rarity"]].head(MAX_ROWS)
    print(display.to_string(index=False))

    if total > MAX_ROWS:
        print(f"\n... {total - MAX_ROWS} more results not shown. Use --set or --number to narrow.")
    elif total == 1:
        card_id = df.iloc[0]["id"]
        print(f"\n  card_id: {card_id}")
    else:
        print(f"\n{total} results. Add --set or --number to narrow to one card.")


if __name__ == "__main__":
    main()
