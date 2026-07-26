"""
Phase 0 verification: JustTCG set/number match-rate spot-check.

Measures how often our card metadata (name, number, set_name) finds an exact
set+number match in the JustTCG API — i.e. how often pricing avoids the
`cards[0]` fallback that can silently return the wrong card's price.

Run locally (needs JUSTTCG_API_KEY in .env):
    uv run python eval/spot_check_pricing.py

Samples ~30 cards stratified across sets from data/cards.csv. If cards.csv is
missing, falls back to a small built-in vintage sample.

Records per-card outcome:
    exact        — set AND number matched (the good path)
    number_only  — number matched but set didn't (fallback tier 2)
    first_result — neither matched; would use cards[0] (price is suspect)
    no_results   — API returned nothing for the name query
    error        — request failed

Interpretation (per CLAUDE.md Phase 0): if the exact-match rate is poor,
number normalization (pre-existing fix #6) is critical, not nice-to-have.
Compare raw vs normalized matching in the output to quantify exactly that.
"""

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from pricing.price_lookup import API_BASE, API_KEY, set_match

SAMPLE_SIZE = 30
SEED = 42

# Used only if data/cards.csv is absent. Vintage cards with well-known numbers.
FALLBACK_SAMPLE = [
    {"name": "Alakazam",       "number": "1",  "set_name": "Base"},
    {"name": "Blastoise",      "number": "2",  "set_name": "Base"},
    {"name": "Charizard",      "number": "4",  "set_name": "Base"},
    {"name": "Mewtwo",         "number": "10", "set_name": "Base"},
    {"name": "Pikachu",        "number": "58", "set_name": "Base"},
    {"name": "Snorlax",        "number": "11", "set_name": "Jungle"},
    {"name": "Scyther",        "number": "10", "set_name": "Jungle"},
    {"name": "Aerodactyl",     "number": "1",  "set_name": "Fossil"},
    {"name": "Zapdos",         "number": "15", "set_name": "Fossil"},
    {"name": "Dark Charizard", "number": "4",  "set_name": "Team Rocket"},
]


def normalize_number(num: str) -> str:
    """Strip '/total' suffix and leading zeros: '004/102' -> '4'."""
    base = str(num).split("/")[0].strip()
    stripped = base.lstrip("0")
    return stripped if stripped else base


def sample_cards() -> list[dict]:
    csv_path = Path(__file__).parent.parent / "data" / "cards.csv"
    if not csv_path.exists():
        print(f"NOTE: {csv_path} not found — using built-in vintage sample "
              f"({len(FALLBACK_SAMPLE)} cards). Modern sets untested!\n")
        return FALLBACK_SAMPLE

    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # Stratify: sample one card per set from up to SAMPLE_SIZE sets,
    # spread across the full set list (old and new).
    sets = sorted(df["set_id"].unique())
    step = max(1, len(sets) // SAMPLE_SIZE)
    chosen_sets = sets[::step][:SAMPLE_SIZE]
    rows = []
    for s in chosen_sets:
        rows.append(df[df["set_id"] == s].sample(1, random_state=SEED).iloc[0])
    return [{"name": r["name"], "number": r["number"], "set_name": r["set_name"]}
            for r in rows]


def classify(card: dict, api_cards: list[dict], use_norm: bool) -> str:
    num = normalize_number(card["number"]) if use_norm else str(card["number"])

    def api_num(c):
        return normalize_number(c.get("number", "")) if use_norm else str(c.get("number", ""))

    if any(api_num(c) == num and set_match(card["set_name"], c.get("set_name", ""))
           for c in api_cards):
        return "exact"
    if any(api_num(c) == num for c in api_cards):
        return "number_only"
    return "first_result"


def fetch_card_results(name: str) -> list | dict:
    """Query the API by card name, backing off and retrying on 429.

    The free tier throttles per-minute bursts — a 429 here is a pacing
    signal, not a failure, so wait it out (honouring Retry-After when
    present) rather than recording an error."""
    for attempt in range(5):
        resp = httpx.get(f"{API_BASE}/cards",
                         params={"game": "pokemon", "q": name},
                         headers={"X-API-Key": API_KEY}, timeout=15)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After") or 0) or 30 * (attempt + 1)
            print(f"  [429] rate limited — waiting {wait}s before retrying {name}...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("still rate limited after 5 retries — daily cap likely hit")


def main():
    if not API_KEY:
        sys.exit("JUSTTCG_API_KEY not set — add it to .env first.")

    cards = sample_cards()
    print(f"Checking {len(cards)} cards against JustTCG...\n")

    outcomes_raw: Counter = Counter()
    outcomes_norm: Counter = Counter()
    rows = []

    for card in cards:
        try:
            body = fetch_card_results(card["name"])
            api_cards = body.get("data", body) if isinstance(body, dict) else body
        except Exception as e:
            print(f"  [error] {card['name']}: {e}")
            outcomes_raw["error"] += 1
            outcomes_norm["error"] += 1
            continue

        if not isinstance(api_cards, list) or not api_cards:
            result_raw = result_norm = "no_results"
        else:
            result_raw = classify(card, api_cards, use_norm=False)
            result_norm = classify(card, api_cards, use_norm=True)

        outcomes_raw[result_raw] += 1
        outcomes_norm[result_norm] += 1
        marker = {"exact": "✓", "number_only": "~"}.get(result_norm, "✗")
        print(f"  {marker}  {card['name']:<24} #{card['number']:<8} {card['set_name']:<28} "
              f"raw={result_raw:<13} normalized={result_norm}")
        rows.append({**card, "raw": result_raw, "normalized": result_norm})
        time.sleep(2)  # pace below the free tier's per-minute burst limit

    n = sum(outcomes_norm.values())
    print(f"\n{'─' * 60}")
    for label, counts in (("Raw matching (current code)", outcomes_raw),
                          ("Normalized numbers (fix #6)", outcomes_norm)):
        exact = counts.get("exact", 0)
        print(f"{label}:")
        for k in ("exact", "number_only", "first_result", "no_results", "error"):
            if counts.get(k):
                print(f"    {k:<13} {counts[k]:>3}  ({counts[k] / n:.0%})")
        print(f"    → exact-match rate: {exact / n:.0%}\n")

    print("Record these rates in the CLAUDE.md Session log (Phase 0).")


if __name__ == "__main__":
    main()
