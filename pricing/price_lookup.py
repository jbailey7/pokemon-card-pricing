import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

CACHE_FILE = Path("pricing/price_cache.json")
CACHE_TTL  = 12 * 60 * 60  # 12 hours

API_BASE = "https://api.justtcg.com/v1"
API_KEY  = os.getenv("JUSTTCG_API_KEY", "")

FOIL_KEYWORDS     = ("holo", "foil")
RARE_EDITION_KEYS = ("1st", "1sted", "shadowless")


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def is_foil(rarity: str) -> bool:
    return "holo" in rarity.lower()


def is_rare_edition(printing: str) -> bool:
    p = printing.lower()
    return any(kw in p for kw in RARE_EDITION_KEYS)


def set_match(our_name: str, api_name: str) -> bool:
    def norm(s):
        return "".join(c for c in s.lower() if c.isalnum())
    a, b = norm(our_name), norm(api_name)
    return a in b or b in a


def edition_sort_key(printing: str) -> tuple:
    # Unlimited before 1st Edition / Shadowless
    return (1 if is_rare_edition(printing) else 0, printing.lower())


def to_variant_dict(v: dict) -> dict:
    return {
        "label":      v.get("printing", "Unknown"),
        "market":     v.get("price"),
        "low_90d":    v.get("minPrice90d"),
        "high_90d":   v.get("maxPrice90d"),
        "change_90d": v.get("priceChange90d"),
        "condition":  v.get("condition", "Near Mint"),
        "printing":   v.get("printing", ""),
        "updated_at": v.get("lastUpdated"),
    }


def fetch_variants(card: dict) -> list[dict] | None:
    name      = card["name"]
    number    = str(card["number"])
    set_name  = card["set_name"]
    want_foil = is_foil(card.get("rarity", ""))

    params  = {"game": "pokemon", "q": name}
    headers = {"X-API-Key": API_KEY}

    try:
        resp = httpx.get(f"{API_BASE}/cards", params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        log.warning("request failed for %s: %s", card['id'], e)
        return None

    cards = body.get("data", body) if isinstance(body, dict) else body
    if not isinstance(cards, list) or not cards:
        log.warning("no results for '%s'", name)
        return None

    log.debug("%d result(s) for '%s' — looking for #%s in '%s'", len(cards), name, number, set_name)

    best = next(
        (c for c in cards
         if str(c.get("number", "")) == number and set_match(set_name, c.get("set_name", ""))),
        None,
    )
    if best is None:
        best = next((c for c in cards if str(c.get("number", "")) == number), None)
    if best is None:
        best = cards[0]

    matched_set = best.get("set_name", "?")
    matched_num = best.get("number", "?")
    if matched_set != set_name or matched_num != number:
        log.warning("wanted %s #%s, matched %s #%s", set_name, number, matched_set, matched_num)

    raw = best.get("variants") or [best]
    log.debug("%d variant(s): %s", len(raw), [(v.get('condition'), v.get('printing')) for v in raw])

    if want_foil:
        typed = [v for v in raw if any(kw in v.get("printing", "").lower() for kw in FOIL_KEYWORDS)]
    else:
        typed = [v for v in raw if not any(kw in v.get("printing", "").lower() for kw in FOIL_KEYWORDS)]

    if not typed:
        typed = raw

    condition_order = ["near mint", "lightly played", "moderately played", "heavily played", "damaged"]

    def sort_key(v):
        p = v.get("printing", "")
        c = v.get("condition", "").lower()
        c_idx = next((i for i, s in enumerate(condition_order) if s in c), 99)
        return (edition_sort_key(p)[0], p.lower(), c_idx)

    typed.sort(key=sort_key)

    variants = [to_variant_dict(v) for v in typed]
    for v in variants:
        log.debug("  %s: market=%s", v['label'], v['market'])

    return variants


def get_variants(card: dict) -> list[dict] | None:
    if not API_KEY:
        return None

    cache   = load_cache()
    card_id = card["id"]
    now     = time.time()

    entry = cache.get(card_id)
    if entry and now - entry.get("ts", 0) < CACHE_TTL:
        log.debug("cache hit for %s", card_id)
        return entry.get("variants")

    variants = fetch_variants(card)
    if variants is not None:
        cache[card_id] = {"variants": variants, "ts": now}
        save_cache(cache)

    return variants


def tcgplayer_url(card: dict) -> str:
    params = urlencode({
        "q": f"{card['name']} {card['set_name']}",
        "productLineName": "pokemon",
    })
    return f"https://www.tcgplayer.com/search/pokemon/product?{params}"
