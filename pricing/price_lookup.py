import json
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = "https://api.pokemontcg.io/v2"

# When multiple price variants exist, try them in this order depending on rarity
HOLO_VARIANTS   = ["holofoil", "1stEditionHolofoil", "normal", "reverseHolofoil", "1stEditionNormal"]
NORMAL_VARIANTS = ["normal", "1stEditionNormal", "reverseHolofoil", "holofoil", "1stEditionHolofoil"]


class PriceLookup:
    def __init__(self, cache_path: str | Path = "pricing/price_cache.json") -> None:
        self.api_key = os.environ.get("POKEMONTCG_API_KEY")
        if not self.api_key:
            raise ValueError("POKEMONTCG_API_KEY is not set in the environment")

        self.cache_path = Path(cache_path)
        self.cache: dict = self.load_cache()

    def load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def fetch_card(self, card_id: str) -> dict | None:
        resp = httpx.get(
            f"{BASE_URL}/cards/{card_id}",
            headers={"X-Api-Key": self.api_key},
            timeout=10,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("data")

    def pick_prices(self, tcgplayer: dict, rarity: str) -> dict | None:
        prices = tcgplayer.get("prices", {})
        if not prices:
            return None

        is_holo = "holo" in rarity.lower() or rarity.lower() in ("rare ultra", "rare secret")
        variants = HOLO_VARIANTS if is_holo else NORMAL_VARIANTS

        for variant in variants:
            if variant in prices and prices[variant]:
                p = prices[variant]
                return {
                    "variant": variant,
                    "low":     p.get("low"),
                    "mid":     p.get("mid"),
                    "high":    p.get("high"),
                    "market":  p.get("market"),
                }

        return None

    def lookup(self, card: dict) -> dict | None:
        card_id = card["id"]

        if card_id in self.cache:
            log.debug("Cache hit for %s", card_id)
            return self.cache[card_id]

        try:
            data = self.fetch_card(card_id)
        except httpx.HTTPError as exc:
            log.error("API request failed for %s: %s", card_id, exc)
            return None

        if data is None:
            log.warning("Card not found: %s", card_id)
            return None

        tcgplayer = data.get("tcgplayer")
        if not tcgplayer:
            log.warning("No TCGPlayer data for %s", card_id)
            return None

        prices = self.pick_prices(tcgplayer, card.get("rarity", ""))
        if prices is None:
            log.warning("No price data for %s", card_id)
            return None

        result = {
            "tcgplayer_url": tcgplayer.get("url"),
            "updated_at":    tcgplayer.get("updatedAt"),
            **prices,
        }

        self.cache[card_id] = result
        self.save_cache()

        return result
