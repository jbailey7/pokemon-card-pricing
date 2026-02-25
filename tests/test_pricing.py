import time

import pytest

import pricing.price_lookup as price_lookup
from pricing.price_lookup import (
    edition_sort_key,
    is_foil,
    is_rare_edition,
    load_cache,
    save_cache,
    get_variants,
    set_match,
    tcgplayer_url,
    to_variant_dict,
)


class TestIsFoil:
    def test_holo_rarity(self):
        assert is_foil("Rare Holo")

    def test_non_holo_rarity(self):
        assert not is_foil("Rare")

    def test_common_card(self):
        assert not is_foil("Common")

    def test_case_insensitive(self):
        assert is_foil("RARE HOLO V")

    def test_holo_in_middle(self):
        assert is_foil("Rare Holo VMAX")


class TestIsRareEdition:
    def test_1st_edition(self):
        assert is_rare_edition("1st Edition Holofoil")

    def test_shadowless(self):
        assert is_rare_edition("Shadowless Holofoil")

    def test_unlimited_is_not_rare(self):
        assert not is_rare_edition("Unlimited Holofoil")

    def test_case_insensitive(self):
        assert is_rare_edition("SHADOWLESS")

    def test_plain_string(self):
        assert not is_rare_edition("Normal")


class TestSetMatch:
    def test_exact_match(self):
        assert set_match("Base Set", "Base Set")

    def test_our_name_subset_of_api_name(self):
        # "baseset" is in "baseset2"
        assert set_match("Base Set", "Base Set 2")

    def test_no_match(self):
        assert not set_match("Jungle", "Base Set")

    def test_strips_special_characters(self):
        # "swordshield" == "swordshield" after stripping "&" and space
        assert set_match("Sword & Shield", "Sword Shield")

    def test_case_insensitive(self):
        assert set_match("base set", "BASE SET")


class TestEditionSortKey:
    def test_unlimited_sorts_before_1st_edition(self):
        assert edition_sort_key("Unlimited Holofoil") < edition_sort_key("1st Edition Holofoil")

    def test_unlimited_sorts_before_shadowless(self):
        assert edition_sort_key("Unlimited") < edition_sort_key("Shadowless")

    def test_1st_and_shadowless_both_sort_after_unlimited(self):
        unlimited = edition_sort_key("Unlimited")
        assert unlimited < edition_sort_key("1st Edition")
        assert unlimited < edition_sort_key("Shadowless Holofoil")


class TestToVariantDict:
    def test_maps_all_fields(self):
        raw = {
            "printing":       "Unlimited Holofoil",
            "price":          12.50,
            "minPrice90d":    9.00,
            "maxPrice90d":    18.00,
            "priceChange90d": -8.3,
            "condition":      "Near Mint",
            "lastUpdated":    1740000000,
        }
        v = to_variant_dict(raw)
        assert v["label"]      == "Unlimited Holofoil"
        assert v["market"]     == 12.50
        assert v["low_90d"]    == 9.00
        assert v["high_90d"]   == 18.00
        assert v["change_90d"] == -8.3
        assert v["condition"]  == "Near Mint"
        assert v["printing"]   == "Unlimited Holofoil"
        assert v["updated_at"] == 1740000000

    def test_missing_price_fields_are_none(self):
        v = to_variant_dict({})
        assert v["market"]     is None
        assert v["low_90d"]    is None
        assert v["high_90d"]   is None
        assert v["change_90d"] is None

    def test_default_condition_is_near_mint(self):
        v = to_variant_dict({})
        assert v["condition"] == "Near Mint"

    def test_default_label_is_unknown(self):
        v = to_variant_dict({})
        assert v["label"] == "Unknown"


class TestTcgplayerUrl:
    def test_contains_card_name(self):
        card = {"name": "Charizard", "set_name": "Base Set"}
        assert "Charizard" in tcgplayer_url(card)

    def test_contains_game_parameter(self):
        card = {"name": "Pikachu", "set_name": "Base Set"}
        assert "pokemon" in tcgplayer_url(card)

    def test_no_raw_spaces(self):
        card = {"name": "Dark Charizard", "set_name": "Team Rocket"}
        assert " " not in tcgplayer_url(card)

    def test_correct_base_url(self):
        card = {"name": "Mewtwo", "set_name": "Base Set"}
        url = tcgplayer_url(card)
        assert url.startswith("https://www.tcgplayer.com/search/pokemon/product?")


class TestCache:
    def test_load_cache_returns_empty_when_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")
        assert load_cache() == {}

    def test_save_and_load_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")
        data = {"base1-4": {"variants": [{"label": "Unlimited"}], "ts": 1000.0}}
        save_cache(data)
        assert load_cache() == data

    def test_load_cache_returns_empty_on_corrupt_file(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("not valid json")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", cache_path)
        assert load_cache() == {}


class TestGetVariants:
    def test_returns_none_when_no_api_key(self, monkeypatch):
        monkeypatch.setattr(price_lookup, "API_KEY", "")
        card = {"id": "base1-4", "name": "Charizard", "set_name": "Base Set",
                "number": "4", "rarity": "Rare Holo"}
        assert get_variants(card) is None

    def test_returns_cached_result_without_network_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")

        variants = [{"label": "Unlimited Holofoil", "market": 12.50}]
        save_cache({"base1-4": {"variants": variants, "ts": time.time()}})

        card = {"id": "base1-4", "name": "Charizard", "set_name": "Base Set",
                "number": "4", "rarity": "Rare Holo"}
        assert get_variants(card) == variants

    def test_expired_cache_triggers_fetch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")

        # ts=0 is long expired
        save_cache({"base1-4": {"variants": [], "ts": 0}})

        fetched = []

        def fake_fetch(card):
            fetched.append(card["id"])
            return None

        monkeypatch.setattr(price_lookup, "fetch_variants", fake_fetch)

        card = {"id": "base1-4", "name": "Charizard", "set_name": "Base Set",
                "number": "4", "rarity": "Rare Holo"}
        get_variants(card)
        assert fetched == ["base1-4"]
