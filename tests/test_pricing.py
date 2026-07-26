import time
from types import SimpleNamespace

import pytest

import pricing.price_lookup as price_lookup
from pricing.price_lookup import (
    edition_sort_key,
    fetch_variants,
    is_foil,
    is_rare_edition,
    load_cache,
    normalize_number,
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

    # Modern foil-only rarities without "holo" in the name (fix #5)
    def test_rare_ultra(self):
        assert is_foil("Rare Ultra")

    def test_rare_secret(self):
        assert is_foil("Rare Secret")

    def test_illustration_rare(self):
        assert is_foil("Illustration Rare")

    def test_special_illustration_rare(self):
        assert is_foil("Special Illustration Rare")

    def test_double_rare(self):
        assert is_foil("Double Rare")

    def test_hyper_rare(self):
        assert is_foil("Hyper Rare")

    def test_amazing_rare(self):
        assert is_foil("Amazing Rare")

    def test_plain_rare_is_not_foil(self):
        assert not is_foil("Rare")

    def test_uncommon_is_not_foil(self):
        assert not is_foil("Uncommon")


class TestNormalizeNumber:
    def test_plain_number_unchanged(self):
        assert normalize_number("4") == "4"

    def test_strips_total_suffix(self):
        assert normalize_number("4/102") == "4"

    def test_strips_leading_zeros(self):
        assert normalize_number("004") == "4"

    def test_strips_both(self):
        assert normalize_number("004/102") == "4"

    def test_all_zeros_preserved(self):
        assert normalize_number("0") == "0"

    def test_promo_numbers_preserved(self):
        assert normalize_number("SWSH039") == "SWSH039"

    def test_tg_numbers_preserved(self):
        assert normalize_number("TG12") == "TG12"

    def test_int_input(self):
        assert normalize_number(4) == "4"

    def test_equivalence(self):
        assert normalize_number("4/102") == normalize_number("004")


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


CARD = {"id": "base1-4", "name": "Charizard", "set_name": "Base Set",
        "number": "4", "rarity": "Rare Holo"}


def make_result(**overrides) -> dict:
    result = {
        "variants":                 [{"label": "Unlimited Holofoil", "market": 12.50}],
        "matched_set_name":         "Base Set",
        "matched_number":           "4",
        "set_number_mismatch":      False,
        "printing_filter_fallback": False,
    }
    result.update(overrides)
    return result


class TestGetVariants:
    def test_returns_none_when_no_api_key(self, monkeypatch):
        monkeypatch.setattr(price_lookup, "API_KEY", "")
        assert get_variants(CARD) is None

    def test_returns_cached_result_without_network_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")

        result = make_result()
        save_cache({"base1-4": {"result": result, "ts": time.time()}})

        assert get_variants(CARD) == result

    def test_expired_cache_triggers_fetch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")

        # ts=0 is long expired
        save_cache({"base1-4": {"result": make_result(), "ts": 0}})

        fetched = []

        def fake_fetch(card):
            fetched.append(card["id"])
            return None

        monkeypatch.setattr(price_lookup, "fetch_variants", fake_fetch)

        get_variants(CARD)
        assert fetched == ["base1-4"]

    def test_legacy_cache_entry_triggers_refetch(self, monkeypatch, tmp_path):
        """Pre-mismatch-flag cache entries stored {'variants': ...} — must refetch."""
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")

        save_cache({"base1-4": {"variants": [{"label": "old"}], "ts": time.time()}})

        fetched = []
        monkeypatch.setattr(price_lookup, "fetch_variants",
                            lambda c: fetched.append(c["id"]) or make_result())

        result = get_variants(CARD)
        assert fetched == ["base1-4"]
        assert result == make_result()

    def test_successful_fetch_is_cached_in_new_shape(self, monkeypatch, tmp_path):
        monkeypatch.setattr(price_lookup, "API_KEY", "fake-key")
        monkeypatch.setattr(price_lookup, "CACHE_FILE", tmp_path / "cache.json")
        monkeypatch.setattr(price_lookup, "fetch_variants", lambda c: make_result())

        get_variants(CARD)
        entry = load_cache()["base1-4"]
        assert "result" in entry
        assert entry["result"] == make_result()


def make_fake_httpx(api_cards: list[dict]):
    """Stub httpx.get returning a JustTCG-shaped payload."""
    def fake_get(url, params=None, headers=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": api_cards},
        )
    return SimpleNamespace(get=fake_get)


class TestFetchVariants:
    def api_card(self, **overrides):
        card = {
            "set_name": "Base Set",
            "number":   "4",
            "variants": [
                {"printing": "Unlimited Holofoil", "condition": "Near Mint", "price": 100.0},
                {"printing": "Unlimited",          "condition": "Near Mint", "price": 40.0},
            ],
        }
        card.update(overrides)
        return card

    def test_exact_match_no_mismatch_flag(self, monkeypatch):
        monkeypatch.setattr(price_lookup, "httpx", make_fake_httpx([self.api_card()]))
        result = fetch_variants(CARD)
        assert result["set_number_mismatch"] is False
        assert result["matched_set_name"] == "Base Set"
        assert result["matched_number"] == "4"

    def test_number_format_difference_still_matches(self, monkeypatch):
        """API returns '4/102', we have '4' — normalization must bridge it (fix #6)."""
        monkeypatch.setattr(price_lookup, "httpx",
                            make_fake_httpx([self.api_card(number="4/102")]))
        result = fetch_variants(CARD)
        assert result["set_number_mismatch"] is False

    def test_wrong_set_falls_back_with_mismatch_flag(self, monkeypatch):
        """No set/number match anywhere → cards[0] fallback must be flagged (fix #1)."""
        monkeypatch.setattr(price_lookup, "httpx",
                            make_fake_httpx([self.api_card(set_name="Jungle", number="99")]))
        result = fetch_variants(CARD)
        assert result["set_number_mismatch"] is True
        assert result["matched_set_name"] == "Jungle"
        assert result["matched_number"] == "99"

    def test_number_only_match_flags_mismatch(self, monkeypatch):
        """Right number, wrong set → tier-2 fallback, still flagged."""
        monkeypatch.setattr(price_lookup, "httpx",
                            make_fake_httpx([self.api_card(set_name="Neo Genesis", number="4")]))
        result = fetch_variants(CARD)
        assert result["set_number_mismatch"] is True
        assert result["matched_set_name"] == "Neo Genesis"

    def test_foil_card_gets_foil_variants(self, monkeypatch):
        monkeypatch.setattr(price_lookup, "httpx", make_fake_httpx([self.api_card()]))
        result = fetch_variants(CARD)  # Rare Holo → want foil
        assert all("holo" in v["printing"].lower() or "foil" in v["printing"].lower()
                   for v in result["variants"])
        assert result["printing_filter_fallback"] is False

    def test_empty_printing_filter_sets_fallback_flag(self, monkeypatch):
        """Foil-only card but API has only non-foil variants → flagged, not silent (fix #5)."""
        api = self.api_card(variants=[
            {"printing": "Unlimited", "condition": "Near Mint", "price": 40.0},
        ])
        monkeypatch.setattr(price_lookup, "httpx", make_fake_httpx([api]))
        result = fetch_variants(CARD)  # Rare Holo → wants foil, none exist
        assert result["printing_filter_fallback"] is True
        assert len(result["variants"]) == 1  # fell back to all variants

    def test_no_results_returns_none(self, monkeypatch):
        monkeypatch.setattr(price_lookup, "httpx", make_fake_httpx([]))
        assert fetch_variants(CARD) is None

    def test_request_error_returns_none(self, monkeypatch):
        def raise_get(*a, **kw):
            raise RuntimeError("connection failed")
        monkeypatch.setattr(price_lookup, "httpx", SimpleNamespace(get=raise_get))
        assert fetch_variants(CARD) is None
