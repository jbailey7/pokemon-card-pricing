from pricing.display import CONDITION_ORDER, condition_idx, fmt_price, price_order_violations


class TestFmtPrice:
    def test_formats_to_two_decimal_places(self):
        assert fmt_price(12.5) == "$12.50"

    def test_none_returns_dash(self):
        assert fmt_price(None) == "—"

    def test_zero(self):
        assert fmt_price(0.0) == "$0.00"

    def test_whole_number(self):
        assert fmt_price(100) == "$100.00"

    def test_rounding(self):
        assert fmt_price(9.999) == "$10.00"


class TestConditionIdx:
    def test_near_mint_is_first(self):
        assert condition_idx("Near Mint") == 0

    def test_damaged_is_last(self):
        assert condition_idx("Damaged") == len(CONDITION_ORDER) - 1

    def test_unknown_condition_returns_99(self):
        assert condition_idx("Mint") == 99

    def test_all_conditions_in_ascending_order(self):
        indices = [condition_idx(c) for c in CONDITION_ORDER]
        assert indices == sorted(indices)


class TestPriceOrderViolations:
    def test_no_violations_with_correct_order(self):
        variants = [
            {"condition": "Near Mint",        "market": 10.0},
            {"condition": "Lightly Played",   "market": 8.0},
            {"condition": "Moderately Played","market": 5.0},
        ]
        conditions = ["Near Mint", "Lightly Played", "Moderately Played"]
        assert price_order_violations(variants, conditions) == []

    def test_detects_single_violation(self):
        variants = [
            {"condition": "Near Mint",      "market": 2.0},
            {"condition": "Lightly Played", "market": 60.0},
        ]
        conditions = ["Near Mint", "Lightly Played"]
        violations = price_order_violations(variants, conditions)
        assert len(violations) == 1
        assert violations[0] == ("Near Mint", 2.0, "Lightly Played", 60.0)

    def test_detects_multiple_violations(self):
        variants = [
            {"condition": "Near Mint",        "market": 1.0},
            {"condition": "Lightly Played",   "market": 5.0},
            {"condition": "Moderately Played","market": 3.0},
            {"condition": "Heavily Played",   "market": 10.0},
        ]
        conditions = ["Near Mint", "Lightly Played", "Moderately Played", "Heavily Played"]
        violations = price_order_violations(variants, conditions)
        assert len(violations) == 2

    def test_skips_conditions_with_no_price(self):
        variants = [
            {"condition": "Near Mint",      "market": None},
            {"condition": "Lightly Played", "market": 60.0},
        ]
        conditions = ["Near Mint", "Lightly Played"]
        assert price_order_violations(variants, conditions) == []

    def test_equal_prices_are_not_a_violation(self):
        variants = [
            {"condition": "Near Mint",      "market": 5.0},
            {"condition": "Lightly Played", "market": 5.0},
        ]
        conditions = ["Near Mint", "Lightly Played"]
        assert price_order_violations(variants, conditions) == []

    def test_empty_conditions_returns_empty(self):
        assert price_order_violations([], []) == []
