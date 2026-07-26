"""Presentation helpers for price data, shared by any UI layer.

Ported from main.py (Streamlit app) so they survive the move to the
FastAPI + HTML/JS frontend. Pure functions — no framework dependencies.
"""

CONDITION_ORDER = ["Near Mint", "Lightly Played", "Moderately Played", "Heavily Played", "Damaged"]


def fmt_price(val) -> str:
    return f"${val:.2f}" if val is not None else "—"


def condition_idx(cond: str) -> int:
    try:
        return CONDITION_ORDER.index(cond)
    except ValueError:
        return 99


def price_order_violations(printing_variants: list, conditions: list) -> list[tuple]:
    """Detect condition pairs where a worse condition is priced above a better one.

    Usually indicates sparse recent sales for one condition — worth surfacing
    to the user so they verify on TCGPlayer rather than trusting the number.
    """
    price_map = {v["condition"]: v["market"] for v in printing_variants if v.get("market") is not None}
    violations = []
    for i in range(len(conditions) - 1):
        better, worse = conditions[i], conditions[i + 1]
        p_better = price_map.get(better)
        p_worse  = price_map.get(worse)
        if p_better is not None and p_worse is not None and p_worse > p_better:
            violations.append((better, p_better, worse, p_worse))
    return violations
