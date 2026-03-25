from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

from model.inference import CardIdentifier
from pricing.price_lookup import get_variants, tcgplayer_url, _api_key

CHECKPOINT = "checkpoints/best_model.pt"
INDEX_DIR  = "index/"


@st.cache_resource
def load_identifier():
    return CardIdentifier(CHECKPOINT, INDEX_DIR)


CONDITION_ORDER = ["Near Mint", "Lightly Played", "Moderately Played", "Heavily Played", "Damaged"]


def fmt_price(val) -> str:
    return f"${val:.2f}" if val is not None else "—"


def condition_idx(cond: str) -> int:
    try:
        return CONDITION_ORDER.index(cond)
    except ValueError:
        return 99


def price_order_violations(printing_variants: list, conditions: list) -> list[tuple]:
    price_map = {v["condition"]: v["market"] for v in printing_variants if v.get("market") is not None}
    violations = []
    for i in range(len(conditions) - 1):
        better, worse = conditions[i], conditions[i + 1]
        p_better = price_map.get(better)
        p_worse  = price_map.get(worse)
        if p_better is not None and p_worse is not None and p_worse > p_better:
            violations.append((better, p_better, worse, p_worse))
    return violations


def main():
    st.set_page_config(page_title="Pokémon Card Pricer", layout="centered")
    st.title("Pokémon Card Pricer")
    st.caption("Upload a photo of a card to identify it and check current prices.")

    for key in ("file_id", "results", "confirmed", "variants"):
        if key not in st.session_state:
            st.session_state[key] = None

    uploaded = st.file_uploader(
        "Card photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        for key in ("file_id", "results", "confirmed", "variants"):
            st.session_state[key] = None
        return

    file_id = (uploaded.name, uploaded.size)
    if file_id != st.session_state.file_id:
        st.session_state.file_id   = file_id
        st.session_state.results   = None
        st.session_state.confirmed = None
        st.session_state.variants  = None

    image = Image.open(uploaded)

    if st.session_state.results is None:
        try:
            identifier = load_identifier()
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return
        with st.spinner("Identifying card..."):
            st.session_state.results = identifier.predict(image, k=3)

    results = st.session_state.results

    if st.session_state.confirmed is None:
        col_img, col_matches = st.columns([1, 2])

        with col_img:
            st.image(image, caption="Your photo", use_container_width=True)

        with col_matches:
            st.subheader("Which card is this?")
            for i, r in enumerate(results):
                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if Path(r["img_path"]).exists():
                            st.image(r["img_path"], use_container_width=True)
                    with c2:
                        st.write(f"**{r['name']}**")
                        st.write(f"{r['set_name']} · #{r['number']}")
                        st.write(r["rarity"])
                        st.progress(r["score"], text=f"Confidence: {r['score']:.0%}")
                        if st.button("This is my card", key=f"pick_{i}", use_container_width=True):
                            st.session_state.confirmed = r
                            st.rerun()
        return

    card = st.session_state.confirmed

    if st.session_state.variants is None:
        with st.spinner("Fetching prices..."):
            st.session_state.variants = get_variants(card)

    all_variants = st.session_state.variants

    c_img, c_info = st.columns([1, 2])

    with c_img:
        if Path(card["img_path"]).exists():
            st.image(card["img_path"], use_container_width=True)

    with c_info:
        st.subheader(card["name"])
        st.write(f"{card['set_name']} · #{card['number']} · {card['rarity']}")

        if all_variants:
            printings = list(dict.fromkeys(v["printing"] for v in all_variants))

            if len(printings) > 1:
                sel_printing = st.radio("Edition / printing", printings, horizontal=True)
            else:
                sel_printing = printings[0]

            printing_variants = [v for v in all_variants if v["printing"] == sel_printing]
            conditions = sorted(
                list(dict.fromkeys(v["condition"] for v in printing_variants)),
                key=condition_idx,
            )

            if len(conditions) > 1:
                sel_condition = st.radio("Condition", conditions, horizontal=True)
            else:
                sel_condition = conditions[0]

            prices = next(
                (v for v in printing_variants if v["condition"] == sel_condition),
                printing_variants[0],
            )

            market = prices.get("market")
            lo     = prices.get("low_90d")
            hi     = prices.get("high_90d")
            change = prices.get("change_90d")

            m1, m2, m3 = st.columns(3)
            with m1:
                delta_str = f"{change:+.1f}% (90d)" if change is not None else None
                st.metric("Market price", fmt_price(market), delta=delta_str)
            with m2:
                st.metric("90d low", fmt_price(lo))
            with m3:
                st.metric("90d high", fmt_price(hi))

            violations = price_order_violations(printing_variants, conditions)
            if violations:
                lines = [f"{worse} ({fmt_price(p_worse)}) > {better} ({fmt_price(p_better)})"
                         for better, p_better, worse, p_worse in violations]
                st.warning(
                    "Prices appear out of order: " + "; ".join(lines) + ". "
                    "This usually means sparse recent sales for one condition — verify on TCGPlayer.",
                    icon="⚠️",
                )

            updated_at = prices.get("updated_at")
            if updated_at:
                as_of = datetime.fromtimestamp(updated_at, tz=timezone.utc).strftime("%-d %b %Y")
                st.caption(f"as of {as_of} · via JustTCG")
            else:
                st.caption("via JustTCG")
        elif not _api_key():
            st.info("No price data — add a JUSTTCG_API_KEY to .env to enable pricing.")
        else:
            st.warning("No price data returned for this card. It may not be in the JustTCG database, or the card number/set couldn't be matched confidently.")

        st.link_button("View on TCGPlayer", tcgplayer_url(card), use_container_width=True)

    if st.button("← Try another card"):
        st.session_state.results   = None
        st.session_state.confirmed = None
        st.session_state.variants  = None
        st.rerun()


if __name__ == "__main__":
    main()
