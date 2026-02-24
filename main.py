from pathlib import Path

import streamlit as st
from PIL import Image

from model.inference import CardIdentifier
from pricing.price_lookup import tcgplayer_url

CHECKPOINT = "checkpoints/best_model.pt"
INDEX_DIR  = "index/"


@st.cache_resource
def load_identifier():
    return CardIdentifier(CHECKPOINT, INDEX_DIR)


def main():
    st.set_page_config(page_title="Pokémon Card Pricer", layout="centered")
    st.title("Pokémon Card Pricer")
    st.caption("Upload a photo of a card to identify it and find it on TCGPlayer.")

    for key in ("file_id", "results", "confirmed"):
        if key not in st.session_state:
            st.session_state[key] = None

    uploaded = st.file_uploader(
        "Card photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        for key in ("file_id", "results", "confirmed"):
            st.session_state[key] = None
        return

    # Reset state when a new file is uploaded
    file_id = (uploaded.name, uploaded.size)
    if file_id != st.session_state.file_id:
        st.session_state.file_id   = file_id
        st.session_state.results   = None
        st.session_state.confirmed = None

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

    # Stage 1: pick the correct card from the top-3 candidates
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
                        card_img_path = Path(r["img_path"])
                        if card_img_path.exists():
                            st.image(str(card_img_path), use_container_width=True)
                    with c2:
                        st.write(f"**{r['name']}**")
                        st.write(f"{r['set_name']} · #{r['number']}")
                        st.write(r["rarity"])
                        st.progress(r["score"], text=f"Confidence: {r['score']:.0%}")
                        if st.button("This is my card", key=f"pick_{i}", use_container_width=True):
                            st.session_state.confirmed = r
                            st.rerun()
        return

    # Stage 2: show the confirmed card and link to TCGPlayer
    card = st.session_state.confirmed

    c_img, c_info = st.columns([1, 2])
    with c_img:
        card_img_path = Path(card["img_path"])
        if card_img_path.exists():
            st.image(str(card_img_path), use_container_width=True)

    with c_info:
        st.subheader(card["name"])
        st.write(f"{card['set_name']} · #{card['number']} · {card['rarity']}")
        st.link_button("View prices on TCGPlayer", tcgplayer_url(card), use_container_width=True)

    if st.button("← Try another card"):
        st.session_state.results   = None
        st.session_state.confirmed = None
        st.rerun()


if __name__ == "__main__":
    main()
