# pokemon-card-pricing

A machine learning system that identifies a Pokémon card from a photo and looks up its current market value.


## Overview

**Model — Card Identification (Computer Vision)**
Takes a photo of a Pokémon card and identifies exactly which card it is: name, set, card number, and rarity.

**Price Lookup**
Uses the identified card to query PriceCharting for current market prices: ungraded, PSA 9, and PSA 10.

**Output:**
Point your camera at a card then get the card identity and current market prices in seconds.


## Project Structure

```
pokemon-card-pricing/
├── data/
│   ├── download_cards.py     — download card metadata + images from GitHub / CDN
│   ├── cards.csv             — 20,078 card records (local only)
│   └── images/               — 20,021 hi-res card images (local only)
├── model/
│   ├── constants.py          — shared constants 
│   ├── embedding_model.py    — EfficientNet-B0 + projection head
│   ├── dataset.py            — CardDataset, PKSampler, augmentation pipeline
│   ├── train.py              — training loop with online hard triplet mining
│   ├── build_index.py        — embed all cards -> FAISS nearest-neighbour index
│   └── inference.py          — CardIdentifier class for production use
├── notebooks/
│   └── train.ipynb           — self-contained Google Colab training notebook
├── main.py                   — entry point (Streamlit app, future phase)
└── pyproject.toml
```


## Setup

```bash
# Install dependencies (requires Python 3.12, uv)
uv sync

# Download card metadata and images (~14 GB, takes ~45 min)
uv run python data/download_cards.py

# Metadata only (no images, ~5 MB, instant)
uv run python data/download_cards.py --sets-only
```


## Training (Google Colab)

Open `notebooks/train.ipynb` in Google Colab with a **T4 GPU runtime**. The notebook:
1. Installs pinned dependencies
2. Clones this repo
3. Downloads card data from the CDN (or uses a Drive cache from a previous session)
4. Trains the embedding model (30 epochs, ~2–3 hours on T4)
5. Builds the FAISS index
6. Validates inference on a test card

**Local smoke test** (CPU, 200 cards, 2 epochs):
```bash
uv run python -m model.train \
    --data-dir data/ \
    --checkpoint-dir checkpoints/ \
    --epochs-total 2 \
    --limit 200
```

**Build the FAISS index** after training:
```bash
uv run python -m model.build_index \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/ \
    --output-dir index/
```


## How It Works

### Phase 1 — Data

**Source:** [`PokemonTCG/pokemon-tcg-data`](https://github.com/PokemonTCG/pokemon-tcg-data) — a static GitHub dump of all card metadata. Images are downloaded from the CDN at `images.pokemontcg.io`.

Using the static repo rather than the live API avoids timeout and reliability issues with `api.pokemontcg.io`.

| Stat | Value |
|------|-------|
| Cards in dataset | 20,078 |
| Images downloaded | 20,021 |
| Image failures | 57 (promo cards with no CDN image) |
| Supertypes | 16,961 Pokémon · 2,731 Trainer · 386 Energy |

Images are stored as `data/images/{set_id}/{number}_hires.png`.


### Phase 2 — Data Processing

**Augmentation pipeline** (training only, applied on the fly):

| Step | Transform | Purpose |
|------|-----------|---------|
| 1 | `Resize(256)` | Standardise input |
| 2 | `RandomPerspective(distortion=0.4, p=0.8)` | Simulate angled photo |
| 3 | `RandomRotation(±15°)` | Slight card tilt |
| 4 | `RandomResizedCrop(224, scale=0.7–1.0)` | Zoom / partial view |
| 5 | `ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.05)` | Lighting variation |
| 6 | `RandomGrayscale(p=0.05)` | Occasional desaturation |
| 7 | `GaussianBlur(σ=0.1–2.0)` | Focus/blur variation |
| 8 | `ToTensor` + `Normalize` (ImageNet stats) | Model input format |

**No `RandomHorizontalFlip`** — the set symbol and card number position are semantically meaningful.

Val/inference uses `Resize(224)` → `CenterCrop(224)` → `Normalize` only.


### Phase 3 — Modelling 

**Architecture: EfficientNet-B0 + Projection Head + FAISS**

```
Input (224×224 RGB)
  → EfficientNet-B0 backbone (pretrained ImageNet, classifier removed)
  → AdaptiveAvgPool2d → 1280-d feature vector
  → Linear(1280→512) → BatchNorm1d → ReLU → Linear(512→512)
  → L2 normalise → 512-d embedding
```

**Training objective:** Online batch-hard triplet loss
- Batch sampler: **PKSampler** — P=16 classes × K=4 augmented views = batch size 64
- For each anchor, find the *hardest positive* (max distance, same card) and *hardest negative* (min distance, different card) within the batch
- Loss = `mean(ReLU(d_pos − d_neg + 0.3))`

**Progressive unfreezing:**
- Epochs 1–5: backbone frozen, train projection head only (`lr=1e-3`)
- Epochs 6–30: last 2 EfficientNet blocks unfrozen (`lr=1e-4`, CosineAnnealingLR)

**Validation metric:** Top-1 and Top-3 retrieval accuracy (not triplet loss) — embed all val images, find nearest neighbours, check if the correct card is returned.

**FAISS index:** `IndexFlatL2(512)` — exact nearest-neighbour search over 20k embeddings (~39 MB).


### Phase 4 — Inference

```python
from model.inference import CardIdentifier
from PIL import Image

identifier = CardIdentifier(
    checkpoint="checkpoints/best_model.pt",
    index_dir="index/",
)

img = Image.open("my_card.jpg")
results = identifier.predict(img, k=3)

for r in results:
    print(f"#{r['rank']} {r['name']} ({r['set_name']} #{r['number']}) — score {r['score']:.3f}")
```

Returns top-3 candidates with a score in `(0, 1]` (higher = more similar). Top-3 is used rather than top-1 to handle the **reprint ambiguity** problem — many cards share identical artwork across sets (e.g., Pikachu appears in dozens of sets). The user can confirm the correct match.


### Phase 5 — Price Lookup (planned)

Query the **PriceCharting API** with the identified `(card_name, set, number)` to retrieve:
- Ungraded market price
- PSA 9 price
- PSA 10 price


### Phase 6 — Evaluation (planned)

- Top-1 and Top-3 accuracy on a real-world photo test set (~200–500 manually photographed cards)
- Accuracy by: set era (vintage vs. modern), rarity, visual similarity to other cards
- Domain gap analysis: confidence on clean reference images vs. real photos


### Phase 7 — Demo App (planned)

Streamlit app (`main.py`):
- Upload a photo of a card
- Model returns top-3 candidates with confidence scores
- User confirms the correct card
- App displays: name, set, number, rarity + ungraded / PSA 9 / PSA 10 prices


## Known Limitations

| Limitation | Mitigation |
|------------|-----------|
| **Reprint ambiguity** — many cards share identical artwork across sets | Return top-3 results; user confirms |
| **Domain gap** — trained on clean official art, tested on real photos | Aggressive augmentation during training |
| **57 promo cards** have no CDN image and are excluded from training | Listed in `data/failed_images.txt` |
| **Very low resolution or heavy glare** photos | Model may return low confidence scores |
