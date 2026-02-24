# pokemon-card-pricing

A machine learning system that identifies a Pokémon card from a photo and looks up its current market price.


## Overview

**Card Identification (Computer Vision)**
Takes a photo of a Pokémon card and identifies exactly which card it is: name, set, card number, and rarity.

**Price Lookup**
Fetches Near Mint market price and 90-day low/high from the JustTCG API (free tier), then links to the card's TCGPlayer listing.

**Output:**
Point your camera at a card and get the identity, current market price, 90-day range, and a direct TCGPlayer link.


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
│   ├── build_index.py        — embed all cards -> numpy embeddings array
│   └── inference.py          — CardIdentifier class
├── pricing/
│   ├── price_lookup.py       — JustTCG API client + TCGPlayer URL builder
│   └── price_cache.json      — 12-hour price cache (local only)
├── main.py                   — Streamlit app
└── pyproject.toml
```


## Setup

```bash
# Install dependencies (requires Python 3.12, uv)
uv sync

# Add your JustTCG API key — free at https://justtcg.com/dashboard
cp .env.example .env
# edit .env and set JUSTTCG_API_KEY=...

# Download card metadata and images (~14 GB, takes ~45 min)
uv run python data/download_cards.py

# Metadata only (no images, ~5 MB, instant)
uv run python data/download_cards.py --sets-only
```

The app works without an API key — pricing will just be unavailable and only the TCGPlayer link is shown.


## Training

Training was done locally on the full 20k card dataset.

```bash
uv run python -m model.train \
    --data-dir data/ \
    --checkpoint-dir checkpoints/ \
    --epochs-total 30
```

**Smoke test** (CPU, 200 cards, 2 epochs):
```bash
uv run python -m model.train \
    --data-dir data/ \
    --checkpoint-dir checkpoints/ \
    --epochs-total 2 \
    --limit 200
```

**Build the embeddings index** after training:
```bash
uv run python -m model.build_index \
    --checkpoint checkpoints/best_model.pt \
    --data-dir data/ \
    --output-dir index/
```


## How It Works

### Data

**Source:** [`PokemonTCG/pokemon-tcg-data`](https://github.com/PokemonTCG/pokemon-tcg-data) — a static GitHub dump of all card metadata. Images are downloaded from the CDN at `images.pokemontcg.io`.

Using the static repo rather than the live API avoids timeout and reliability issues with `api.pokemontcg.io`.

| Stat | Value |
|------|-------|
| Cards in dataset | 20,078 |
| Images downloaded | 20,021 |
| Image failures | 57 (promo cards with no CDN image) |
| Supertypes | 16,961 Pokémon · 2,731 Trainer · 386 Energy |

Images are stored as `data/images/{set_id}/{number}_hires.png`.


### Data Processing

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


### Model

**Architecture: EfficientNet-B0 + Projection Head + numpy nearest-neighbour search**

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

**Validation metric:** Top-1 and Top-3 retrieval accuracy — embed all val images, find nearest neighbours, check if the correct card is returned.

**Nearest-neighbour search:** embeddings are saved as `index/card_embeddings.npy` (20k × 512 float32). At inference time a single matrix multiply against the query vector ranks all cards by cosine similarity in well under a second on CPU.


### Inference

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

Returns top-3 candidates with a confidence score in `(0, 1]`. Top-3 is used rather than top-1 to handle the **reprint ambiguity** problem — many cards share identical artwork across sets. The user confirms the correct match before prices are fetched.


### Price Lookup

After the user confirms which card they have, the app fetches Near Mint pricing from the **JustTCG API** (free tier: 1,000 req/month) and displays:

- **Market price** — current Near Mint market value with 90-day % change
- **90-day range** — min and max over the last 90 days
- **TCGPlayer link** — direct search link for the card

```python
from pricing.price_lookup import get_prices, tcgplayer_url

prices = get_prices(card)  # card is a dict from CardIdentifier.predict()
# {
#   "market":     12.50,
#   "low_90d":    9.00,
#   "high_90d":   18.00,
#   "change_90d": -8.3,   # % change over 90 days
#   "condition":  "Near Mint",
#   "printing":   "Foil",
# }

url = tcgplayer_url(card)
# "https://www.tcgplayer.com/search/pokemon/product?q=Charizard+Base+Set&productLineName=pokemon"
```

**Variant selection:** holos get the `Foil` price track; everything else uses `Normal`, based on the card's rarity.

**Caching:** results are stored in `pricing/price_cache.json` keyed by card ID with a 12-hour TTL, so the free-tier request budget isn't a concern in normal use.


## Price Data Quality

Prices come from the **JustTCG free tier**, which aggregates recent completed sales from TCGPlayer. The data is generally reliable for popular cards with high sales volume, but can look unusual in a few situations:

**Sparse sales data.** Market price is a weighted average of recent sales. If a particular condition (e.g. Near Mint) has had very few sales in the last 90 days, a single low-ball transaction can drag the average down significantly. This can produce counterintuitive results like a Lightly Played copy appearing more expensive than Near Mint. The app flags this with a warning when it detects prices out of the expected NM ≥ LP ≥ MP ≥ HP ≥ Damaged order.

**Staleness.** Prices are cached locally for 12 hours and JustTCG updates their data on their own schedule. For fast-moving cards (new set releases, tournament results), the displayed price may lag the live TCGPlayer market by up to a day. When in doubt, click the TCGPlayer link to see current listings.

**Free tier limitations.** The JustTCG free tier provides 1,000 requests per month. The 12-hour local cache means routine use stays well within this, but the data is sourced from TCGPlayer's market rather than being live or guaranteed complete.

### Upgrading to a paid data source

If you need consistently reliable prices across all conditions and rarities, the options are:

| Option | What you get | Cost |
|--------|-------------|------|
| **JustTCG paid tier** | Same API, higher rate limits, priority data freshness | From ~$10/mo |
| **TCGPlayer Pro API** | Direct access to TCGPlayer's own pricing data | Invite-only / requires approval |
| **TCGCSV** | Bulk CSV/JSON snapshots of current listings, no API key required | Free, but no historical data |

For this project, swapping in a different source only requires updating `pricing/price_lookup.py` — `get_variants()` is the single function that talks to the external API, and `main.py` only depends on the dict structure it returns.


## Known Limitations

| Limitation | Mitigation |
|------------|-----------|
| **Reprint ambiguity** — many cards share identical artwork across sets | Return top-3 results; user confirms before prices are fetched |
| **Domain gap** — trained on clean official art, tested on real photos | Aggressive augmentation during training |
| **57 promo cards** have no CDN image and are excluded from training | Listed in `data/failed_images.txt` |
| **Very low resolution or heavy glare** photos | Model returns low confidence scores |
| **No graded (PSA/CGC) pricing** | Out of scope; TCGPlayer covers the primary use case |
