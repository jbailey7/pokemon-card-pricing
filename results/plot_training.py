from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = Path(__file__).parent
df   = pd.read_csv(HERE / "training_log.csv")

UNFREEZE_EPOCH = 6   # backbone partially unfrozen at the start of this epoch
BEST_EPOCH     = 24  # checkpoint with highest val_top1

fig, (ax_loss, ax_acc) = plt.subplots(
    1, 2,
    figsize=(11, 4.5),
    constrained_layout=True,
)
fig.suptitle("Training Run — EfficientNet-B0 + Triplet Loss", fontsize=13, fontweight="bold")

# ── shared annotation helpers ──────────────────────────────────────────────

def add_unfreeze_line(ax):
    ax.axvline(
        UNFREEZE_EPOCH - 0.5, color="steelblue", linewidth=1.2,
        linestyle="--", alpha=0.7,
    )
    ax.text(
        UNFREEZE_EPOCH - 0.3, ax.get_ylim()[1] * 0.97,
        "backbone\nunfrozen", color="steelblue", fontsize=7.5,
        va="top", ha="left",
    )


# ── left panel: triplet loss ───────────────────────────────────────────────

ax_loss.plot(df["epoch"], df["loss"], color="#e07b39", linewidth=2, label="Triplet loss")
ax_loss.plot(
    df["epoch"], df["active_pct"] / 100,
    color="gray", linewidth=1.2, linestyle=":", alpha=0.7, label="Active triplets %",
)

ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss / Active triplet fraction")
ax_loss.set_title("Training loss")
ax_loss.set_xlim(1, 30)
ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_loss.legend(fontsize=8)
ax_loss.grid(axis="y", linewidth=0.5, alpha=0.4)

add_unfreeze_line(ax_loss)


# ── right panel: validation accuracy ──────────────────────────────────────

ax_acc.plot(df["epoch"], df["val_top1"], color="#2e8b57", linewidth=2, label="Top-1")
ax_acc.plot(df["epoch"], df["val_top3"], color="#2e8b57", linewidth=2,
            linestyle="--", alpha=0.65, label="Top-3")

best = df[df["epoch"] == BEST_EPOCH].iloc[0]
ax_acc.scatter(
    [BEST_EPOCH], [best["val_top1"]],
    marker="*", s=200, color="#2e8b57", zorder=5,
    label=f"Best Top-1 = {best['val_top1']:.1%}",
)
ax_acc.scatter(
    [BEST_EPOCH], [best["val_top3"]],
    marker="*", s=200, color="#2e8b57", zorder=5, alpha=0.65,
    label=f"Best-epoch Top-3 = {best['val_top3']:.1%}",
)

ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Retrieval accuracy")
ax_acc.set_title("Validation accuracy (2,002-card hold-out)")
ax_acc.set_xlim(1, 30)
ax_acc.set_ylim(0, 1)
ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax_acc.legend(fontsize=8)
ax_acc.grid(axis="y", linewidth=0.5, alpha=0.4)

add_unfreeze_line(ax_acc)

out = HERE / "training_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
