"""
p1_15_figure_distance_decay.py
Publication-quality distance-decay figure (Cell Reports style).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE  = str(Path(__file__).resolve().parents[1])
IN    = os.path.join(BASE, "results", "phase1", "p1_distance_decay.tsv")
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
OUTD  = str(REGEN_DIR)
os.makedirs(OUTD, exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN, sep="\t")
labels = df["bin_label"].tolist()
x = np.arange(len(labels))

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       8,
    "axes.labelsize":  8,
    "axes.titlesize":  9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi":      300,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
})

COL_ADQTL   = "#c0392b"
COL_CONTROL = "#9aa0a6"

# ── figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.5))
suptitle(
    fig,
    "Figure S_.  Cis-haplotype epistasis enrichment decays with\n"
    "focal-background SNP distance",
    fontsize=8, y=1.01, ha="center"
)

# ── Panel A: bar chart of rates ────────────────────────────────────────────────
w = 0.35
rate_a = df["rate_adqtl"].values * 100
rate_c = df["rate_control"].values * 100
n_a    = df["n_adqtl"].values.astype(int)
n_c    = df["n_control"].values.astype(int)

bars_a = ax1.bar(x - w/2, rate_a, width=w, color=COL_ADQTL,   label="adQTL",   zorder=3)
bars_c = ax1.bar(x + w/2, rate_c, width=w, color=COL_CONTROL, label="Control", zorder=3, alpha=0.85)

# n labels above bars
for i, (ba, bc) in enumerate(zip(bars_a, bars_c)):
    ax1.text(ba.get_x() + ba.get_width()/2, ba.get_height() + 0.8,
             f"n={n_a[i]}", ha="center", va="bottom", fontsize=6, color=COL_ADQTL, rotation=90)
    ax1.text(bc.get_x() + bc.get_width()/2, bc.get_height() + 0.8,
             f"n={n_c[i]}", ha="center", va="bottom", fontsize=6, color="#555555", rotation=90)

ax1.axhline(5, color="black", lw=0.8, ls="--", label="5% (null)", zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha="right")
ax1.set_xlabel("Focal-background SNP distance")
ax1.set_ylabel("Loci with significant interaction (%)")
ax1.set_ylim(0, 80)
ax1.set_title("A  Interaction rate by distance bin", loc="left", fontweight="bold", fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(frameon=False)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax1.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

# ── Panel B: OR with CI ────────────────────────────────────────────────────────
OR    = df["OR"].values
OR_lo = df["OR_lo"].values
OR_hi = df["OR_hi"].values
fp    = df["fisher_p"].values

# gradient: dark red -> light red
n_bins = len(labels)
reds   = [matplotlib.cm.Reds(0.9 - 0.5 * i / (n_bins - 1)) for i in range(n_bins)]

for i in range(n_bins):
    if np.isfinite(OR[i]):
        ax2.errorbar(
            x[i], OR[i],
            yerr=[[OR[i] - OR_lo[i]], [OR_hi[i] - OR[i]]],
            fmt="o", color=reds[i], ms=7, lw=1.5, capsize=4, capthick=1.5, zorder=3
        )
        # Fisher p annotation
        if fp[i] < 0.001:
            pstr = "p<0.001"
        elif fp[i] < 0.01:
            pstr = f"p={fp[i]:.3f}"
        else:
            pstr = f"p={fp[i]:.2f}"
        ax2.text(x[i], OR_hi[i] * 1.05, pstr, ha="center", va="bottom",
                 fontsize=6.5, color="#333333")

ax2.axhline(1, color="black", lw=0.8, ls="--", zorder=2)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha="right")
ax2.set_xlabel("Focal-background SNP distance")
ax2.set_ylabel("Odds ratio (adQTL vs control)")
ax2.set_yscale("log")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
ax2.set_title("B  OR (adQTL vs control) by distance bin", loc="left", fontweight="bold", fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

# ── save ───────────────────────────────────────────────────────────────────────
fig.tight_layout()
for ext in ["png", "pdf"]:
    fp_out = os.path.join(OUTD, f"Figure_DistanceDecay.{ext}")
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
announce(__file__, "Figure_DistanceDecay")

plt.close(fig)
print("Done.")
