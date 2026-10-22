"""
p1_19_figure_simulation.py
4-panel publication-quality figure from generative simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results/phase1/p1_simulation_results.tsv"), sep="\t")

# ── palette
COL_NULL  = "#95a5a6"
COL_LD    = "#7f8c8d"
COL_GXE   = "#2980b9"
COL_EPI   = "#c0392b"
COL_CTRL  = "#e74c3c"

# ── four primary model labels
MODEL_LABELS = ["A", "B", "C", "D_g0.3_df0.5"]
MODEL_NAMES  = ["Null\n(A)", "LD artifact\n(B)", "GxE\n(C)", "Cis-epistasis\n(D)"]
MODEL_COLORS = [COL_NULL, COL_LD, COL_GXE, COL_EPI]

def get_grp(label):
    return df[df["model_label"] == label]

def adqtl_rate(grp):
    return (grp["p_het"] < 0.05).mean()

def int_rate(grp):
    return (grp["p_int"] < 0.05).mean()

def int_rate_anc_ctrl(grp):
    col = "p_int_anc"
    if col not in grp.columns:
        return np.nan
    valid = grp[col].dropna()
    if len(valid) == 0:
        return np.nan
    return (valid < 0.05).mean()

def compute_OR(grp):
    adqtl_mask  = grp["p_het"] < 0.05
    nonadq_mask = grp["p_het"] > 0.3
    if adqtl_mask.sum() < 5 or nonadq_mask.sum() < 5:
        return np.nan
    r1 = (grp.loc[adqtl_mask,  "p_int"] < 0.05).mean()
    r0 = (grp.loc[nonadq_mask, "p_int"] < 0.05).mean()
    eps = 1e-9
    if r0 < eps:
        return np.nan
    return (r1 / (1 - r1 + eps)) / (r0 / (1 - r0 + eps))

# ──────────────────────────────────────────────
# Panel A: adQTL detection rate
# ──────────────────────────────────────────────
rates_A = [adqtl_rate(get_grp(l)) for l in MODEL_LABELS]

# ──────────────────────────────────────────────
# Panel B: focal×bg interaction rate + ancestry control
# ──────────────────────────────────────────────
rates_B_main = [int_rate(get_grp(l)) for l in MODEL_LABELS]
rates_B_ctrl = [int_rate_anc_ctrl(get_grp(l)) for l in MODEL_LABELS]

# ──────────────────────────────────────────────
# Panel C: OR adQTL vs non-adQTL
# ──────────────────────────────────────────────
ors_C = [compute_OR(get_grp(l)) for l in MODEL_LABELS]

# ──────────────────────────────────────────────
# Panel D: power curve for Model D (gamma × Delta_f vs adQTL rate)
# ──────────────────────────────────────────────
d_rows = df[df["model"].astype(str) == "D"].copy()
d_rows["gxdf"] = d_rows["gamma"] * d_rows["delta_f"]
power_curve = d_rows.groupby("gxdf").apply(lambda g: (g["p_het"] < 0.05).mean()).reset_index()
power_curve.columns = ["gxdf", "adqtl_rate"]
power_curve = power_curve.sort_values("gxdf")

# ──────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers

suptitle(
    fig,
    "Figure S_. Generative simulation establishes identifiability of cis-haplotype epistasis",
    fontsize=13, fontweight="bold", y=1.01
)

x = np.arange(len(MODEL_LABELS))
w = 0.6

# --- Panel A ---
ax = axes[0]
bars = ax.bar(x, rates_A, width=w, color=MODEL_COLORS, edgecolor="white", linewidth=0.8)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="5% null rate")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=9)
ax.set_ylabel("adQTL detection rate", fontsize=10)
ax.set_title("A  |  adQTL detection rate\n(p_het < 0.05)", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(max(rates_A)*1.15, 0.12))
ax.legend(fontsize=8)
for bar, val in zip(bars, rates_A):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# --- Panel B ---
ax = axes[1]
bar_w = 0.28
bars_main = ax.bar(x - bar_w/2, rates_B_main, width=bar_w,
                   color=MODEL_COLORS, edgecolor="white", linewidth=0.8, label="Marginal")
# ancestry-controlled bars only for C and D (models 2 and 3)
ctrl_colors = ["lightgray", "lightgray", "#5dade2", "#e59866"]
bars_ctrl = ax.bar(x + bar_w/2, rates_B_ctrl, width=bar_w,
                   color=ctrl_colors, edgecolor="white", linewidth=0.8,
                   linestyle="--", label="+ Ancestry ctrl")
ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=9)
ax.set_ylabel("focal×bg interaction rate", fontsize=10)
ax.set_title("B  |  focal×bg interaction rate\n(p_int < 0.05)", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(max(rates_B_main)*1.15, 0.12))
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# --- Panel C ---
ax = axes[2]
ors_plot = [min(o, 50) if (o is not None and not np.isnan(o) and np.isfinite(o)) else 0 for o in ors_C]
bars = ax.bar(x, ors_plot, width=w, color=MODEL_COLORS, edgecolor="white", linewidth=0.8)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="OR = 1")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, fontsize=9)
ax.set_ylabel("Odds ratio (adQTL vs non-adQTL\nfor focal×bg interaction)", fontsize=9)
ax.set_title("C  |  Enrichment OR\n(adQTL vs non-adQTL)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
for bar, val in zip(bars, ors_plot):
    label = f"{val:.1f}" if (val > 0 and np.isfinite(val)) else "n/a"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            label, ha="center", va="bottom", fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# --- Panel D ---
ax = axes[3]
if len(power_curve) > 0:
    ax.plot(power_curve["gxdf"], power_curve["adqtl_rate"],
            "o-", color=COL_EPI, linewidth=2, markersize=7, label="Model D")
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1.0, label="5% null")
    ax.axhline(0.80, color="#27ae60", linestyle=":", linewidth=1.2, label="80% power")
    # mark realistic gamma=0.3, delta_f=0.3 → gxdf=0.09
    ax.axvline(0.09, color="navy", linestyle=":", linewidth=1.0, alpha=0.7, label="γ=0.3, Δf=0.3")
    ax.set_xlabel("γ × Δf  (interaction × freq divergence)", fontsize=10)
    ax.set_ylabel("adQTL detection rate", fontsize=10)
    ax.set_title("D  |  Power curve (Model D)\nadQTL detection vs γ × Δf", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()

fig_dir = str(REGEN_DIR)
for ext in ["png", "pdf"]:
    path = os.path.join(fig_dir, f"Figure_Simulation.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
announce(__file__, "Figure_Simulation")

plt.close(fig)

# ── Print key numbers
print("\n=== KEY NUMBERS ===")
for i, (label, name) in enumerate(zip(MODEL_LABELS, MODEL_NAMES)):
    grp = get_grp(label)
    or_str = f"{ors_C[i]:.2f}" if (ors_C[i] is not None and not np.isnan(ors_C[i]) and np.isfinite(ors_C[i])) else "nan"
    print(f"{name.replace(chr(10),' ')}: adQTL={rates_A[i]:.3f}, int={rates_B_main[i]:.3f}, OR={or_str}")

print("\nModel D power at realistic gamma x Deltaf values:")
for _, row in power_curve.iterrows():
    print(f"  gxdf={row['gxdf']:.3f}: adQTL rate={row['adqtl_rate']:.3f}")
