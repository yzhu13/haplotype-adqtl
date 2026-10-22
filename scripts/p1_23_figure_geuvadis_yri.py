"""
p1_23_figure_geuvadis_yri.py

3-panel figure for Geuvadis YRI replication analysis.

Panel A: P-value histogram (-log10 p_het) for adQTL vs control loci
Panel B: Bar chart of median |z_het| for adQTL vs control (+ Mann-Whitney p)
Panel C: Scatter plot of MAGE ObsDelta vs Geuvadis EUR-YRI delta (Spearman r)

Output: figures/paper/Figure_GeuvadisYRI.png + .pdf
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers

BASE = pathlib.Path(__file__).resolve().parents[1]
IN_TSV    = BASE / "results" / "phase1" / "p1_geuvadis_yri_replication.tsv"
OUT_DIR   = REGEN_DIR

COL_ADQTL   = "#c0392b"
COL_CONTROL = "#555555"

# ─── Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv(IN_TSV, sep="\t")
df["geuv_delta"] = df["beta_eur_geuvadis"] - df["beta_yri_geuvadis"]
df["abs_z_het"] = df["z_het"].abs()

ad = df[df["set"] == "adqtl"].dropna(subset=["p_het", "z_het"]).copy()
ct = df[df["set"] == "control"].dropna(subset=["p_het", "z_het"]).copy()

print(f"adQTL n={len(ad)}, control n={len(ct)}")

# ─── Stats ───────────────────────────────────────────────────────────────────

mwu_stat, mwu_p = stats.mannwhitneyu(
    ad["abs_z_het"].dropna(), ct["abs_z_het"].dropna(), alternative="greater"
)
print(f"Mann-Whitney (adQTL > control |z_het|): p={mwu_p:.3g}")

both = df.dropna(subset=["ObsDelta_mage", "geuv_delta"])
rho_all, rp_all = stats.spearmanr(both["ObsDelta_mage"], both["geuv_delta"])
ad_b  = both[both["set"] == "adqtl"]
ct_b  = both[both["set"] == "control"]
rho_a, rp_a = stats.spearmanr(ad_b["ObsDelta_mage"], ad_b["geuv_delta"]) if len(ad_b) > 5 else (np.nan, np.nan)
rho_c, rp_c = stats.spearmanr(ct_b["ObsDelta_mage"], ct_b["geuv_delta"]) if len(ct_b) > 5 else (np.nan, np.nan)

print(f"Spearman (all): rho={rho_all:.3f} p={rp_all:.3g}")
print(f"Spearman (adQTL): rho={rho_a:.3f} p={rp_a:.3g}")
print(f"Spearman (control): rho={rho_c:.3f} p={rp_c:.3g}")

# ─── Figure ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Panel A: histogram of -log10(p_het) ──────────────────────────────────────

ax_a = fig.add_subplot(gs[0])

ad_logp = -np.log10(ad["p_het"].clip(lower=1e-10))
ct_logp = -np.log10(ct["p_het"].clip(lower=1e-10))
max_logp = max(ad_logp.max(), ct_logp.max())
bins = np.linspace(0, max_logp + 0.5, 20)

ax_a.hist(ct_logp, bins=bins, color=COL_CONTROL, alpha=0.55, label="Control", density=True)
ax_a.hist(ad_logp, bins=bins, color=COL_ADQTL,   alpha=0.70, label="adQTL",   density=True)
ax_a.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1.0, label="p=0.05")

ax_a.set_xlabel(r"$-\log_{10}$(p$_{\rm het}$)", fontsize=12)
ax_a.set_ylabel("Density", fontsize=12)
ax_a.set_title("A  EUR-YRI heterogeneity p-value\ndistribution", fontsize=11, loc="left", fontweight="bold")
ax_a.legend(fontsize=9, framealpha=0.8)

rate_ad = (ad["p_het"] < 0.05).mean()
rate_ct = (ct["p_het"] < 0.05).mean()
ax_a.text(0.97, 0.96,
          f"p<0.05: adQTL {rate_ad:.1%}\ncontrol {rate_ct:.1%}",
          transform=ax_a.transAxes, ha="right", va="top", fontsize=9,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

# ── Panel B: bar chart of median |z_het| ─────────────────────────────────────

ax_b = fig.add_subplot(gs[1])

med_ad = ad["abs_z_het"].median()
med_ct = ct["abs_z_het"].median()
se_ad  = ad["abs_z_het"].std() / np.sqrt(len(ad))
se_ct  = ct["abs_z_het"].std() / np.sqrt(len(ct))

bars = ax_b.bar([0, 1], [med_ad, med_ct],
               color=[COL_ADQTL, COL_CONTROL],
               yerr=[se_ad, se_ct],
               capsize=5, width=0.5, edgecolor="black", linewidth=0.8)
ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(["adQTL loci", "Control loci"], fontsize=11)
ax_b.set_ylabel("Median |z$_{{het}}$| (EUR-YRI)", fontsize=12)
ax_b.set_title("B  Effect-size heterogeneity", fontsize=11, loc="left", fontweight="bold")

# Annotate Mann-Whitney p
if mwu_p < 0.001:
    p_label = f"p={mwu_p:.2e}"
elif mwu_p < 0.05:
    p_label = f"p={mwu_p:.3f}"
else:
    p_label = f"p={mwu_p:.2f}"

y_top = max(med_ad + se_ad, med_ct + se_ct) * 1.15
ax_b.plot([0, 1], [y_top, y_top], color="black", linewidth=0.8)
ax_b.text(0.5, y_top * 1.02, f"MWU {p_label}", ha="center", va="bottom", fontsize=9)
ax_b.set_ylim(0, y_top * 1.2)

# ── Panel C: scatter ObsDelta vs Geuvadis delta ───────────────────────────────

ax_c = fig.add_subplot(gs[2])

for subset, color, label, rho_v, rp_v in [
    (ct_b, COL_CONTROL, "Control", rho_c, rp_c),
    (ad_b, COL_ADQTL,   "adQTL",   rho_a, rp_a),
]:
    if len(subset) == 0:
        continue
    ax_c.scatter(subset["ObsDelta_mage"], subset["geuv_delta"],
                 color=color, alpha=0.55, s=18, label=label, zorder=2 if color == COL_ADQTL else 1)

# Reference line
lim = max(abs(both["ObsDelta_mage"].max()), abs(both["geuv_delta"].max())) * 1.1
ax_c.axhline(0, color="gray", linewidth=0.5)
ax_c.axvline(0, color="gray", linewidth=0.5)
ax_c.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.4, label="y=x")

ax_c.set_xlabel("MAGE ObsDelta (EUR−AFR)", fontsize=12)
ax_c.set_ylabel("Geuvadis EUR−YRI delta", fontsize=12)
ax_c.set_title("C  Direction replication", fontsize=11, loc="left", fontweight="bold")
ax_c.legend(fontsize=9, framealpha=0.8)

def fmt_rho(rho, p):
    if np.isnan(rho):
        return "n/a"
    ps = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
    return f"ρ={rho:.2f}, p={ps}"

txt = (f"adQTL: {fmt_rho(rho_a, rp_a)}\n"
       f"control: {fmt_rho(rho_c, rp_c)}")
ax_c.text(0.03, 0.97, txt, transform=ax_c.transAxes, ha="left", va="top",
          fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

# ─── Suptitle & save ─────────────────────────────────────────────────────────

suptitle(
    fig,
    "Figure S_. Independent Geuvadis YRI replication confirms\n"
    "EUR-YRI eQTL heterogeneity at MAGE-defined adQTL loci",
    fontsize=12, y=1.03
)

for fmt in ("png", "pdf"):
    out_path = OUT_DIR / f"Figure_GeuvadisYRI.{fmt}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
announce(__file__, "Figure_GeuvadisYRI")

plt.close(fig)
print("Done.")
