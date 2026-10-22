"""
Validation V3: distribution of cis-epistasis P values
=============================================
adQTL vs control: density plot of best-background-SNP interaction P values,
showing that adQTL P values are smaller overall (left-shifted), beyond a single "X% significant" number.

Outputs:
  figures/validation/v3_pval_density.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE    = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "figures" / "validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CIS_FILE = BASE / "results" / "phase1" / "p1_cis_interaction.tsv"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})
COL_ADQ  = "#d62728"
COL_CTRL = "#1f77b4"


def main():
    df = pd.read_csv(CIS_FILE, sep="\t")
    adq  = df[df["set"] == "adqtl"]
    ctrl = df[df["set"] == "control"]

    # log10 p values
    adq_logp  = -np.log10(adq["min_int_p"].clip(lower=1e-300)).dropna()
    ctrl_logp = -np.log10(ctrl["min_int_p"].clip(lower=1e-300)).dropna()

    # MWU test
    mwu = stats.mannwhitneyu(adq_logp, ctrl_logp, alternative="greater")

    # Bonferroni threshold line
    # original p threshold: bonf_p < 0.05 -> min_int_p < 0.05/n_bg_tested (varies)
    # use median n_bg_tested
    med_n_bg = df["n_bg_tested"].median()
    bonf_cutoff = -np.log10(0.05 / med_n_bg)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: density/histogram
    ax = axes[0]
    bins = np.linspace(0, max(adq_logp.max(), ctrl_logp.max())+0.5, 40)
    ax.hist(ctrl_logp, bins=bins, density=True, alpha=0.55, color=COL_CTRL,
            label=f"concordant eQTL (n={len(ctrl)})", edgecolor="white", lw=0.3)
    ax.hist(adq_logp,  bins=bins, density=True, alpha=0.55, color=COL_ADQ,
            label=f"adQTL (n={len(adq)})", edgecolor="white", lw=0.3)
    ax.axvline(bonf_cutoff, color="black", lw=1.2, ls="--",
               label=f"Bonferroni threshold\n(-log10 p = {bonf_cutoff:.1f})")
    ax.set_xlabel("-log10(best cis-interaction p)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("A  cis-interaction p-value distribution", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5)
    ax.text(0.97, 0.97,
            f"MWU p (adQTL > ctrl) = {mwu.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # Panel B: ECDF comparison (QQ-style)
    ax2 = axes[1]
    for logp, label, col in [
        (ctrl_logp, f"concordant eQTL (n={len(ctrl)})", COL_CTRL),
        (adq_logp,  f"adQTL (n={len(adq)})",            COL_ADQ),
    ]:
        sorted_p = np.sort(logp)
        ecdf = np.arange(1, len(sorted_p)+1) / len(sorted_p)
        ax2.plot(sorted_p, 1-ecdf, lw=2, color=col, label=label)
    ax2.axvline(bonf_cutoff, color="black", lw=1.2, ls="--", label="Bonferroni threshold")
    ax2.set_xlabel("-log10(best cis-interaction p)", fontsize=10)
    ax2.set_ylabel("Fraction with p above threshold\n(1 - ECDF)", fontsize=9)
    ax2.set_title("B  Cumulative distribution\n(adQTL right-shifted = more significant)", fontsize=10, fontweight="bold", loc="left")
    ax2.legend(fontsize=8.5)
    # shade the difference
    # interpolate both ECDFs on common grid
    xgrid = np.linspace(0, max(adq_logp.max(), ctrl_logp.max()), 500)
    def ecdf_survival(logp_vals, xg):
        sorted_p = np.sort(logp_vals)
        surv = np.array([(sorted_p >= x).mean() for x in xg])
        return surv
    sa = ecdf_survival(adq_logp.values, xgrid)
    sc = ecdf_survival(ctrl_logp.values, xgrid)
    ax2.fill_between(xgrid, sc, sa, where=sa>sc, alpha=0.15, color=COL_ADQ,
                     label="adQTL excess")
    ax2.set_ylim(0, 1.02)

    fig.suptitle("cis-epistasis interaction p-value distributions: adQTL vs concordant eQTL",
                 fontsize=10.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "v3_pval_density.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"Figure -> {out}")
    print(f"MWU p = {mwu.pvalue:.3e}")
    print("[v3_interaction_pval_dist] DONE.")


if __name__ == "__main__":
    main()
