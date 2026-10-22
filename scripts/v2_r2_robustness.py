"""
Validation V2: robustness to the r² threshold
==================================
Re-run the cis-epistasis enrichment with r² < 0.1 (strictly independent) instead of r² < 0.5,
showing that the enrichment does not depend on the threshold choice.

Inputs:
  results/phase1/p1_cis_interaction.tsv — includes the bg_r2_focal column

Outputs:
  results/validation/v2_r2_robustness.tsv
  figures/validation/v2_r2_robustness.png  (Supplementary Figure)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "validation"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: shared output policy
FIG_DIR = REGEN_DIR  # this render becomes draft Figure S6
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CIS_FILE = BASE / "results" / "phase1" / "p1_cis_interaction.tsv"

PLT_PARAMS = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
}
plt.rcParams.update(PLT_PARAMS)
COL_ADQ  = "#d62728"
COL_CTRL = "#1f77b4"


def sig_rate(df, bonf_thresh=0.05):
    """fraction with Bonferroni-corrected cis-interaction p < bonf_thresh"""
    return (df["bonf_p"] < bonf_thresh).mean()


def enrichment_test(adq_df, ctrl_df, bonf_thresh=0.05):
    """Fisher exact test: sig_rate adqtl vs control"""
    n_a = len(adq_df);  k_a = (adq_df["bonf_p"] < bonf_thresh).sum()
    n_c = len(ctrl_df); k_c = (ctrl_df["bonf_p"] < bonf_thresh).sum()
    ct = [[k_a, n_a-k_a], [k_c, n_c-k_c]]
    or_, p = stats.fisher_exact(ct, alternative="greater")
    return {"n_adqtl": n_a, "n_sig_adqtl": int(k_a), "rate_adqtl": round(k_a/n_a, 3),
            "n_ctrl":   n_c, "n_sig_ctrl":  int(k_c), "rate_ctrl":  round(k_c/n_c, 3),
            "OR": round(or_, 2), "fisher_p": p}


def main():
    df = pd.read_csv(CIS_FILE, sep="\t")
    print(f"Loaded p1_cis_interaction: {len(df)} rows")
    print(f"bg_r2_focal range: {df['bg_r2_focal'].min():.4f} - {df['bg_r2_focal'].max():.4f}")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]   # r² cutoffs to compare
    rows = []
    for r2_cut in thresholds:
        sub = df[df["bg_r2_focal"] <= r2_cut]
        adq  = sub[sub["set"] == "adqtl"]
        ctrl = sub[sub["set"] == "control"]
        res  = enrichment_test(adq, ctrl)
        res["r2_cutoff"] = r2_cut
        rows.append(res)
        print(f"r2<={r2_cut}: adqtl {res['rate_adqtl']*100:.1f}% ({res['n_adqtl']} vars) "
              f"vs ctrl {res['rate_ctrl']*100:.1f}% ({res['n_ctrl']} vars)  "
              f"OR={res['OR']:.1f}  p={res['fisher_p']:.2e}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "v2_r2_robustness.tsv", sep="\t", index=False)

    # ── Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: sig rate vs r² cutoff
    ax = axes[0]
    ax.plot(result_df["r2_cutoff"], result_df["rate_adqtl"]*100,
            "o-", color=COL_ADQ,  lw=2, ms=7, label="adQTL")
    ax.plot(result_df["r2_cutoff"], result_df["rate_ctrl"]*100,
            "s-", color=COL_CTRL, lw=2, ms=7, label="concordant eQTL (control)")
    # shade the gap
    ax.fill_between(result_df["r2_cutoff"],
                    result_df["rate_ctrl"]*100,
                    result_df["rate_adqtl"]*100,
                    alpha=0.12, color=COL_ADQ)
    # mark original cutoff
    ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7, label="Original cutoff (r²<0.5)")
    ax.axvline(0.1, color="darkgreen", lw=1, ls=":", alpha=0.8, label="Strict cutoff (r²<0.1)")
    ax.set_xlabel("Max r² (focal-background SNP)", fontsize=10)
    ax.set_ylabel("% with significant cis-interaction\n(Bonferroni p<0.05)", fontsize=9)
    ax.set_title("A  Enrichment robust to r² threshold", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Panel B: OR vs r² cutoff with p-value annotations
    ax2 = axes[1]
    ors = result_df["OR"].values
    ps  = result_df["fisher_p"].values
    colors = [COL_ADQ if p < 0.05 else "lightcoral" for p in ps]
    bars = ax2.bar(result_df["r2_cutoff"].astype(str), ors, color=colors,
                   width=0.4, edgecolor="black", linewidth=0.7)
    ax2.axhline(1, color="gray", lw=1, ls="--")
    ax2.set_xlabel("Max r² cutoff", fontsize=10)
    ax2.set_ylabel("Odds ratio (adQTL vs control)", fontsize=9)
    ax2.set_title("B  OR across r² thresholds", fontsize=10, fontweight="bold", loc="left")
    for bar, or_, p in zip(bars, ors, ps):
        p_str = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
        ax2.text(bar.get_x() + bar.get_width()/2, or_ + 0.1,
                 f"OR={or_:.1f}\n{p_str}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, max(ors)*1.4)

    suptitle(fig, "Supplementary Figure: cis-epistasis enrichment robust to background SNP r² threshold",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "v2_r2_robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    announce(__file__, "v2_r2_robustness")
    print("[v2_r2_robustness] DONE.")


if __name__ == "__main__":
    main()
