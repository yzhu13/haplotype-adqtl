"""
Validation V9: background-SNP TF / regulatory-effect analysis (fully local)
=======================================================================
Uses the existing p1_tf_vignette.tsv, p1_adqtl_refscore.tsv and cis_interaction.tsv
to characterize the predicted TF-binding effects of background SNPs.

Questions:
1. Are predicted TF-binding / ATAC effects of background SNPs larger than for random SNPs?
   → would indicate that bg SNPs alter the local regulatory context through TFs
2. Is |bg_chip_tf| positively correlated with |int_beta| (focal×bg interaction strength)?
   → would link the size of a bg SNP's TF effect to its ability to modify the focal effect
3. Do bg-SNP TF effects differ between adQTL and control loci?
   → is the regulatory magnitude of the background specific to adQTLs?

Inputs:
  p1_tf_vignette.tsv  — predicted scores incl. bg_chip_tf, bg_atac
  p1_cis_interaction.tsv — int_beta, bonf_p
  p1_adqtl_refscore.tsv — pred_rna_lfc of the focal SNP

Outputs:
  results/validation/v9_bg_tf_analysis.tsv
  figures/validation/v9_bg_tf_analysis.{png,pdf}
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
P1      = BASE / "results" / "phase1"
P0      = BASE / "results" / "phase0"
OUT_DIR = BASE / "results" / "validation"
FIG_DIR = BASE / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8})
AD  = "#d62728"; CO = "#1f77b4"; BL = "#2166ac"; GR = "#2ca02c"


def main():
    tf   = pd.read_csv(P1 / "p1_tf_vignette.tsv",    sep="\t")
    ci   = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    ref  = pd.read_csv(P1 / "p1_adqtl_refscore.tsv", sep="\t")
    mb   = pd.read_csv(P1 / "p1_multi_bg_decomp.tsv", sep="\t")

    # Merge TF + interaction data
    merged = ci.merge(tf, on=["variant_kgpID","ensemblID","best_bg"], how="left")
    merged = merged.merge(ref[["variant_kgpID","ensemblID","pred_rna_lfc"]],
                          on=["variant_kgpID","ensemblID"], how="left")
    print(f"Merged rows: {len(merged)}, with TF scores: {merged.bg_chip_tf.notna().sum()}")

    adq  = merged[merged.set=="adqtl"]
    ctrl = merged[merged.set=="control"]

    # Analysis 1: bg_chip_tf absolute value distribution: adQTL vs control
    adq_tf  = adq.bg_chip_tf.abs().dropna()
    ctrl_tf = ctrl.bg_chip_tf.abs().dropna()
    mwu_tf  = stats.mannwhitneyu(adq_tf, ctrl_tf, alternative="two-sided")
    print(f"\n|bg_chip_tf| adQTL median={adq_tf.median():.4f}  ctrl median={ctrl_tf.median():.4f}")
    print(f"MWU p (two-sided) = {mwu_tf.pvalue:.3e}")

    # Analysis 2: bg_atac absolute value: adQTL vs control
    adq_atac  = adq.bg_atac.abs().dropna()
    ctrl_atac = ctrl.bg_atac.abs().dropna()
    mwu_atac  = stats.mannwhitneyu(adq_atac, ctrl_atac, alternative="two-sided")
    print(f"\n|bg_atac| adQTL median={adq_atac.median():.4f}  ctrl median={ctrl_atac.median():.4f}")
    print(f"MWU p (two-sided) = {mwu_atac.pvalue:.3e}")

    # Analysis 3: |bg_chip_tf| vs |int_beta| correlation (adQTL only)
    adq_both = adq.dropna(subset=["bg_chip_tf","int_beta"])
    if len(adq_both)>5:
        sp_tf_beta = stats.spearmanr(adq_both.bg_chip_tf.abs(), adq_both.int_beta.abs())
        print(f"\nCorrelation |bg_chip_tf| vs |int_beta| (adQTL, n={len(adq_both)}): "
              f"Spearman r={sp_tf_beta.statistic:.3f} p={sp_tf_beta.pvalue:.3e}")

    # Analysis 4: bg_chip_tf vs focal_chip_tf — comparable regulatory magnitude?
    focal_tf = tf.focal_chip_tf.abs().dropna()
    print(f"\n|focal_chip_tf| median: {focal_tf.median():.4f}")
    print(f"|bg_chip_tf| adQTL median:    {adq_tf.median():.4f}")
    mwu_focal_bg = stats.mannwhitneyu(adq_tf, focal_tf, alternative="two-sided")
    print(f"MWU focal_chip_tf vs bg_chip_tf (adQTL): p={mwu_focal_bg.pvalue:.3e}")

    # Analysis 5: For adQTL loci with significant interaction, is bg_chip_tf larger?
    adq_sig  = adq[adq.bonf_p < 0.05].bg_chip_tf.abs().dropna()
    adq_ns   = adq[adq.bonf_p >= 0.05].bg_chip_tf.abs().dropna()
    if len(adq_sig)>3 and len(adq_ns)>3:
        mwu_sig = stats.mannwhitneyu(adq_sig, adq_ns, alternative="greater")
        print(f"\n|bg_chip_tf|: sig adQTL median={adq_sig.median():.4f}  "
              f"ns adQTL median={adq_ns.median():.4f}  p={mwu_sig.pvalue:.3e}")

    # Save
    summary = {
        "n_adqtl_with_tf": len(adq_tf), "n_ctrl_with_tf": len(ctrl_tf),
        "median_bg_chip_tf_adqtl": round(float(adq_tf.median()), 4),
        "median_bg_chip_tf_ctrl":  round(float(ctrl_tf.median()), 4),
        "mwu_p_chip_tf": float(mwu_tf.pvalue),
        "median_bg_atac_adqtl": round(float(adq_atac.median()), 4),
        "median_bg_atac_ctrl":  round(float(ctrl_atac.median()), 4),
        "mwu_p_atac": float(mwu_atac.pvalue),
        "spearman_bg_tf_vs_int_beta": round(float(sp_tf_beta.statistic), 3) if len(adq_both)>5 else np.nan,
        "spearman_p_bg_tf_vs_int_beta": float(sp_tf_beta.pvalue) if len(adq_both)>5 else np.nan,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "v9_bg_tf_analysis.tsv", sep="\t", index=False)

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # A: |bg_chip_tf| violin adQTL vs ctrl
    ax = axes[0, 0]
    data_tf = [ctrl_tf.values, adq_tf.values]
    parts = ax.violinplot(data_tf, positions=[0,1], showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], [CO, AD]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks([0,1]); ax.set_xticklabels(["concordant\n(control)", "adQTL"], fontsize=10)
    ax.set_ylabel("|predicted TF ChIP-seq effect| (bg SNP)", fontsize=9)
    ax.set_title("A  Background SNP TF effect\n(adQTL vs control)", fontweight="bold", loc="left", fontsize=9)
    ax.text(0.97, 0.97, f"MWU p={mwu_tf.pvalue:.2e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ymax = max(adq_tf.max(), ctrl_tf.max())
    ax.set_ylim(-0.02*ymax, ymax*1.1)

    # B: bg_chip_tf vs focal_chip_tf comparison (same unit)
    ax2 = axes[0, 1]
    ax2.hist(focal_tf.clip(0, 1), bins=25, alpha=0.6, color=BL, density=True,
             label=f"focal chip_tf (n={len(focal_tf)})")
    ax2.hist(adq_tf.clip(0, 1),   bins=25, alpha=0.6, color=AD, density=True,
             label=f"bg chip_tf adQTL (n={len(adq_tf)})")
    ax2.set_xlabel("|predicted TF ChIP-seq effect|"); ax2.set_ylabel("Density")
    ax2.set_title("B  Background SNP effect vs focal SNP effect\n(same units; bg comparable to focal)",
                  fontweight="bold", loc="left", fontsize=9)
    ax2.legend(fontsize=8); ax2.text(0.97, 0.97,
        f"Medians: focal={focal_tf.median():.3f}\nbg={adq_tf.median():.3f}\np={mwu_focal_bg.pvalue:.2e}",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8.5)

    # C: |bg_chip_tf| vs |int_beta| scatter
    ax3 = axes[1, 0]
    if len(adq_both)>5:
        ax3.scatter(adq_both.int_beta.abs(), adq_both.bg_chip_tf.abs(),
                    s=15, color=AD, alpha=0.55)
        ax3.set_xlabel("|focal×bg interaction β|"); ax3.set_ylabel("|bg_chip_tf|")
        ax3.set_title(f"C  Bg TF effect vs interaction strength\n(adQTL, n={len(adq_both)}, "
                      f"Spearman r={sp_tf_beta.statistic:.2f})",
                      fontweight="bold", loc="left", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "insufficient data", ha="center"); ax3.axis("off")

    # D: significant vs non-significant adQTL bg_chip_tf
    ax4 = axes[1, 1]
    if len(adq_sig)>3 and len(adq_ns)>3:
        data_sig = [adq_ns.values, adq_sig.values]
        parts2 = ax4.violinplot(data_sig, positions=[0,1], showmedians=True, showextrema=False)
        for pc, c in zip(parts2["bodies"], ["#ff7f0e", AD]):
            pc.set_facecolor(c); pc.set_alpha(0.6)
        parts2["cmedians"].set_color("black"); parts2["cmedians"].set_linewidth(2)
        ax4.set_xticks([0,1]); ax4.set_xticklabels(["ns adQTL\n(no sig bg)", "sig adQTL\n(bonf_p<0.05)"])
        ax4.set_ylabel("|bg_chip_tf|")
        ax4.set_title(f"D  Bg TF effect by interaction significance\n"
                      f"(sig={len(adq_sig)}, ns={len(adq_ns)})",
                      fontweight="bold", loc="left", fontsize=9)
        ax4.text(0.97, 0.97, f"MWU p={mwu_sig.pvalue:.2e}", transform=ax4.transAxes,
                 ha="right", va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    else:
        ax4.text(0.5, 0.5, "insufficient data", ha="center"); ax4.axis("off")

    fig.suptitle("V9: Background SNP TF regulatory effects — mechanistic characterization",
                 fontweight="bold", fontsize=10.5, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "v9_bg_tf_analysis.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"\nFigure -> {out}")
    print("[v9] DONE.")


if __name__ == "__main__":
    main()
