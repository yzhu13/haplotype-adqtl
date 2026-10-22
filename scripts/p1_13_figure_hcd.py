"""
Phase 1 - Step 13: Publication-quality HCD figure (4 panels).

Reads results/phase1/p1_hcd.tsv and produces:
  figures/paper/Figure_HCD.png  (300 DPI)
  figures/paper/Figure_HCD.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RES1 = BASE / "results" / "phase1"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
FIGS = REGEN_DIR

HCD_TSV = RES1 / "p1_hcd.tsv"

# Cell Reports palette
COL_ADQTL   = "#c0392b"
COL_CONTROL = "#9aa0a6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def strip_jitter(ax, data, x_center, color, jitter=0.08, alpha=0.5, size=3, zorder=3):
    """Overlay strip plot with jitter on an axis."""
    xs = np.random.default_rng(42).uniform(x_center - jitter, x_center + jitter, size=len(data))
    ax.scatter(xs, data, s=size, color=color, alpha=alpha, zorder=zorder, linewidths=0)


def annotate_mw(ax, data_a, data_b, x0, x1, y_top, label_a="adQTL", label_b="control"):
    """Annotate Mann-Whitney U p-value between two groups."""
    stat, p = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
    p_str = f"p = {p:.2e}" if p >= 1e-4 else f"p = {p:.1e}"
    # bracket
    y_br = y_top * 1.05
    ax.plot([x0, x0, x1, x1], [y_br, y_br * 1.02, y_br * 1.02, y_br],
            color="black", lw=0.8)
    ax.text((x0 + x1) / 2, y_br * 1.03, p_str, ha="center", va="bottom", fontsize=7)
    return p


def main():
    print("Loading HCD results ...", flush=True)
    df = pd.read_csv(HCD_TSV, sep="\t")
    print(f"  Rows: {len(df)}, sets: {df['set'].value_counts().to_dict()}")

    adq = df[df["set"] == "adqtl"].copy()
    ctl = df[df["set"] == "control"].copy()

    adq_hcd = adq["hcd_conditional"].dropna()
    ctl_hcd = ctl["hcd_conditional"].dropna()

    # ---- figure setup -------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    suptitle(
        fig,
        "Figure 6.  Ancestry-differential eQTL loci harbor greater haplotype context divergence between populations",
        fontsize=9, y=1.01, ha="center"
    )

    # =========================================================================
    # Panel A: Violin + strip of HCD by group
    # =========================================================================
    ax = axes[0]
    ax.set_title("A  HCD by eQTL class", fontweight="bold", loc="left")

    groups = [adq_hcd.values, ctl_hcd.values]
    positions = [1, 2]
    colors = [COL_ADQTL, COL_CONTROL]
    labels = ["adQTL", "Control"]

    vp = ax.violinplot(groups, positions=positions, showmedians=False,
                       showextrema=False, widths=0.5)
    for i, (body, col) in enumerate(zip(vp["bodies"], colors)):
        body.set_facecolor(col)
        body.set_alpha(0.35)
        body.set_edgecolor(col)

    # medians
    for pos, data, col in zip(positions, groups, colors):
        med = np.median(data)
        ax.hlines(med, pos - 0.2, pos + 0.2, colors=col, linewidth=2, zorder=5)
        strip_jitter(ax, data, pos, col)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("HCD (conditional JSD)")
    ax.set_xlim(0.4, 2.6)

    # Mann-Whitney annotation
    y_top = max(np.nanmax(adq_hcd), np.nanmax(ctl_hcd))
    mw_p = annotate_mw(ax, adq_hcd.values, ctl_hcd.values, 1, 2, y_top)

    # Legend patches
    patch_a = mpatches.Patch(color=COL_ADQTL, alpha=0.7, label=f"adQTL (n={len(adq_hcd)})")
    patch_c = mpatches.Patch(color=COL_CONTROL, alpha=0.7, label=f"Control (n={len(ctl_hcd)})")
    ax.legend(handles=[patch_a, patch_c], frameon=False, loc="upper right")

    print(f"Panel A: adQTL median={np.median(adq_hcd):.4f}, control median={np.median(ctl_hcd):.4f}, MW p={mw_p:.3e}")

    # =========================================================================
    # Panel B: Scatter HCD vs |ObsDelta| (adQTL only)
    # =========================================================================
    ax = axes[1]
    ax.set_title("B  HCD vs |ObsDelta| (adQTLs)", fontweight="bold", loc="left")

    sub = adq[["hcd_conditional", "obs_delta", "bonf_p"]].dropna()
    x = sub["hcd_conditional"].values
    y = np.abs(sub["obs_delta"].values)
    bp = sub["bonf_p"].values

    # Color by -log10(bonf_p)
    lp = -np.log10(np.clip(bp, 1e-10, 1))
    sc = ax.scatter(x, y, c=lp, cmap="Reds", s=20, alpha=0.7, linewidths=0,
                    vmin=0, vmax=np.percentile(lp, 95))
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("-log10(Bonf. p)", fontsize=7)

    # Spearman
    r, p = stats.spearmanr(x, y)
    p_str = f"p = {p:.2e}" if p >= 1e-4 else f"p = {p:.1e}"
    ax.text(0.05, 0.92, f"Spearman r = {r:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.set_xlabel("Conditional HCD (JSD)")
    ax.set_ylabel("|ObsDelta EUR-AFR|")

    print(f"Panel B: Spearman r={r:.3f}, p={p:.3e}")

    # =========================================================================
    # Panel C: Conditional vs Unconditional HCD comparison (paired violin)
    # =========================================================================
    ax = axes[2]
    ax.set_title("C  Conditional vs Unconditional HCD", fontweight="bold", loc="left")

    # Pair across all loci with both values
    both = df[["hcd_conditional", "hcd_unconditional", "set"]].dropna()

    # Side-by-side violin for adQTL and control
    # 4 violins: cond-adq, uncond-adq, cond-ctl, uncond-ctl
    cond_adq  = both[both["set"] == "adqtl"]["hcd_conditional"].values
    uncond_adq = both[both["set"] == "adqtl"]["hcd_unconditional"].values
    cond_ctl  = both[both["set"] == "control"]["hcd_conditional"].values
    uncond_ctl = both[both["set"] == "control"]["hcd_unconditional"].values

    pos_ca, pos_ua, pos_cc, pos_uc = 1, 2, 3.5, 4.5
    vp2 = ax.violinplot(
        [cond_adq, uncond_adq, cond_ctl, uncond_ctl],
        positions=[pos_ca, pos_ua, pos_cc, pos_uc],
        showmedians=False, showextrema=False, widths=0.5
    )
    style = [
        (COL_ADQTL, 0.6), (COL_ADQTL, 0.2),
        (COL_CONTROL, 0.6), (COL_CONTROL, 0.2)
    ]
    for body, (col, alpha) in zip(vp2["bodies"], style):
        body.set_facecolor(col)
        body.set_alpha(alpha)
        body.set_edgecolor(col)

    for pos, data, col in zip(
        [pos_ca, pos_ua, pos_cc, pos_uc],
        [cond_adq, uncond_adq, cond_ctl, uncond_ctl],
        [COL_ADQTL]*2 + [COL_CONTROL]*2
    ):
        ax.hlines(np.median(data), pos - 0.2, pos + 0.2, colors=col, lw=2, zorder=5)

    ax.set_xticks([1.5, 4.0])
    ax.set_xticklabels(["adQTL", "Control"])
    ax.set_ylabel("HCD (JSD)")

    leg_patches = [
        mpatches.Patch(facecolor=COL_ADQTL, alpha=0.6, label="Conditional"),
        mpatches.Patch(facecolor=COL_ADQTL, alpha=0.2, label="Unconditional"),
    ]
    ax.legend(handles=leg_patches, frameon=False, loc="upper right", fontsize=6)

    # Wilcoxon test: paired cond vs uncond within adQTLs
    if len(cond_adq) > 3 and len(uncond_adq) > 3:
        wstat, wp = stats.wilcoxon(cond_adq[:min(len(cond_adq), len(uncond_adq))],
                                    uncond_adq[:min(len(cond_adq), len(uncond_adq))])
        wp_str = f"Wilcoxon p = {wp:.2e}" if wp >= 1e-4 else f"Wilcoxon p = {wp:.1e}"
        ax.text(0.05, 0.95, wp_str, transform=ax.transAxes, fontsize=6, va="top",
                color=COL_ADQTL)

    # =========================================================================
    # Panel D: HCD by sign-flip status (adQTLs only)
    # =========================================================================
    ax = axes[3]
    ax.set_title("D  HCD by sign-flip (adQTLs)", fontweight="bold", loc="left")

    adq_complete = adq[["hcd_conditional", "obs_delta", "bonf_p"]].dropna()

    # Load per-ancestry betas to get sign-flip info
    ref_file = BASE / "results" / "phase1" / "p1_adqtl_refscore.tsv"
    ref = pd.read_csv(ref_file, sep="\t")

    adq2 = adq.merge(
        ref[["variant_kgpID", "ensemblID", "beta_EUR", "beta_AFR"]],
        on=["variant_kgpID", "ensemblID"], how="left"
    ).dropna(subset=["hcd_conditional", "beta_EUR", "beta_AFR"])

    # Sign-flip: beta_EUR and beta_AFR have opposite signs
    adq2["sign_flip"] = np.sign(adq2["beta_EUR"]) != np.sign(adq2["beta_AFR"])
    flip = adq2[adq2["sign_flip"]]["hcd_conditional"].values
    same = adq2[~adq2["sign_flip"]]["hcd_conditional"].values

    grps = [flip, same]
    pos2 = [1, 2]
    col2 = ["#8e44ad", "#2980b9"]
    labs = [f"Sign-flip\n(n={len(flip)})", f"Same-sign\n(n={len(same)})"]

    if len(flip) > 2 and len(same) > 2:
        vp3 = ax.violinplot(grps, positions=pos2, showmedians=False,
                            showextrema=False, widths=0.5)
        for body, col in zip(vp3["bodies"], col2):
            body.set_facecolor(col)
            body.set_alpha(0.4)
            body.set_edgecolor(col)
        for pos, data, col in zip(pos2, grps, col2):
            ax.hlines(np.median(data), pos - 0.2, pos + 0.2, colors=col, lw=2, zorder=5)
            strip_jitter(ax, data, pos, col)

        mwstat, mwp = stats.mannwhitneyu(flip, same, alternative="two-sided")
        mwp_str = f"MW p = {mwp:.2e}"
        y_top2 = max(np.nanmax(flip), np.nanmax(same))
        ax.text(0.5, 0.95, mwp_str, transform=ax.transAxes, fontsize=7,
                ha="center", va="top")
        print(f"Panel D: sign-flip n={len(flip)}, same-sign n={len(same)}, MW p={mwp:.3e}")
    else:
        ax.text(0.5, 0.5, "Insufficient data\nfor sign-flip",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="grey")
        print("Panel D: insufficient data for sign-flip analysis")

    ax.set_xticks(pos2)
    ax.set_xticklabels(labs)
    ax.set_ylabel("Conditional HCD (JSD)")

    # ---- save ---------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ["png", "pdf"]:
        out = FIGS / f"Figure_HCD.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    announce(__file__, "Figure_HCD")

    plt.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
