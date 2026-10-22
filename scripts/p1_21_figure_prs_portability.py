"""
p1_21_figure_prs_portability.py
Publication-quality 3-panel figure for PRS portability simulation.

Panel A: AFR R² vs fraction of epistatic loci (lines per strategy)
Panel B: AFR R² gain relative to standard GWAS at 25% epistatic loci
Panel C: Mechanism schematic (EUR vs AFR effective focal effect)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path(__file__).resolve().parents[1]
PHASE1  = BASE / "results" / "phase1"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
FIG_DIR = REGEN_DIR

COLORS = {
    "standard_gwas":     "#777777",
    "finemapped_causal": "#2980b9",
    "ancestry_specific": "#27ae60",
    "haplotype_aware":   "#c0392b",
}
LABELS = {
    "standard_gwas":     "S1: Standard GWAS PRS",
    "finemapped_causal": "S2: Fine-mapped causal PRS",
    "ancestry_specific": "S3: Ancestry-specific PRS (oracle)",
    "haplotype_aware":   "S4: Haplotype-aware PRS",
}
LINESTYLES = {
    "standard_gwas":     (0, (5, 2)),
    "finemapped_causal": "-",
    "ancestry_specific": "-.",
    "haplotype_aware":   "-",
}
MARKERS = {
    "standard_gwas":     "s",
    "finemapped_causal": "^",
    "ancestry_specific": "D",
    "haplotype_aware":   "o",
}
STRATEGIES = ["standard_gwas", "finemapped_causal", "ancestry_specific", "haplotype_aware"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":  9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    df = pd.read_csv(PHASE1 / "p1_prs_simulation.tsv", sep="\t")
    fracs  = sorted(df["frac_epistatic"].unique())
    eur_df = df[df["ancestry"] == "EUR"]
    afr_df = df[df["ancestry"] == "AFR"]

    fig = plt.figure(figsize=(13.5, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40,
                            left=0.07, right=0.97, top=0.87, bottom=0.17)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A: AFR R² vs fraction epistatic
    # ══════════════════════════════════════════════════════════════════════════

    # EUR reference (standard_gwas is representative)
    eur_ref = eur_df[eur_df["strategy"] == "standard_gwas"] \
                .set_index("frac_epistatic")["r2"].reindex(fracs)
    ax_a.axhline(eur_ref.mean(), color="#333333", linewidth=1.0, linestyle=":",
                 alpha=0.6, zorder=1, label="EUR R² (reference, all strats ≈ same)")

    for strat in STRATEGIES:
        r2_afr = afr_df[afr_df["strategy"] == strat] \
                    .set_index("frac_epistatic")["r2"].reindex(fracs)
        sd_afr = afr_df[afr_df["strategy"] == strat] \
                    .set_index("frac_epistatic")["r2_sd"].reindex(fracs)
        ax_a.plot(fracs, r2_afr.values,
                  color=COLORS[strat], linestyle=LINESTYLES[strat],
                  linewidth=2.0, marker=MARKERS[strat], markersize=5.5,
                  label=LABELS[strat], zorder=3)
        ax_a.fill_between(fracs,
                          r2_afr.values - sd_afr.values / np.sqrt(50),
                          r2_afr.values + sd_afr.values / np.sqrt(50),
                          color=COLORS[strat], alpha=0.15, zorder=2)

    ax_a.set_xlabel("Fraction of loci with cis-epistasis")
    ax_a.set_ylabel("R² in AFR test set")
    ax_a.set_title("A   AFR predictive accuracy by strategy")
    ax_a.set_xticks(fracs)
    ax_a.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax_a.legend(loc="upper right", frameon=False, fontsize=7.5,
                handlelength=2.5)

    # annotate S1=S2 at 25% mark
    frac25 = 0.25
    r2_s1  = afr_df[(afr_df["strategy"]=="standard_gwas") &
                    (afr_df["frac_epistatic"]==frac25)]["r2"].values[0]
    r2_s2  = afr_df[(afr_df["strategy"]=="finemapped_causal") &
                    (afr_df["frac_epistatic"]==frac25)]["r2"].values[0]
    ax_a.annotate("S1 = S2\n(fine-mapping\ndoesn't help)",
                  xy=(frac25, (r2_s1+r2_s2)/2), fontsize=7,
                  xytext=(frac25 + 0.06, r2_s1 - 0.04),
                  color="#2980b9",
                  arrowprops=dict(arrowstyle="->", color="#2980b9", lw=0.8))

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B: AFR R² gain relative to S1 at 25% epistatic loci
    # (shows how much each strategy improves over the baseline)
    # ══════════════════════════════════════════════════════════════════════════
    sub025  = df[df["frac_epistatic"] == 0.25]
    afr025  = sub025[sub025["ancestry"] == "AFR"].set_index("strategy")["r2"]
    eur025  = sub025[sub025["ancestry"] == "EUR"].set_index("strategy")["r2"]

    r2_s1_afr = afr025.get("standard_gwas", 0)
    gains = [afr025.get(s, 0) - r2_s1_afr for s in STRATEGIES]

    bars = ax_b.bar(range(4), gains,
                    color=[COLORS[s] for s in STRATEGIES],
                    width=0.6, edgecolor="white", linewidth=0.5)

    for i, (bar, g) in enumerate(zip(bars, gains)):
        va = "bottom" if g >= 0 else "top"
        offset = 0.0003 if g >= 0 else -0.0003
        ax_b.text(bar.get_x() + bar.get_width()/2, g + offset,
                  f"{g:+.4f}", ha="center", va=va, fontsize=7.5,
                  color=COLORS[STRATEGIES[i]])

    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(["S1\nStandard", "S2\nFine-\nmapped",
                           "S3\nAncestry-\nspecific", "S4\nHaplotype-\naware"],
                         fontsize=8)
    ax_b.set_ylabel(r"$\Delta$R² vs standard GWAS PRS (AFR)")
    ax_b.set_title("B   R² gain over standard PRS\nat 25% epistatic loci (AFR)")
    ax_b.axhline(0, color="gray", linewidth=0.8, linestyle="-")

    # Add ΔR² for S1->S4 and S1->S3
    ax_b.annotate("S1 ≡ S2:\nfine-mapping\nno benefit",
                  xy=(1, gains[1] + 0.0005), fontsize=7, color="#2980b9",
                  ha="center", va="bottom")

    r2_s1_eur = eur025.get("standard_gwas", 0)
    eur_afr_gap = r2_s1_eur - r2_s1_afr
    ax_b.text(0.5, min(gains) - 0.002,
              f"EUR−AFR R² gap for S1/S2: {eur_afr_gap:+.4f}",
              ha="center", va="top", fontsize=7.5, color="#333333",
              transform=ax_b.get_xaxis_transform())

    # ══════════════════════════════════════════════════════════════════════════
    # Panel C: Mechanism schematic
    # Shows focal variant effect on trait as function of background SNP dosage
    # EUR carriers have different average background -> different "effective β"
    # ══════════════════════════════════════════════════════════════════════════
    beta_f   =  0.50
    gamma    = -0.60
    bg_dos   = np.linspace(0, 2, 100)

    maf_eur  = 0.65    # high EUR background MAF (example from real adQTL loci)
    maf_afr  = 0.20    # lower AFR background MAF
    E_bg_eur = 2 * maf_eur
    E_bg_afr = 2 * maf_afr

    effect_curve = beta_f + gamma * bg_dos  # focal variant effect given bg dosage

    ax_c.plot(bg_dos, effect_curve, "-", color="#555555", linewidth=2.0,
              label=r"Focal effect = $\beta_f + \gamma \cdot bg$")

    # EUR average bg
    eff_eur = beta_f + gamma * E_bg_eur
    ax_c.axvline(E_bg_eur, color=COLORS["standard_gwas"], linewidth=1.5,
                 linestyle="--", alpha=0.8)
    ax_c.plot(E_bg_eur, eff_eur, "^", color=COLORS["standard_gwas"],
              markersize=11, zorder=5,
              label=f"EUR avg bg (MAF={maf_eur})\n-> GWAS β̂ = {eff_eur:.2f}")

    # AFR average bg
    eff_afr = beta_f + gamma * E_bg_afr
    ax_c.axvline(E_bg_afr, color=COLORS["haplotype_aware"], linewidth=1.5,
                 linestyle="--", alpha=0.8)
    ax_c.plot(E_bg_afr, eff_afr, "s", color=COLORS["haplotype_aware"],
              markersize=11, zorder=5,
              label=f"AFR avg bg (MAF={maf_afr})\n-> True AFR β = {eff_afr:.2f}")

    # Arrow showing the effect discrepancy
    ax_c.annotate("",
                  xy=(E_bg_eur, eff_eur),
                  xytext=(E_bg_eur, eff_afr),
                  arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax_c.text(E_bg_eur + 0.08, (eff_eur + eff_afr)/2,
              f"PRS bias\n= γ × 2Δf\n= {gamma:.1f}×{E_bg_eur-E_bg_afr:.2f}\n= {gamma*(E_bg_eur-E_bg_afr):.2f}",
              fontsize=7.5, color="black", va="center")

    ax_c.set_xlabel("Background SNP dosage (bg)")
    ax_c.set_ylabel(r"Focal variant effect on trait Y ($\times$ focal dosage)")
    ax_c.set_title("C   Epistasis mechanism: same causal focal SNP,\ndifferent effective β in EUR vs AFR")
    ax_c.set_xticks([0, E_bg_afr, 1, E_bg_eur, 2])
    ax_c.set_xticklabels(["0", f"{E_bg_afr:.1f}\n(AFR\navg bg)", "1",
                           f"{E_bg_eur:.1f}\n(EUR\navg bg)", "2"], fontsize=7.5)
    ax_c.legend(loc="upper right", frameon=False, fontsize=7.5)

    # Shaded region showing the bias is due to wrong background expectation
    ax_c.fill_betweenx([eff_afr, eff_eur], E_bg_eur, E_bg_eur,
                        alpha=0.0)  # invisible — just keeping for reference

    # Note box
    ax_c.text(0.02, 0.05,
              "Fine-mapping to the true focal SNP\ndoes NOT fix this — the bias\nlives in the EUR-calibrated β̂",
              transform=ax_c.transAxes, fontsize=7.5, color="#c0392b",
              bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="#c0392b", alpha=0.8))

    # ── global title ──────────────────────────────────────────────────────────
    suptitle(
        fig,
        "Figure S_. Cis-haplotype epistasis degrades PRS portability, "
        "not correctable by fine-mapping alone",
        fontsize=10, y=0.98, fontweight="bold",
    )

    for fmt in ("png", "pdf"):
        out = FIG_DIR / f"Figure_PRS_Portability.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    announce(__file__, "Figure_PRS_Portability")

    # ── print key numbers ─────────────────────────────────────────────────────
    print("\n=== Key numbers at 25% epistatic loci ===")
    print(f"{'Strategy':28s}  EUR R²    AFR R²    gap (EUR-AFR)")
    for s in STRATEGIES:
        e = eur025.get(s, np.nan)
        a = afr025.get(s, np.nan)
        print(f"  {s:25s}  {e:.4f}    {a:.4f}    {e-a:+.4f}")

    s1_afr = afr025.get("standard_gwas", 0)
    print(f"\n  AFR R² gain S1->S2 (fine-mapping): "
          f"{afr025.get('finemapped_causal',0)-s1_afr:+.4f}")
    print(f"  AFR R² gain S1->S3 (ancestry-specific): "
          f"{afr025.get('ancestry_specific',0)-s1_afr:+.4f}")
    print(f"  AFR R² gain S1->S4 (haplotype-aware): "
          f"{afr025.get('haplotype_aware',0)-s1_afr:+.4f}")

    plt.close("all")


if __name__ == "__main__":
    main()
