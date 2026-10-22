"""
Figure S5 — robustness and validation of the cis-haplotype epistasis signal.

Rebuilds all six panels from the source tables and computes every statistic at run
time (nothing hardcoded). Panel A uses the testable-loci denominators (159/189) for both
the bars and the OR/P annotation.

Panels:
  A  V5  Geuvadis EUR replication of the focal x background interaction
  B  V5  MAGE-vs-Geuvadis interaction direction concordance
  C  V7  focal x background significance before vs after ancestry covariate control
  D  V8  same-data vs cross-validated cis-background fraction, per locus
  E  V8  the same two distributions, showing the bias correction
  F  V9  background-SNP vs focal-SNP predicted TF ChIP-seq effect

Conventions
-----------
Two-sided Fisher exact tests throughout.

No in-figure "Figure S5." title by default: Cell Press typesets legends separately, and
a baked-in number goes stale when figures are renumbered. Pass --title if a journal wants
it burned into the image.

Output
------
Writes figures/FigureS5_v2.{png,pdf} by default -- it does NOT overwrite the
FigureS5.png currently embedded in the draft. Compare the two, then re-run with
--overwrite to replace it.

Usage:
    python scripts/v10_figure_s5_robustness.py
    python scripts/v10_figure_s5_robustness.py --overwrite      # replace figures/FigureS5.*
    python scripts/v10_figure_s5_robustness.py --title          # burn "Figure S5." into the image
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from lib_figure_io import REGEN_DIR

BASE = Path(__file__).resolve().parents[1]
P1 = BASE / "results" / "phase1"
VAL = BASE / "results" / "validation"
FIG = BASE / "figures"

for fam in ["Arial", "Helvetica", "DejaVu Sans"]:
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
AD = "#c0392b"    # adQTL
CO = "#9aa0a6"    # control
BL = "#2166ac"    # secondary / focal
OR_ = "#e08214"   # "original / biased" series

# collected at run time so the caption text can never drift from the figure
STATS = {}


def panel(ax, letter, title):
    ax.text(-0.16, 1.10, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=6)


def fisher_2x2(hit_a, n_a, hit_b, n_b):
    odds, p2 = stats.fisher_exact([[hit_a, n_a - hit_a], [hit_b, n_b - hit_b]])
    return odds, p2


# ------------------------------------------------------------------ A, B (V5)

def panel_a_b(ax_a, ax_b):
    v5 = pd.read_csv(VAL / "v5_geuvadis_focal_bg.tsv", sep="\t")
    a = v5[v5.set == "adqtl"]
    c = v5[v5.set == "control"]

    # denominator = loci where the interaction was estimable in Geuvadis.
    # The old figure drew the bars on all rows (162/191) but annotated OR/P from
    # this fit -- the two must agree.
    pa, pc = a.geuv_int_p.dropna(), c.geuv_int_p.dropna()
    ha, na = int((pa < 0.05).sum()), len(pa)
    hc, nc = int((pc < 0.05).sum()), len(pc)
    ra, rc = 100 * ha / na, 100 * hc / nc
    odds, p2 = fisher_2x2(ha, na, hc, nc)
    STATS["A"] = dict(adqtl=(ha, na, ra), control=(hc, nc, rc), OR=odds, p_two=p2)

    ax_a.bar([0, 1], [ra, rc], color=[AD, CO], width=0.6)
    for i, (r, h, n) in enumerate([(ra, ha, na), (rc, hc, nc)]):
        ax_a.text(i, r + 0.5, f"{r:.1f}%\n({h}/{n})", ha="center", fontsize=7)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["adQTL", "concordant\ncontrol"])
    ax_a.set_ylabel("% with Geuvadis focal × bg\ninteraction P < 0.05")
    ax_a.set_ylim(0, max(ra, rc) * 1.45)
    ax_a.text(0.97, 0.95, f"OR = {odds:.2f}\ntwo-sided P = {p2:.3f}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=7,
              bbox=dict(boxstyle="round,pad=0.35", fc="#fdf6e3", ec="#b58900", lw=0.6))
    panel(ax_a, "A", "Geuvadis replication\n(276 EUR donors independent of MAGE)")

    # B: does the interaction point the same way in MAGE and Geuvadis?
    sig = a[a.mage_bonf_p < 0.05].sign_conc.dropna()
    ns = a[a.mage_bonf_p >= 0.05].sign_conc.dropna()
    vals = [100 * sig.mean(), 100 * ns.mean()]
    ns_counts = [(int(sig.sum()), len(sig)), (int(ns.sum()), len(ns))]
    STATS["B"] = dict(mage_sig=(ns_counts[0], vals[0]), mage_ns=(ns_counts[1], vals[1]))

    ax_b.bar([0, 1], vals, color=[AD, "#a6bddb"], width=0.6)
    for i, (v, (h, n)) in enumerate(zip(vals, ns_counts)):
        ax_b.text(i, v + 1.2, f"{v:.1f}%\n({h}/{n})", ha="center", fontsize=7)
    ax_b.axhline(50, ls="--", color="k", lw=0.7)
    # keep clear of the bar-top labels, which sit just above the 50% line
    ax_b.text(-0.46, 46.5, "50% chance", fontsize=6.5, ha="left", va="top")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["MAGE\nBonf-significant", "MAGE\nnon-significant"])
    ax_b.set_ylabel("% interaction sign concordance\n(MAGE vs Geuvadis)")
    ax_b.set_ylim(0, 80)
    panel(ax_b, "B", "Direction concordance")


# ------------------------------------------------------------------ C (V7)

def panel_c(ax):
    v7 = pd.read_csv(VAL / "v7_ancestry_model.tsv", sep="\t")
    d = v7.dropna(subset=["p_focal_bg_orig", "p_focal_bg_anc"]).copy()
    # guard against exact zeros before log-transforming
    floor = np.nextafter(0, 1)
    d["x"] = -np.log10(d.p_focal_bg_orig.clip(lower=floor))
    d["y"] = -np.log10(d.p_focal_bg_anc.clip(lower=floor))
    ad, co = d[d.set == "adqtl"], d[d.set == "control"]

    # the manuscript's r = 0.76 is the adQTL subset; the all-rows value is 0.723.
    r_ad = stats.spearmanr(ad.x, ad.y).statistic
    r_all = stats.spearmanr(d.x, d.y).statistic
    keep_ad = int((ad.p_focal_bg_anc < 0.05).sum())
    keep_co = int((co.p_focal_bg_anc < 0.05).sum())
    STATS["C"] = dict(r_adqtl=r_ad, n_adqtl=len(ad), r_all=r_all, n_all=len(d),
                      keep_adqtl=(keep_ad, len(ad)), keep_control=(keep_co, len(co)))

    ax.scatter(co.x, co.y, s=9, color=BL, alpha=0.45, lw=0, label=f"control (n={len(co)})")
    ax.scatter(ad.x, ad.y, s=11, color=AD, alpha=0.6, lw=0, label=f"adQTL (n={len(ad)})")
    lim = max(d.x.max(), d.y.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="k", lw=0.7)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    # Arial has no subscript glyphs -- use mathtext, which matplotlib renders itself
    ax.set_xlabel("focal $\\times$ bg  $-\\log_{10}P$  (original)")
    ax.set_ylabel("focal $\\times$ bg  $-\\log_{10}P$\n(+ ancestry covariate)")
    ax.text(0.03, 0.97, f"Spearman r = {r_ad:.3f} (adQTL)\n"
                        f"{100 * keep_ad / len(ad):.1f}% adQTL, "
                        f"{100 * keep_co / len(co):.1f}% control retain P < 0.05",
            transform=ax.transAxes, va="top", fontsize=6.8)
    ax.legend(frameon=False, loc="lower right", markerscale=1.4)
    panel(ax, "C", "Robust to ancestry covariate control")


# ------------------------------------------------------------------ D, E (V8)

def panel_d_e(ax_d, ax_e):
    v8 = pd.read_csv(VAL / "v8_splitsample_decomp.tsv", sep="\t")
    d = v8.dropna(subset=["frac_cis_multi_orig", "frac_cv"])
    r = stats.spearmanr(d.frac_cis_multi_orig, d.frac_cv).statistic
    conc = 100 * (np.sign(d.frac_cis_multi_orig) == np.sign(d.frac_cv)).mean()
    m_orig, m_cv = d.frac_cis_multi_orig.median(), d.frac_cv.median()
    STATS["DE"] = dict(r=r, n=len(d), median_orig=m_orig, median_cv=m_cv, sign_conc=conc)

    ax_d.scatter(d.frac_cis_multi_orig, d.frac_cv, s=14, color=AD, alpha=0.6, lw=0)
    lo = min(d.frac_cis_multi_orig.min(), d.frac_cv.min())
    hi = max(d.frac_cis_multi_orig.max(), d.frac_cv.max())
    ax_d.plot([lo, hi], [lo, hi], ls="--", color="k", lw=0.7, label="y = x (no bias)")
    ax_d.set_xlabel("apparent fraction (same data)")
    ax_d.set_ylabel("cross-validated fraction\n(split-sample)")
    ax_d.text(0.03, 0.97, f"Spearman r = {r:.2f}  (n = {len(d)})\n"
                          f"median {m_orig:.2f} → {m_cv:.2f}\n"
                          f"{conc:.1f}% sign-concordant",
              transform=ax_d.transAxes, va="top", fontsize=6.8)
    ax_d.legend(frameon=False, loc="lower right")
    panel(ax_d, "D", "Bias-reduced decomposition")

    bins = np.linspace(-0.6, 1.6, 26)
    ax_e.hist(d.frac_cis_multi_orig.clip(-0.6, 1.6), bins=bins, color=OR_, alpha=0.65,
              label=f"apparent (median {m_orig:.2f})")
    ax_e.hist(d.frac_cv.clip(-0.6, 1.6), bins=bins, color=AD, alpha=0.6,
              label=f"cross-validated (median {m_cv:.2f})")
    ax_e.axvline(m_orig, color=OR_, lw=1.4)
    ax_e.axvline(m_cv, color=AD, lw=1.4)
    ax_e.set_xlabel("fraction of ObsΔ explained by cis-background")
    ax_e.set_ylabel("# adQTL loci")
    ax_e.legend(frameon=False, loc="upper right")
    panel(ax_e, "E", "Selection bias correction")


# ------------------------------------------------------------------ F (V9)

def panel_f(ax):
    # The focal-vs-background comparison lives in the vignette table. Do NOT use
    # v9_bg_tf_analysis.tsv's adQTL-vs-control contrast: its control arm is n = 4.
    tv = pd.read_csv(P1 / "p1_tf_vignette.tsv", sep="\t")
    bg = tv.bg_chip_tf.dropna()
    fo = tv.focal_chip_tf.dropna()
    p = stats.mannwhitneyu(fo, bg, alternative="two-sided").pvalue
    STATS["F"] = dict(n_bg=len(bg), n_focal=len(fo), median_bg=bg.median(),
                      median_focal=fo.median(), mwu_p=p)

    bins = np.linspace(0, max(bg.max(), fo.max()), 22)
    ax.hist(fo, bins=bins, color=BL, alpha=0.55, label=f"focal SNP (n = {len(fo)})")
    ax.hist(bg, bins=bins, color=AD, alpha=0.55, label=f"background SNP (n = {len(bg)})")
    ax.axvline(fo.median(), color=BL, lw=1.4)
    ax.axvline(bg.median(), color=AD, lw=1.4)
    ax.set_xlabel("|AlphaGenome TF ChIP-seq effect|")
    ax.set_ylabel("# variants")
    ax.text(0.97, 0.97, f"median {fo.median():.3f} vs {bg.median():.3f}\nMWU P = {p:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8)
    ax.legend(frameon=False, loc="center right")
    panel(ax, "F", "Background SNPs match focal SNPs\nin predicted TF disruption")


def main():
    overwrite = "--overwrite" in sys.argv
    with_title = "--title" in sys.argv

    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42)
    panel_a_b(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]))
    panel_d_e(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    panel_f(fig.add_subplot(gs[1, 2]))

    if with_title:
        # only on request: a baked-in number is what went stale last time
        fig.suptitle("Figure S5.  Robustness and validation of the cis-haplotype "
                     "epistasis signal", fontsize=9.5, fontweight="bold",
                     x=0.02, ha="left", y=1.0)

    # same policy as every other figure script (see lib_figure_io): render into
    # figures/regenerated/ and let finalize_figures.py promote it deliberately.
    for ext in ["png", "pdf"]:
        out = (FIG if overwrite else REGEN_DIR) / f"FigureS5.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[figS5] wrote {out}")
    plt.close(fig)

    a, c_, de, f = STATS["A"], STATS["C"], STATS["DE"], STATS["F"]
    print("\n--- numbers drawn (cross-check against the legend) ---")
    print(f"  A  adQTL {a['adqtl'][0]}/{a['adqtl'][1]} = {a['adqtl'][2]:.1f}%  "
          f"control {a['control'][0]}/{a['control'][1]} = {a['control'][2]:.1f}%  "
          f"OR={a['OR']:.2f}  two-sided P={a['p_two']:.4f}")
    print(f"  B  MAGE-sig {STATS['B']['mage_sig'][1]:.1f}%  "
          f"MAGE-ns {STATS['B']['mage_ns'][1]:.1f}%")
    print(f"  C  Spearman r={c_['r_adqtl']:.3f} (adQTL n={c_['n_adqtl']}); "
          f"all rows r={c_['r_all']:.3f} (n={c_['n_all']}); "
          f"retain P<0.05: adQTL {c_['keep_adqtl'][0]}/{c_['keep_adqtl'][1]}, "
          f"control {c_['keep_control'][0]}/{c_['keep_control'][1]}")
    print(f"  DE r={de['r']:.3f} n={de['n']}  median {de['median_orig']:.4f} -> "
          f"{de['median_cv']:.4f}  sign-conc {de['sign_conc']:.1f}%")
    print(f"  F  focal median {f['median_focal']:.4f} vs bg {f['median_bg']:.4f}  "
          f"MWU P={f['mwu_p']:.4f}  n={f['n_focal']}")
    if not overwrite:
        print("\n  (wrote into figures/regenerated/; re-run with --overwrite, or use "
              "finalize_figures.py, to replace the draft's FigureS5)")


if __name__ == "__main__":
    main()
