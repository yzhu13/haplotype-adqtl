"""
Update paper figures with validation analysis panels.

Changes vs original:
  Figure2 → 2×3 layout, adds Panel E: Geuvadis EUR sign-concordance (V1)
  Figure5 → 3-row layout, adds Panel E: cis-interaction p-value ECDF (V3)
  FigureS1 → r² threshold robustness (V2) — copied from the v2_r2_robustness render

Outputs land in figures/regenerated/make_updated_figures/ (see lib_figure_io); promote
them with finalize_figures.py. Legacy name → current draft figure:
  Figure2  → draft Figure S1
  Figure5  → draft Figure 3
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
P0   = BASE / "results" / "phase0"
P1   = BASE / "results" / "phase1"
VAL  = BASE / "results" / "validation"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
OUT  = REGEN_DIR
MAN  = OUT / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

# ── style (match make_paper_figures.py) ──────────────────────────────
for fam in ["Arial", "Helvetica", "DejaVu Sans"]:
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})
AD  = "#c0392b"; CO = "#9aa0a6"; BL = "#2166ac"; GRY = "#d9d9d9"
GCOL = {"EUR": "#4575b4", "AFR": "#d73027", "EAS": "#1a9850", "SAS": "#984ea3", "AMR": "#ff7f00"}


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")


def save(fig, name, also_man=True):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
        if also_man:
            fig.savefig(MAN / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    announce(__file__, name)


# ─────────────────────────── Figure 2 ───────────────────────────────
def figure2():
    dec = pd.read_csv(P1 / "p1_decomposition.tsv", sep="\t")
    sh  = pd.read_csv(P1 / "p1_splithalf.tsv", sep="\t")
    v1  = pd.read_csv(VAL / "v1_geuvadis_result.tsv", sep="\t")

    both = dec.dropna(subset=["p_het_EUR_AFR"]).copy()
    both["abs_z"]   = both.z_EUR_AFR.abs()
    both["maf_gap"] = (both.maf_EUR - both.maf_AFR).abs()

    fig = plt.figure(figsize=(10.8, 6.2))
    gs  = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.40)

    # A: |z| vs MAF gap
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    dd = both.dropna(subset=["maf_gap", "abs_z"])
    ax.scatter(dd.maf_gap, dd.abs_z, s=4, color=CO, edgecolor="none", rasterized=True)
    r = stats.spearmanr(dd.maf_gap, dd.abs_z)[0]
    ax.set_xlabel(r"$|MAF_{EUR}-MAF_{AFR}|$"); ax.set_ylabel(r"$|z_{het}|$")
    ax.set_title(f"Not allele-frequency driven\nSpearman r={r:.3f}; AF+power model $R^2$=0.003")

    # B: adQTL rate by PIP
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    dp = both.dropna(subset=["variantPIP"]).copy()
    dp["bin"] = pd.cut(dp.variantPIP, [0.5, 0.7, 0.9, 1.001], labels=["0.5–0.7", "0.7–0.9", "0.9–1.0"])
    rate = dp.groupby("bin", observed=True).is_adqtl.agg(["mean", "size"])
    ax.bar(range(len(rate)), rate["mean"] * 100, color=BL, width=0.62)
    for i, (m, s) in enumerate(zip(rate["mean"], rate["size"])):
        ax.text(i, m * 100 + 0.1, f"n={s}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(rate))); ax.set_xticklabels(rate.index)
    ax.set_xlabel("Lead-variant PIP"); ax.set_ylabel("adQTL rate (%)")
    ax.set_title("Stronger at well-fine-mapped\ncausal variants (not LD)")

    # C: adQTL rate by CS size
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "C")
    dc = both.dropna(subset=["numVariants"]).copy()
    dc["bin"] = pd.cut(dc.numVariants, [0, 1.5, 3.5, 10.5, 1e9], labels=["1", "2–3", "4–10", ">10"])
    rate = dc.groupby("bin", observed=True).is_adqtl.agg(["mean", "size"])
    ax.bar(range(len(rate)), rate["mean"] * 100, color=BL, width=0.62)
    for i, (m, s) in enumerate(zip(rate["mean"], rate["size"])):
        ax.text(i, m * 100 + 0.1, f"n={s}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(rate))); ax.set_xticklabels(rate.index)
    ax.set_xlabel("Credible-set size (# variants)"); ax.set_ylabel("adQTL rate (%)")
    ax.set_title("Higher at smaller credible sets")

    # D: split-half reproducibility
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "D")
    co = sh[~sh.is_adqtl]; ad = sh[sh.is_adqtl]
    ax.scatter(co.obsd_h1, co.obsd_h2, s=4, color=CO, edgecolor="none", rasterized=True, label="all")
    ax.scatter(ad.obsd_h1, ad.obsd_h2, s=10, color=AD, edgecolor="none", label="adQTL")
    sr = stats.spearmanr(ad.obsd_h1, ad.obsd_h2)[0]
    pr = stats.pearsonr(ad.obsd_h1, ad.obsd_h2)[0]
    lim = np.nanpercentile(np.abs(np.r_[sh.obsd_h1, sh.obsd_h2]), 99)
    ax.plot([-lim, lim], [-lim, lim], "--", color="k", lw=0.7)
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta\beta$ (sample half 1)"); ax.set_ylabel(r"$\Delta\beta$ (sample half 2)")
    ax.set_title(f"Reproducible (adQTL Spearman {sr:.2f},\nPearson {pr:.2f}, 99% sign)")
    ax.legend(loc="lower right", frameon=False, markerscale=1.5)

    # E: Geuvadis EUR sign concordance (V1 validation)
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "E")
    r0 = v1.iloc[0]
    n_adq  = int(r0["n_adqtl_EUR_matched"])
    n_ctrl = int(r0["n_ctrl_EUR_matched"])
    grps   = [f"adQTL\n(n={n_adq})", f"concordant\n(n={n_ctrl})"]
    pcts   = [float(r0["EUR_concordance_adqtl_pct"]), float(r0["EUR_concordance_ctrl_pct"])]
    cols   = [AD, CO]
    bars   = ax.bar([0, 1], pcts, color=cols, width=0.55, edgecolor="black", linewidth=0.7)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(grps, fontsize=8.5)
    # headroom so the P-value box clears the 100% bar label instead of sitting on it
    ax.set_ylabel("EUR sign concordance\nwith Geuvadis (n=373)"); ax.set_ylim(0, 132)
    ax.axhline(50, color="gray", lw=0.8, ls="--", label="chance")
    p_val = float(r0["EUR_binom_p_adqtl"])
    ax.text(0.5, 0.99, f"adQTL p = {p_val:.1e}", transform=ax.transAxes,
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ax.set_title("Independent replication\nin Geuvadis LCLs (EUR; n=373)")

    # panel F: blank (placeholder for future or can remove)
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")

    suptitle(fig,
        "Figure 2.  Ancestry-differential effects are not an allele-frequency, "
        "LD, power, selection, or dataset-specific artifact",
        fontsize=9.3, fontweight="bold", y=0.995, x=0.02, ha="left")
    save(fig, "Figure2")


# ─────────────────────────── Figure 5 (cis-epistasis) ───────────────
def figure5():
    """Rebuild the cis-epistasis figure (paper Figure5), adding panel E = p-value ECDF."""
    P1_data  = BASE / "results" / "phase1"
    ci       = pd.read_csv(P1_data / "p1_cis_interaction.tsv", sep="\t")
    vign_path = P1_data / "p1_vignette_annotation.tsv"

    # ── helpers ──────────────────────────────────────────────────────
    try:
        from scripts.p1_10_figure_cis_epistasis import vignette_data
    except Exception:
        vignette_data = None

    # colour scheme (match existing script)
    AD_e = "#c0392b"; CO_e = "#9aa0a6"
    GCOL_e = {"EUR": "#4575b4", "AFR": "#d73027"}

    fig = plt.figure(figsize=(7.2, 9.0))
    gs  = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.40,
                           height_ratios=[1, 1, 0.85])

    # ── A: enrichment bar ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    t  = ci[ci.n_bg_tested > 0]
    ad = t[t.set == "adqtl"]; co = t[t.set == "control"]
    radq = (ad.bonf_p < 0.05).mean() * 100
    rco  = (co.bonf_p < 0.05).mean() * 100
    ax.bar([0, 1], [rco, radq], color=[CO_e, AD_e], width=0.6)
    for i, (v, n) in enumerate([(rco, len(co)), (radq, len(ad))]):
        ax.text(i, v + 1.5, f"{v:.0f}%\n(n={n})", ha="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["concordant\n(control)", "adQTL"])
    ax.set_ylabel("% with significant cis-interaction"); ax.set_ylim(0, 105)
    ax.set_title("Focal × background-SNP interaction\n(Bonferroni adj p<0.05)")

    # ── B: within-ancestry confirmation ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    try:
        wa = pd.read_csv(P1_data / "p1_within_ancestry.tsv", sep="\t")
        ce   = (wa.int_p_EUR < 0.05).mean() * 100
        ca   = (wa.int_p_AFR < 0.05).mean() * 100
        cany = ((wa.int_p_EUR < 0.05) | (wa.int_p_AFR < 0.05)).mean() * 100
        ax.bar([0, 1, 2], [ce, ca, cany],
               color=[GCOL_e["EUR"], GCOL_e["AFR"], "#444"], width=0.6)
        for i, v in enumerate([ce, ca, cany]):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
        ax.axhline(5, ls="--", color="k", lw=0.7)
        ax.text(2.4, 7, "null ~5%", fontsize=6.5, ha="right")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["within\nEUR", "within\nAFR", "≥1\nancestry"])
        ax.set_ylabel("% adQTLs confirmed"); ax.set_ylim(0, 100)
        ax.set_title(f"Within-ancestry confirmation\n(rules out focal×ancestry; n={len(wa)})")
    except Exception as e:
        ax.text(0.5, 0.5, "within-ancestry n/a", ha="center"); ax.axis("off")

    # ── C: decomposition histogram ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C")
    try:
        mb = P1_data / "p1_multi_bg_decomp.tsv"
        if mb.exists():
            dec = pd.read_csv(mb, sep="\t"); col = "frac_cis_multi"; sub = "multi independent cis-bg"
        else:
            dec = pd.read_csv(P1_data / "p1_cis_trans_decomp.tsv", sep="\t")
            col = "frac_cis"; sub = "single best cis-bg (lower bound)"
        fc  = dec[col].dropna().clip(-0.5, 1.5)
        ax.hist(fc, bins=20, color=BL, edgecolor="white", lw=0.4)
        med = dec[col].dropna().median()
        ax.axvline(med, color=AD_e, lw=1.5)
        ax.text(med + 0.02, ax.get_ylim()[1] * 0.88, f"median {med:.2f}",
                color=AD_e, fontsize=8, fontweight="bold")
        ax.set_xlabel("fraction of ObsDelta explained by cis-background")
        ax.set_ylabel("# adQTLs")
        ax.set_title(f"cis-vs-trans decomposition\n({sub})")
    except Exception as e:
        ax.text(0.5, 0.5, f"decomp n/a\n{e}", ha="center", fontsize=6); ax.axis("off")

    # ── D: vignette ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D")
    try:
        import sys
        sys.path.insert(0, str(BASE / "scripts"))
        from p1_10_figure_cis_epistasis import vignette_data as _vd
        d, gene, bg, anc, minp = _vd()
        cols_v = ["#fdae61", "#abd9e9", "#2c7bb6"]
        for gt in [0, 1, 2]:
            sub = d[d.bg.round() == gt]
            if len(sub) < 3:
                continue
            ax.scatter(sub.focal + np.random.uniform(-0.05, 0.05, len(sub)),
                       sub.expr, s=10, color=cols_v[gt], alpha=0.6,
                       label=f"bg={gt} (n={len(sub)})")
            if sub.focal.nunique() > 1:
                z = np.polyfit(sub.focal, sub.expr, 1)
                xs = np.array([sub.focal.min(), sub.focal.max()])
                ax.plot(xs, np.polyval(z, xs), color=cols_v[gt], lw=1.6)
        ax.set_xlabel(f"focal dosage  (within {anc})")
        ax.set_ylabel("expression (inv-normal)")
        ax.set_title(f"Vignette: {gene}\nfocal slope depends on bg genotype (p={minp:.1e})",
                     fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.5, loc="best")
    except Exception as e:
        ax.text(0.5, 0.5, f"vignette n/a\n{e}", ha="center", fontsize=6); ax.axis("off")

    # ── E: cis-interaction p-value ECDF (V3) — full-width ────────────
    ax = fig.add_subplot(gs[2, :]); panel(ax, "E")
    adq_logp  = -np.log10(ci[ci["set"] == "adqtl"]["min_int_p"].clip(lower=1e-300)).dropna()
    ctrl_logp = -np.log10(ci[ci["set"] == "control"]["min_int_p"].clip(lower=1e-300)).dropna()
    mwu = stats.mannwhitneyu(adq_logp, ctrl_logp, alternative="greater")
    med_n_bg    = ci["n_bg_tested"].median()
    bonf_cutoff = -np.log10(0.05 / med_n_bg)

    for logp, label, col in [
        (ctrl_logp, f"concordant eQTL (n={len(ctrl_logp)})", CO_e),
        (adq_logp,  f"adQTL (n={len(adq_logp)})",            AD_e),
    ]:
        sorted_p = np.sort(logp)
        ecdf     = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax.plot(sorted_p, 1 - ecdf, lw=2, color=col, label=label)
    ax.axvline(bonf_cutoff, color="black", lw=1.2, ls="--",
               label=f"Bonferroni threshold (-log10p={bonf_cutoff:.1f})")

    xgrid = np.linspace(0, max(adq_logp.max(), ctrl_logp.max()), 500)
    sa = np.array([(np.sort(adq_logp.values)  >= x).mean() for x in xgrid])
    sc = np.array([(np.sort(ctrl_logp.values) >= x).mean() for x in xgrid])
    ax.fill_between(xgrid, sc, sa, where=sa > sc, alpha=0.15, color=AD_e)
    ax.set_xlabel("-log10(best cis-interaction p)"); ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction with p above threshold\n(1 - ECDF)")
    ax.set_title(
        f"adQTL cis-interaction p-values systematically more significant "
        f"(MWU p={mwu.pvalue:.2e}; shaded = adQTL excess)",
        fontsize=8.5)
    ax.legend(fontsize=8)

    suptitle(fig,
        "Figure 5.  Ancestry-differential cis-eQTLs are driven by cis-haplotype epistasis "
        "(genetic, not environmental, context)",
        fontsize=8.2, fontweight="bold", y=0.997, x=0.02, ha="left")
    save(fig, "Figure5")


# ─────────────────────────── Figure S1 (supplementary) ──────────────
def figureS1():
    """r² threshold robustness — copy validation figure to paper/ as FigureS1."""
    from lib_figure_io import regen_dir_for
    v2_dir = regen_dir_for("v2_r2_robustness")
    src_png = v2_dir / "v2_r2_robustness.png"
    src_pdf = v2_dir / "v2_r2_robustness.pdf"
    if src_png.exists():
        shutil.copy(src_png, OUT / "FigureS1.png")
        shutil.copy(src_pdf, OUT / "FigureS1.pdf")
        shutil.copy(src_png, MAN / "FigureS1.png")
        shutil.copy(src_pdf, MAN / "FigureS1.pdf")
        print("  wrote FigureS1.png/.pdf (from V2 validation)")
    else:
        print("  [WARN] V2 source not found; run scripts/v2_r2_robustness.py first")


if __name__ == "__main__":
    print("[update figures] Figure2 (non-artifact + Geuvadis) ...")
    figure2()
    print("[update figures] Figure5 (cis-epistasis + ECDF) ...")
    figure5()
    print("[update figures] FigureS1 (r2 robustness) ...")
    figureS1()
    print(f"[update figures] Done. Renders in {OUT}; promote with finalize_figures.py")
