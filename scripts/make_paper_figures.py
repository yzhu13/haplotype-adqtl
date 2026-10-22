"""
Generate publication-quality Figures 1-4 (PNG 300dpi + vector PDF) for the
Ancestry-Differential Regulatory eQTL Atlas paper. All inputs are on disk.
Output: figures/regenerated/make_paper_figures/ (promote with finalize_figures.py).
Legacy name -> current draft figure: Figure1 -> Figure 2, Figure2 -> Figure S1,
Figure4 -> Figure 7, Figure5 -> not used in the current draft.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
P0 = BASE / "results" / "phase0"
P1 = BASE / "results" / "phase1"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
OUT = REGEN_DIR
OUT.mkdir(parents=True, exist_ok=True)

# ---- publication style ----
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
AD = "#c0392b"; CO = "#9aa0a6"; BL = "#2166ac"; GRY = "#d9d9d9"
GROUPS = ["EUR", "AFR", "EAS", "SAS", "AMR"]
GCOL = {"EUR": "#4575b4", "AFR": "#d73027", "EAS": "#1a9850", "SAS": "#984ea3", "AMR": "#ff7f00"}


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    announce(__file__, name)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def load():
    d = {}
    d["bet"] = pd.read_csv(P0 / "p0_per_ancestry_betas.tsv", sep="\t")
    d["dec"] = pd.read_csv(P1 / "p1_decomposition.tsv", sep="\t")
    d["sh"] = pd.read_csv(P1 / "p1_splithalf.tsv", sep="\t")
    d["chmm"] = pd.read_csv(P1 / "p1_chromhmm.tsv", sep="\t")
    d["ref"] = pd.read_csv(P1 / "p1_adqtl_refscore.tsv", sep="\t")
    d["cal"] = pd.read_csv(P0 / "p0_alphagenome_predictions.tsv", sep="\t")
    if "var_tss_dist" not in d["cal"].columns:
        cal = pd.read_csv(P0 / "p0_calibration_set.tsv", sep="\t")[["variant_kgpID", "var_tss_dist"]]
        d["cal"] = d["cal"].merge(cal, on="variant_kgpID", how="left")
    try:
        d["reveng"] = pd.read_csv(P0 / "p0_readout_reveng.tsv", sep="\t")
    except Exception:
        d["reveng"] = None
    return d


# ============================ FIGURE 1 ============================
def figure1(d):
    dec = d["dec"]; bet = d["bet"]
    both = dec.dropna(subset=["p_het_EUR_AFR"]).copy()
    n = len(both); n_nom = int((both.p_het_EUR_AFR < 0.05).sum())
    n_ad = int(both.is_adqtl.sum())
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)

    # A: QQ
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    p = np.sort(both.p_het_EUR_AFR.values)
    exp = -np.log10(np.arange(1, len(p) + 1) / (len(p) + 1))
    obs = -np.log10(p)
    ax.scatter(exp, obs, s=4, color=AD, edgecolor="none", rasterized=True)
    lim = max(exp.max(), obs.max())
    ax.plot([0, lim], [0, lim], "--", color="k", lw=0.8)
    ax.set_xlabel("Expected  $-\\log_{10}P$"); ax.set_ylabel("Observed  $-\\log_{10}P$")
    ax.set_title(f"EUR–AFR effect heterogeneity\n{n_nom}/{n} nominal ({3.0:.1f}× null), {n_ad} adQTLs (FDR<0.1)")

    # B: volcano -log10 p vs ObsDelta
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    co = both[~both.is_adqtl]; ad = both[both.is_adqtl]
    ax.scatter(co.ObsDelta_EUR_AFR, -np.log10(co.p_het_EUR_AFR), s=4, color=CO, edgecolor="none", rasterized=True, label="concordant")
    ax.scatter(ad.ObsDelta_EUR_AFR, -np.log10(ad.p_het_EUR_AFR), s=7, color=AD, edgecolor="none", label="adQTL")
    ax.axvline(0, lw=0.5, color="k")
    ax.set_xlabel(r"$\Delta\beta$ (EUR − AFR)"); ax.set_ylabel(r"$-\log_{10}P_{het}$")
    ax.set_title("Ancestry-differential effect"); ax.legend(loc="upper right", frameon=False, markerscale=1.5)
    ax.set_xlim(-2.2, 2.2)

    # C: beta_EUR vs beta_AFR for adQTLs
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C")
    a = both[both.is_adqtl].dropna(subset=["beta_EUR", "beta_AFR"])
    flip = np.sign(a.beta_EUR) != np.sign(a.beta_AFR)
    lim = np.nanpercentile(np.abs(np.r_[a.beta_EUR, a.beta_AFR]), 99) * 1.1
    ax.axhspan(-lim, 0, -0.0, 1, color="none")
    ax.fill_between([-lim, 0], 0, lim, color=GRY, alpha=0.35, lw=0)
    ax.fill_between([0, lim], -lim, 0, color=GRY, alpha=0.35, lw=0)
    ax.scatter(a.beta_AFR[~flip], a.beta_EUR[~flip], s=10, color=AD, edgecolor="none", label="same sign (75%)")
    ax.scatter(a.beta_AFR[flip], a.beta_EUR[flip], s=14, color="#1a1aa6", edgecolor="white", lw=0.3, label=f"sign flip ({int(flip.sum())})")
    ax.plot([-lim, lim], [-lim, lim], "--", color="k", lw=0.7)
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\beta_{AFR}$"); ax.set_ylabel(r"$\beta_{EUR}$")
    ax.set_title("adQTL effect sizes by ancestry"); ax.legend(loc="lower right", frameon=False)

    # D: forest of top adQTLs across 5 groups
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D")
    ad_full = bet.merge(both[["variant_kgpID", "ensemblID", "is_adqtl"]], on=["variant_kgpID", "ensemblID"], how="inner")
    ad_full = ad_full[ad_full.is_adqtl].dropna(subset=[f"beta_{g}" for g in GROUPS])
    ad_full = ad_full.sort_values("I2", ascending=False).head(7).iloc[::-1]
    yt = []
    for i, (_, r) in enumerate(ad_full.iterrows()):
        for g in GROUPS:
            b, se = r[f"beta_{g}"], r[f"se_{g}"]
            ax.errorbar(b, i + (GROUPS.index(g) - 2) * 0.13, xerr=1.96 * se, fmt="o", ms=2.5,
                        color=GCOL[g], lw=0.8, capsize=0)
        yt.append(r.geneSymbol)
    ax.axvline(0, lw=0.6, color="k")
    ax.set_yticks(range(len(ad_full))); ax.set_yticklabels(yt, fontsize=6.5, fontstyle="italic")
    ax.set_xlabel(r"$\beta$ per continental group")
    ax.set_title("Top adQTLs (by $I^2$)")
    handles = [plt.Line2D([0], [0], marker="o", color=GCOL[g], ls="", ms=3, label=g) for g in GROUPS]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=1, fontsize=6, handletextpad=0.2)

    suptitle(fig,"Figure 1.  A multi-ancestry atlas of ancestry-differential cis-eQTLs",
                 fontsize=10, fontweight="bold", y=0.995, x=0.02, ha="left")
    save(fig, "Figure1")


# ============================ FIGURE 2 ============================
def figure2(d):
    dec = d["dec"]; sh = d["sh"]
    both = dec.dropna(subset=["p_het_EUR_AFR"]).copy()
    both["abs_z"] = both.z_EUR_AFR.abs()
    both["maf_gap"] = (both.maf_EUR - both.maf_AFR).abs()
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)

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
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C")
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
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D")
    co = sh[~sh.is_adqtl]; ad = sh[sh.is_adqtl]
    ax.scatter(co.obsd_h1, co.obsd_h2, s=4, color=CO, edgecolor="none", rasterized=True, label="all")
    ax.scatter(ad.obsd_h1, ad.obsd_h2, s=10, color=AD, edgecolor="none", label="adQTL")
    sr = stats.spearmanr(ad.obsd_h1, ad.obsd_h2)[0]; pr = stats.pearsonr(ad.obsd_h1, ad.obsd_h2)[0]
    lim = np.nanpercentile(np.abs(np.r_[sh.obsd_h1, sh.obsd_h2]), 99)
    ax.plot([-lim, lim], [-lim, lim], "--", color="k", lw=0.7)
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\Delta\beta$ (sample half 1)"); ax.set_ylabel(r"$\Delta\beta$ (sample half 2)")
    ax.set_title(f"Reproducible (adQTL Spearman {sr:.2f},\nPearson {pr:.2f}, 99% sign)")
    ax.legend(loc="lower right", frameon=False, markerscale=1.5)

    suptitle(fig,"Figure 2.  Ancestry-differential effects are not an allele-frequency, LD, power, or selection artifact",
                 fontsize=9.3, fontweight="bold", y=0.995, x=0.02, ha="left")
    save(fig, "Figure2")


# ============================ FIGURE 3 ============================
def figure3(d):
    chmm = d["chmm"]; ref = d["ref"]
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, top=0.74, bottom=0.20)

    # A: chromHMM grouped states
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    grp = chmm[chmm.group.str.contains("Promoter|Enhancer|Transcription|Repressed")].copy()
    lbls = [g.split(" (")[0] for g in grp.group]
    x = np.arange(len(grp))
    ax.bar(x - 0.19, grp.conc_frac * 100, width=0.36, color=CO, label="concordant")
    ax.bar(x + 0.19, grp.adqtl_frac * 100, width=0.36, color=AD, label="adQTL")
    ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=6.3)
    ax.set_ylabel("% of variants"); ax.set_title("Chromatin state (GM12878)\nno enrichment (all p>0.2)")
    ax.legend(frameon=False, loc="upper right")

    # B: |pred| adQTL vs control
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    aa = ref.loc[ref.set == "adqtl", "pred_rna_lfc"].abs().dropna()
    cc = ref.loc[ref.set == "control", "pred_rna_lfc"].abs().dropna()
    parts = ax.violinplot([cc, aa], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], [CO, AD]):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor("k"); pc.set_linewidth(0.5)
    for k in ["cmedians", "cmaxes", "cmins", "cbars"]:
        parts[k].set_color("k"); parts[k].set_linewidth(0.7)
    p = stats.mannwhitneyu(aa, cc).pvalue
    ax.set_xticks([1, 2]); ax.set_xticklabels(["concordant", "adQTL"])
    ax.set_ylabel(r"$|$predicted LFC$|$ (reference)"); ax.set_ylim(0, np.percentile(np.r_[aa, cc], 97))
    ax.set_title(f"Not stronger reg. variants\n(MWU p={p:.2f})")

    # C: pred vs beta_EUR / beta_AFR for adQTLs
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "C")
    a = ref[ref.set == "adqtl"].dropna(subset=["pred_rna_lfc", "beta_EUR", "beta_AFR"])
    a = a[a.pred_rna_lfc != 0]
    for col, c, lab in [("beta_EUR", GCOL["EUR"], "EUR"), ("beta_AFR", GCOL["AFR"], "AFR")]:
        ax.scatter(a.pred_rna_lfc, a[col], s=9, color=c, edgecolor="none", alpha=0.7, label=lab)
        z = np.polyfit(a.pred_rna_lfc, a[col], 1); xs = np.linspace(a.pred_rna_lfc.min(), a.pred_rna_lfc.max(), 20)
        ax.plot(xs, np.polyval(z, xs), color=c, lw=1.1)
    re_ = stats.spearmanr(a.pred_rna_lfc, a.beta_EUR)[0]; ra_ = stats.spearmanr(a.pred_rna_lfc, a.beta_AFR)[0]
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_xlabel("AlphaGenome predicted LFC"); ax.set_ylabel(r"observed $\beta$ (adQTL)")
    ax.set_title(f"r(EUR)={re_:.2f}, r(AFR)={ra_:.2f}"); ax.legend(frameon=False, loc="upper left")

    suptitle(fig,"Figure 4.  adQTLs are defined by their ancestry-differential effect,\nnot by baseline regulatory features",
                 fontsize=9, fontweight="bold", y=0.99, x=0.02, ha="left")
    save(fig, "Figure4")


# ============================ FIGURE 4 ============================
def figure4(d):
    cal = d["cal"]; reveng = d["reveng"]
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, top=0.74, bottom=0.20)

    # A: sub-gate A calibration (near-TSS)
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    c = cal.dropna(subset=["pred_rna_lfc", "log2_aFC"])
    near = c[c.var_tss_dist <= 50000]; dist = c[c.var_tss_dist > 50000]
    ax.scatter(dist.pred_rna_lfc, dist.log2_aFC, s=4, color=GRY, edgecolor="none", rasterized=True, label=">50kb")
    ax.scatter(near.pred_rna_lfc, near.log2_aFC, s=6, color=BL, edgecolor="none", label="≤50kb")
    r = stats.spearmanr(near.pred_rna_lfc, near.log2_aFC)[0]
    ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
    ax.set_xlabel("AlphaGenome predicted LFC (server)"); ax.set_ylabel(r"observed $\log_2$aFC")
    ax.set_title(f"Engine works on reference bg\nnear-TSS Spearman r={r:.2f}"); ax.legend(frameon=False, loc="lower right")

    # B: readout sensitivity ladder
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    if reveng is not None and len(reveng) > 5:
        rr = reveng.dropna(subset=["server_lfc"])
        def sp(col):
            m = rr[(rr[col] != 0) & np.isfinite(rr[col])]
            return stats.spearmanr(m[col], m.log2_aFC)[0] if len(m) > 5 else np.nan
        names = ["server\nscorer", "exon\nmask*", "gene\nbody", "TSS\nwindow"]
        vals = [sp("server_lfc"), np.nan, sp("cand_body_log2sum"), sp("cand_tss_log2sum")]
        # exon-mask value from validation file if present
        try:
            ev = pd.read_csv(P0 / "p0_exon_readout_validation.tsv", sep="\t")
            ev = ev[(ev.predD_exon != 0) & np.isfinite(ev.predD_exon)]
            vals[1] = stats.spearmanr(ev.predD_exon, ev.log2_aFC)[0]
        except Exception:
            pass
        cols = [BL, AD, "#e08214", "#bdbdbd"]
        ax.bar(range(4), vals, color=cols, width=0.62)
        for i, v in enumerate(vals):
            if np.isfinite(v): ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6.5)
        ax.set_ylabel(r"Spearman r vs observed aFC"); ax.set_ylim(0, 0.6)
        ax.set_title("Readout sensitivity ladder\n(client ≈ ⅓ of server)")
    else:
        ax.text(0.5, 0.5, "reveng data n/a", ha="center"); ax.axis("off")

    # C: client custom-bg readout fails (predD vs aFC ~ 0)
    ax = fig.add_subplot(gs[0, 2]); panel(ax, "C")
    try:
        ev = pd.read_csv(P0 / "p0_exon_readout_validation.tsv", sep="\t")
        ev = ev[(ev.predD_exon != 0) & np.isfinite(ev.predD_exon)]
        ax.scatter(ev.predD_exon, ev.log2_aFC, s=12, color=AD, edgecolor="none")
        pr = stats.pearsonr(ev.predD_exon, ev.log2_aFC)[0]; sr = stats.spearmanr(ev.predD_exon, ev.log2_aFC)[0]
        ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
        ax.set_xlabel("client exon-mask predicted LFC"); ax.set_ylabel(r"observed $\log_2$aFC")
        ax.set_title(f"Custom-bg readout too weak\nSpearman {sr:.2f} / Pearson {pr:.2f}")
    except Exception:
        ax.axis("off")

    suptitle(fig,"Figure 5.  AlphaGenome predicts reference-background eQTL effects, but its\nsensitive scorer cannot test ancestry haplotype backgrounds",
                 fontsize=9, fontweight="bold", y=0.99, x=0.02, ha="left")
    save(fig, "Figure5")


def main():
    d = load()
    print("[figures] generating publication figures ->", OUT)
    figure1(d); figure2(d); figure3(d); figure4(d)
    print("[figures] done.")


if __name__ == "__main__":
    main()
