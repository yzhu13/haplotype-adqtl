"""
Phase 1 / Direction 1 — Figure 5: cis-haplotype epistasis is the mechanism of
ancestry-differential cis-eQTLs. Publication-quality (PNG 300dpi + PDF).

Panels:
  A  cis-interaction enrichment: % adQTL vs control with significant focal×bg interaction.
  B  within-ancestry confirmation: % confirmed in EUR / AFR / >=1 (rules out focal×ancestry).
  C  cis-vs-trans decomposition: distribution of fraction of ObsDelta explained by the
     (single best) cis background SNP.
  D  vignette: for the top confirmed locus, focal eQTL slope changes with background
     genotype within a single ancestry (the cis-epistasis signature).

Inputs: results/phase1/{p1_cis_interaction, p1_within_ancestry, p1_cis_trans_decomp}.tsv
Output: figures/paper/Figure5.{png,pdf}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
P1 = BASE / "results" / "phase1"; OUT = REGEN_DIR

for fam in ["Arial", "Helvetica", "DejaVu Sans"]:
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
AD = "#c0392b"; CO = "#9aa0a6"; BL = "#2166ac"; GCOL = {"EUR": "#4575b4", "AFR": "#d73027"}


def panel(ax, L):
    ax.text(-0.2, 1.06, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")


def vignette_data():
    """fetch focal+best_bg+expr for the top confirmed locus; return (df, gene, bg, ancestry)."""
    from p0_06_per_ancestry_betas import load_expr, load_covar, META
    from p0_05_fetch_genotypes import mage_sample_order, gt_to_dosage, VCF_TMPL
    from lib_remote_vcf import RemoteTabixVCF
    wa = pd.read_csv(P1 / "p1_within_ancestry.tsv", sep="\t")
    wa["minp"] = wa[["int_p_EUR", "int_p_AFR"]].min(axis=1)
    wa = wa.sort_values("minp")
    r = wa.iloc[0]
    anc = "EUR" if r.int_p_EUR <= r.int_p_AFR else "AFR"
    expr = load_expr()
    meta = pd.read_csv(META, sep="\t"); grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    samples = [s for s in mage_sample_order() if grp.get(s) == anc and s in expr.columns]
    fd = pd.read_csv(BASE / "data/processed/lead_variant_dosages.tsv", sep="\t").drop_duplicates("variant_kgpID").set_index("variant_kgpID")
    bc, bp = "chr" + r.best_bg.split(":")[0], int(r.best_bg.split(":")[1])
    vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=bc)); vsamp = {s: i for i, s in enumerate(vcf.samples())}
    bgdos = {}
    for ln in vcf.query(bc, bp, bp):
        f = ln.split("\t")
        if f[2] == r.best_bg:
            gts = f[9:]; bgdos = {s: gt_to_dosage(gts[vsamp[s]]) for s in samples if s in vsamp}; break
    d = pd.DataFrame({"expr": expr.loc[r.ensemblID, samples].astype(float).values,
                      "focal": fd.loc[r.variant_kgpID, samples].astype(float).values,
                      "bg": [bgdos.get(s, np.nan) for s in samples]})
    return d.dropna(), r.ensemblID, r.best_bg, anc, r.minp


def main():
    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    fig = plt.figure(figsize=(7.2, 6.0)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.36)

    # A enrichment
    ax = fig.add_subplot(gs[0, 0]); panel(ax, "A")
    t = ci[ci.n_bg_tested > 0]
    ad = t[t.set == "adqtl"]; co = t[t.set == "control"]
    radq = (ad.bonf_p < 0.05).mean() * 100; rco = (co.bonf_p < 0.05).mean() * 100
    ax.bar([0, 1], [rco, radq], color=[CO, AD], width=0.6)
    for i, (v, n) in enumerate([(rco, len(co)), (radq, len(ad))]):
        ax.text(i, v + 1.5, f"{v:.0f}%\n(n={n})", ha="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["concordant\n(control)", "adQTL"])
    ax.set_ylabel("% with significant cis-interaction"); ax.set_ylim(0, 105)
    ax.set_title("Focal × background-SNP interaction\n(Bonferroni adj p<0.05)")

    # B within-ancestry confirmation
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "B")
    try:
        wa = pd.read_csv(P1 / "p1_within_ancestry.tsv", sep="\t")
        ce = (wa.int_p_EUR < 0.05).mean() * 100; ca = (wa.int_p_AFR < 0.05).mean() * 100
        cany = ((wa.int_p_EUR < 0.05) | (wa.int_p_AFR < 0.05)).mean() * 100
        ax.bar([0, 1, 2], [ce, ca, cany], color=[GCOL["EUR"], GCOL["AFR"], "#444"], width=0.6)
        for i, v in enumerate([ce, ca, cany]):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
        ax.axhline(5, ls="--", color="k", lw=0.7); ax.text(2.4, 7, "null ~5%", fontsize=6.5, ha="right")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["within\nEUR", "within\nAFR", "≥1\nancestry"])
        ax.set_ylabel("% adQTLs confirmed"); ax.set_ylim(0, 100)
        ax.set_title(f"Within-ancestry confirmation\n(rules out focal×ancestry; n={len(wa)})")
    except Exception as e:
        ax.text(0.5, 0.5, "within-ancestry n/a", ha="center"); ax.axis("off")

    # C decomposition (multi-bg if available, else single-bg)
    ax = fig.add_subplot(gs[1, 0]); panel(ax, "C")
    try:
        mb = P1 / "p1_multi_bg_decomp.tsv"
        if mb.exists():
            dec = pd.read_csv(mb, sep="\t"); col = "frac_cis_multi"; sub = "multi independent cis-bg"
        else:
            dec = pd.read_csv(P1 / "p1_cis_trans_decomp.tsv", sep="\t"); col = "frac_cis"; sub = "single best cis-bg (lower bound)"
        fc = dec[col].dropna().clip(-0.5, 1.5)
        ax.hist(fc, bins=20, color=BL, edgecolor="white", lw=0.4)
        med = dec[col].dropna().median()
        ax.axvline(med, color=AD, lw=1.5); ax.text(med + 0.02, ax.get_ylim()[1]*0.88, f"median {med:.2f}", color=AD, fontsize=8, fontweight="bold")
        ax.set_xlabel("fraction of ObsΔ explained by cis-background"); ax.set_ylabel("# adQTLs")
        ax.set_title(f"cis-vs-trans decomposition\n({sub})")
    except Exception as e:
        ax.text(0.5, 0.5, f"decomp n/a\n{e}", ha="center", fontsize=6); ax.axis("off")

    # D vignette
    ax = fig.add_subplot(gs[1, 1]); panel(ax, "D")
    try:
        d, gene, bg, anc, minp = vignette_data()
        cols = ["#fdae61", "#abd9e9", "#2c7bb6"]
        for gt in [0, 1, 2]:
            sub = d[d.bg.round() == gt]
            if len(sub) < 3:
                continue
            ax.scatter(sub.focal + np.random.uniform(-0.05, 0.05, len(sub)), sub.expr, s=10,
                       color=cols[gt], alpha=0.6, label=f"bg={gt} (n={len(sub)})")
            if sub.focal.nunique() > 1:
                z = np.polyfit(sub.focal, sub.expr, 1); xs = np.array([sub.focal.min(), sub.focal.max()])
                ax.plot(xs, np.polyval(z, xs), color=cols[gt], lw=1.6)
        ax.set_xlabel(f"focal dosage  (within {anc})"); ax.set_ylabel("expression (inv-normal)")
        ax.set_title(f"Vignette: {gene}\nfocal slope depends on bg genotype (p={minp:.1e})", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.5, loc="best")
    except Exception as e:
        ax.text(0.5, 0.5, f"vignette n/a\n{e}", ha="center", fontsize=6); ax.axis("off")

    suptitle(fig, "Figure 3.  Ancestry-differential cis-eQTLs are enriched for cis-haplotype epistasis",
             fontsize=8.2, fontweight="bold", y=0.995, x=0.02, ha="left")
    fig.savefig(OUT / "Figure3.png", bbox_inches="tight"); fig.savefig(OUT / "Figure3.pdf", bbox_inches="tight")
    announce(__file__, "Figure3")


if __name__ == "__main__":
    main()
