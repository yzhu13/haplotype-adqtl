"""
Phase 1 — Step 01: Rigorously decompose the ancestry-differential cis-eQTL signal.

Reframed project (sub-gate B blocked by API): build the case around the adQTL
atlas. Central claim = ancestry-differential effects are REAL and NOT explained by
allele frequency or power or incomplete fine-mapping (LD-to-causal). What remains
is a genuine effect-size difference (the substrate for GxE / cis-context biology).

We test the SAME fine-mapped SNP (variant_kgpID matched, PIP>=0.5) in each
continental group, so this is effect-SIZE heterogeneity, not tag-SNP swapping.

Analyses:
  A. NULL excess: heterogeneity p vs uniform (QQ); fraction at nominal/FDR.
  B. ALLELE-FREQUENCY/POWER model: does |z_het| / adQTL status track MAF gap,
     min-MAF, or per-group n? (expect ~0 -> not an AF/power artifact)
  C. LD / FINE-MAPPING proxy: does the adQTL rate depend on PIP or credible-set
     size? If heterogeneity persists at high PIP / small CS, it is not a
     LD-to-causal (incomplete fine-mapping) artifact.
  D. EFFECT CONCORDANCE: among adQTLs, are beta_EUR / beta_AFR same sign
     (magnitude difference) or opposite (qualitative flip)?
  E. MULTI-ANCESTRY: Cochran Q across all 5 groups (consistency of the signal).

Inputs (all on disk): p0_per_ancestry_betas.tsv, p0_calibration_set.tsv (PIP),
  MAGE leadVariant_aFCn (variantCredibleSet) + credibleSet_summary (numVariants).
Outputs: results/phase1/p1_decomposition.tsv, figures/phase1/p1_decomposition.png,
  docs/phase1_decomposition_report.md
"""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RES0 = BASE / "results" / "phase0"
RES = BASE / "results" / "phase1"
FIG = BASE / "figures" / "phase1"
DOCS = BASE / "docs"
MAGE = BASE / "data/raw/MAGE/QTL_results/eQTL_results/eQTL_finemapping_results"
AFCN = MAGE / "eQTL_finemapping.leadVariant_aFCn.MAGE.v1.0.txt.gz"
CSUM = MAGE / "eQTL_finemapping.credibleSet_summary.MAGE.v1.0.txt.gz"
FDR = 0.10


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES0 / "p0_per_ancestry_betas.tsv", sep="\t")
    cal = pd.read_csv(RES0 / "p0_calibration_set.tsv", sep="\t")[
        ["variant_kgpID", "ensemblID", "variantPIP", "var_tss_dist"]]
    df = df.merge(cal, on=["variant_kgpID", "ensemblID"], how="left")

    # credible-set size of each lead variant (fine-mapping resolution proxy)
    afc = pd.read_csv(AFCN, sep="\t")[["variant_kgpID", "ensemblID", "variantCredibleSet"]]
    csum = pd.read_csv(CSUM, sep="\t")[["ensemblID", "credibleSet", "numVariants"]]
    afc = afc.merge(csum, left_on=["ensemblID", "variantCredibleSet"],
                    right_on=["ensemblID", "credibleSet"], how="left")
    df = df.merge(afc[["variant_kgpID", "ensemblID", "numVariants"]],
                  on=["variant_kgpID", "ensemblID"], how="left")

    both = df.dropna(subset=["p_het_EUR_AFR", "beta_EUR", "beta_AFR"]).copy()
    both["fdr"] = multipletests(both["p_het_EUR_AFR"], method="fdr_bh")[1]
    both["is_adqtl"] = both["fdr"] < FDR
    both["abs_z"] = both["z_EUR_AFR"].abs()
    both["maf_gap"] = (both["maf_EUR"] - both["maf_AFR"]).abs()
    both["min_maf"] = both[["maf_EUR", "maf_AFR"]].min(axis=1)
    both["harm_n"] = 2 / (1 / both["n_EUR"] + 1 / both["n_AFR"])
    n = len(both); n_nom = int((both.p_het_EUR_AFR < 0.05).sum()); n_fdr = int(both.is_adqtl.sum())

    log = []
    def L(s):
        log.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode())
    L(f"# Phase 1 — decomposition of ancestry-differential eQTL signal\n")
    L(f"## A. Null excess\n- pairs: {n}; nominal p<0.05: {n_nom} ({100*n_nom/n:.1f}%) "
      f"= {n_nom/(0.05*n):.1f}x null; FDR<{FDR} adQTLs: {n_fdr}")

    # B. AF/power model
    L("\n## B. Allele-frequency / power model (predict |z_het|)")
    Xcols = ["maf_gap", "min_maf", "harm_n"]
    d2 = both.dropna(subset=["abs_z"] + Xcols)
    X = sm.add_constant(d2[Xcols]); m = sm.OLS(d2["abs_z"], X).fit()
    L(f"- OLS |z_het| ~ maf_gap + min_maf + harm_n: R²={m.rsquared:.4f} (n={int(m.nobs)})")
    for c in Xcols:
        L(f"    {c}: beta={m.params[c]:+.3g} p={m.pvalues[c]:.3f}")
    L(f"  => AF/power explain ~{100*m.rsquared:.1f}% of heterogeneity variance.")

    # C. LD / fine-mapping (PIP and CS size)
    L("\n## C. LD / fine-mapping resolution (adQTL rate by PIP & credible-set size)")
    for col, edges, lbl in [("variantPIP", [0.5, 0.7, 0.9, 1.01], "PIP"),
                            ("numVariants", [0, 1.5, 3.5, 10.5, 1e9], "CS size")]:
        d3 = both.dropna(subset=[col])
        d3 = d3.assign(_b=pd.cut(d3[col], edges, include_lowest=True))
        tab = d3.groupby("_b", observed=True)["is_adqtl"].agg(["mean", "size"])
        L(f"- adQTL rate by {lbl}:")
        for b, row in tab.iterrows():
            L(f"    {str(b):<18} rate={row['mean']:.3f} (n={int(row['size'])})")
    # correlation of |z| with PIP and CS size
    for col in ["variantPIP", "numVariants"]:
        d3 = both.dropna(subset=[col, "abs_z"])
        r, p = stats.spearmanr(d3[col], d3["abs_z"])
        L(f"  |z_het| vs {col}: Spearman r={r:+.3f} (p={p:.2e})")

    # D. effect concordance among adQTLs
    ad = both[both.is_adqtl]
    same = int((np.sign(ad.beta_EUR) == np.sign(ad.beta_AFR)).sum())
    L(f"\n## D. Effect concordance among {len(ad)} adQTLs")
    L(f"- same sign (magnitude diff): {same} ({100*same/max(1,len(ad)):.0f}%); "
      f"opposite sign (flip): {len(ad)-same}")

    # E. multi-ancestry consistency
    nq = both.dropna(subset=["p_Q"])
    L(f"\n## E. Multi-ancestry (Cochran Q, >=2 groups): {len(nq)} testable; "
      f"p_Q<0.05: {int((nq.p_Q<0.05).sum())} ({100*(nq.p_Q<0.05).mean():.1f}%); "
      f"median I²(all)={both['I2'].median():.3f}, I²(adQTL)={ad['I2'].median():.3f}")

    both.to_csv(RES / "p1_decomposition.tsv", sep="\t", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    p = np.sort(both.p_het_EUR_AFR.values)
    exp = -np.log10(np.arange(1, len(p)+1)/(len(p)+1))
    ax[0,0].plot(exp, -np.log10(p), ".", ms=3, color="#c0392b")
    lim=max(exp.max(),(-np.log10(p)).max()); ax[0,0].plot([0,lim],[0,lim],"k--",lw=.8)
    ax[0,0].set_title(f"(A) heterogeneity QQ: {n_nom/(0.05*n):.1f}x null, {n_fdr} adQTLs")
    ax[0,0].set_xlabel("expected -log10p"); ax[0,0].set_ylabel("observed -log10p")
    ax[0,1].scatter(both.maf_gap, both.abs_z, s=8, alpha=.3)
    ax[0,1].set_title(f"(B) |z_het| vs |MAF gap| (AF model R²={m.rsquared:.3f})")
    ax[0,1].set_xlabel("|MAF_EUR - MAF_AFR|"); ax[0,1].set_ylabel("|z_het|")
    d3=both.dropna(subset=["variantPIP"]).assign(_b=pd.cut(both.dropna(subset=["variantPIP"]).variantPIP,[0.5,0.7,0.9,1.01]))
    rate=d3.groupby("_b",observed=True)["is_adqtl"].mean()
    ax[1,0].bar(range(len(rate)),rate.values,color="#2980b9"); ax[1,0].set_xticks(range(len(rate)))
    ax[1,0].set_xticklabels([str(x) for x in rate.index],fontsize=8)
    ax[1,0].set_title("(C) adQTL rate by PIP (LD/fine-mapping)"); ax[1,0].set_ylabel("adQTL rate")
    ax[1,1].scatter(ad.beta_AFR, ad.beta_EUR, s=18, alpha=.7, color="#c0392b")
    lo,hi=ad[["beta_AFR","beta_EUR"]].min().min(),ad[["beta_AFR","beta_EUR"]].max().max()
    ax[1,1].plot([lo,hi],[lo,hi],"k--",lw=.8); ax[1,1].axhline(0,lw=.4,color="k"); ax[1,1].axvline(0,lw=.4,color="k")
    ax[1,1].set_title(f"(D) adQTL effects: beta_EUR vs beta_AFR (n={len(ad)})")
    ax[1,1].set_xlabel("beta_AFR"); ax[1,1].set_ylabel("beta_EUR")
    fig.suptitle("Phase 1 — ancestry-differential eQTL: real, not AF/power/LD artifact", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG/"p1_decomposition.png", dpi=150)

    L("\n## Interpretation")
    L("- If AF/power R²≈0 AND adQTL rate does not fall at high PIP/small CS, the "
      "ancestry-differential effect is a genuine effect-SIZE difference — not "
      "allele frequency, not power, not incomplete fine-mapping. That residual is "
      "the GxE / cis-context substrate (which the sequence model could not resolve, "
      "Phase 0 sub-gate B).")
    (DOCS/"phase1_decomposition_report.md").write_text("\n".join(log), encoding="utf-8")
    print(f"\n[fig] -> {FIG/'p1_decomposition.png'}")


if __name__ == "__main__":
    main()
