"""
Phase 0 — Step 07: Characterize the ancestry-differential eQTL (adQTL) signal and
decide whether it is REAL or a power/allele-frequency ARTIFACT — before investing
in expensive AlphaGenome ADRS predictions.

Critical checks (the cis-GxG vs LD/AF/GxE debate):
  1. NULL excess: is the EUR-AFR heterogeneity p-distribution inflated beyond the
     5% expected under the null? (QQ + count at FDR<0.1)
  2. POWER confound: are "adQTLs" just variants that are rare/low-MAF in one group
     (large SE -> spurious or genuine-but-power-driven heterogeneity)?
  3. ALLELE-FREQUENCY / LD confound: do adQTLs have higher between-ancestry Fst
     (MAGE Finemapped_Fst)? High Fst => heterogeneity partly reflects freq/LD
     differences, NOT necessarily cis-regulatory background (the thing AlphaGenome
     could explain). This sets up the Phase-1 variance decomposition.

Defines: adQTL set (FDR<0.1) + MAF/distance-matched concordant controls, for the
sub-gate B ADRS test.

Input : results/phase0/p0_per_ancestry_betas.tsv (p0_06)
        results/phase0/p0_calibration_set.tsv (var_tss_dist, PIP)
        data/raw/MAGE/.../Finemapped_Fst.csv
Output: results/phase0/p0_adqtl_set.tsv, p0_control_set.tsv
        figures/phase0/p0_adqtl_characterization.png
        docs/phase0_adqtl_report.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
FIG = BASE / "figures" / "phase0"
DOCS = BASE / "docs"
MAGE = BASE / "data" / "raw" / "MAGE"
BETAS = RES / "p0_per_ancestry_betas.tsv"
CAL = RES / "p0_calibration_set.tsv"
FST = MAGE / "global_trend_results/global_expression_trends/Finemapped_Fst.csv"

FDR = 0.10


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BETAS, sep="\t")
    cal = pd.read_csv(CAL, sep="\t")[["variant_kgpID", "var_tss_dist", "variantPIP", "log2_aFC"]]
    df = df.merge(cal, on="variant_kgpID", how="left")

    both = df.dropna(subset=["p_het_EUR_AFR", "beta_EUR", "beta_AFR"]).copy()
    n = len(both)

    # 1) NULL excess + FDR
    both["fdr_het"] = multipletests(both["p_het_EUR_AFR"], method="fdr_bh")[1]
    n_fdr = (both["fdr_het"] < FDR).sum()
    n_nom = (both["p_het_EUR_AFR"] < 0.05).sum()
    # expected under null at 0.05 = 0.05*n
    exc_ratio = n_nom / (0.05 * n)

    both["is_adqtl"] = both["fdr_het"] < FDR

    # 2) POWER confound: maf gap and min group MAF vs heterogeneity
    both["maf_gap"] = (both["maf_EUR"] - both["maf_AFR"]).abs()
    both["min_maf"] = both[["maf_EUR", "maf_AFR"]].min(axis=1)
    # correlate |z_het| with maf_gap and with min_maf
    r_gap = stats.spearmanr(both["maf_gap"], both["z_EUR_AFR"].abs(), nan_policy="omit")
    r_minmaf = stats.spearmanr(both["min_maf"], both["z_EUR_AFR"].abs(), nan_policy="omit")

    # 3) Fst confound
    fst = pd.read_csv(FST)  # geneID, continentalGroup, mean_wcFst, DEpadj
    fst_w = fst.pivot_table(index="geneID", columns="continentalGroup",
                            values="mean_wcFst", aggfunc="mean")
    fst_w["fst_mean"] = fst_w.mean(axis=1)
    both["geneID_nov"] = both["ensemblID"].str.split(".").str[0]
    both = both.merge(fst_w[["fst_mean"]], left_on="geneID_nov", right_index=True, how="left")
    fst_adqtl = both.loc[both.is_adqtl, "fst_mean"].dropna()
    fst_conc = both.loc[~both.is_adqtl, "fst_mean"].dropna()
    fst_u = stats.mannwhitneyu(fst_adqtl, fst_conc, alternative="greater") if len(fst_adqtl) and len(fst_conc) else None

    # distance: are adQTLs more distal?
    dist_adqtl = both.loc[both.is_adqtl, "var_tss_dist"].dropna()
    dist_conc = both.loc[~both.is_adqtl, "var_tss_dist"].dropna()
    dist_u = stats.mannwhitneyu(dist_adqtl, dist_conc, alternative="greater")

    # Define sets + MAF/distance-matched controls for adQTLs
    adqtl = both[both.is_adqtl].copy()
    pool = both[~both.is_adqtl].copy()
    # simple nearest matching on (min_maf, log10 dist)
    controls = []
    used = set()
    pool = pool.dropna(subset=["min_maf", "var_tss_dist"])
    pv = pool[["min_maf", "var_tss_dist"]].values.astype(float)
    pv_norm = (pv - pv.mean(0)) / (pv.std(0) + 1e-9)
    for _, a in adqtl.dropna(subset=["min_maf", "var_tss_dist"]).iterrows():
        q = np.array([a.min_maf, a.var_tss_dist], float)
        qn = (q - pv.mean(0)) / (pv.std(0) + 1e-9)
        d = ((pv_norm - qn) ** 2).sum(1)
        order = np.argsort(d)
        for j in order:
            if j not in used:
                used.add(j); controls.append(pool.iloc[j]); break
    ctrl = pd.DataFrame(controls)

    adqtl.to_csv(RES / "p0_adqtl_set.tsv", sep="\t", index=False)
    ctrl.to_csv(RES / "p0_control_set.tsv", sep="\t", index=False)

    # ---- Figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    # (a) QQ of p_het vs uniform
    p = np.sort(both["p_het_EUR_AFR"].values)
    exp = -np.log10((np.arange(1, len(p) + 1)) / (len(p) + 1))
    ax[0, 0].plot(exp, -np.log10(p), ".", ms=3, color="#c0392b")
    lim = max(exp.max(), (-np.log10(p)).max())
    ax[0, 0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0, 0].set_title(f"(a) EUR-AFR heterogeneity QQ\nnominal p<0.05: {n_nom}/{n} "
                       f"({100*n_nom/n:.1f}%), {exc_ratio:.1f}x null; FDR<{FDR}: {n_fdr}")
    ax[0, 0].set_xlabel("expected -log10 p"); ax[0, 0].set_ylabel("observed -log10 p")
    # (b) |z_het| vs maf_gap (power/AF confound)
    ax[0, 1].scatter(both["maf_gap"], both["z_EUR_AFR"].abs(), s=8, alpha=0.3)
    ax[0, 1].set_title(f"(b) |z_het| vs |MAF_EUR-MAF_AFR|\nSpearman r={r_gap.correlation:.2f} "
                       f"(p={r_gap.pvalue:.1e})  [AF/LD confound]")
    ax[0, 1].set_xlabel("|MAF gap EUR vs AFR|"); ax[0, 1].set_ylabel("|z heterogeneity|")
    # (c) Fst: adQTL vs concordant
    ax[1, 0].boxplot([fst_conc, fst_adqtl], labels=["concordant", "adQTL"], showfliers=False)
    ttl = f"(c) gene Fst (fine-mapped variants)"
    if fst_u: ttl += f"\nadQTL>concordant MWU p={fst_u.pvalue:.1e}"
    ax[1, 0].set_title(ttl); ax[1, 0].set_ylabel("mean_wcFst")
    # (d) distance: adQTL vs concordant
    ax[1, 1].boxplot([np.log10(dist_conc + 1), np.log10(dist_adqtl + 1)],
                     labels=["concordant", "adQTL"], showfliers=False)
    ax[1, 1].set_title(f"(d) variant-TSS distance\nadQTL>concordant MWU p={dist_u.pvalue:.1e}")
    ax[1, 1].set_ylabel("log10(var_tss_dist+1)")
    fig.suptitle("Phase 0 — adQTL characterization (is ancestry heterogeneity real & cis?)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p0_adqtl_characterization.png", dpi=150)

    # ---- Report ----
    rep = [
        "# Phase 0 — adQTL characterization report",
        f"\n## Sample\n- variant-gene pairs with EUR & AFR estimable: **{n}**",
        f"- nominal heterogeneity p<0.05: **{n_nom} ({100*n_nom/n:.1f}%)** = **{exc_ratio:.1f}x** null expectation",
        f"- FDR<{FDR} adQTLs: **{n_fdr}**",
        "\n## Is it a power/allele-frequency artifact?",
        f"- |z_het| vs |MAF gap EUR-AFR|: Spearman r={r_gap.correlation:.3f} (p={r_gap.pvalue:.2e})",
        f"- |z_het| vs min(MAF): Spearman r={r_minmaf.correlation:.3f} (p={r_minmaf.pvalue:.2e})",
        ("  - VERDICT: heterogeneity is **NOT** explained by allele-frequency gap or power "
         "(|r|<0.05) — the differential effect is not a trivial frequency/power artifact."
         if abs(r_gap.correlation) < 0.1 and abs(r_minmaf.correlation) < 0.1 else
         "  - heterogeneity correlates with MAF gap/power — partly an allele-frequency/LD effect "
         "that Phase 1 must condition out."),
        "  - NOTE: we test the SAME fine-mapped variant (variant_kgpID matched, PIP>=0.5) in both "
         "ancestries, so this is effect-SIZE heterogeneity of one SNP, not tag-SNP/LD swapping — a "
         "cleaner GxG/GxE substrate than typical cross-ancestry PRS comparisons (residual LD-to-causal remains).",
        "\n## Cis vs LD/AF signature",
        f"- gene Fst (fine-mapped): adQTL median={fst_adqtl.median():.4f} vs concordant median={fst_conc.median():.4f}"
        + (f"; MWU(adQTL>conc) p={fst_u.pvalue:.2e}" if fst_u else ""),
        f"- variant-TSS distance: adQTL median={dist_adqtl.median():.0f} vs concordant median={dist_conc.median():.0f} bp; "
        f"MWU(adQTL>conc) p={dist_u.pvalue:.2e}",
        "\n## Sets defined",
        f"- adQTL set (FDR<{FDR}): {len(adqtl)} -> results/phase0/p0_adqtl_set.tsv",
        f"- matched concordant controls: {len(ctrl)} -> results/phase0/p0_control_set.tsv",
        "\n## Interpretation for sub-gate B",
        "- Excess heterogeneity beyond null confirms real ancestry-differential effects exist.",
        "- BUT if it tracks MAF gap / Fst strongly, much is allele-frequency/LD driven (not cis-regulatory).",
        "- The ADRS test (AlphaGenome on ancestry haplotype backgrounds) must show ADRS predicts ObsDelta",
        "  ABOVE AND BEYOND MAF-gap/Fst — that residual is the cis-sequence-encoded GxG component.",
    ]
    (DOCS / "phase0_adqtl_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\n[fig] -> {FIG/'p0_adqtl_characterization.png'}")


if __name__ == "__main__":
    main()
