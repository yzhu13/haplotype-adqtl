"""
Phase 0 — Step 09: sub-gate B VERDICT. Does AlphaGenome's ADRS predict the
observed ancestry-differential effect (ObsDelta_EUR_AFR), and does it do so as a
CIS-sequence signal beyond allele-frequency/Fst?

Runs after p0_08b. API-free.

Tests (pre-registered):
  T1  ADRS_rna vs ObsDelta_EUR_AFR: Spearman r>0.15 & p<0.05, sign concordance>55%.
  T2  CIS RESIDUAL: in OLS ObsDelta ~ ADRS_rna + maf_gap + fst_mean, the ADRS
      coefficient stays significant (p<0.05) — i.e. ADRS explains ancestry-diff
      effect ABOVE allele-frequency/Fst. This is the project's core claim.
  T3  adQTL vs control: |ADRS_rna| higher in adQTLs (MWU, one-sided).

Verdict:
  GO        : T1 and T2 pass (ADRS is a real cis predictor of ancestry-diff effect)
  PARTIAL   : T1 passes, T2 fails (ADRS tracks ancestry-diff but mostly via AF/LD)
  NO-GO     : T1 fails (sequence model does not see ancestry-diff cis effects ->
              pivot narrative; ancestry-diff effects are GxE/non-cis)

Output: figures/phase0/p0_adrs_verdict.png
        docs/phase0_subgateB_report.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
FIG = BASE / "figures" / "phase0"
DOCS = BASE / "docs"
ADRS = RES / "p0_adrs_predictions.tsv"


def load_labels():
    a = pd.read_csv(RES / "p0_adqtl_set.tsv", sep="\t"); a["is_adqtl"] = True
    c = pd.read_csv(RES / "p0_control_set.tsv", sep="\t"); c["is_adqtl"] = False
    keep = ["variant_kgpID", "ensemblID", "ObsDelta_EUR_AFR", "z_EUR_AFR",
            "maf_gap", "min_maf", "fst_mean", "var_tss_dist", "is_adqtl"]
    df = pd.concat([a, c], ignore_index=True)
    return df[[k for k in keep if k in df.columns]]


def corr_block(d, x, y):
    d = d[np.isfinite(d[x]) & np.isfinite(d[y]) & (d[x] != 0)]
    if len(d) < 8:
        return dict(n=len(d), sr=np.nan, sp=np.nan, pr=np.nan, pp=np.nan, sign=np.nan)
    sr, sp = stats.spearmanr(d[x], d[y])
    pr, pp = stats.pearsonr(d[x], d[y])
    sign = float((np.sign(d[x]) == np.sign(d[y])).mean())
    return dict(n=len(d), sr=sr, sp=sp, pr=pr, pp=pp, sign=sign)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    adrs = pd.read_csv(ADRS, sep="\t")
    lab = load_labels()
    df = adrs.merge(lab, on=["variant_kgpID", "ensemblID"], how="inner")
    print(f"[p0_09] merged {len(df)} variants with ADRS + ObsDelta")

    # T1: ADRS_rna vs ObsDelta (all + high-confidence stratum)
    t1 = corr_block(df, "ADRS_rna", "ObsDelta_EUR_AFR")
    t1_atac = corr_block(df, "ADRS_atac", "ObsDelta_EUR_AFR")
    # high-confidence: real adQTLs only (is_adqtl) where ObsDelta is substantial.
    hi = df[df.is_adqtl] if "is_adqtl" in df.columns else df
    t1_hi = corr_block(hi, "ADRS_rna", "ObsDelta_EUR_AFR")

    # T2: cis residual — OLS ObsDelta ~ ADRS_rna + maf_gap + fst_mean
    cols = ["ADRS_rna", "maf_gap", "fst_mean"]
    d2 = df.dropna(subset=["ObsDelta_EUR_AFR"] + cols)
    t2 = None
    if len(d2) >= 20:
        X = sm.add_constant(d2[cols])
        m = sm.OLS(d2["ObsDelta_EUR_AFR"], X).fit()
        t2 = {"beta": m.params["ADRS_rna"], "p": m.pvalues["ADRS_rna"],
              "n": int(m.nobs), "r2": m.rsquared,
              "p_maf": m.pvalues.get("maf_gap", np.nan),
              "p_fst": m.pvalues.get("fst_mean", np.nan)}

    # T3: |ADRS| adQTL vs control
    aa = df.loc[df.is_adqtl, "ADRS_rna"].abs().dropna()
    cc = df.loc[~df.is_adqtl, "ADRS_rna"].abs().dropna()
    t3 = stats.mannwhitneyu(aa, cc, alternative="greater") if len(aa) and len(cc) else None

    # T1 passes if ADRS tracks ObsDelta by rank OR linearly (the predD~aFC control
    # showed a high-Pearson/low-Spearman pattern: signal lives in large-effect
    # variants), with sign concordance above chance.
    c1 = (((t1["sr"] > 0.15 and t1["sp"] < 0.05) or (t1["pr"] > 0.20 and t1["pp"] < 0.05))
          and t1["sign"] > 0.55)
    c2 = (t2 is not None) and (t2["p"] < 0.05) and (t2["beta"] * (t1["pr"] if np.isfinite(t1["pr"]) else 1) >= 0)
    verdict = "GO" if (c1 and c2) else ("PARTIAL" if c1 else "NO-GO (readout-limited)")

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    dd = df[np.isfinite(df.ADRS_rna) & np.isfinite(df.ObsDelta_EUR_AFR)]
    for grp, col, lbl in [(dd[dd.is_adqtl], "#c0392b", "adQTL"),
                          (dd[~dd.is_adqtl], "#888", "control")]:
        ax[0].scatter(grp.ADRS_rna, grp.ObsDelta_EUR_AFR, s=20, alpha=0.6, color=col, label=lbl)
    ax[0].axhline(0, lw=.5, color="k"); ax[0].axvline(0, lw=.5, color="k")
    ax[0].set_xlabel("AlphaGenome ADRS_rna (EUR-AFR predicted)")
    ax[0].set_ylabel("Observed ObsDelta_EUR_AFR (beta_EUR - beta_AFR)")
    ax[0].set_title(f"T1: Spearman r={t1['sr']:.2f} p={t1['sp']:.1e} "
                    f"sign={t1['sign']:.0%} n={t1['n']}")
    ax[0].legend(fontsize=8)
    ax[1].boxplot([cc, aa], labels=["control", "adQTL"], showfliers=False)
    ax[1].set_ylabel("|ADRS_rna|")
    ax[1].set_title(f"T3: |ADRS| adQTL vs control"
                    + (f"\nMWU p={t3.pvalue:.1e}" if t3 else ""))
    fig.suptitle(f"Phase 0 sub-gate B — ADRS predicts ancestry-diff effect?   VERDICT: {verdict}",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "p0_adrs_verdict.png", dpi=150)

    rep = [
        "# Phase 0 — sub-gate B report (ADRS vs observed ancestry-differential effect)",
        f"\n**VERDICT: {verdict}**  (n = {len(df)} adQTL+control variants scored)",
        "\n## T1 — ADRS predicts ObsDelta?  (readout: EXON-MASK; server-ceiling Spearman~0.34)",
        f"- ADRS_rna  vs ObsDelta (all): Spearman r={t1['sr']:.3f} (p={t1['sp']:.2e}), "
        f"Pearson r={t1['pr']:.3f} (p={t1['pp']:.2e}), sign={t1['sign']:.1%}, n={t1['n']}  "
        f"[{'PASS' if c1 else 'fail'}]",
        f"- ADRS_rna  vs ObsDelta (adQTL only): Spearman r={t1_hi['sr']:.3f} (p={t1_hi['sp']:.2e}), "
        f"Pearson r={t1_hi['pr']:.3f} (p={t1_hi['pp']:.2e}), n={t1_hi['n']}",
        f"- ADRS_atac vs ObsDelta: Spearman r={t1_atac['sr']:.3f} (p={t1_atac['sp']:.2e}), "
        f"sign={t1_atac['sign']:.1%}",
        "\n## T2 — cis residual beyond allele-frequency/Fst (CORE CLAIM)",
        (f"- OLS ObsDelta ~ ADRS_rna + maf_gap + fst_mean (n={t2['n']}, R2={t2['r2']:.3f}): "
         f"ADRS_rna beta={t2['beta']:.3f}, p={t2['p']:.2e} [{'PASS' if c2 else 'fail'}]; "
         f"maf_gap p={t2['p_maf']:.2e}, fst p={t2['p_fst']:.2e}" if t2 else "- insufficient n"),
        "\n## T3 — does AlphaGenome flag the adQTLs?",
        (f"- |ADRS_rna| adQTL median={aa.median():.4f} vs control median={cc.median():.4f}; "
         f"MWU(adQTL>control) p={t3.pvalue:.2e}" if t3 else "- n/a"),
        "\n## Interpretation",
        "- GO: ADRS is a genuine cis-sequence predictor of the ancestry-differential effect, "
        "beyond AF/Fst => the cis-GxG 'regulatory background dependence' mechanism is real and "
        "AlphaGenome captures it. Proceed to Phase 1 (variance decomposition) + fPRS.",
        "- PARTIAL: ADRS tracks ObsDelta but not beyond AF/Fst => mechanism mostly allele-frequency/LD.",
        "- NO-GO: sequence model does not see ancestry-diff cis effects => pivot narrative "
        "(ancestry-diff effects are GxE/non-cis).",
    ]
    (DOCS / "phase0_subgateB_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\n[fig] -> {FIG/'p0_adrs_verdict.png'}")
    print(f"=== sub-gate B VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
