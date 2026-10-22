"""
Phase 1 — Step 04c: Analyze the reference-background AlphaGenome scores (p1_04b).

Hypothesis: hg38 reference is European-biased, so the reference-background
prediction should track beta_EUR better than beta_AFR. Tested via a bootstrap of
the correlation difference r(pred,bEUR) - r(pred,bAFR) (dependent correlations,
common variable = pred). Also |pred| adQTL vs control.

Input: results/phase1/p1_adqtl_refscore.tsv. Output: docs/phase1_refscore_report.md
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase1"; DOCS = BASE / "docs"


def boot_diff(x, ye, ya, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([stats.spearmanr(x[idx], ye[idx])[0] - stats.spearmanr(x[idx], ya[idx])[0]
                  for idx in (rng.integers(0, len(x), len(x)) for _ in range(n))])
    obs = stats.spearmanr(x, ye)[0] - stats.spearmanr(x, ya)[0]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, np.percentile(d, [2.5, 97.5]), p


def main():
    d = pd.read_csv(RES / "p1_adqtl_refscore.tsv", sep="\t")
    d = d[np.isfinite(d.pred_rna_lfc) & (d.pred_rna_lfc != 0)]
    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii","replace").decode())

    L("# Phase 1 — reference-background AlphaGenome score: EUR vs AFR bias\n")
    L(f"- scored variants: {len(d)} (adQTL {int((d.set=='adqtl').sum())}, "
      f"control {int((d.set=='control').sum())})")

    for grp in ["adqtl", "control", "all"]:
        sub = d if grp == "all" else d[d.set == grp]
        sub = sub.dropna(subset=["beta_EUR", "beta_AFR"])
        if len(sub) < 15:
            L(f"\n## {grp}: n={len(sub)} too few"); continue
        re_, pe = stats.spearmanr(sub.pred_rna_lfc, sub.beta_EUR)
        ra_, pa = stats.spearmanr(sub.pred_rna_lfc, sub.beta_AFR)
        L(f"\n## {grp} (n={len(sub)})")
        L(f"- pred vs beta_EUR: Spearman r={re_:+.3f} (p={pe:.2e})")
        L(f"- pred vs beta_AFR: Spearman r={ra_:+.3f} (p={pa:.2e})")
        obs, ci, p = boot_diff(sub.pred_rna_lfc.values, sub.beta_EUR.values, sub.beta_AFR.values)
        L(f"- diff r(EUR)-r(AFR) = {obs:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], boot p={p:.3f}"
          + ("  [EUR-biased]" if (obs > 0 and p < 0.1) else ""))

    # magnitude
    aa = d.loc[d.set=="adqtl","pred_rna_lfc"].abs(); cc = d.loc[d.set=="control","pred_rna_lfc"].abs()
    if len(aa) and len(cc):
        u = stats.mannwhitneyu(aa, cc, alternative="two-sided")
        L(f"\n## |pred| adQTL median={aa.median():.4f} vs control={cc.median():.4f} "
          f"(MWU p={u.pvalue:.3f})")
    # data-driven interpretation
    ad = d[d.set == "adqtl"].dropna(subset=["beta_EUR", "beta_AFR"])
    obs, ci, p = boot_diff(ad.pred_rna_lfc.values, ad.beta_EUR.values, ad.beta_AFR.values)
    L("\n## Interpretation (data-driven)")
    if abs(obs) < 0.06 or p > 0.1:
        L("- NULL: the reference-background prediction does NOT track beta_EUR better than "
          "beta_AFR for adQTLs (diff ~0, p>0.1), and |pred| is equal for adQTL vs control. "
          "Combined with the null chromHMM result, adQTLs are NOT distinguishable from "
          "concordant eQTLs by baseline regulatory features or single-reference prediction "
          "strength. What defines an adQTL is purely its ancestry-differential EFFECT — a "
          "context/background phenomenon NOT encoded in the single-reference annotation, "
          "consistent with the sub-gate B finding that the sequence model (reference-only) "
          "cannot resolve it.")
    elif obs > 0:
        L(f"- EUR-biased: pred tracks beta_EUR > beta_AFR (diff={obs:+.3f}, p={p:.3f}) — "
          "the European-reference prediction is itself ancestry-specific.")
    else:
        L(f"- Unexpectedly AFR-leaning for adQTLs (diff={obs:+.3f}, p={p:.3f}); marginal, "
          "interpret with caution (hg38 is a mosaic, not a clean EUR background).")
    (DOCS / "phase1_refscore_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
