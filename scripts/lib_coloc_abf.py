"""
Pure-Python coloc.abf (Giambartolomei et al. 2014 PLoS Genet; Wakefield 2009 ABF).

No LD / SuSiE required: combines per-SNP Wakefield approximate Bayes factors
(from beta/se) over a shared SNP set for two traits into PP.H0-H4.

  PP.H0: neither trait has a causal variant in the region
  PP.H1: only trait 1 has a causal variant
  PP.H2: only trait 2 has a causal variant
  PP.H3: both traits have causal variants, but DIFFERENT ones
  PP.H4: both traits share the SAME causal variant  <- coloc evidence

Default priors match the `coloc` R package: p1=p2=1e-4 (prior prob a given SNP
is causal for trait 1 / trait 2), p12=1e-5 (prior prob it's causal for both),
W=0.15**2 (prior variance of the effect size, Wakefield's default).
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def log_abf(beta, se, W=0.15 ** 2):
    """Per-SNP log Wakefield ABF (log Bayes factor for association vs null)."""
    V = se ** 2
    z = beta / se
    r = W / (W + V)
    return 0.5 * (np.log(1 - r) + r * z ** 2)


def _log_sub_exp(a, b):
    """log(exp(a) - exp(b)), a >= b, stable."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.exp(b - a)
    diff = np.clip(diff, None, 1 - 1e-300)  # guard a==b -> log(0)
    return a + np.log1p(-diff)


def coloc_abf(df1, df2, p1=1e-4, p2=1e-4, p12=1e-5, W=0.15 ** 2):
    """
    df1, df2: DataFrames indexed by a shared SNP id, with columns 'beta','se'.
    Returns dict with PP.H0..PP.H4, n_snps, and the top SNP per trait.
    """
    snps = df1.index.intersection(df2.index)
    if len(snps) == 0:
        return {"PP.H0": np.nan, "PP.H1": np.nan, "PP.H2": np.nan,
                "PP.H3": np.nan, "PP.H4": np.nan, "n_snps": 0,
                "top_snp_1": None, "top_snp_2": None}

    d1 = df1.loc[snps]
    d2 = df2.loc[snps]
    l1 = log_abf(d1["beta"].values, d1["se"].values, W)
    l2 = log_abf(d2["beta"].values, d2["se"].values, W)

    logS1 = logsumexp(l1)
    logS2 = logsumexp(l2)
    logS12 = logsumexp(l1 + l2)

    l0 = 0.0
    lh1 = np.log(p1) + logS1
    lh2 = np.log(p2) + logS2
    log_s1_s2 = logS1 + logS2
    lh3 = (np.log(p1) + np.log(p2)
           + _log_sub_exp(log_s1_s2, logS12))
    lh4 = np.log(p12) + logS12

    logs = np.array([l0, lh1, lh2, lh3, lh4])
    pp = np.exp(logs - logsumexp(logs))

    return {
        "PP.H0": pp[0], "PP.H1": pp[1], "PP.H2": pp[2],
        "PP.H3": pp[3], "PP.H4": pp[4],
        "n_snps": len(snps),
        "top_snp_1": d1["beta"].abs().div(d1["se"]).idxmax(),
        "top_snp_2": d2["beta"].abs().div(d2["se"]).idxmax(),
    }


def _toy_test():
    """Sanity check on synthetic data."""
    rng = np.random.default_rng(0)
    n = 50
    snps = [f"snp{i}" for i in range(n)]
    se1 = np.full(n, 0.1)
    se2 = np.full(n, 0.1)

    # Case A: shared causal variant at snp10 -> expect high PP.H4
    beta1 = rng.normal(0, 0.02, n)
    beta2 = rng.normal(0, 0.02, n)
    beta1[10] = 0.6
    beta2[10] = 0.5
    df1 = pd.DataFrame({"beta": beta1, "se": se1}, index=snps)
    df2 = pd.DataFrame({"beta": beta2, "se": se2}, index=snps)
    res = coloc_abf(df1, df2)
    print("Case A (shared causal snp10):", {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    assert res["PP.H4"] > 0.8, "expected high PP.H4 for shared causal variant"

    # Case B: different causal variants (snp10 vs snp30) -> expect high PP.H3
    beta2b = rng.normal(0, 0.02, n)
    beta2b[30] = 0.6
    df2b = pd.DataFrame({"beta": beta2b, "se": se2}, index=snps)
    res_b = coloc_abf(df1, df2b)
    print("Case B (different causal snps):", {k: round(v, 4) if isinstance(v, float) else v for k, v in res_b.items()})
    assert res_b["PP.H3"] > 0.5, "expected high PP.H3 for distinct causal variants"

    # Case C: no signal in either trait -> expect high PP.H0
    df1c = pd.DataFrame({"beta": rng.normal(0, 0.02, n), "se": se1}, index=snps)
    df2c = pd.DataFrame({"beta": rng.normal(0, 0.02, n), "se": se2}, index=snps)
    res_c = coloc_abf(df1c, df2c)
    print("Case C (no signal):", {k: round(v, 4) if isinstance(v, float) else v for k, v in res_c.items()})
    assert res_c["PP.H0"] > 0.5, "expected high PP.H0 for noise-only"

    print("\nlib_coloc_abf sanity checks PASSED.")


if __name__ == "__main__":
    _toy_test()
