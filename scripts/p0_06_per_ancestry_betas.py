"""
Phase 0 — Step 06: Re-estimate eQTL effect sizes SEPARATELY within each
continental group, to get the OBSERVED ancestry-differential effect — the ground
truth for sub-gate B (does AlphaGenome's ADRS predict it?).

MAGE ships only a POOLED log2_aFC; here we stratify by continentalGroup and run
per-group OLS at each eGene lead variant:

    expr_inv_normal ~ alt_dosage + sex + PC1..PC5 + PEER1..PEER{k}

Per group g: beta_g, se_g, p_g, n_g, maf_g (groups with MAF<0.05 or <5 minor
carriers are skipped — effect undefined).

Ancestry-differential signal:
  - ObsDelta_EUR_AFR = beta_EUR - beta_AFR (primary)
  - pairwise heterogeneity z = (b1-b2)/sqrt(se1^2+se2^2), p_het (two-sided)
  - Cochran's Q across all available groups -> Q, p_Q, I2

API-free. Defines the adQTL (ancestry-differential) vs concordant sets.

Input : data/processed/lead_variant_dosages.tsv (p0_05)
        MAGE expr bed + covariates + sample.metadata
Output: results/phase0/p0_per_ancestry_betas.tsv
        docs/phase0_per_ancestry_betas_summary.md
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
MAGE = BASE / "data" / "raw" / "MAGE"
RES = BASE / "results" / "phase0"
PROC = BASE / "data" / "processed"
DOCS = BASE / "docs"
DOS = PROC / "lead_variant_dosages.tsv"
EXPR_BED = (MAGE / "QTL_results/eQTL_results/expression_quants/"
            "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz")
COVAR = (MAGE / "QTL_results/eQTL_results/expression_quants/eQTL_covariates.tab.gz")
META = MAGE / "sample_library_info/sample.metadata.MAGE.v1.0.txt"

GROUPS = ["EUR", "AFR", "EAS", "SAS", "AMR"]
MIN_MAF = 0.05
MIN_MINOR = 5


def load_expr():
    """gene(ensemblID) x sample DataFrame of inverse-normal expression."""
    rows, ids = [], []
    with gzip.open(EXPR_BED, "rt") as f:
        cols = next(f).rstrip("\n").split("\t")
        samples = cols[4:]
        for line in f:
            c = line.rstrip("\n").split("\t")
            ids.append(c[3])
            rows.append(np.array(c[4:], dtype=float))
    return pd.DataFrame(rows, index=ids, columns=samples)


def load_covar(n_peer):
    """sample x covariate DataFrame (sex, PC1-5, PEER1..n_peer)."""
    cov = pd.read_csv(COVAR, sep="\t", index_col=0)  # rows=covariate, cols=sample
    want = ["sex"] + [f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, n_peer + 1)]
    want = [w for w in want if w in cov.index]
    cdf = cov.loc[want].T.copy()  # sample x covariate
    if "sex" in cdf.columns:      # encode karyotype XX/XY -> 0/1
        cdf["sex"] = cdf["sex"].map({"XX": 0, "XY": 1}).fillna(
            pd.to_numeric(cdf["sex"], errors="coerce"))
    return cdf.astype(float)


def fit_group(dos, expr, cov, idx):
    """OLS expr~dosage+covariates on samples idx. Returns (beta,se,p,n,maf) or None."""
    d = dos[idx].astype(float)
    ok = np.isfinite(d) & np.isfinite(expr[idx].astype(float))
    if ok.sum() < 20:
        return None
    d = d[ok]
    maf = d.sum() / (2 * len(d))
    maf = min(maf, 1 - maf)
    minor = min((d > 0).sum(), (d < 2).sum())
    if maf < MIN_MAF or minor < MIN_MINOR:
        return None
    y = expr[idx].astype(float)[ok]
    X = np.column_stack([d, cov.loc[np.array(idx)[ok]].values])
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None
    return (float(res.params.iloc[1]), float(res.bse.iloc[1]),
            float(res.pvalues.iloc[1]), int(ok.sum()), float(maf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-peer", type=int, default=15)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    print("[p0_06] loading expr, covariates, dosages, metadata ...")
    expr = load_expr()
    cov = load_covar(args.n_peer)
    dos = pd.read_csv(DOS, sep="\t")
    meta = pd.read_csv(META, sep="\t")
    grp_of = dict(zip(meta["sample_kgpID"], meta["continentalGroup"]))

    # samples shared across all three
    samples = [s for s in expr.columns if s in cov.index and s in dos.columns]
    print(f"  shared samples: {len(samples)}")
    group_idx = {g: [s for s in samples if grp_of.get(s) == g] for g in GROUPS}
    print("  per-group n:", {g: len(v) for g, v in group_idx.items()})

    cov = cov.loc[samples]
    out = []
    for _, r in dos.iterrows():
        vid, gene = r["variant_kgpID"], r["ensemblID"]
        if gene not in expr.index:
            continue
        dvec = r[samples].to_dict()
        dser = pd.Series(dvec)
        evec = expr.loc[gene, samples]
        rec = {"variant_kgpID": vid, "ensemblID": gene, "geneSymbol": r.get("geneSymbol", "")}
        betas = {}
        for g in GROUPS:
            idx = group_idx[g]
            fit = fit_group(dser, evec, cov, idx)
            if fit:
                b, se, p, ng, maf = fit
                rec[f"beta_{g}"], rec[f"se_{g}"], rec[f"p_{g}"] = b, se, p
                rec[f"n_{g}"], rec[f"maf_{g}"] = ng, maf
                betas[g] = (b, se)
            else:
                for k in ("beta", "se", "p", "n", "maf"):
                    rec[f"{k}_{g}"] = np.nan
        # EUR vs AFR difference + heterogeneity
        if "EUR" in betas and "AFR" in betas:
            (bE, sE), (bA, sA) = betas["EUR"], betas["AFR"]
            rec["ObsDelta_EUR_AFR"] = bE - bA
            z = (bE - bA) / np.sqrt(sE**2 + sA**2)
            rec["z_EUR_AFR"] = z
            rec["p_het_EUR_AFR"] = 2 * stats.norm.sf(abs(z))
        else:
            rec["ObsDelta_EUR_AFR"] = rec["z_EUR_AFR"] = rec["p_het_EUR_AFR"] = np.nan
        # Cochran's Q across all available groups
        bs = [(betas[g][0], betas[g][1]) for g in GROUPS if g in betas]
        if len(bs) >= 2:
            w = np.array([1 / se**2 for _, se in bs])
            b = np.array([bb for bb, _ in bs])
            bbar = (w * b).sum() / w.sum()
            Q = (w * (b - bbar) ** 2).sum()
            df = len(bs) - 1
            rec["Q"], rec["p_Q"] = Q, stats.chi2.sf(Q, df)
            rec["I2"] = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
            rec["n_groups"] = len(bs)
        else:
            rec["Q"] = rec["p_Q"] = rec["I2"] = np.nan
            rec["n_groups"] = len(bs)
        out.append(rec)

    df = pd.DataFrame(out)
    df.to_csv(RES / "p0_per_ancestry_betas.tsv", sep="\t", index=False)
    print(f"[p0_06] {len(df)} variant-gene pairs -> {RES/'p0_per_ancestry_betas.tsv'}")

    # summary
    both = df.dropna(subset=["p_het_EUR_AFR"])
    n_sig = (both["p_het_EUR_AFR"] < 0.05).sum()
    nq = df.dropna(subset=["p_Q"])
    summ = [
        "# Phase 0 — Per-ancestry eQTL beta summary",
        f"\n- pairs tested: {len(df)}",
        f"- with both EUR & AFR estimable: {len(both)}",
        f"- EUR-AFR heterogeneity p<0.05: {n_sig} ({100*n_sig/max(1,len(both)):.1f}%)",
        f"- Cochran Q estimable (>=2 groups): {len(nq)}; p_Q<0.05: {(nq['p_Q']<0.05).sum()} "
        f"({100*(nq['p_Q']<0.05).mean():.1f}%)" if len(nq) else "- Cochran Q: n/a",
        f"- median |ObsDelta_EUR_AFR|: {both['ObsDelta_EUR_AFR'].abs().median():.3f}" if len(both) else "",
        "\nThese define the adQTL (heterogeneous) vs concordant sets for sub-gate B.",
        "\n*Covariates:* dosage + sex + PC1-5 + PEER1-%d, per-group OLS." % args.n_peer,
    ]
    (DOCS / "phase0_per_ancestry_betas_summary.md").write_text("\n".join(summ), encoding="utf-8")
    print("\n".join(summ))


if __name__ == "__main__":
    main()
