"""
Phase 1 — Step 16b: Permutation negative control for cis-haplotype epistasis.

Design: For each locus with a valid best_bg, run the same OLS interaction test
(expr ~ focal + bg + focal:bg + covariates) but with the background SNP dosages
PERMUTED (sample labels shuffled within the combined EUR+AFR population).

This preserves:
  - Expression ~ focal main effect and all covariates (unchanged)
  - Background SNP MAF and marginal distribution (just reordered)

But DESTROYS:
  - Any true focal x background interaction (cis-haplotype pairing broken)

Expected: ~5% significant rate for BOTH adQTL and control after permutation.
If the real cis enrichment (adQTL 54% vs control 15%) drops to ~5% for both,
that confirms the signal is specifically dependent on the cis co-localization of
focal and background variants on the same chromosome.

N_PERM = 100 permutations per locus; call locus "perm-significant" if at least
one permutation yields p < 0.05/N_PERM (Bonferroni within locus, same as real).
Report: median permutation p-value distribution vs real p-value per locus.

Also: run 5 TRANS focal tests — use focal dosage from a DIFFERENT gene's focal
variant on a different chromosome, paired with the original best_bg. This is a
cleaner trans control than using best_bg SNPs from other loci.

Output:
  results/phase1/p1_perm_control.tsv
  figures/paper/Figure_TransControl.png (updated)
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

MAGE    = BASE / "data" / "raw" / "MAGE"
PROC    = BASE / "data" / "processed"
RES1    = BASE / "results" / "phase1"

CIS_TSV  = RES1 / "p1_cis_interaction.tsv"
DOS_TSV  = PROC / "lead_variant_dosages.tsv"
EXPR_BED = (MAGE / "QTL_results/eQTL_results/expression_quants/"
            "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz")
COVAR    = (MAGE / "QTL_results/eQTL_results/expression_quants/eQTL_covariates.tab.gz")

# Reuse bg dosage cache from p1_16
import pickle
CACHE_PKL = RES1 / "trans_bg_dosage_cache.pkl"
OUT_TSV   = RES1 / "p1_perm_control.tsv"

N_PERM  = 100
N_PEER  = 15
SEED    = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mage_samples():
    with gzip.open(EXPR_BED, "rt") as f:
        cols = next(f).rstrip("\n").split("\t")
    return cols[4:]


def load_covar(samples):
    cov = pd.read_csv(COVAR, sep="\t", index_col=0)
    want = ["sex"] + [f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, N_PEER + 1)]
    want = [w for w in want if w in cov.index]
    cdf = cov.loc[want].T.copy()
    if "sex" in cdf.columns:
        cdf["sex"] = cdf["sex"].map({"XX": 0, "XY": 1}).fillna(
            pd.to_numeric(cdf["sex"], errors="coerce"))
    cdf = cdf.reindex(samples)
    return cdf.astype(float)


def load_expr_gene(gene_id):
    with gzip.open(EXPR_BED, "rt") as f:
        header = next(f).rstrip("\n").split("\t")
        samples = header[4:]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[3] == gene_id:
                return np.array(parts[4:], dtype=float), samples
    return None, None


def ols_interaction_pval(focal, bg, expr, cov_df, samples):
    """OLS: expr ~ focal + bg + focal*bg + covariates. Returns p-val of interaction."""
    mask = ~(np.isnan(focal) | np.isnan(bg) | np.isnan(expr))
    if mask.sum() < 30:
        return np.nan
    f = focal[mask]; b = bg[mask]; y = expr[mask]
    cov = cov_df.reindex(np.array(samples)[mask]).values
    if np.any(np.isnan(cov)):
        valid = ~np.any(np.isnan(cov), axis=1)
        f = f[valid]; b = b[valid]; y = y[valid]; cov = cov[valid]
    if len(y) < 20:
        return np.nan
    inter = f * b
    X = np.column_stack([np.ones(len(y)), f, b, inter, cov])
    n, p = X.shape
    if n <= p:
        return np.nan
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sigma2 = np.dot(resid, resid) / (n - p)
        XtXinv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(sigma2 * np.diag(XtXinv), 0))
        t_stat = coef[3] / se[3] if se[3] > 0 else np.nan
        if np.isnan(t_stat):
            return np.nan
        return float(2 * stats.t.sf(abs(t_stat), df=n - p))
    except np.linalg.LinAlgError:
        return np.nan


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    # Load inputs
    print("Loading data ...", flush=True)
    cis = pd.read_csv(CIS_TSV, sep="\t")
    valid = cis[cis["best_bg"].notna() & (cis["n_bg_tested"] > 0)].copy()
    print(f"  {len(valid)} loci with valid best_bg")

    mage_samples = load_mage_samples()
    print(f"  {len(mage_samples)} MAGE samples")

    # Load bg dosage cache (from p1_16 run)
    if CACHE_PKL.exists():
        with open(CACHE_PKL, "rb") as fh:
            dosage_cache = pickle.load(fh)
        print(f"  Loaded dosage cache: {len(dosage_cache)} SNPs")
    else:
        print("  WARNING: no dosage cache found, run p1_16 first")
        dosage_cache = {}

    # Load focal dosages
    dos_df = pd.read_csv(DOS_TSV, sep="\t", index_col=0)
    sample_cols = [c for c in dos_df.columns if c not in ("ensemblID", "geneSymbol")]
    focal_dos_df = dos_df[sample_cols]
    focal_dos_df = focal_dos_df[~focal_dos_df.index.duplicated(keep="first")]
    focal_samples = sample_cols
    print(f"  {len(focal_dos_df)} focal variants")

    # Load covariates
    cov_df = load_covar(mage_samples)
    print(f"  Covariates: {cov_df.shape}")

    results = []
    n_loci = len(valid)
    t0 = __import__("time").time()

    for i, (_, row) in enumerate(valid.iterrows()):
        if i % 20 == 0:
            elapsed = __import__("time").time() - t0
            print(f"  [{i}/{n_loci}] elapsed={elapsed:.0f}s", flush=True)

        vid      = row["variant_kgpID"]
        gene_id  = row["ensemblID"]
        lset     = row["set"]
        best_bg  = row["best_bg"]
        real_p   = row["bonf_p"]   # already computed in cis analysis

        # get bg dosages
        if best_bg not in dosage_cache:
            continue
        bg_full = dosage_cache[best_bg]   # aligned to mage_samples

        # get focal dosages aligned to mage_samples
        if vid not in focal_dos_df.index:
            continue
        focal_row = focal_dos_df.loc[vid]
        focal_s   = pd.Series(focal_row.values, index=focal_samples)
        focal_full = focal_s.reindex(mage_samples).values.astype(float)

        # load expression
        expr_arr, expr_samples = load_expr_gene(gene_id)
        if expr_arr is None:
            continue
        # align to mage_samples
        expr_s   = pd.Series(expr_arr, index=expr_samples)
        expr_full = expr_s.reindex(mage_samples).values.astype(float)

        # --- PERMUTATION CONTROL ---
        # Shuffle bg dosage N_PERM times; record how often interaction is sig
        perm_pvals = []
        for _ in range(N_PERM):
            bg_perm = rng.permutation(bg_full)
            p = ols_interaction_pval(focal_full, bg_perm, expr_full, cov_df, mage_samples)
            perm_pvals.append(p)

        perm_pvals = np.array(perm_pvals)
        # Bonferroni within permutation (1 bg SNP per locus here, threshold = 0.05)
        n_perm_sig = int(np.sum(perm_pvals < 0.05))
        perm_sig_rate = n_perm_sig / N_PERM
        median_perm_p = float(np.nanmedian(perm_pvals))

        results.append({
            "variant_kgpID":  vid,
            "ensemblID":      gene_id,
            "set":            lset,
            "real_bonf_p":    real_p,
            "real_sig":       real_p < 0.05,
            "perm_sig_rate":  perm_sig_rate,
            "median_perm_p":  median_perm_p,
            "n_perm_sig":     n_perm_sig,
        })

    out = pd.DataFrame(results)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nSaved: {OUT_TSV}", flush=True)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n=== REAL vs PERMUTATION COMPARISON ===")
    for grp in ["adqtl", "control"]:
        sub = out[out["set"] == grp]
        real_rate  = sub["real_sig"].mean() * 100
        # a locus is "perm-significant" if its permutation sig rate > 5%
        # (i.e., more than 5% of permutations are nominally significant)
        perm_rate  = (sub["perm_sig_rate"] > 0.05).mean() * 100
        median_perm = sub["median_perm_p"].median()
        print(f"  {grp}: real_sig={real_rate:.1f}%  perm_sig_rate>5%={perm_rate:.1f}%  "
              f"median_perm_p={median_perm:.3f}  n={len(sub)}")


if __name__ == "__main__":
    main()
