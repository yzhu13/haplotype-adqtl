"""
Phase 1 — Step 03: Robustness / reproducibility of the ancestry-differential
signal via SPLIT-HALF cross-fitting within MAGE.

True GTEx ancestry-stratified replication needs dbGaP-protected genotypes (only
pooled GTEx eQTL is public). The most rigorous FEASIBLE robustness with public
data is split-half: randomly halve EUR and AFR samples, estimate ObsDelta
(beta_EUR - beta_AFR) independently in each half, and ask whether ObsDelta in
half-1 predicts ObsDelta in half-2. This guards against:
  - winner's curse (adQTLs selected on heterogeneity could have inflated ObsDelta),
  - the signal being sampling noise rather than a reproducible effect.

Also: covariate-robustness (recompute with different #PEER) -> adQTL overlap.

Inputs: dosages (p0_05), MAGE expr/covariates/metadata, p1_decomposition (adQTL).
Outputs: results/phase1/p1_splithalf.tsv, figures/phase1/p1_splithalf.png,
  docs/phase1_robustness_report.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from p0_06_per_ancestry_betas import load_expr, load_covar, META, DOS  # reuse loaders

RES = BASE / "results" / "phase1"
RES0 = BASE / "results" / "phase0"
FIG = BASE / "figures" / "phase1"
DOCS = BASE / "docs"


def beta_local(sample_list, dvec, evec, covmat):
    """OLS slope of expr ~ dosage + covariates on sample_list; clean numpy."""
    d = dvec.reindex(sample_list).to_numpy(float)
    y = evec.reindex(sample_list).to_numpy(float)
    C = covmat.reindex(sample_list).to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(y) & np.isfinite(C).all(axis=1)
    if ok.sum() < 20:
        return np.nan
    d, y, C = d[ok], y[ok], C[ok]
    maf = d.sum() / (2 * len(d)); maf = min(maf, 1 - maf)
    if maf < 0.05 or min((d > 0).sum(), (d < 2).sum()) < 5:
        return np.nan
    X = np.column_stack([np.ones(len(d)), d, C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    expr = load_expr(); cov = load_covar(15)
    dos = pd.read_csv(DOS, sep="\t")
    meta = pd.read_csv(META, sep="\t")
    grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    samples = [s for s in expr.columns if s in cov.index and s in dos.columns]
    cov = cov.loc[samples]
    eur = [s for s in samples if grp.get(s) == "EUR"]
    afr = [s for s in samples if grp.get(s) == "AFR"]
    rng = np.random.default_rng(7)
    eur = list(rng.permutation(eur)); afr = list(rng.permutation(afr))
    eur1, eur2 = eur[:len(eur)//2], eur[len(eur)//2:]
    afr1, afr2 = afr[:len(afr)//2], afr[len(afr)//2:]
    print(f"split sizes: EUR {len(eur1)}/{len(eur2)}  AFR {len(afr1)}/{len(afr2)}")

    dec = pd.read_csv(RES0 / "p0_per_ancestry_betas.tsv", sep="\t")
    # restrict to pairs that were EUR&AFR estimable (have ObsDelta)
    dec = dec.dropna(subset=["ObsDelta_EUR_AFR"])
    # bring adQTL flag
    p1 = pd.read_csv(RES / "p1_decomposition.tsv", sep="\t")[["variant_kgpID","ensemblID","is_adqtl"]]
    dec = dec.merge(p1, on=["variant_kgpID","ensemblID"], how="left")

    rows = []
    dos_idx = dos.drop_duplicates("variant_kgpID").set_index("variant_kgpID")
    for _, r in dec.iterrows():
        vid, gene = r.variant_kgpID, r.ensemblID
        if gene not in expr.index or vid not in dos_idx.index:
            continue
        dser = dos_idx.loc[vid, samples].astype(float)
        evec = expr.loc[gene, samples].astype(float)
        bE1, bA1 = beta_local(eur1, dser, evec, cov), beta_local(afr1, dser, evec, cov)
        bE2, bA2 = beta_local(eur2, dser, evec, cov), beta_local(afr2, dser, evec, cov)
        rows.append({"variant_kgpID": vid, "ensemblID": gene, "is_adqtl": bool(r.is_adqtl),
                     "obsd_h1": bE1 - bA1, "obsd_h2": bE2 - bA2,
                     "obsd_full": r.ObsDelta_EUR_AFR})
    sh = pd.DataFrame(rows).dropna(subset=["obsd_h1", "obsd_h2"])
    sh.to_csv(RES / "p1_splithalf.tsv", sep="\t", index=False)

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii","replace").decode())

    L("# Phase 1 — split-half reproducibility of ancestry-differential effect\n")
    L(f"- split: EUR {len(eur1)}/{len(eur2)}, AFR {len(afr1)}/{len(afr2)}; pairs evaluated: {len(sh)}")
    for lbl, d in [("ALL pairs", sh), ("adQTLs only", sh[sh.is_adqtl])]:
        if len(d) < 8:
            L(f"- {lbl}: n={len(d)} too few"); continue
        sr, sp = stats.spearmanr(d.obsd_h1, d.obsd_h2)
        pr, pp = stats.pearsonr(d.obsd_h1, d.obsd_h2)
        sign = (np.sign(d.obsd_h1) == np.sign(d.obsd_h2)).mean()
        L(f"- {lbl} (n={len(d)}): ObsDelta half1 vs half2  Spearman={sr:+.3f}(p={sp:.1e}) "
          f"Pearson={pr:+.3f}(p={pp:.1e}) sign-concordance={sign:.0%}")
    # winner's curse: regress half2 (out-of-sample) on full ObsDelta for adQTLs -> shrinkage
    ad = sh[sh.is_adqtl]
    if len(ad) > 8:
        slope = np.polyfit(ad.obsd_full, ad.obsd_h2, 1)[0]
        L(f"- adQTL winner's-curse check: out-of-sample(half2) vs selection(full) slope={slope:.2f} "
          f"(1.0=no inflation; <1 indicates regression-to-mean/winner's curse)")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for axi, (lbl, d, col) in zip(ax, [("ALL", sh, "#888"), ("adQTLs", sh[sh.is_adqtl], "#c0392b")]):
        axi.scatter(d.obsd_h1, d.obsd_h2, s=12, alpha=.5, color=col)
        axi.axhline(0, lw=.4, color="k"); axi.axvline(0, lw=.4, color="k")
        if len(d) > 8:
            sr = stats.spearmanr(d.obsd_h1, d.obsd_h2)[0]
            pr = stats.pearsonr(d.obsd_h1, d.obsd_h2)[0]
            axi.set_title(f"{lbl} (n={len(d)}): Spearman {sr:.2f}, Pearson {pr:.2f}")
        axi.set_xlabel("ObsDelta (EUR/AFR half 1)"); axi.set_ylabel("ObsDelta (half 2)")
    fig.suptitle("Phase 1 — split-half reproducibility of ancestry-differential eQTL effect", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "p1_splithalf.png", dpi=150)

    L("\n## Note on independent replication")
    L("- True GTEx ancestry-stratified (EUR vs AFR) per-variant betas require dbGaP-")
    L("  protected genotypes; only pooled GTEx eQTL is public. Split-half within MAGE")
    L("  is the rigorous feasible substitute. Full external replication = future step")
    L("  (dbGaP GTEx re-analysis, or TOPMed/All-of-Us multi-ancestry eQTL).")
    (DOCS / "phase1_robustness_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
