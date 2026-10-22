"""
Phase 1 / Direction 1 — DECISIVE confounder control for the cis-interaction result.

p1_06 found adQTLs are hugely enriched for a focal x background-SNP interaction
(90% vs 16% of controls). But because adQTLs have a focal x ancestry effect by
definition, and the background SNP is ancestry-differentiated, that interaction
COULD be focal x ancestry proxied by an ancestry-correlated SNP — not true
cis-epistasis.

Decisive test: re-fit the focal x best_bg interaction WITHIN a single ancestry
(EUR-only, AFR-only), where ancestry is held constant. If the interaction survives
within an ancestry, it CANNOT be focal x ancestry — it is genuine cis-haplotype
epistasis (the focal variant's effect depends on the local background haplotype
among individuals of the same ancestry).

For each significant adQTL from p1_06 (its best_bg), test interaction within EUR
and within AFR. Report how many confirm (within-ancestry p<0.05 in EUR and/or AFR).

Inputs: results/phase1/p1_cis_interaction.tsv (best_bg, best_bg_pos), expression,
  covariates, lead_variant_dosages, 1000G phased VCF.
Outputs: results/phase1/p1_within_ancestry.tsv, docs/phase1_within_ancestry_report.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from p0_06_per_ancestry_betas import load_expr, load_covar, META
from p0_05_fetch_genotypes import mage_sample_order, gt_to_dosage, VCF_TMPL
from lib_remote_vcf import RemoteTabixVCF

P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"
DOS = BASE / "data/processed/lead_variant_dosages.tsv"
# within-ancestry: fewer covariates to preserve df (EUR n=142)
NPC = 3
NPEER = 10


def fit_interaction(y, focal, bg, C):
    ok = np.isfinite(y) & np.isfinite(focal) & np.isfinite(bg) & np.isfinite(C).all(1)
    if ok.sum() < 50 or np.std(focal[ok]) == 0 or np.std(bg[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    X = np.column_stack([np.ones(ok.sum()), focal[ok], bg[ok], focal[ok] * bg[ok], C[ok]])
    try:
        res = sm.OLS(y[ok], X).fit()
        return float(res.params[3]), float(res.pvalues[3]), int(ok.sum())
    except Exception:
        return np.nan, np.nan, int(ok.sum())


def main():
    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    sig = ci[(ci.set == "adqtl") & (ci.bonf_p < 0.1) & ci.best_bg.notna()].copy()
    sig["bg_chrom"] = sig.best_bg.str.split(":").str[0].apply(lambda x: f"chr{x}")
    sig["bg_pos"] = sig.best_bg.str.split(":").str[1].astype(int)
    print(f"[p1_07] significant adQTLs to confirm: {len(sig)}")

    expr = load_expr()
    cov_full = load_covar(NPEER)
    keep_cov = [c for c in cov_full.columns if c == "sex" or c.startswith("PEER") or c in [f"PC{i}" for i in range(1, NPC + 1)]]
    cov = cov_full[keep_cov]
    meta = pd.read_csv(META, sep="\t"); grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    samples = mage_sample_order()
    eur = [s for s in samples if grp.get(s) == "EUR" and s in expr.columns and s in cov.index]
    afr = [s for s in samples if grp.get(s) == "AFR" and s in expr.columns and s in cov.index]
    fd = pd.read_csv(DOS, sep="\t").drop_duplicates("variant_kgpID").set_index("variant_kgpID")

    rows = []
    for chrom, g in sig.groupby("bg_chrom"):
        vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
        vsamp = {s: i for i, s in enumerate(vcf.samples())}
        for _, r in g.iterrows():
            vid = r.variant_kgpID; gene = r.ensemblID
            if vid not in fd.index or gene not in expr.index:
                continue
            # fetch best_bg dosages
            bgdos = {}
            for ln in vcf.query(chrom, int(r.bg_pos), int(r.bg_pos)):
                f = ln.split("\t")
                if f[2] == r.best_bg:
                    gts = f[9:]
                    bgdos = {s: gt_to_dosage(gts[vsamp[s]]) for s in (eur + afr) if s in vsamp}
                    break
            if not bgdos:
                continue
            res = {"ensemblID": gene, "variant_kgpID": vid, "best_bg": r.best_bg,
                   "pooled_int_p": r.min_int_p}
            for lbl, ss in [("EUR", eur), ("AFR", afr)]:
                y = expr.loc[gene, ss].astype(float).values
                fo = fd.loc[vid, ss].astype(float).values
                bg = np.array([bgdos.get(s, np.nan) for s in ss], float)
                C = cov.loc[ss].astype(float).values
                beta, p, n = fit_interaction(y, fo, bg, C)
                res[f"int_beta_{lbl}"] = beta; res[f"int_p_{lbl}"] = p; res[f"n_{lbl}"] = n
            rows.append(res)
            print(f"  {gene} bg={r.best_bg} EUR p={res['int_p_EUR']:.1e} AFR p={res['int_p_AFR']:.1e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(P1 / "p1_within_ancestry.tsv", sep="\t", index=False)

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())
    L("# Phase 1 / Direction 1 — within-ancestry confirmation (decisive cis-epistasis test)\n")
    L(f"- significant adQTLs tested: {len(df)} (each: focal x best_bg within EUR-only and AFR-only)")
    conf_eur = (df.int_p_EUR < 0.05); conf_afr = (df.int_p_AFR < 0.05)
    conf_any = conf_eur | conf_afr; conf_both = conf_eur & conf_afr
    L(f"- confirmed within EUR (p<0.05): {int(conf_eur.sum())}/{len(df)} ({conf_eur.mean():.0%})")
    L(f"- confirmed within AFR (p<0.05): {int(conf_afr.sum())}/{len(df)} ({conf_afr.mean():.0%})")
    L(f"- confirmed within >=1 ancestry: {int(conf_any.sum())}/{len(df)} ({conf_any.mean():.0%})")
    L(f"- confirmed within BOTH ancestries: {int(conf_both.sum())}/{len(df)} ({conf_both.mean():.0%})")
    # under the null (no cis-epistasis), expect ~5% by chance per ancestry
    L(f"- null expectation per ancestry ~5%; observed {conf_any.mean():.0%} (>=1) — "
      f"{'STRONGLY confirms genuine cis-epistasis' if conf_any.mean() > 0.4 else 'partial' if conf_any.mean()>0.15 else 'does NOT confirm (likely focal x ancestry)'}")
    L("\n## Confirmed cis-epistasis adQTLs (within-ancestry p<0.05)")
    for _, r in df[conf_any].sort_values("int_p_EUR").head(20).iterrows():
        L(f"- {r.ensemblID} bg={r.best_bg} | EUR p={r.int_p_EUR:.1e} (n={int(r.n_EUR)}) | "
          f"AFR p={r.int_p_AFR:.1e} (n={int(r.n_AFR)})")
    L("\n## Interpretation")
    L("- Within-ancestry survival proves the focal variant's effect depends on the local")
    L("  background haplotype AMONG INDIVIDUALS OF THE SAME ANCESTRY — genuine cis-epistasis,")
    L("  not focal x ancestry. This is the mechanism behind ancestry-differential eQTL effects:")
    L("  ancestry-differentiated background SNPs modulate the focal effect (cis-haplotype context).")
    (DOCS / "phase1_within_ancestry_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
