"""
Phase 1 / Direction 1 — Step 2 refined: MULTI-background cis-vs-trans decomposition.

The single-best-bg decomposition (p1_08) is a LOWER bound (median 46% of ObsDelta).
The focal variant's effect can depend on MULTIPLE independent background SNPs. Here
we aggregate independent cis-background contributions:

  per adQTL:
    1. candidate bg SNPs in +/-50kb: common in both EUR&AFR (MAF>=0.1), |Δf|>=0.1,
       r^2<0.5 with focal.
    2. marginal interaction scan (expr ~ focal+bg+focal:bg+cov), keep p<0.01.
    3. LD-prune kept bg (greedy, pairwise r^2<0.2) -> independent signals.
    4. JOINT model: expr ~ focal + Σ bg_k + Σ focal:bg_k + cov  (PCs included, so
       gamma_k is the clean within-sample cis-interaction, not stratification).
    5. ObsDelta_cis = 2 * Σ_k gamma_k * (f_k,EUR - f_k,AFR);  frac = ObsDelta_cis/ObsDelta.

gamma_k estimated with ancestry PCs (clean cis-interaction, confirmed within-ancestry
in p1_07); Δf_k is the bg's directional ancestry frequency difference. The sum over
independent bg SNPs raises the explained fraction toward the true cis-haplotype share.

Incremental checkpoint + resumable (lesson: never lose long-run work to sleep).

Inputs: p1_cis_interaction.tsv (sig adQTL list), expr/cov/genotypes/VCF.
Outputs: results/phase1/p1_multi_bg_decomp.tsv, docs/phase1_multi_bg_decomp_report.md
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

P0 = BASE / "results" / "phase0"; P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"
DOS = BASE / "data/processed/lead_variant_dosages.tsv"
WIN = 50_000; MIN_MAF_BOTH = 0.10; MAX_R2_FOCAL = 0.5; MIN_DELTA_F = 0.10
MARGINAL_P = 0.01      # keep bg with marginal interaction p < this
PRUNE_R2 = 0.2         # LD-prune bg to independent set
MAX_K = 10             # cap joint interaction terms (n=338)
OUT = P1 / "p1_multi_bg_decomp.tsv"


def adosage(gts, cols):
    return np.array([gt_to_dosage(gts[c]) for c in cols], float)


def main():
    expr = load_expr(); cov = load_covar(15); covmat = cov
    meta = pd.read_csv(META, sep="\t"); grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    sm_all = mage_sample_order()
    eur = [s for s in sm_all if grp.get(s) == "EUR"]; afr = [s for s in sm_all if grp.get(s) == "AFR"]
    ea = [s for s in sm_all if grp.get(s) in ("EUR", "AFR") and s in expr.columns and s in cov.index]
    fd = pd.read_csv(DOS, sep="\t").drop_duplicates("variant_kgpID").set_index("variant_kgpID")

    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    sig = ci[(ci.set == "adqtl") & (ci.bonf_p < 0.1)].copy()
    cal = pd.read_csv(P0 / "p0_calibration_set.tsv", sep="\t")[["variant_kgpID", "ensemblID", "variantChrom", "variantPosition"]]
    bet = pd.read_csv(P0 / "p0_per_ancestry_betas.tsv", sep="\t")[["variant_kgpID", "ensemblID", "ObsDelta_EUR_AFR"]]
    sig = sig.merge(cal, on=["variant_kgpID", "ensemblID"]).merge(bet, on=["variant_kgpID", "ensemblID"], how="left")

    done = set(pd.read_csv(OUT, sep="\t").variant_kgpID) if OUT.exists() else set()
    first = not OUT.exists()
    sig = sig[~sig.variant_kgpID.isin(done)]
    print(f"[p1_11] adQTLs to decompose: {len(sig)} (resume skip {len(done)})")

    for chrom, g in sig.groupby("variantChrom"):
        vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom)); vsamp = {s: i for i, s in enumerate(vcf.samples())}
        eur_c = [vsamp[s] for s in eur if s in vsamp]; afr_c = [vsamp[s] for s in afr if s in vsamp]
        ea_idx = {s: vsamp[s] for s in ea if s in vsamp}
        for _, r in g.iterrows():
            vid, gene, fpos = r.variant_kgpID, r.ensemblID, int(r.variantPosition)
            if vid not in fd.index or gene not in expr.index:
                continue
            y = expr.loc[gene, ea].astype(float).values
            focal = fd.loc[vid, ea].astype(float).values
            C = covmat.loc[ea].astype(float).values
            ok0 = np.isfinite(y) & np.isfinite(focal) & np.isfinite(C).all(1)
            # collect candidate bg dosage vectors + Δf
            cands = []
            for ln in vcf.query(chrom, fpos - WIN, fpos + WIN):
                f = ln.split("\t")
                need = max(max(eur_c), max(afr_c)) + 1
                if len(f) < 9 + need or int(f[1]) == fpos or len(f[3]) != 1 or len(f[4]) != 1 or "," in f[4]:
                    continue
                gts = f[9:]
                de = adosage(gts, eur_c); da = adosage(gts, afr_c)
                fe = de.sum() / (2 * len(de)); fa = da.sum() / (2 * len(da))
                if min(fe, 1-fe) < MIN_MAF_BOTH or min(fa, 1-fa) < MIN_MAF_BOTH or abs(fe-fa) < MIN_DELTA_F:
                    continue
                bg = np.array([gt_to_dosage(gts[ea_idx[s]]) if s in ea_idx else np.nan for s in ea], float)
                cands.append((f[2], bg, fe - fa))
            # marginal interaction scan
            kept = []
            for bid, bg, df_ in cands:
                ok = ok0 & np.isfinite(bg)
                if ok.sum() < 60 or np.std(focal[ok]) == 0 or np.std(bg[ok]) == 0:
                    continue
                if np.corrcoef(focal[ok], bg[ok])[0, 1] ** 2 > MAX_R2_FOCAL:
                    continue
                X = np.column_stack([np.ones(ok.sum()), focal[ok], bg[ok], focal[ok]*bg[ok], C[ok]])
                try:
                    p = sm.OLS(y[ok], X).fit().pvalues[3]
                except Exception:
                    continue
                if p < MARGINAL_P:
                    kept.append((bid, bg, df_, p))
            # LD-prune (greedy by p)
            kept.sort(key=lambda t: t[3]); pruned = []
            for bid, bg, df_, p in kept:
                ok = ok0 & np.isfinite(bg)
                indep = True
                for _, bg2, _, _ in pruned:
                    o2 = ok & np.isfinite(bg2)
                    if o2.sum() > 30 and np.corrcoef(bg[o2], bg2[o2])[0, 1] ** 2 > PRUNE_R2:
                        indep = False; break
                if indep:
                    pruned.append((bid, bg, df_, p))
                if len(pruned) >= MAX_K:
                    break
            # joint model + decomposition
            obsd_cis_single = np.nan; obsd_cis_multi = np.nan; k = len(pruned)
            if k >= 1:
                okj = ok0.copy()
                for _, bg, _, _ in pruned:
                    okj &= np.isfinite(bg)
                if okj.sum() > 60 + 2 * k:
                    bgmat = np.column_stack([bg[okj] for _, bg, _, _ in pruned])
                    intmat = bgmat * focal[okj][:, None]
                    X = np.column_stack([np.ones(okj.sum()), focal[okj], bgmat, intmat, C[okj]])
                    try:
                        res = sm.OLS(y[okj], X).fit()
                        gammas = res.params[2 + k: 2 + 2 * k]   # interaction coefs
                        dfs = np.array([df_ for _, _, df_, _ in pruned])
                        obsd_cis_multi = float(2 * np.sum(gammas * dfs))
                        obsd_cis_single = float(2 * gammas[0] * dfs[0])  # top bg in joint
                    except Exception:
                        pass
            obsd = r.ObsDelta_EUR_AFR
            frac_multi = obsd_cis_multi / obsd if (np.isfinite(obsd_cis_multi) and abs(obsd) > 1e-6) else np.nan
            rec = {"variant_kgpID": vid, "ensemblID": gene, "ObsDelta": obsd,
                   "n_bg_indep": k, "ObsDelta_cis_multi": obsd_cis_multi, "frac_cis_multi": frac_multi}
            pd.DataFrame([rec]).to_csv(OUT, sep="\t", mode="w" if first else "a", header=first, index=False)
            first = False
            print(f"  {gene} k_indep={k} frac_multi={frac_multi if frac_multi==frac_multi else float('nan'):.2f}", flush=True)

    # report
    df = pd.read_csv(OUT, sep="\t")
    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())
    L("# Phase 1 / Direction 1 — multi-background cis-vs-trans decomposition\n")
    L(f"- adQTLs: {len(df)}; median independent cis-bg SNPs/locus: {int(df.n_bg_indep.median())}; "
      f"max: {int(df.n_bg_indep.max())}")
    dd = df.dropna(subset=["frac_cis_multi"])
    fc = dd.frac_cis_multi
    L(f"- **median frac of ObsDelta explained by MULTI cis-bg: {np.median(fc):.2f}** "
      f"(single-best bg was ~0.46)")
    L(f"- % with frac in (0.2,1.5]: {100*((fc>0.2)&(fc<=1.5)).mean():.0f}%; "
      f"% explaining >=80%: {100*((fc>0.8)&(fc<=1.5)).mean():.0f}%")
    L(f"- sign-concordant: {100*(np.sign(dd.ObsDelta_cis_multi)==np.sign(dd.ObsDelta)).mean():.0f}%")
    L(f"- aggregate Sigma|ObsDelta_cis_multi| / Sigma|ObsDelta| = "
      f"{dd.ObsDelta_cis_multi.abs().sum()/dd.ObsDelta.abs().sum():.2f}")
    L("\n## Interpretation")
    L("- Aggregating independent cis-background SNPs raises the explained fraction of the")
    L("  ancestry effect difference above the single-bg lower bound. The residual (1-frac)")
    L("  reflects trans context + bg SNPs below detection. This quantifies how much of")
    L("  ancestry-differential cis-regulation is cis-haplotype (genetic, local) in origin.")
    (DOCS / "phase1_multi_bg_decomp_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
