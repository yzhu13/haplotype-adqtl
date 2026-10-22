"""
Phase 1 / Direction 1 — feasibility test: is the ancestry-differential effect of an
adQTL explained by cis-haplotype-background dependence (cis-epistasis)?

In LCLs (controlled cellular environment), GxE is largely controlled out, so a
surviving ancestry-differential effect points to GENETIC context. We test this
DIRECTLY (no AlphaGenome): for each adQTL focal variant V and gene G, fit

    expr_G ~ focal + bg + focal:bg + sex + PC1-5 + PEER1-15      (EUR+AFR samples)

for each nearby background SNP `bg`, and test the focal:bg interaction.

To decouple cis-haplotype from the ancestry label, background SNPs must be
polymorphic WITHIN both EUR and AFR (MAF>=0.1 each) and only weakly correlated
with the focal variant (r^2<0.5). We prioritize ancestry-differentiated background
SNPs (|MAF_EUR - MAF_AFR|>=0.1), since only those can contribute to ObsDelta.

Feasibility verdict: are adQTLs enriched (vs matched controls) for a significant
cis-interaction? If yes, Direction 1 is viable and reorganizes the project.

Scope (feasibility): top ~50 adQTLs (smallest p_het) + ~50 controls.

Outputs: results/phase1/p1_cis_interaction.tsv, docs/phase1_cis_interaction_report.md
"""
import sys, bisect
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from p0_06_per_ancestry_betas import load_expr, load_covar, META
from p0_05_fetch_genotypes import mage_sample_order, gt_to_dosage, VCF_TMPL
from lib_remote_vcf import RemoteTabixVCF

P0 = BASE / "results" / "phase0"; P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"
DOS = BASE / "data/processed/lead_variant_dosages.tsv"

import argparse
WIN = 50_000          # +/- background window
MIN_MAF_BOTH = 0.10   # bg SNP common in both EUR and AFR
MAX_R2_FOCAL = 0.5    # bg only weakly correlated with focal
MIN_DELTA_F = 0.10    # ancestry-differentiated bg (can contribute to ObsDelta)
_ap = argparse.ArgumentParser(); _ap.add_argument("--n-per-set", type=int, default=50)
N_PER_SET = _ap.parse_args().n_per_set


def maf(d):
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    f = d.sum() / (2 * len(d))
    return min(f, 1 - f)


def main():
    expr = load_expr(); cov = load_covar(15)
    meta = pd.read_csv(META, sep="\t"); grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    samples_all = mage_sample_order()
    eur = [s for s in samples_all if grp.get(s) == "EUR"]
    afr = [s for s in samples_all if grp.get(s) == "AFR"]
    ea = [s for s in samples_all if grp.get(s) in ("EUR", "AFR") and s in expr.columns and s in cov.index]
    print(f"[p1_06] EUR+AFR samples usable: {len(ea)}")

    # focal dosages
    fd = pd.read_csv(DOS, sep="\t").drop_duplicates("variant_kgpID").set_index("variant_kgpID")

    # select sets
    dec = pd.read_csv(P1 / "p1_decomposition.tsv", sep="\t")
    cal = pd.read_csv(P0 / "p0_calibration_set.tsv", sep="\t")[
        ["variant_kgpID", "ensemblID", "variantChrom", "variantPosition"]]
    bet = pd.read_csv(P0 / "p0_per_ancestry_betas.tsv", sep="\t")[
        ["variant_kgpID", "ensemblID", "p_het_EUR_AFR", "ObsDelta_EUR_AFR", "maf_EUR", "maf_AFR"]]
    adq = dec[dec.is_adqtl].merge(bet, on=["variant_kgpID", "ensemblID"], how="left", suffixes=("", "_b"))
    adq = adq.sort_values("p_het_EUR_AFR").head(N_PER_SET).assign(set="adqtl")
    ctrl_ids = set(pd.read_csv(P0 / "p0_control_set.tsv", sep="\t").variant_kgpID)
    ctrl = dec[dec.variant_kgpID.isin(ctrl_ids)].merge(bet, on=["variant_kgpID", "ensemblID"], how="left", suffixes=("", "_b"))
    ctrl = ctrl.head(N_PER_SET).assign(set="control")
    sets = pd.concat([adq, ctrl], ignore_index=True).merge(cal, on=["variant_kgpID", "ensemblID"], how="left").dropna(subset=["variantPosition"])
    print(f"[p1_06] loci to test: adQTL {int((sets.set=='adqtl').sum())}, control {int((sets.set=='control').sum())}")

    cov_cols = [c for c in cov.columns]
    OUTTSV = P1 / "p1_cis_interaction.tsv"
    done = set()
    if OUTTSV.exists():
        done = set(pd.read_csv(OUTTSV, sep="\t").variant_kgpID)
        print(f"[p1_06] resuming: {len(done)} loci already done")
    first = not OUTTSV.exists()

    def write_row(rec):
        nonlocal first
        pd.DataFrame([rec]).to_csv(OUTTSV, sep="\t", mode="w" if first else "a",
                                   header=first, index=False)
        first = False

    sets = sets[~sets.variant_kgpID.isin(done)]
    for chrom, grpdf in sets.groupby("variantChrom"):
        try:
            vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
        except Exception as e:
            print(f"  ERROR open {chrom}: {e}"); continue
        vsamp = {s: i for i, s in enumerate(vcf.samples())}
        eur_c = [vsamp[s] for s in eur if s in vsamp]; afr_c = [vsamp[s] for s in afr if s in vsamp]
        ea_c = {s: vsamp[s] for s in ea if s in vsamp}
        for _, r in grpdf.iterrows():
            vid = r.variant_kgpID; fpos = int(r.variantPosition); gene = r.ensemblID
            if vid not in fd.index or gene not in expr.index:
                continue
            focal = fd.loc[vid, ea].astype(float)
            y = expr.loc[gene, ea].astype(float)
            C = cov.loc[ea, cov_cols].astype(float)
            # fetch background SNPs
            cand = []
            for ln in vcf.query(chrom, fpos - WIN, fpos + WIN):
                f = ln.split("\t")
                if len(f) < 9 + max(max(eur_c), max(afr_c)) + 1:
                    continue
                pos = int(f[1])
                if pos == fpos or len(f[3]) != 1 or len(f[4]) != 1 or "," in f[4]:
                    continue
                gts = f[9:]
                fe = maf(np.array([gt_to_dosage(gts[c]) for c in eur_c], float))
                fa = maf(np.array([gt_to_dosage(gts[c]) for c in afr_c], float))
                if not (fe >= MIN_MAF_BOTH and fa >= MIN_MAF_BOTH):
                    continue
                if abs(fe - fa) < MIN_DELTA_F:
                    continue
                bg = np.array([gt_to_dosage(gts[ea_c[s]]) if s in ea_c else np.nan for s in ea], float)
                cand.append((f[2], pos, bg, fe, fa))
            # test interactions
            best_p = np.nan; best = None; n_test = 0
            for bid, bpos, bg, fe, fa in cand:
                ok = np.isfinite(focal.values) & np.isfinite(y.values) & np.isfinite(bg) & np.isfinite(C.values).all(1)
                if ok.sum() < 60:
                    continue
                fo = focal.values[ok]; bgo = bg[ok]
                if np.std(fo) == 0 or np.std(bgo) == 0:
                    continue
                r2 = np.corrcoef(fo, bgo)[0, 1] ** 2
                if r2 > MAX_R2_FOCAL:
                    continue
                X = np.column_stack([np.ones(ok.sum()), fo, bgo, fo * bgo, C.values[ok]])
                try:
                    res = sm.OLS(y.values[ok], X).fit()
                except Exception:
                    continue
                pint = res.pvalues[3]; n_test += 1
                if not np.isfinite(best_p) or pint < best_p:
                    best_p = pint; best = (bid, bpos, float(res.params[3]), fe, fa, r2)
            adj = best_p * n_test if (np.isfinite(best_p) and n_test > 0) else np.nan
            write_row({"variant_kgpID": vid, "ensemblID": gene, "set": r.set,
                       "p_het": r.p_het_EUR_AFR, "ObsDelta": r.ObsDelta_EUR_AFR,
                       "best_bg_pos": best[1] if best else np.nan,
                       "n_bg_tested": n_test, "min_int_p": best_p, "bonf_p": adj,
                       "best_bg": best[0] if best else "", "int_beta": best[2] if best else np.nan,
                       "bg_maf_EUR": best[3] if best else np.nan, "bg_maf_AFR": best[4] if best else np.nan,
                       "bg_r2_focal": best[5] if best else np.nan})
            print(f"  [{r.set}] {gene} n_bg={n_test} min_int_p={best_p if not np.isnan(best_p) else float('nan'):.1e}", flush=True)

    df = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())
    L("# Phase 1 / Direction 1 — cis-haplotype interaction feasibility\n")
    L(f"- loci tested: adQTL {int((df.set=='adqtl').sum())}, control {int((df.set=='control').sum())}")
    L(f"- median background SNPs tested/locus: {int(df.n_bg_tested.median())}")
    tested = df[df.n_bg_tested > 0]
    for thr, lab in [(0.05, "Bonferroni adj p<0.05"), (0.1, "adj p<0.1")]:
        ad = tested[tested.set == "adqtl"]; co = tested[tested.set == "control"]
        a1 = int((ad.bonf_p < thr).sum()); c1 = int((co.bonf_p < thr).sum())
        odds, p = stats.fisher_exact([[a1, len(ad)-a1], [c1, len(co)-c1]], alternative="greater")
        L(f"- {lab}: adQTL {a1}/{len(ad)} ({a1/max(1,len(ad)):.0%}) vs control {c1}/{len(co)} "
          f"({c1/max(1,len(co)):.0%})  OR={odds:.2f} p={p:.3f}")
    sig = tested[(tested.set=='adqtl') & (tested.bonf_p < 0.1)].sort_values("bonf_p")
    L(f"\n## adQTLs with significant cis-interaction (adj p<0.1): {len(sig)}")
    for _, r in sig.head(15).iterrows():
        L(f"- {r.ensemblID} int_p={r.min_int_p:.1e} (adj {r.bonf_p:.1e}) bg={r.best_bg} "
          f"int_beta={r.int_beta:+.2f} bg_MAF EUR={r.bg_maf_EUR:.2f}/AFR={r.bg_maf_AFR:.2f}")
    L("\n## Interpretation")
    L("- Enrichment of cis-interactions in adQTLs vs controls => ancestry-differential effects are")
    L("  (partly) cis-haplotype-dependent (cis-epistasis), tested directly from phased genotypes,")
    L("  bypassing the AlphaGenome tooling limit. Null => points to trans genetic context.")
    L("- FEASIBILITY: a positive or trending signal justifies scaling to all adQTLs + the cis-vs-trans")
    L("  decomposition of ObsDelta. (n per set small; this is a power-limited feasibility probe.)")
    (DOCS / "phase1_cis_interaction_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
