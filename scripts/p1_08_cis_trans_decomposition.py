"""
Phase 1 / Direction 1 — Step 2: quantitative cis-vs-trans decomposition of ObsDelta.

For an adQTL whose focal effect depends on a background SNP via interaction gamma:
    focal effect for an individual with bg dosage b  =  beta0 + gamma * b
    => mean focal effect in ancestry A  ~  beta0 + gamma * E[bg | A] = beta0 + gamma*2*fA
    => ObsDelta = beta_EUR - beta_AFR  ~  2*gamma*(f_EUR - f_AFR)    [the cis-haplotype part]

So the cis-haplotype-mediated component of the ancestry effect difference is
    ObsDelta_cis = 2 * gamma * (f_EUR - f_AFR)
and the fraction of ObsDelta explained by this (single best) background SNP is
    frac_cis = ObsDelta_cis / ObsDelta      (a LOWER bound: only the single best bg).
Residual (1 - frac_cis) ~ trans / multi-bg / other.

f_EUR, f_AFR are the DIRECTIONAL alt-allele frequencies of the bg SNP (re-fetched).
gamma is the pooled interaction coefficient (int_beta from p1_06).

Inputs: results/phase1/p1_cis_interaction.tsv (int_beta, ObsDelta, best_bg),
  p1_within_ancestry.tsv (confirmed flag). Output: p1_cis_trans_decomp.tsv + report.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from p0_06_per_ancestry_betas import META
from p0_05_fetch_genotypes import mage_sample_order, gt_to_dosage, VCF_TMPL
from lib_remote_vcf import RemoteTabixVCF

P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"


def alt_freq(gts, cols):
    a = n = 0
    for c in cols:
        for p in gts[c].split(":", 1)[0].replace("/", "|").split("|"):
            if p != ".":
                a += int(p != "0"); n += 1
    return a / n if n else np.nan


def main():
    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    ci = ci[(ci.set == "adqtl") & (ci.bonf_p < 0.1) & ci.best_bg.notna()].copy()
    # confirmed-within-ancestry flag (if available)
    try:
        wa = pd.read_csv(P1 / "p1_within_ancestry.tsv", sep="\t")
        wa["confirmed"] = (wa.int_p_EUR < 0.05) | (wa.int_p_AFR < 0.05)
        ci = ci.merge(wa[["variant_kgpID", "best_bg", "confirmed"]], on=["variant_kgpID", "best_bg"], how="left")
    except Exception:
        ci["confirmed"] = np.nan
    ci["bg_chrom"] = "chr" + ci.best_bg.str.split(":").str[0]
    ci["bg_pos"] = ci.best_bg.str.split(":").str[1].astype(int)

    meta = pd.read_csv(META, sep="\t"); grp = dict(zip(meta.sample_kgpID, meta.continentalGroup))
    samples = mage_sample_order()
    eurset = {s for s in samples if grp.get(s) == "EUR"}; afrset = {s for s in samples if grp.get(s) == "AFR"}

    rows = []
    for chrom, g in ci.groupby("bg_chrom"):
        vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
        vsamp = {s: i for i, s in enumerate(vcf.samples())}
        eur_c = [vsamp[s] for s in eurset if s in vsamp]; afr_c = [vsamp[s] for s in afrset if s in vsamp]
        for _, r in g.iterrows():
            fE = fA = np.nan
            for ln in vcf.query(chrom, int(r.bg_pos), int(r.bg_pos)):
                f = ln.split("\t")
                if f[2] == r.best_bg:
                    gts = f[9:]; fE = alt_freq(gts, eur_c); fA = alt_freq(gts, afr_c); break
            if not np.isfinite(fE):
                continue
            gamma = r.int_beta; obsd = r.ObsDelta
            obsd_cis = 2 * gamma * (fE - fA)
            frac = obsd_cis / obsd if (obsd not in (0, np.nan) and np.isfinite(obsd) and abs(obsd) > 1e-6) else np.nan
            rows.append({"ensemblID": r.ensemblID, "variant_kgpID": r.variant_kgpID, "best_bg": r.best_bg,
                         "confirmed": r.get("confirmed", np.nan), "gamma": gamma,
                         "f_EUR": fE, "f_AFR": fA, "delta_f": fE - fA,
                         "ObsDelta": obsd, "ObsDelta_cis": obsd_cis, "frac_cis": frac})
    df = pd.DataFrame(rows)
    df.to_csv(P1 / "p1_cis_trans_decomp.tsv", sep="\t", index=False)

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())
    L("# Phase 1 / Direction 1 — cis-vs-trans decomposition of ObsDelta\n")
    L(f"- adQTLs decomposed: {len(df)} (confirmed within-ancestry: {int((df.confirmed==True).sum())})")
    for lbl, d in [("all significant-cis adQTLs", df), ("within-ancestry-confirmed only", df[df.confirmed == True])]:
        dd = d.dropna(subset=["frac_cis"])
        if len(dd) < 3:
            L(f"\n## {lbl}: n={len(dd)} too few"); continue
        fc = dd.frac_cis.clip(-0.5, 1.5)  # clip extreme ratios for summary
        L(f"\n## {lbl} (n={len(dd)})")
        L(f"- median frac of ObsDelta explained by single best cis-bg: {np.median(dd.frac_cis):.2f}")
        L(f"- % with frac_cis in (0.2, 1.5] (substantial cis): {100*((dd.frac_cis>0.2)&(dd.frac_cis<=1.5)).mean():.0f}%")
        L(f"- sign-concordant (ObsDelta_cis same sign as ObsDelta): {100*(np.sign(dd.ObsDelta_cis)==np.sign(dd.ObsDelta)).mean():.0f}%")
        tot = dd.ObsDelta_cis.abs().sum() / dd.ObsDelta.abs().sum()
        L(f"- aggregate Sigma|ObsDelta_cis| / Sigma|ObsDelta| = {tot:.2f}")
    L("\n## Interpretation")
    L("- frac_cis is a LOWER bound (single best background SNP only); the true cis-haplotype")
    L("  contribution (multiple bg SNPs) is larger. A substantial median frac + high sign-")
    L("  concordance => cis-haplotype epistasis explains a major share of the ancestry effect")
    L("  difference; the residual indicates trans / additional cis background.")
    (DOCS / "phase1_decomposition_cis_trans_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
