"""
Phase 2 -- Step 01: per-ancestry regional (TSS +/- 500kb) nominal eQTL scan for
the Pan-UKB coloc pilot loci.

For each pilot gene x ancestry (EUR, AFR), fetch phased 1000G genotypes for all
SNVs in the +/-500kb window around the gene's TSS, and run, per SNP:

    expr_inv_normal ~ dosage + sex + PC1-5 + PEER1-15   (ancestry subset)

via the Frisch-Waugh-Lovell trick (residualize y and all SNP dosage columns on
the covariate design once, then per-SNP beta/se/p in closed form) -- this makes
a ~1Mb x ~10k-SNP window x 2 ancestries x 11 genes scan tractable.

Same covariates/MAF filters as p0_06_per_ancestry_betas.py.

Output: results/phase2/p2_regional_eqtl/{gene}_{ancestry}.tsv
  columns: variant_kgpID, chrom, pos, ref, alt, beta, se, pval, n, maf
Checkpointed per gene x ancestry (skip if output file exists).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_remote_vcf import RemoteTabixVCF
from p0_05_fetch_genotypes import VCF_TMPL, mage_sample_order, gt_to_dosage
from p0_06_per_ancestry_betas import load_expr, load_covar, GROUPS, MIN_MAF, MIN_MINOR

BASE = Path(__file__).resolve().parents[1]
MAGE = BASE / "data" / "raw" / "MAGE"
META = MAGE / "sample_library_info/sample.metadata.MAGE.v1.0.txt"
OUT_DIR = BASE / "results" / "phase2" / "p2_regional_eqtl"

WINDOW = 500_000

# (geneSymbol, ensemblID, chrom, TSS, lead_variant_kgpID)
LOCI = [
    ("AMZ1",          "ENSG00000174945.14", "chr7",  2679521,   "7:2709535:T:C"),
    ("PHYKPL",        "ENSG00000175309.15", "chr5",  178232801, "5:178232623:C:A"),
    ("NR2F6",         "ENSG00000160113.6",  "chr19", 17245918,  "19:17237270:G:A"),
    ("CDK10",         "ENSG00000185324.22", "chr16", 89680736,  "16:89698141:G:A"),
    ("CTD-2054N24.2", "ENSG00000259363.6",  "chr15", 99805792,  "15:99860384:T:C"),
    ("FNBP1P1",       "ENSG00000257800.1",  "chr2",  74120679,  "2:74120539:T:G"),
    ("EPB41L2",       "ENSG00000079819.19", "chr6",  131063321, "6:131075076:G:A"),
    ("ZNF341",        "ENSG00000131061.14", "chr20", 33731656,  "20:33775731:G:T"),
    ("LRRC61",        "ENSG00000127399.15", "chr7",  150323262, "7:150323437:C:T"),
    ("PMS2P5",        "ENSG00000123965.13", "chr7",  74894115,  "7:74936194:A:G"),
    ("ANKH",          "ENSG00000154122.14", "chr5",  14871777,  "5:14877365:A:G"),
]


def fetch_window_genotypes(chrom, beg, end, samples):
    """Return (df_meta[variant_kgpID,chrom,pos,ref,alt], dosage_matrix[n_snv x n_sample])."""
    vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
    vcf_samples = vcf.samples()
    sidx = {s: i for i, s in enumerate(vcf_samples)}
    keep = [s for s in samples if s in sidx]
    col_pos = [sidx[s] for s in keep]

    n_vcf_samples = len(vcf_samples)
    meta_rows, dos_rows = [], []
    for ln in vcf.query(chrom, beg, end):
        f = ln.split("\t")
        if len(f) < 9 + n_vcf_samples:
            continue  # truncated line at BGZF chunk boundary
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1 or "," in alt:
            continue  # SNVs only
        gts = f[9:]
        dos = [gt_to_dosage(gts[c]) for c in col_pos]
        if any(d is None for d in dos):
            continue
        meta_rows.append((f[2], chrom, int(f[1]), ref, alt))
        dos_rows.append(dos)

    meta = pd.DataFrame(meta_rows, columns=["variant_kgpID", "chrom", "pos", "ref", "alt"])
    dosmat = np.array(dos_rows, dtype=float)  # n_snv x n_keep
    return meta, dosmat, keep


def fwl_scan(dosmat, y, Z):
    """Frisch-Waugh-Lovell: per-row (SNP) beta/se/pval/maf of y ~ dosage + Z (no intercept
    needed in Z; we add one). dosmat: n_snv x n; y: n; Z: n x p (covariates)."""
    n = len(y)
    Z1 = np.column_stack([np.ones(n), Z])  # n x (p+1)
    # projection: resid = X - Z1 (Z1'Z1)^-1 Z1'X
    ZtZ_inv = np.linalg.pinv(Z1.T @ Z1)
    Hz = Z1 @ ZtZ_inv @ Z1.T  # n x n (small n, fine)
    ry = y - Hz @ y  # n
    rD = dosmat - dosmat @ Hz.T  # n_snv x n  (rows residualized)

    rDD = (rD * rD).sum(axis=1)  # n_snv
    rDy = (rD * ry[None, :]).sum(axis=1)  # n_snv

    beta = np.divide(rDy, rDD, out=np.full_like(rDy, np.nan), where=rDD > 1e-12)
    pred = beta[:, None] * rD
    sse = ((ry[None, :] - pred) ** 2).sum(axis=1)
    df = n - Z1.shape[1] - 1
    se = np.sqrt(np.divide(sse, df * rDD, out=np.full_like(rDD, np.nan), where=rDD > 1e-12))
    tstat = beta / se
    pval = 2 * stats.t.sf(np.abs(tstat), df)
    maf = dosmat.mean(axis=1) / 2
    maf = np.minimum(maf, 1 - maf)
    return beta, se, pval, maf


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = mage_sample_order()
    print(f"[p2_01] MAGE samples: {len(samples)}")

    expr = load_expr()
    cov = load_covar(n_peer=15)
    meta_df = pd.read_csv(META, sep="\t")
    grp_of = dict(zip(meta_df["sample_kgpID"], meta_df["continentalGroup"]))

    for gene, ensg, chrom, tss, lead_vid in LOCI:
        beg, end = max(1, tss - WINDOW), tss + WINDOW
        for anc in ["EUR", "AFR"]:
            out_f = OUT_DIR / f"{gene}_{anc}.tsv"
            if out_f.exists():
                print(f"[p2_01] {gene} {anc}: exists, skip")
                continue

            print(f"[p2_01] {gene} ({ensg}) {chrom}:{beg}-{end} {anc} ...")
            anc_samples = [s for s in samples
                            if grp_of.get(s) == anc and s in expr.columns and s in cov.index]
            try:
                meta, dosmat_all, keep = fetch_window_genotypes(chrom, beg, end, anc_samples)
            except Exception as e:
                print(f"  ERROR fetching genotypes: {e}")
                continue
            if len(meta) == 0:
                print("  no SNVs in window, skip")
                meta_out = pd.DataFrame(columns=["variant_kgpID", "chrom", "pos", "ref", "alt",
                                                   "beta", "se", "pval", "n", "maf"])
                meta_out.to_csv(out_f, sep="\t", index=False)
                continue

            y = expr.loc[ensg, keep].astype(float).values
            Z = cov.loc[keep].values

            # MAF / minor-allele-count filter
            allele_sum = dosmat_all.sum(axis=1)
            minor_n = np.minimum((dosmat_all > 0).sum(axis=1), (dosmat_all < 2).sum(axis=1))
            maf_all = allele_sum / (2 * dosmat_all.shape[1])
            maf_all = np.minimum(maf_all, 1 - maf_all)
            ok = (maf_all >= MIN_MAF) & (minor_n >= MIN_MINOR)
            meta_ok = meta[ok].reset_index(drop=True)
            dosmat = dosmat_all[ok]
            print(f"  {len(meta)} SNVs in window, {len(meta_ok)} pass MAF>={MIN_MAF}/minor>={MIN_MINOR}, n={len(keep)}")

            if len(meta_ok) == 0:
                meta_out = pd.DataFrame(columns=["variant_kgpID", "chrom", "pos", "ref", "alt",
                                                   "beta", "se", "pval", "n", "maf"])
                meta_out.to_csv(out_f, sep="\t", index=False)
                continue

            beta, se, pval, maf = fwl_scan(dosmat, y, Z)
            meta_ok["beta"] = beta
            meta_ok["se"] = se
            meta_ok["pval"] = pval
            meta_ok["n"] = len(keep)
            meta_ok["maf"] = maf
            meta_ok.to_csv(out_f, sep="\t", index=False)

            # sanity: check lead variant if present
            lead = meta_ok[meta_ok.variant_kgpID == lead_vid]
            if len(lead):
                print(f"  lead variant {lead_vid}: beta={lead.iloc[0]['beta']:.3f} "
                      f"se={lead.iloc[0]['se']:.3f} (p0_06 comparison expected)")

    print("\n[p2_01] done.")


if __name__ == "__main__":
    main()
