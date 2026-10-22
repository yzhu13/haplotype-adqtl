"""
Phase 2 -- Step 03: run coloc.abf between the p2_01 regional eQTL scans and the
p2_02 Pan-UKB GWAS sumstats, per locus x ancestry x matched phenotype.

For each (gene, ancestry, phenocode):
  - load results/phase2/p2_regional_eqtl/{gene}_{ancestry}.tsv  (eQTL beta/se,
    indexed by hg38 variant_kgpID)
  - load results/phase2/p2_panukb/{gene}_{ancestry}_{phenocode}.tsv (GWAS
    beta/se, indexed by hg38-lifted variant_kgpID)
  - merge on variant_kgpID (chrom:pos:ref:alt -- requires matching REF/ALT
    orientation; SNPs where Pan-UKB ref/alt is swapped relative to 1000G are
    dropped here as a conservative first pass, flagged via n_snps vs n_eqtl/n_gwas)
  - coloc_abf(eqtl, gwas) -> PP.H0-H4

Output: results/phase2/p2_coloc.tsv
  one row per gene x ancestry x phenocode, with PP.H0-H4, n_snps, n_eqtl,
  n_gwas, top_snp_eqtl, top_snp_gwas.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_coloc_abf import coloc_abf
from p2_01_regional_eqtl_nominal import LOCI
from p2_02_fetch_panukb import LOCUS_PHENOS, ANCESTRIES

BASE = Path(__file__).resolve().parents[1]
EQTL_DIR = BASE / "results" / "phase2" / "p2_regional_eqtl"
GWAS_DIR = BASE / "results" / "phase2" / "p2_panukb"
OUT = BASE / "results" / "phase2" / "p2_coloc.tsv"


def main():
    rows = []
    for gene, ensg, chrom, tss, lead_vid in LOCI:
        for anc in ANCESTRIES:
            eqtl_f = EQTL_DIR / f"{gene}_{anc}.tsv"
            if not eqtl_f.exists():
                print(f"[p2_03] {gene} {anc}: missing eQTL file, skip")
                continue
            eqtl = pd.read_csv(eqtl_f, sep="\t")
            if len(eqtl) == 0:
                print(f"[p2_03] {gene} {anc}: empty eQTL table, skip")
                continue
            eqtl = eqtl.drop_duplicates("variant_kgpID").set_index("variant_kgpID")

            for phenocode in LOCUS_PHENOS[gene]:
                gwas_f = GWAS_DIR / f"{gene}_{anc}_{phenocode}.tsv"
                if not gwas_f.exists():
                    print(f"[p2_03] {gene} {anc} pheno{phenocode}: missing GWAS file, skip")
                    continue
                gwas = pd.read_csv(gwas_f, sep="\t")
                if len(gwas) == 0:
                    print(f"[p2_03] {gene} {anc} pheno{phenocode}: empty GWAS table, skip")
                    continue
                gwas = gwas.drop_duplicates("variant_kgpID").set_index("variant_kgpID")

                res = coloc_abf(eqtl[["beta", "se"]], gwas[["beta", "se"]])
                rec = {
                    "geneSymbol": gene, "ancestry": anc, "phenocode": phenocode,
                    "lead_variant_kgpID": lead_vid,
                    "n_eqtl": len(eqtl), "n_gwas": len(gwas),
                    **res,
                }
                rows.append(rec)
                print(f"[p2_03] {gene} {anc} pheno{phenocode}: n_snps={res['n_snps']} "
                      f"(eqtl={len(eqtl)}, gwas={len(gwas)}) PP.H4={res['PP.H4']:.3f}"
                      if res["n_snps"] else
                      f"[p2_03] {gene} {anc} pheno{phenocode}: n_snps=0 (eqtl={len(eqtl)}, gwas={len(gwas)})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"\n[p2_03] {len(df)} locus x ancestry x phenotype results -> {OUT}")


if __name__ == "__main__":
    main()
