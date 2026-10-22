"""
Phase 2 -- Step 02: fetch Pan-UKB per-ancestry GWAS summary stats for the
+/-500kb window around each pilot locus's TSS.

Pan-UKB sumstats are GRCh37 (hg19); MAGE/1000G (and p2_01's eQTL output) are
GRCh38 (hg38). We liftover the hg38 TSS+/-500kb window to hg19 to query Pan-UKB
(scripts/lib_remote_panukb.py), then liftover each returned SNP's hg19 position
back to hg38 to produce a `variant_kgpID` ("chrom:pos:ref:alt", chrom as a bare
number, no 'chr' prefix) that can be merged against p2_01's eQTL output.

Manually matched candidate-locus -> Pan-UKB phenotype(s) (anthropometric, the
best AFR-QC-passing matches to each locus's GWAS Catalog trait overlap):
  AMZ1, PHYKPL, CTD-2054N24.2, FNBP1P1, EPB41L2, LRRC61, ANKH -> height (50)
  NR2F6                                                        -> height (50), body fat % (23099)
  CDK10, ZNF341                                                -> waist circumference (48)
                                                                   (hip circumference, phenocode 49,
                                                                   fails AFR QC -> use waist as the
                                                                   adiposity-trait proxy)
  PMS2P5                                                       -> BMI (21001), body fat % (23099)

Output: results/phase2/p2_panukb/{gene}_{ancestry}_{phenocode}.tsv
  columns: variant_kgpID, chr_hg19, pos_hg19, pos_hg38, ref, alt, beta, se, pval, af
Checkpointed per (gene, ancestry, phenocode) file.
"""

import sys
from pathlib import Path

import pandas as pd
from pyliftover import LiftOver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_remote_panukb import RemotePanUKB
from p2_01_regional_eqtl_nominal import LOCI, WINDOW

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "phase2" / "p2_panukb"

# gene -> list of (phenocode, filename)
PHENO_FILES = {
    "50":    "continuous-50-both_sexes-irnt.tsv.bgz",     # Standing height
    "21001": "continuous-21001-both_sexes-irnt.tsv.bgz",  # BMI
    "23099": "continuous-23099-both_sexes-irnt.tsv.bgz",  # Body fat percentage
    "48":    "continuous-48-both_sexes-irnt.tsv.bgz",     # Waist circumference
}

LOCUS_PHENOS = {
    "AMZ1":          ["50"],
    "PHYKPL":        ["50"],
    "NR2F6":         ["50", "23099"],
    "CDK10":         ["48"],
    "CTD-2054N24.2": ["50"],
    "FNBP1P1":       ["50"],
    "EPB41L2":       ["50"],
    "ZNF341":        ["48"],
    "LRRC61":        ["50"],
    "PMS2P5":        ["21001", "23099"],
    "ANKH":          ["50"],
}

# Pan-UKB ancestry groups available per phenotype (all 4 used here have AFR QC pass)
ANCESTRIES = ["EUR", "AFR"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lo_38to19 = LiftOver("hg38", "hg19")
    lo_19to38 = LiftOver("hg19", "hg38")

    for gene, ensg, chrom, tss, lead_vid in LOCI:
        chrom_num = chrom.replace("chr", "")
        hg38_beg, hg38_end = max(1, tss - WINDOW), tss + WINDOW

        c1 = lo_38to19.convert_coordinate(chrom, hg38_beg)
        c2 = lo_38to19.convert_coordinate(chrom, hg38_end)
        if not c1 or not c2:
            print(f"[p2_02] {gene}: liftover of window bounds failed, skip")
            continue
        hg19_chrom = c1[0][0].replace("chr", "")
        hg19_positions = sorted([c1[0][1], c2[0][1]])
        hg19_beg, hg19_end = hg19_positions[0], hg19_positions[1]
        print(f"[p2_02] {gene}: hg38 {chrom}:{hg38_beg}-{hg38_end} "
              f"-> hg19 {hg19_chrom}:{hg19_beg}-{hg19_end}")

        for phenocode in LOCUS_PHENOS[gene]:
            if all((OUT_DIR / f"{gene}_{anc}_{phenocode}.tsv").exists() for anc in ANCESTRIES):
                print(f"  pheno {phenocode}: all ancestry outputs exist, skip")
                continue
            pu = RemotePanUKB(PHENO_FILES[phenocode])
            try:
                df = pu.query_region(hg19_chrom, hg19_beg, hg19_end)
            except Exception as e:
                print(f"  ERROR querying pheno {phenocode}: {e}")
                continue
            print(f"  pheno {phenocode}: {len(df)} variants in hg19 window")
            if len(df) == 0:
                continue

            # liftover hg19 -> hg38 once per row
            pos38 = []
            for p in df["pos"].astype(int):
                c = lo_19to38.convert_coordinate(f"chr{hg19_chrom}", p)
                pos38.append(c[0][1] if c else None)
            df = df.copy()
            df["pos_hg38"] = pos38
            df = df.dropna(subset=["pos_hg38"])
            df["pos_hg38"] = df["pos_hg38"].astype(int)
            df["variant_kgpID"] = (chrom_num + ":" + df["pos_hg38"].astype(str)
                                    + ":" + df["ref"] + ":" + df["alt"])

            for anc in ANCESTRIES:
                out_f = OUT_DIR / f"{gene}_{anc}_{phenocode}.tsv"
                if out_f.exists():
                    print(f"    {gene} {anc} {phenocode}: exists, skip")
                    continue
                bcol, scol, pcol, acol = f"beta_{anc}", f"se_{anc}", f"neglog10_pval_{anc}", f"af_{anc}"
                sub = df[["variant_kgpID", "chr", "pos", "pos_hg38", "ref", "alt",
                          bcol, scol, pcol, acol]].copy()
                sub = sub.rename(columns={"chr": "chr_hg19", "pos": "pos_hg19",
                                           bcol: "beta", scol: "se",
                                           pcol: "neglog10_pval", acol: "af"})
                for c in ["beta", "se", "neglog10_pval", "af"]:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")
                sub = sub.dropna(subset=["beta", "se"])
                sub.to_csv(out_f, sep="\t", index=False)
                print(f"    {gene} {anc} {phenocode}: {len(sub)} usable variants -> {out_f.name}")

    print("\n[p2_02] done.")


if __name__ == "__main__":
    main()
