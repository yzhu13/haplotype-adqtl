"""
Phase 0 — Step 05: Fetch genotype dosages for the eGene lead variants from the
1000G high-coverage 3202-sample PHASED VCFs (EBI), for the 731 MAGE samples.

API-free (does NOT touch AlphaGenome).

Uses the pure-Python remote tabix slicer (lib_remote_vcf) — only the needed BGZF
blocks are HTTP-range-fetched, no whole-chromosome download.

Match: VCF ID column == MAGE variant_kgpID ("chr:pos:ref:alt"), guaranteeing
correct REF/ALT orientation. Dosage = count of ALT alleles (phased a|b).

Input : results/phase0/p0_calibration_set.tsv (lead variants)
        data/raw/MAGE/.../sample.metadata + expr bed (sample order)
Output: data/processed/lead_variant_dosages.tsv  (variant_kgpID x 731 samples)
        checkpointed per variant; resumable.

Usage:
  python p0_05_fetch_genotypes.py --chrom chr22      # one chromosome (test)
  python p0_05_fetch_genotypes.py                    # all autosomes
"""

import argparse
import gzip
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_remote_vcf import RemoteTabixVCF

BASE = Path(__file__).resolve().parents[1]
MAGE = BASE / "data" / "raw" / "MAGE"
RES = BASE / "results" / "phase0"
PROC = BASE / "data" / "processed"
CAL = RES / "p0_calibration_set.tsv"
EXPR_BED = (MAGE / "QTL_results/eQTL_results/expression_quants/"
            "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz")
OUT = PROC / "lead_variant_dosages.tsv"

VCF_BASE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20201028_3202_phased/")
VCF_TMPL = VCF_BASE + "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"


def mage_sample_order():
    """Sample IDs in expression-matrix column order (kgpIDs)."""
    with gzip.open(EXPR_BED, "rt") as f:
        cols = next(f).rstrip("\n").split("\t")
    return cols[4:]


def gt_to_dosage(gt_field):
    """'a|b[:...]' -> alt-allele count (0/1/2), or None if missing."""
    gt = gt_field.split(":", 1)[0]
    sep = "|" if "|" in gt else "/"
    parts = gt.split(sep)
    try:
        return sum(1 for p in parts if p not in (".", "0"))
    except Exception:
        return None


def fetch_one(vcf, chrom, r, keep, col_pos):
    """Query one variant; return its dosage row dict or None if not found."""
    vid = r.variant_kgpID
    pos = int(r.variantPosition)
    rec = None
    for ln in vcf.query(chrom, pos, pos):
        f = ln.split("\t")
        if f[2] == vid:
            rec = f
            break
    if rec is None:
        return None
    gts = rec[9:]
    dos = {keep[i]: gt_to_dosage(gts[col_pos[i]]) for i in range(len(keep))}
    return {"variant_kgpID": vid, "ensemblID": r.ensemblID,
            "geneSymbol": r.geneSymbol, **dos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default=None, help="Restrict to one chrom (e.g. chr22).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel query threads.")
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)

    samples = mage_sample_order()
    print(f"[p0_05] MAGE samples: {len(samples)}")

    cal = pd.read_csv(CAL, sep="\t")
    cal = cal[(cal.variantRef.str.len() == 1) & (cal.variantAlt.str.len() == 1)]  # SNVs
    if args.chrom:
        cal = cal[cal.variantChrom == args.chrom]
    print(f"[p0_05] lead SNVs to fetch: {len(cal):,}")

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, sep="\t", usecols=["variant_kgpID"])["variant_kgpID"])
        print(f"  resuming: {len(done):,} already fetched")

    first = not OUT.exists()
    n = miss = 0
    for chrom, grp in cal.groupby("variantChrom"):
        todo = grp[~grp.variant_kgpID.isin(done)]
        if len(todo) == 0:
            continue
        print(f"[{chrom}] opening VCF ({len(todo)} variants) ...")
        try:
            vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
        except Exception as e:
            print(f"  ERROR opening {chrom}: {e}")
            continue
        vcf_samples = vcf.samples()
        sidx = {s: i for i, s in enumerate(vcf_samples)}
        keep = [s for s in samples if s in sidx]
        if len(keep) != len(samples):
            print(f"  WARN: {len(samples)-len(keep)} MAGE samples missing from VCF")
        col_pos = [sidx[s] for s in keep]

        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, vcf, chrom, r, keep, col_pos): r.variant_kgpID
                    for _, r in todo.iterrows()}
            for fut in as_completed(futs):
                vid = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  ERROR {vid}: {e}")
                    continue
                if row is None:
                    miss += 1
                    continue
                with lock:
                    pd.DataFrame([row]).to_csv(OUT, sep="\t", mode="w" if first else "a",
                                               header=first, index=False)
                    first = False
                    n += 1
                    if n % 100 == 0:
                        print(f"  fetched {n} (missing {miss}) ... last {vid}", flush=True)

    print(f"\n[p0_05] done. {n} variants fetched, {miss} not found -> {OUT}")


if __name__ == "__main__":
    main()
