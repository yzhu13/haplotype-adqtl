"""
Phase 0 — Step 08a: Build ancestry-specific haplotype BACKGROUNDS for the ADRS
test (sub-gate B). API-FREE — can run while AlphaGenome is busy.

For each adQTL / control lead variant, fetch the LOCAL common SNPs (+/- BG_HALF)
from the 1000G phased VCF and compute, per site, the EUR-major and AFR-major
allele — using the SAME MAGE EUR/AFR individuals whose betas we measured (p0_06).
Sites where EUR-major != AFR-major are the ancestry-differentiated background that
could change the focal variant's regulatory consequence (the cis-GxG substrate).

We store a COMPACT substitution table (not full sequences). p0_08b reconstructs
the 4 sequences (EUR/AFR x ref/alt) on hg38 and runs AlphaGenome.

Output: data/processed/adrs_backgrounds.jsonl
  one JSON line per variant:
    {variant_kgpID, ensemblID, chrom, window_start, var_rel, tss_rel,
     focal_ref, focal_alt, n_common, n_diff,
     subs: [[rel_pos, eur_allele, afr_allele], ...]}   # only sites with a non-ref allele to set
"""

import argparse
import gzip
import json
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
META = MAGE / "sample_library_info/sample.metadata.MAGE.v1.0.txt"
OUT = PROC / "adrs_backgrounds.jsonl"

VCF_TMPL = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20201028_3202_phased/"
            "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz")

BG_HALF = 25_000      # local background window half-width (bp)
MIN_MAF = 0.05        # only flip "common" background SNPs


def alt_freq(gts, cols):
    """alt-allele frequency over sample columns `cols` (phased 'a|b')."""
    a = n = 0
    for c in cols:
        gt = gts[c].split(":", 1)[0]
        for p in gt.replace("/", "|").split("|"):
            if p == ".":
                continue
            a += (p != "0")
            n += 1
    return (a / n) if n else None


def build_one(vcf, chrom, r, eur_cols, afr_cols):
    fpos = int(r.variantPosition)
    ws = int(r.window_start)
    need = max(max(eur_cols), max(afr_cols)) + 1  # min GT columns for valid line
    subs = []
    n_common = n_diff = 0
    for ln in vcf.query(chrom, fpos - BG_HALF, fpos + BG_HALF):
        f = ln.split("\t")
        if len(f) < 9 + need:                # truncated line at a chunk boundary
            continue
        pos, ref, alt = int(f[1]), f[3], f[4]
        if pos == fpos:                      # skip the focal variant itself
            continue
        if len(ref) != 1 or len(alt) != 1 or "," in alt:   # biallelic SNV only
            continue
        gts = f[9:]
        fe = alt_freq(gts, eur_cols)
        fa = alt_freq(gts, afr_cols)
        if fe is None or fa is None:
            continue
        if min(fe, 1 - fe) < MIN_MAF and min(fa, 1 - fa) < MIN_MAF:
            continue                          # not common in either ancestry
        n_common += 1
        eur_allele = alt if fe >= 0.5 else ref
        afr_allele = alt if fa >= 0.5 else ref
        # only store if at least one ancestry-major differs from hg38 ref
        if eur_allele == ref and afr_allele == ref:
            continue
        if eur_allele != afr_allele:
            n_diff += 1
        subs.append([pos - 1 - ws, eur_allele, afr_allele])  # rel_pos 0-based in window
    return {"variant_kgpID": r.variant_kgpID, "ensemblID": r.ensemblID,
            "chrom": chrom, "window_start": ws,
            "var_rel": int(r.var_rel), "tss_rel": int(r.tss_rel),
            "focal_ref": r.variantRef, "focal_alt": r.variantAlt,
            "n_common": n_common, "n_diff": n_diff, "subs": subs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["adqtl", "control", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META, sep="\t")
    eur = set(meta.loc[meta.continentalGroup == "EUR", "sample_kgpID"])
    afr = set(meta.loc[meta.continentalGroup == "AFR", "sample_kgpID"])

    cols = ["variant_kgpID", "ensemblID", "variantChrom", "variantPosition",
            "variantRef", "variantAlt", "window_start", "var_rel", "tss_rel"]
    frames = []
    if args.set in ("adqtl", "both"):
        frames.append(pd.read_csv(RES / "p0_adqtl_set.tsv", sep="\t"))
    if args.set in ("control", "both"):
        frames.append(pd.read_csv(RES / "p0_control_set.tsv", sep="\t"))
    sets = pd.concat(frames, ignore_index=True)
    # need calibration coords (window_start etc.) — merge from calibration set
    cal = pd.read_csv(RES / "p0_calibration_set.tsv", sep="\t")
    sets = sets[["variant_kgpID", "ensemblID"]].merge(
        cal[cols], on=["variant_kgpID", "ensemblID"], how="left").dropna(subset=["window_start"])
    sets = sets.drop_duplicates("variant_kgpID")
    print(f"[p0_08a] variants to build: {len(sets)}")

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {json.loads(l)["variant_kgpID"] for l in f if l.strip()}
        print(f"  resuming: {len(done)} done")
    sets = sets[~sets.variant_kgpID.isin(done)]

    lock = threading.Lock()
    fout = open(OUT, "a")
    n = 0
    for chrom, grp in sets.groupby("variantChrom"):
        print(f"[{chrom}] {len(grp)} variants ...", flush=True)
        vcf = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom))
        vsamp = {s: i for i, s in enumerate(vcf.samples())}
        eur_cols = [vsamp[s] for s in eur if s in vsamp]
        afr_cols = [vsamp[s] for s in afr if s in vsamp]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_one, vcf, chrom, r, eur_cols, afr_cols): r.variant_kgpID
                    for _, r in grp.iterrows()}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"  ERROR {futs[fut]}: {e}")
                    continue
                with lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    n += 1
                    if n % 25 == 0:
                        print(f"  built {n} (last {rec['variant_kgpID']}, "
                              f"n_common={rec['n_common']} n_diff={rec['n_diff']})", flush=True)
    fout.close()
    print(f"[p0_08a] done. {n} backgrounds -> {OUT}")


if __name__ == "__main__":
    main()
