"""
Phase 0 — Step 03: Run AlphaGenome on the eGene lead-variant calibration set and
score the PREDICTED cis-regulatory effect, to be compared against MAGE's observed
pooled log2_aFC in p0_04.

Engine: AlphaGenome `score_variant` with the RECOMMENDED variant scorers
(GeneMaskLFCScorer for RNA_SEQ — gene-aware & strand-aware; CenterMaskScorer for
ATAC). These are the model authors' recommended aggregations, far more robust
than a hand-rolled TSS window.
Cell type: GM12878 LCL = EFO:0002784 (matched to MAGE LCLs); also GTEx
EBV-transformed lymphocytes EFO:0000572 for RNA.

Readouts per variant (from the tidy score table, target gene_id x LCL ontology):
  - pred_rna_lfc : gene-level RNA-seq log2 fold-change (raw_score), averaged over
        the LCL RNA tracks. This is the matched readout for an eQTL effect (aFC).
  - pred_atac_lfc: ATAC center-mask log2 change at the variant in GM12878 —
        chromatin accessibility, the regime where AlphaGenome is strongest
       . Secondary readout.

First pass = SNVs only (indels deferred). Stratified --sample-n keeps the initial
go/no-go run cheap.

Uses a watchdog + checkpoint-resume pattern (gRPC calls can hang).

Input : results/phase0/p0_calibration_set.tsv  (from p0_02)
Output: results/phase0/p0_alphagenome_predictions.tsv  (checkpointed, resumable)

Usage:
  python p0_03_run_alphagenome_calibration.py --sample-n 300      # first gate
  python p0_03_run_alphagenome_calibration.py                     # full 7,451
"""

import argparse
import os
import sys
import time
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alphagenome.models import dna_client, variant_scorers as vs
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV as P1_ENV

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
CAL = RES / "p0_calibration_set.tsv"
OUT = RES / "p0_alphagenome_predictions.tsv"

LCL_RNA_ONTOLOGY = {"EFO:0002784", "EFO:0000572"}  # GM12878 + GTEx LCL
LCL_ATAC_ONTOLOGY = {"EFO:0002784"}                # GM12878 ATAC
RNA_SCORER = vs.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]   # GeneMaskLFCScorer
ATAC_SCORER = vs.RECOMMENDED_VARIANT_SCORERS["ATAC"]     # CenterMaskScorer w=501

# ── Watchdog (gRPC can hang silently; outer restart loop resumes from checkpoint) ──
WATCHDOG_TIMEOUT = 600
_last = time.time()
_lock = threading.Lock()


def _touch():
    global _last
    with _lock:
        _last = time.time()


def _watchdog():
    while True:
        time.sleep(30)
        with _lock:
            idle = time.time() - _last
        if idle > WATCHDOG_TIMEOUT:
            print(f"\n[WATCHDOG] no progress {idle:.0f}s — exiting for restart.", flush=True)
            os._exit(99)


def stratified_sample(df, n, seed=0):
    """Sample n rows stratified across |log2_aFC| deciles (keeps weak & strong)."""
    if n is None or n >= len(df):
        return df
    q = pd.qcut(df["log2_aFC"].abs(), 10, labels=False, duplicates="drop")
    per = max(1, n // (q.nunique()))
    out = (df.assign(_q=q).groupby("_q", group_keys=False)
             .apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
             .drop(columns="_q"))
    return out.sample(min(n, len(out)), random_state=seed)


def extract_lcl_scores(tidy, ensembl_id):
    """From the tidy score table, pull the target gene's LCL RNA LFC and the LCL
    ATAC center-mask change. Returns (rna_lfc, atac_lfc, n_rna_tracks, n_atac_tracks).
    gene_id in tidy is unversioned; strip the .N version from ensembl_id."""
    gid = ensembl_id.split(".")[0]
    rna = tidy[(tidy.output_type == "RNA_SEQ")
               & (tidy.gene_id == gid)
               & (tidy.ontology_curie.isin(LCL_RNA_ONTOLOGY))]
    atac = tidy[(tidy.output_type == "ATAC")
                & (tidy.ontology_curie.isin(LCL_ATAC_ONTOLOGY))]
    rna_lfc = float(rna.raw_score.mean()) if len(rna) else float("nan")
    atac_lfc = float(atac.raw_score.mean()) if len(atac) else float("nan")
    return rna_lfc, atac_lfc, len(rna), len(atac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=None,
                    help="Stratified subsample size (default: all SNVs).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_dotenv(P1_ENV)
    api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        sys.exit(f"ERROR: ALPHAGENOME_API_KEY not found (looked in {P1_ENV})")

    cal = pd.read_csv(CAL, sep="\t")
    snv = cal[(cal.variantRef.str.len() == 1) & (cal.variantAlt.str.len() == 1)].copy()
    print(f"[p0_03] calibration SNVs: {len(snv):,} / {len(cal):,} total")
    todo = stratified_sample(snv, args.sample_n, args.seed)

    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT, sep="\t")
        done = set(prev["variant_kgpID"])
        print(f"  resuming: {len(done):,} already scored")
    todo = todo[~todo["variant_kgpID"].isin(done)]
    print(f"  to score: {len(todo):,}")
    if len(todo) == 0:
        print("  nothing to do.")
        return

    client = dna_client.create(api_key, timeout=180)
    first = not OUT.exists()
    t0 = time.time()
    n = 0
    threading.Thread(target=_watchdog, daemon=True).start()
    _touch()

    for _, r in todo.iterrows():
        try:
            iv = genome.Interval(r.variantChrom, int(r.window_start), int(r.window_end))
            v = genome.Variant(r.variantChrom, int(r.variantPosition),
                               r.variantRef, r.variantAlt)
            scores = client.score_variant(iv, v, variant_scorers=[RNA_SCORER, ATAC_SCORER])
            tidy = vs.tidy_scores(scores, match_gene_strand=True)
            rna_lfc, atac_lfc, n_rna, n_atac = extract_lcl_scores(tidy, r.ensemblID)

            rec = {"variant_kgpID": r.variant_kgpID, "ensemblID": r.ensemblID,
                   "geneSymbol": r.geneSymbol, "log2_aFC": r.log2_aFC,
                   "variantPIP": r.variantPIP, "var_tss_dist": r.var_tss_dist,
                   "pred_rna_lfc": rna_lfc, "pred_atac_lfc": atac_lfc,
                   "n_rna_tracks": n_rna, "n_atac_tracks": n_atac}

            pd.DataFrame([rec]).to_csv(OUT, sep="\t", mode="w" if first else "a",
                                       header=first, index=False)
            first = False
            n += 1
            _touch()
            rate = n / (time.time() - t0)
            eta = (len(todo) - n) / rate / 60 if rate > 0 else float("inf")
            print(f"  [{n}/{len(todo)}] {r.geneSymbol:<14} "
                  f"obs={r.log2_aFC:+.2f} predRNA={rna_lfc:+.4f} "
                  f"predATAC={atac_lfc:+.4f} (nR={n_rna}) "
                  f"rate={rate:.2f}/s ETA={eta:.0f}m", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted — checkpoint saved.")
            break
        except Exception as e:
            print(f"  ERROR {r.variant_kgpID}: {e}")
            traceback.print_exc()
            continue

    print(f"\n[p0_03] done. {n} new predictions -> {OUT}")


if __name__ == "__main__":
    main()
