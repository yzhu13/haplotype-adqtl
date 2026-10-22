"""
Phase 0 — Step 08b: Compute ADRS (Ancestry-Differential Regulatory Score) with
AlphaGenome and test whether it predicts the OBSERVED ancestry-differential effect
(ObsDelta_EUR_AFR from p0_06). This is the REAL sub-gate B test. NEEDS THE API.

For each adQTL / control variant we build 4 sequences from hg38 + the cached
ancestry backgrounds (p0_08a):
    eur_ref = EUR-background, focal=ref      eur_alt = EUR-background, focal=alt
    afr_ref = AFR-background, focal=ref      afr_alt = AFR-background, focal=alt
predict all 4 (LCL GM12878 tracks), read RNA at the gene TSS and ATAC at the
variant, then:
    PredD_EUR = log2(rna(eur_alt)) - log2(rna(eur_ref))      # variant effect on EUR bg
    PredD_AFR = log2(rna(afr_alt)) - log2(rna(afr_ref))      # variant effect on AFR bg
    ADRS      = PredD_EUR - PredD_AFR                          # predicted ancestry diff

sub-gate B passes if ADRS correlates with ObsDelta_EUR_AFR (and, in p0_09, does so
ABOVE allele-frequency/Fst — the cis-GxG residual).

Politeness + resilience: waits for Project-1 AlphaGenome jobs (shared API key),
watchdog on gRPC hang, checkpoint resume. Run via p0_08b launched in background.

Input : data/processed/adrs_backgrounds.jsonl (p0_08a) + hg38
Output: results/phase0/p0_adrs_predictions.tsv  (checkpointed)
"""

import argparse
import json
import os
import sys
import time
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyfaidx import Fasta

from alphagenome.models import dna_client, dna_output
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV, HG38_FASTA
HG38 = HG38_FASTA
ENV = ALPHAGENOME_ENV

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
RES = BASE / "results" / "phase0"
BG = PROC / "adrs_backgrounds.jsonl"
OUT = RES / "p0_adrs_predictions.tsv"
GENES = BASE / "data/raw/MAGE/QTL_results/eQTL_results/expression_filteredGenes.MAGE.v1.0.txt.gz"
EXONS = PROC / "gene_exons.json"

CONTEXT = 1_048_576
RNA_ONT = ["EFO:0002784", "EFO:0000572"]
ATAC_ONT = ["EFO:0002784"]
TSS_HALF = 2000
VAR_HALF = 500
EPS = 1e-3

WATCHDOG = 600
_last = time.time(); _lock = threading.Lock()


def _touch():
    global _last
    with _lock:
        _last = time.time()


def _watch():
    while True:
        time.sleep(30)
        with _lock:
            idle = time.time() - _last
        if idle > WATCHDOG:
            print(f"\n[WATCHDOG] idle {idle:.0f}s — exit for restart.", flush=True)
            os._exit(99)


def build_seq(fa, chrom, ws, subs, allele_key, var_rel, focal_base):
    """Reference window with ancestry-major background applied + focal base set."""
    seq = list(fa[chrom][ws:ws + CONTEXT])  # uppercase, as_raw
    if len(seq) < CONTEXT:
        seq += ["N"] * (CONTEXT - len(seq))
    j = 1 if allele_key == "eur" else 2  # sub = [rel_pos, eur_allele, afr_allele]
    for s in subs:
        rp = s[0]
        if 0 <= rp < CONTEXT:
            seq[rp] = s[j]
    if 0 <= var_rel < CONTEXT:
        seq[var_rel] = focal_base
    return "".join(seq)


def readout(out, center, half):
    """sum signal in [center-half, center+half] across tracks -> scalar."""
    v = out.values
    lo, hi = max(0, center - half), min(v.shape[0], center + half)
    return float(v[lo:hi, :].sum())


def readout_region(out, lo, hi):
    """sum signal over [lo, hi) across tracks -> scalar (gene-body RNA readout;
    validated best client proxy for the server GeneMaskLFCScorer, p0_08d)."""
    v = out.values
    lo, hi = max(0, lo), min(v.shape[0], hi)
    return float(v[lo:hi, :].sum())


def load_gene_spans():
    df = pd.read_csv(GENES, sep="\t")
    return {r.ensemblID: (int(r.chromStart), int(r.chromEnd)) for r in df.itertuples()}


def load_exons():
    import json
    return json.loads(EXONS.read_text(encoding="utf-8")) if EXONS.exists() else {}


def exon_mask(exon_list, ws):
    """boolean (CONTEXT,) marking exonic bins; exon coords 1-based genomic."""
    m = np.zeros(CONTEXT, dtype=bool)
    for s, e in exon_list:
        lo, hi = max(0, s - 1 - ws), min(CONTEXT, e - ws)
        if hi > lo:
            m[lo:hi] = True
    return m


def readout_mask(out, mask):
    """sum RNA over exonic bins across tracks -> scalar."""
    return float(out.values[mask, :].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    load_dotenv(ENV)
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key:
        sys.exit("no ALPHAGENOME_API_KEY")

    recs = [json.loads(l) for l in open(BG) if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT, sep="\t")["variant_kgpID"])
    recs = [r for r in recs if r["variant_kgpID"] not in done]
    print(f"[p0_08b] backgrounds to score: {len(recs)} (resume skip {len(done)})")
    if not recs:
        return

    fa = Fasta(str(HG38), as_raw=True, sequence_always_upper=True)
    spans = load_gene_spans()
    exons = load_exons()
    client = dna_client.create(key, timeout=180)
    first = not OUT.exists()
    n = 0
    threading.Thread(target=_watch, daemon=True).start()
    _touch()

    for r in recs:
        try:
            ws, vr = r["window_start"], r["var_rel"]
            chrom = r["chrom"]
            seqs = [
                build_seq(fa, chrom, ws, r["subs"], "eur", vr, r["focal_ref"]),  # eur_ref
                build_seq(fa, chrom, ws, r["subs"], "eur", vr, r["focal_alt"]),  # eur_alt
                build_seq(fa, chrom, ws, r["subs"], "afr", vr, r["focal_ref"]),  # afr_ref
                build_seq(fa, chrom, ws, r["subs"], "afr", vr, r["focal_alt"]),  # afr_alt
            ]
            iv = genome.Interval(chrom, ws, ws + CONTEXT)
            outs = client.predict_sequences(
                seqs,
                requested_outputs=[dna_output.OutputType.RNA_SEQ, dna_output.OutputType.ATAC],
                ontology_terms=RNA_ONT + ATAC_ONT,
                intervals=[iv, iv, iv, iv],
                progress_bar=False, max_workers=args.max_workers,
            )
            # RNA: EXON-MASK sum (best validated client proxy for the server
            # GeneMaskLFCScorer, p0_08f). Fall back to gene-body, then TSS window.
            gid = r["ensemblID"]
            gid0 = gid.split(".")[0]
            if gid0 in exons:
                m = exon_mask(exons[gid0], ws)
                rna = [np.log2(readout_mask(o.rna_seq, m) + EPS) for o in outs]
            elif gid in spans:
                gs, ge = spans[gid]
                rna = [np.log2(readout_region(o.rna_seq, gs - ws, ge - ws) + EPS) for o in outs]
            else:
                rna = [np.log2(readout(o.rna_seq, r["tss_rel"], TSS_HALF) + EPS) for o in outs]
            atac = [np.log2(readout(o.atac, vr, VAR_HALF) + EPS) for o in outs]
            predD_eur_rna = rna[1] - rna[0]
            predD_afr_rna = rna[3] - rna[2]
            predD_eur_at = atac[1] - atac[0]
            predD_afr_at = atac[3] - atac[2]
            rec = {
                "variant_kgpID": r["variant_kgpID"], "ensemblID": r["ensemblID"],
                "n_diff": r["n_diff"], "n_common": r["n_common"],
                "predD_EUR_rna": predD_eur_rna, "predD_AFR_rna": predD_afr_rna,
                "ADRS_rna": predD_eur_rna - predD_afr_rna,
                "predD_EUR_atac": predD_eur_at, "predD_AFR_atac": predD_afr_at,
                "ADRS_atac": predD_eur_at - predD_afr_at,
            }
            pd.DataFrame([rec]).to_csv(OUT, sep="\t", mode="w" if first else "a",
                                       header=first, index=False)
            first = False
            n += 1
            _touch()
            print(f"  [{n}/{len(recs)}] {r['variant_kgpID']} n_diff={r['n_diff']} "
                  f"ADRS_rna={rec['ADRS_rna']:+.4f} ADRS_atac={rec['ADRS_atac']:+.4f}", flush=True)
        except KeyboardInterrupt:
            print("interrupted — checkpoint saved."); break
        except Exception as e:
            print(f"  ERROR {r['variant_kgpID']}: {e}")
            traceback.print_exc()
            continue

    print(f"[p0_08b] done. {n} scored -> {OUT}")


if __name__ == "__main__":
    main()
