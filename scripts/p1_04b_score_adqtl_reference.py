"""
Phase 1 — Step 04b: Score adQTL + control variants with AlphaGenome's WORKING
server scorer (GeneMaskLFCScorer RNA, reference/hg38 background) and ask a sharp,
hypothesis-linked question:

  hg38 reference is European-biased. Does the reference-background prediction
  (pred_rna_lfc) track beta_EUR better than beta_AFR? If yes, the sequence-model
  prediction is itself ancestry-specific (matches the reference's ancestry) —
  indirect support that haplotype background shapes the predicted effect.

Also records |pred| for an adQTL-vs-control magnitude comparison.

NEEDS API (server scoring — this is the path that WORKS). Run under timeout loop.

Inputs: results/phase0/p0_adqtl_set.tsv, p0_control_set.tsv, p0_calibration_set.tsv,
  p0_per_ancestry_betas.tsv. Output: results/phase1/p1_adqtl_refscore.tsv (checkpoint)
"""
import os, sys, time, threading
from pathlib import Path
import numpy as np, pandas as pd
from dotenv import load_dotenv
from alphagenome.models import dna_client, variant_scorers as vs
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV, HG38_FASTA
ENV = ALPHAGENOME_ENV
BASE = Path(__file__).resolve().parents[1]
RES0 = BASE / "results" / "phase0"; RES = BASE / "results" / "phase1"
OUT = RES / "p1_adqtl_refscore.tsv"
LCL = {"EFO:0002784", "EFO:0000572"}
CONTEXT = 1_048_576


def main():
    RES.mkdir(parents=True, exist_ok=True)
    load_dotenv(ENV); key = os.environ["ALPHAGENOME_API_KEY"]
    a = pd.read_csv(RES0/"p0_adqtl_set.tsv", sep="\t"); a["set"]="adqtl"
    c = pd.read_csv(RES0/"p0_control_set.tsv", sep="\t"); c["set"]="control"
    sets = pd.concat([a,c], ignore_index=True)[["variant_kgpID","ensemblID","set"]]
    cal = pd.read_csv(RES0/"p0_calibration_set.tsv", sep="\t")
    bet = pd.read_csv(RES0/"p0_per_ancestry_betas.tsv", sep="\t")[
        ["variant_kgpID","ensemblID","beta_EUR","beta_AFR","ObsDelta_EUR_AFR"]]
    df = sets.merge(cal, on=["variant_kgpID","ensemblID"], how="inner").merge(
        bet, on=["variant_kgpID","ensemblID"], how="left")
    df = df[(df.variantRef.str.len()==1)&(df.variantAlt.str.len()==1)].drop_duplicates("variant_kgpID")

    done = set(pd.read_csv(OUT, sep="\t").variant_kgpID) if OUT.exists() else set()
    df = df[~df.variant_kgpID.isin(done)]
    print(f"[p1_04b] to score: {len(df)} (resume skip {len(done)})")
    if len(df)==0: return

    client = dna_client.create(key, timeout=180)
    scorer = vs.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]
    first = not OUT.exists()
    for _, r in df.iterrows():
        try:
            iv = genome.Interval(r.variantChrom, int(r.window_start), int(r.window_end))
            v = genome.Variant(r.variantChrom, int(r.variantPosition), r.variantRef, r.variantAlt)
            tidy = vs.tidy_scores(client.score_variant(iv, v, variant_scorers=[scorer]),
                                  match_gene_strand=True)
            gid = r.ensemblID.split(".")[0]
            sub = tidy[(tidy.output_type=="RNA_SEQ")&(tidy.gene_id==gid)&(tidy.ontology_curie.isin(LCL))]
            pred = float(sub.raw_score.mean()) if len(sub) else np.nan
            rec = {"variant_kgpID":r.variant_kgpID,"ensemblID":r.ensemblID,"set":r.set,
                   "pred_rna_lfc":pred,"beta_EUR":r.beta_EUR,"beta_AFR":r.beta_AFR,
                   "ObsDelta_EUR_AFR":r.ObsDelta_EUR_AFR,"log2_aFC":r.log2_aFC}
            pd.DataFrame([rec]).to_csv(OUT,sep="\t",mode="w" if first else "a",header=first,index=False)
            first=False
            print(f"  {r.geneSymbol:<12} [{r.set}] pred={pred:+.4f} bEUR={r.beta_EUR:+.2f} bAFR={r.beta_AFR:+.2f}",flush=True)
        except Exception as e:
            print(f"  ERROR {r.variant_kgpID}: {e}"); continue
    print("[p1_04b] done.")


if __name__ == "__main__":
    main()
