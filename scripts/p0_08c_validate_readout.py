"""
Phase 0 — Step 08c: VALIDATE the custom-sequence RNA readout before re-running ADRS.

sub-gate B came back NO-GO, but the control showed the manual +/-2kb TSS-window
readout on predict_sequence recovers only r=0.07 of the pooled aFC (vs r=0.48 for
the gene-aware scorer in sub-gate A) — i.e. the readout, not the biology, failed.

Here we test whether `predict_sequence` + a GENE-BODY readout (sum RNA over the
gene's [start,end] from MAGE expression_filteredGenes, across LCL tracks) recovers
the aFC correlation. We compare TSS-window vs gene-body on the same predictions.

If gene-body recovers r>=~0.3, we adopt it for the ADRS re-run (p0_08b).

API-free? NO — needs predict_sequence. Small n, wrap with timeout.

Output: results/phase0/p0_readout_validation.tsv + printed correlations.
"""
import argparse, gzip, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from dotenv import load_dotenv
from pyfaidx import Fasta
from alphagenome.models import dna_client, dna_output
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV, HG38_FASTA
HG38 = HG38_FASTA; ENV = ALPHAGENOME_ENV
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
GENES = BASE / "data/raw/MAGE/QTL_results/eQTL_results/expression_filteredGenes.MAGE.v1.0.txt.gz"
CAL = RES / "p0_calibration_set.tsv"
OUT = RES / "p0_readout_validation.tsv"
CONTEXT = 1_048_576; RNA_ONT = ["EFO:0002784", "EFO:0000572"]; EPS = 1e-3; TSS_HALF = 2000


def gene_spans():
    df = pd.read_csv(GENES, sep="\t")
    return {r.ensemblID: (int(r.chromStart), int(r.chromEnd)) for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
    load_dotenv(ENV); key = os.environ["ALPHAGENOME_API_KEY"]
    spans = gene_spans()
    cal = pd.read_csv(CAL, sep="\t")
    cal = cal[(cal.variantRef.str.len()==1)&(cal.variantAlt.str.len()==1)]
    # resume
    done=set()
    if OUT.exists(): done=set(pd.read_csv(OUT,sep="\t").variant_kgpID)
    cal = cal[~cal.variant_kgpID.isin(done)]
    # stratified by |aFC|
    cal = cal.assign(_q=pd.qcut(cal.log2_aFC.abs(),8,labels=False,duplicates="drop"))
    todo = cal.groupby("_q",group_keys=False).apply(lambda g:g.sample(min(3,len(g)),random_state=1)).head(a.n)
    fa = Fasta(str(HG38), as_raw=True, sequence_always_upper=True)
    client = dna_client.create(key, timeout=180)
    first = not OUT.exists()
    for _,r in todo.iterrows():
        if r.ensemblID not in spans: continue
        ws=int(r.window_start); ch=r.variantChrom; vr=int(r.var_rel)
        seq=list(fa[ch][ws:ws+CONTEXT])
        if len(seq)<CONTEXT: seq+=["N"]*(CONTEXT-len(seq))
        ref="".join(seq); seq[vr]=r.variantAlt; alt="".join(seq)
        iv=genome.Interval(ch,ws,ws+CONTEXT)
        o=client.predict_sequences([ref,alt],requested_outputs=[dna_output.OutputType.RNA_SEQ],
                                   ontology_terms=RNA_ONT,intervals=[iv,iv],progress_bar=False,max_workers=2)
        gs,ge=spans[r.ensemblID]; gsr=max(0,gs-ws); ger=min(CONTEXT,ge-ws)
        def body(out): return float(out.rna_seq.values[gsr:ger,:].sum())
        def tss(out):
            lo,hi=max(0,int(r.tss_rel)-TSS_HALF),min(CONTEXT,int(r.tss_rel)+TSS_HALF)
            return float(out.rna_seq.values[lo:hi,:].sum())
        rec={"variant_kgpID":r.variant_kgpID,"ensemblID":r.ensemblID,"log2_aFC":r.log2_aFC,
             "predD_body":np.log2((body(o[1])+EPS)/(body(o[0])+EPS)),
             "predD_tss":np.log2((tss(o[1])+EPS)/(tss(o[0])+EPS))}
        pd.DataFrame([rec]).to_csv(OUT,sep="\t",mode="w" if first else "a",header=first,index=False)
        first=False
        print(f"  {r.geneSymbol:<12} aFC={r.log2_aFC:+.2f} body={rec['predD_body']:+.4f} tss={rec['predD_tss']:+.4f}",flush=True)
    # report
    d=pd.read_csv(OUT,sep="\t")
    for col in ["predD_body","predD_tss"]:
        dd=d[(d[col]!=0)&np.isfinite(d[col])]
        r,p=stats.spearmanr(dd[col],dd.log2_aFC)
        print(f"\n{col} vs log2_aFC: Spearman r={r:+.3f} p={p:.2e} sign={(np.sign(dd[col])==np.sign(dd.log2_aFC)).mean():.0%} n={len(dd)}")


if __name__=="__main__": main()
