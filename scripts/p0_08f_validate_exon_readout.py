"""
Phase 0 — Step 08f: Validate the EXON-MASK readout before re-running ADRS.

Target to beat: on the adQTL/control variants, the SERVER scorer recovers pooled
aFC at r=0.34 (p0_08d diagnostic). Our gene-body readout got only r=0.045. Here we
test whether an exon-mask log2-sum readout (RNA summed over GENCODE exons of the
target gene) recovers ~r=0.3 on the SAME (unstratified) variant set.

NEEDS API. Run under bash timeout loop.

Output: results/phase0/p0_exon_readout_validation.tsv + correlation report.
"""
import argparse, json, os
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
EXONS = BASE / "data/processed/gene_exons.json"
GENES = BASE / "data/raw/MAGE/QTL_results/eQTL_results/expression_filteredGenes.MAGE.v1.0.txt.gz"
CAL = RES / "p0_calibration_set.tsv"
OUT = RES / "p0_exon_readout_validation.tsv"
CONTEXT = 1_048_576; LCL = ["EFO:0002784", "EFO:0000572"]; EPS = 1e-3; TSS_HALF = 2000


def exon_rel_positions(exons, ws):
    """boolean index array over [0,CONTEXT) marking exonic bins."""
    mask = np.zeros(CONTEXT, dtype=bool)
    for s, e in exons:
        lo, hi = max(0, s - 1 - ws), min(CONTEXT, e - ws)  # 1-based genomic -> 0-based rel
        if hi > lo:
            mask[lo:hi] = True
    return mask


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=70); a = ap.parse_args()
    load_dotenv(ENV); key = os.environ["ALPHAGENOME_API_KEY"]
    exons = {k: v for k, v in json.loads(EXONS.read_text()).items()}
    gspan = {r.ensemblID: (int(r.chromStart), int(r.chromEnd))
             for r in pd.read_csv(GENES, sep="\t").itertuples()}
    cal = pd.read_csv(CAL, sep="\t")
    # use adQTL + control variants (the set sub-gate B runs on), UNSTRATIFIED
    sets = pd.concat([pd.read_csv(RES/"p0_adqtl_set.tsv",sep="\t"),
                      pd.read_csv(RES/"p0_control_set.tsv",sep="\t")], ignore_index=True)
    v = cal[cal.variant_kgpID.isin(set(sets.variant_kgpID))]
    v = v[(v.variantRef.str.len()==1)&(v.variantAlt.str.len()==1)]
    done = set(pd.read_csv(OUT,sep="\t").variant_kgpID) if OUT.exists() else set()
    v = v[~v.variant_kgpID.isin(done)].drop_duplicates("variant_kgpID").head(a.n + len(done))
    v = v[~v.variant_kgpID.isin(done)]

    fa = Fasta(str(HG38), as_raw=True, sequence_always_upper=True)
    client = dna_client.create(key, timeout=180)
    first = not OUT.exists()
    for _, r in v.iterrows():
        gid = r.ensemblID.split(".")[0]
        if gid not in exons: continue
        ws=int(r.window_start); ch=r.variantChrom; vr=int(r.var_rel)
        seq=list(fa[ch][ws:ws+CONTEXT])
        if len(seq)<CONTEXT: seq+=["N"]*(CONTEXT-len(seq))
        ref="".join(seq); seq[vr]=r.variantAlt; alt="".join(seq)
        iv=genome.Interval(ch,ws,ws+CONTEXT)
        o=client.predict_sequences([ref,alt],requested_outputs=[dna_output.OutputType.RNA_SEQ],
                                   ontology_terms=LCL,intervals=[iv,iv],progress_bar=False,max_workers=2)
        emask = exon_rel_positions(exons[gid], ws)
        Rref=o[0].rna_seq.values; Ralt=o[1].rna_seq.values
        def exon_log2sum(): return float(np.log2((Ralt[emask].sum()+EPS)/(Rref[emask].sum()+EPS)))
        gs,ge=gspan.get(r.ensemblID,(None,None))
        def body():
            if gs is None: return np.nan
            lo,hi=max(0,gs-ws),min(CONTEXT,ge-ws)
            return float(np.log2((Ralt[lo:hi].sum()+EPS)/(Rref[lo:hi].sum()+EPS)))
        tlo,thi=max(0,int(r.tss_rel)-TSS_HALF),min(CONTEXT,int(r.tss_rel)+TSS_HALF)
        def tss(): return float(np.log2((Ralt[tlo:thi].sum()+EPS)/(Rref[tlo:thi].sum()+EPS)))
        rec={"variant_kgpID":r.variant_kgpID,"ensemblID":r.ensemblID,"log2_aFC":r.log2_aFC,
             "predD_exon":exon_log2sum(),"predD_body":body(),"predD_tss":tss()}
        pd.DataFrame([rec]).to_csv(OUT,sep="\t",mode="w" if first else "a",header=first,index=False)
        first=False
        print(f"  {r.geneSymbol:<12} aFC={r.log2_aFC:+.2f} exon={rec['predD_exon']:+.4f} body={rec['predD_body']:+.4f}",flush=True)

    d=pd.read_csv(OUT,sep="\t")
    print(f"\n=== exon-readout validation (n={len(d)}, unstratified adQTL/control; server target r~0.34) ===")
    for col in ["predD_exon","predD_body","predD_tss"]:
        dd=d[(d[col]!=0)&np.isfinite(d[col])]
        if len(dd)<5: print(f"  {col}: n={len(dd)} too few"); continue
        r,p=stats.spearmanr(dd[col],dd.log2_aFC)
        print(f"  {col:11} vs aFC: Spearman r={r:+.3f} p={p:.2e} sign={(np.sign(dd[col])==np.sign(dd.log2_aFC)).mean():.0%} n={len(dd)}")


if __name__=="__main__": main()
