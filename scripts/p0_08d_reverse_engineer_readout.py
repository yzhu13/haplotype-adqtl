"""
Phase 0 — Step 08d: Reverse-engineer the server GeneMaskLFCScorer so we can
replicate its sensitivity on custom (background-injected) sequences.

For each of N reference SNVs, on the SAME variant we obtain:
  (A) SERVER raw_score: score_variant(GeneMaskLFCScorer RNA) -> target gene x LCL
      tracks (this is sub-gate A's sensitive readout, recovers aFC r=0.48).
  (B) predict_sequences([ref, alt]) -> Output arrays -> several CANDIDATE client
      aggregations (over gene body / TSS window / per-bin mean LFC), per LCL track.

Then we correlate each candidate vs the server raw_score AND vs pooled aFC. The
candidate that best matches the server score is the recipe to use in the ADRS
re-run; if none matches, we need exon masks (GENCODE) — reported explicitly.

NEEDS API. Small N, run under bash timeout loop.

Output: results/phase0/p0_readout_reveng.tsv + printed correlation table.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from dotenv import load_dotenv
from pyfaidx import Fasta
from alphagenome.models import dna_client, dna_output, variant_scorers as vs
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV, HG38_FASTA
HG38 = HG38_FASTA; ENV = ALPHAGENOME_ENV
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
GENES = BASE / "data/raw/MAGE/QTL_results/eQTL_results/expression_filteredGenes.MAGE.v1.0.txt.gz"
CAL = RES / "p0_calibration_set.tsv"
OUT = RES / "p0_readout_reveng.tsv"
CONTEXT = 1_048_576; LCL = {"EFO:0002784", "EFO:0000572"}; EPS = 1e-3; TSS_HALF = 2000


def gene_spans():
    df = pd.read_csv(GENES, sep="\t")
    return {r.ensemblID: (int(r.chromStart), int(r.chromEnd)) for r in df.itertuples()}


def lcl_cols(meta_df):
    """indices of LCL RNA tracks in an Output.rna_seq.metadata frame."""
    return meta_df.index[meta_df["ontology_curie"].isin(LCL)].tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
    load_dotenv(ENV); key = os.environ["ALPHAGENOME_API_KEY"]
    spans = gene_spans()
    cal = pd.read_csv(CAL, sep="\t")
    cal = cal[(cal.variantRef.str.len()==1)&(cal.variantAlt.str.len()==1)]
    done = set(pd.read_csv(OUT, sep="\t").variant_kgpID) if OUT.exists() else set()
    cal = cal[~cal.variant_kgpID.isin(done)]
    cal = cal.assign(_q=pd.qcut(cal.log2_aFC.abs(),8,labels=False,duplicates="drop"))
    todo = cal.groupby("_q",group_keys=False).apply(lambda g:g.sample(min(6,len(g)),random_state=2)).head(a.n)

    fa = Fasta(str(HG38), as_raw=True, sequence_always_upper=True)
    client = dna_client.create(key, timeout=180)
    rna_scorer = vs.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]
    first = not OUT.exists()

    for _, r in todo.iterrows():
        if r.ensemblID not in spans: continue
        ws=int(r.window_start); ch=r.variantChrom; vr=int(r.var_rel); gid=r.ensemblID.split(".")[0]
        iv = genome.Interval(ch, ws, ws+CONTEXT)
        v = genome.Variant(ch, int(r.variantPosition), r.variantRef, r.variantAlt)
        # (A) server raw_score (avg over LCL RNA tracks for target gene)
        try:
            tidy = vs.tidy_scores(client.score_variant(iv, v, variant_scorers=[rna_scorer]),
                                  match_gene_strand=True)
            srv = tidy[(tidy.output_type=="RNA_SEQ")&(tidy.gene_id==gid)&(tidy.ontology_curie.isin(LCL))]
            server_lfc = float(srv.raw_score.mean()) if len(srv) else np.nan
        except Exception as e:
            print(f"  score_variant ERROR {r.variant_kgpID}: {e}"); continue
        # (B) predict_sequences ref/alt
        seq=list(fa[ch][ws:ws+CONTEXT])
        if len(seq)<CONTEXT: seq+=["N"]*(CONTEXT-len(seq))
        ref="".join(seq); seq[vr]=r.variantAlt; alt="".join(seq)
        o = client.predict_sequences([ref, alt], requested_outputs=[dna_output.OutputType.RNA_SEQ],
                                     ontology_terms=list(LCL), intervals=[iv,iv],
                                     progress_bar=False, max_workers=2)
        cols = lcl_cols(o[0].rna_seq.metadata)
        Rref = o[0].rna_seq.values[:, cols]; Ralt = o[1].rna_seq.values[:, cols]  # (1Mb, k)
        gs,ge = spans[r.ensemblID]; gsr=max(0,gs-ws); ger=min(CONTEXT,ge-ws)
        tlo,thi = max(0,int(r.tss_rel)-TSS_HALF), min(CONTEXT,int(r.tss_rel)+TSS_HALF)
        def log2sum(A,B,lo,hi): return float(np.log2((B[lo:hi].sum()+EPS)/(A[lo:hi].sum()+EPS)))
        def perbin_mean_lfc(A,B,lo,hi):
            a2=A[lo:hi]; b2=B[lo:hi]
            return float(np.mean(np.log2((b2+EPS)/(a2+EPS))))
        rec = {"variant_kgpID":r.variant_kgpID,"ensemblID":r.ensemblID,"log2_aFC":r.log2_aFC,
               "server_lfc":server_lfc,
               "cand_body_log2sum":log2sum(Rref,Ralt,gsr,ger),
               "cand_tss_log2sum":log2sum(Rref,Ralt,tlo,thi),
               "cand_body_perbinmean":perbin_mean_lfc(Rref,Ralt,gsr,ger),
               "cand_tss_perbinmean":perbin_mean_lfc(Rref,Ralt,tlo,thi),
               "cand_full_log2sum":log2sum(Rref,Ralt,0,CONTEXT)}
        pd.DataFrame([rec]).to_csv(OUT,sep="\t",mode="w" if first else "a",header=first,index=False)
        first=False
        print(f"  {r.geneSymbol:<12} aFC={r.log2_aFC:+.2f} server={server_lfc:+.4f} "
              f"body={rec['cand_body_log2sum']:+.4f} tssPB={rec['cand_tss_perbinmean']:+.4f}",flush=True)

    # report
    d = pd.read_csv(OUT, sep="\t")
    print(f"\n=== reverse-engineering report (n={len(d)}) ===")
    print("Does server_lfc recover aFC? (validates server readout):")
    dd=d.dropna(subset=["server_lfc"])
    r,p=stats.spearmanr(dd.server_lfc,dd.log2_aFC); print(f"  server_lfc vs aFC: r={r:+.3f} p={p:.2e} n={len(dd)}")
    print("\nWhich candidate best matches server_lfc?")
    for c in ["cand_body_log2sum","cand_tss_log2sum","cand_body_perbinmean","cand_tss_perbinmean","cand_full_log2sum"]:
        m=d.dropna(subset=["server_lfc",c]); m=m[(m[c]!=0)]
        if len(m)<5: print(f"  {c}: n={len(m)} too few"); continue
        rs,ps=stats.spearmanr(m[c],m.server_lfc); ra,_=stats.spearmanr(m[c],m.log2_aFC)
        print(f"  {c:22} vs server r={rs:+.3f}(p={ps:.1e}) | vs aFC r={ra:+.3f} | n={len(m)}")


if __name__=="__main__": main()
