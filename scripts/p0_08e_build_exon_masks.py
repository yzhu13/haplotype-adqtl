"""
Phase 0 — Step 08e: Extract per-gene EXON intervals from GENCODE, to build an
exon-mask RNA readout that replicates the server GeneMaskLFCScorer (which the
diagnostic p0_08d showed is the sensitive readout: server r=0.34 on the
adQTL/control set vs our gene-body r=0.045).

Only genes in the calibration/adQTL/control sets are kept (compact). Exons are
merged per gene (union of all transcripts' exons).

Output: data/processed/gene_exons.json
  { "ENSG00000...": [[start,end], ...], ... }   # 1-based, genomic, merged
"""
import gzip, json, re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
GTF = BASE / "data/external/gencode.v44.basic.annotation.gtf.gz"
RES = BASE / "results" / "phase0"
OUT = BASE / "data/processed/gene_exons.json"

GID_RE = re.compile(r'gene_id "([^"]+)"')


def target_genes():
    g = set()
    for f in ["p0_calibration_set.tsv", "p0_adqtl_set.tsv", "p0_control_set.tsv"]:
        p = RES / f
        if p.exists():
            g |= set(pd.read_csv(p, sep="\t")["ensemblID"].str.split(".").str[0])
    return g


def merge(intervals):
    intervals.sort()
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    want = target_genes()
    print(f"[p0_08e] target genes: {len(want)}")
    exons = {}
    n = 0
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) < 9 or c[2] != "exon":
                continue
            m = GID_RE.search(c[8])
            if not m:
                continue
            gid = m.group(1).split(".")[0]
            if gid not in want:
                continue
            exons.setdefault(gid, []).append([int(c[3]), int(c[4])])
            n += 1
    exons = {g: merge(v) for g, v in exons.items()}
    OUT.write_text(json.dumps(exons), encoding="utf-8")
    tot_exons = sum(len(v) for v in exons.values())
    print(f"[p0_08e] {len(exons)} genes, {tot_exons} merged exons (from {n} raw) -> {OUT}")
    # quick stats
    import numpy as np
    span = [sum(e - s + 1 for s, e in v) for v in exons.values()]
    print(f"  exonic length median={int(np.median(span))} bp")


if __name__ == "__main__":
    main()
