"""
Phase 1 / Direction 1 — Step 3: mechanism vignette. Does the background SNP that
modulates an adQTL's focal effect disrupt TF binding?

Uses AlphaGenome's WORKING server scoring (reference background — the path that
works; NOT the blocked custom-background path). For each confirmed cis-epistasis
adQTL, score its best background SNP with CHIP_TF and ATAC CenterMask scorers and
take the max |effect| (TF-binding / accessibility disruption). Compare bg SNPs to
the focal variants (matched, same loci) as an internal reference.

Mechanism hypothesis: the ancestry-differentiated background SNP alters a TF
binding site / local accessibility, which modulates the focal variant's regulatory
output (cis-epistasis).

NEEDS API. Run under timeout loop. Output: results/phase1/p1_tf_vignette.tsv + report.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alphagenome.models import dna_client, variant_scorers as vs
from alphagenome.data import genome

from lib_paths import ALPHAGENOME_ENV as P1env
BASE = Path(__file__).resolve().parents[1]
P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"
CTX = 1_048_576


def parse_variant(vid):
    c, p, r, a = vid.split(":")
    return f"chr{c}", int(p), r, a


def score_tf_atac(client, chrom, pos, ref, alt, scorers):
    half = CTX // 2
    iv = genome.Interval(chrom, max(0, pos - half), pos + half)
    v = genome.Variant(chrom, pos, ref, alt)
    tidy = vs.tidy_scores(client.score_variant(iv, v, variant_scorers=scorers), match_gene_strand=False)
    out = {}
    for ot in ["CHIP_TF", "ATAC"]:
        sub = tidy[tidy.output_type == ot]
        out[ot] = float(sub.raw_score.abs().max()) if len(sub) else np.nan
        # top TF track for vignette. Keep the biosample, the FULL track name, and
        # the TF symbol as separate fields -- the old composite/truncated string
        # led to the cell line being read as the TF (see p1_09b_parse_tf_names.py).
        if ot == "CHIP_TF" and len(sub):
            top = sub.loc[sub.raw_score.abs().idxmax()]
            biosample = str(top.get("biosample_name", ""))
            track = str(top.get("track_name", ""))
            out["top_biosample"] = biosample
            out["top_track"] = track
            out["top_tf_symbol"] = track.split(" TF ChIP-seq ", 1)[1].split()[0] if " TF ChIP-seq " in track else ""
            out["top_tf"] = biosample + " / " + track  # legacy composite, no longer truncated
    return out


def main():
    load_dotenv(P1env); key = os.environ["ALPHAGENOME_API_KEY"]
    wa = pd.read_csv(P1 / "p1_within_ancestry.tsv", sep="\t")
    wa["confirmed"] = (wa.int_p_EUR < 0.05) | (wa.int_p_AFR < 0.05)
    conf = wa[wa.confirmed].copy()
    print(f"[p1_09] confirmed cis-epistasis adQTLs to score: {len(conf)}")

    OUT = P1 / "p1_tf_vignette.tsv"
    done = set(pd.read_csv(OUT, sep="\t").best_bg) if OUT.exists() else set()
    client = dna_client.create(key, timeout=180)
    scorers = [vs.RECOMMENDED_VARIANT_SCORERS["CHIP_TF"], vs.RECOMMENDED_VARIANT_SCORERS["ATAC"]]
    first = not OUT.exists()
    for _, r in conf.iterrows():
        if r.best_bg in done:
            continue
        try:
            bc, bp, br, ba = parse_variant(r.best_bg)
            bg = score_tf_atac(client, bc, bp, br, ba, scorers)
            fc, fp, fr, fa = parse_variant(r.variant_kgpID)
            fo = score_tf_atac(client, fc, fp, fr, fa, scorers)
            rec = {"ensemblID": r.ensemblID, "variant_kgpID": r.variant_kgpID, "best_bg": r.best_bg,
                   "bg_chip_tf": bg.get("CHIP_TF"), "bg_atac": bg.get("ATAC"), "bg_top_tf": bg.get("top_tf", ""),
                   "bg_biosample": bg.get("top_biosample", ""), "bg_tf_symbol": bg.get("top_tf_symbol", ""),
                   "bg_track": bg.get("top_track", ""),
                   "focal_chip_tf": fo.get("CHIP_TF"), "focal_atac": fo.get("ATAC"),
                   "focal_biosample": fo.get("top_biosample", ""), "focal_tf_symbol": fo.get("top_tf_symbol", ""),
                   "focal_track": fo.get("top_track", "")}
            pd.DataFrame([rec]).to_csv(OUT, sep="\t", mode="w" if first else "a", header=first, index=False)
            first = False
            print(f"  {r.ensemblID} bg={r.best_bg} bg_TF={rec['bg_chip_tf']:.3f} focal_TF={rec['focal_chip_tf']:.3f}", flush=True)
        except Exception as e:
            print(f"  ERROR {r.best_bg}: {e}"); continue

    df = pd.read_csv(OUT, sep="\t")
    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())
    L("# Phase 1 / Direction 1 — TF-binding mechanism vignette (background SNPs)\n")
    L(f"- confirmed cis-epistasis adQTLs scored: {len(df)}")
    from scipy import stats
    bg = df.bg_chip_tf.dropna(); fo = df.focal_chip_tf.dropna()
    L(f"- bg-SNP |CHIP_TF| median={bg.median():.3f}; focal |CHIP_TF| median={fo.median():.3f}")
    L(f"- bg-SNP |ATAC| median={df.bg_atac.median():.3f}")
    hi = df[df.bg_chip_tf > df.bg_chip_tf.median()].sort_values("bg_chip_tf", ascending=False)
    L(f"\n## Top vignettes: background SNPs with strong predicted TF-binding disruption")
    for _, r in hi.head(12).iterrows():
        # report the TF symbol, not the biosample. Older tables only have the
        # composite bg_top_tf; p1_09b_parse_tf_names.py recovers the symbol there.
        tf = r.get("bg_tf_symbol") or "?"
        bios = r.get("bg_biosample") or ""
        L(f"- {r.ensemblID:<20} bg={r.best_bg} |CHIP_TF|={r.bg_chip_tf:.2f} |ATAC|={r.bg_atac:.2f}  TF={tf} ({bios})")
    L("\n## Interpretation")
    L("- Background SNPs that modulate adQTL focal effects show predicted TF-binding/accessibility")
    L("  disruption => mechanism: an ancestry-differentiated background SNP alters a local TF site,")
    L("  changing the regulatory context and thus the focal variant's effect (cis-epistasis).")
    (DOCS / "phase1_tf_vignette_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
