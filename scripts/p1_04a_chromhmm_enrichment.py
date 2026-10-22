"""
Phase 1 — Step 04a: Chromatin-state characterization of adQTLs.

Are ancestry-differential eQTL lead variants enriched in particular GM12878 (LCL)
chromatin states vs concordant eQTLs? Uses MAGE's Roadmap 15-state chromHMM
annotation (lead variant per credible set). API-free.

Mechanistic motivation: if adQTLs sit preferentially in active enhancers/promoters
(vs quiescent/repressed), their effect is more plausibly cis-regulatory and
context-dependent.

Inputs: results/phase1/p1_decomposition.tsv (is_adqtl),
  MAGE eQTL_leadVariant.roadmap_15state_chromHMM_annotation.GM12878.
Outputs: results/phase1/p1_chromhmm.tsv, docs/phase1_chromhmm_report.md
"""
from pathlib import Path
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase1"
DOCS = BASE / "docs"
CHMM = BASE / "data/raw/MAGE/QTL_functional_annotation/eQTL_leadVariant.roadmap_15state_chromHMM_annotation.GM12878.MAGE.v1.0.txt.gz"

STATES = ["1_TssA","2_TssAFlnk","3_TxFlnk","4_Tx","5_TxWk","6_EnhG","7_Enh",
          "8_ZNF/Rpts","9_Het","10_TssBiv","11_BivFlnk","12_EnhBiv","13_ReprPC",
          "14_ReprPCWk","15_Quies"]
GROUPS = {
    "Promoter (TssA/Flnk/Biv)": ["1_TssA","2_TssAFlnk","10_TssBiv","11_BivFlnk"],
    "Enhancer (EnhG/Enh/EnhBiv)": ["6_EnhG","7_Enh","12_EnhBiv"],
    "Transcription (TxFlnk/Tx/TxWk)": ["3_TxFlnk","4_Tx","5_TxWk"],
    "Repressed/Quiescent": ["9_Het","13_ReprPC","14_ReprPCWk","15_Quies"],
}


def main():
    dec = pd.read_csv(RES / "p1_decomposition.tsv", sep="\t")[["variant_kgpID","ensemblID","is_adqtl"]]
    ch = pd.read_csv(CHMM, sep="\t")
    states = [s for s in STATES if s in ch.columns]
    df = dec.merge(ch[["variant_kgpID","ensemblID"]+states], on=["variant_kgpID","ensemblID"], how="inner")
    ad = df[df.is_adqtl]; co = df[~df.is_adqtl]
    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii","replace").decode())

    L("# Phase 1 — chromHMM (GM12878) state enrichment: adQTL vs concordant\n")
    L(f"- annotated lead variants: {len(df)} (adQTL {len(ad)}, concordant {len(co)})")

    def enrich(cols, label):
        a = int(ad[cols].max(axis=1).sum()); b = len(ad) - a
        c = int(co[cols].max(axis=1).sum()); d = len(co) - c
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        L(f"- {label:<34} adQTL {a/len(ad):.1%} vs concordant {c/len(co):.1%}  "
          f"OR={odds:.2f} p={p:.3f}")
        return {"group": label, "adqtl_frac": a/len(ad), "conc_frac": c/len(co), "OR": odds, "p": p}

    L("\n## Grouped states")
    rows = [enrich(cols, g) for g, cols in GROUPS.items()]
    L("\n## Individual states (active regulatory)")
    for s in ["1_TssA","7_Enh","6_EnhG","12_EnhBiv","10_TssBiv"]:
        if s in states:
            rows.append(enrich([s], s))
    pd.DataFrame(rows).to_csv(RES / "p1_chromhmm.tsv", sep="\t", index=False)
    sig = [r for r in rows if r["p"] < 0.05]
    L("\n## Interpretation")
    if sig:
        L("- adQTLs show chromatin-state enrichment: " +
          "; ".join(f"{r['group']} OR={r['OR']:.2f} p={r['p']:.3f}" for r in sig))
    else:
        L("- NULL: adQTLs show NO significant chromatin-state enrichment vs concordant "
          "eQTLs (all p>0.05). adQTLs are not a structurally distinct class of regulatory "
          "element — both are bona fide fine-mapped eQTLs in similar states. What "
          "distinguishes adQTLs is their ancestry-dependent EFFECT, not their baseline "
          "chromatin annotation — consistent with a context/background (GxE) mechanism "
          "rather than a different element type.")
    (DOCS / "phase1_chromhmm_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
