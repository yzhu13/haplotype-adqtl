"""
Phase 1 — Step 05 (Option A): broad GWAS-Catalog overlap enrichment for adQTLs.

Are ancestry-differential eQTLs (adQTLs) enriched at GWAS loci — overall and at
NON-EUROPEAN-ancestry GWAS loci specifically (the health-equity angle)?

Method: overlap each eQTL lead variant with genome-wide-significant (P<5e-8) GWAS
Catalog associations (GRCh38), at two stringencies:
  - STRICT: within +/-1 kb of a GWAS hit (variant-level),
  - LOCUS:  within +/-50 kb (locus-level).
Each GWAS study is tagged European-only vs containing non-European ancestry (from
the Catalog ancestry file). Enrichment: adQTL vs MAF/distance-matched CONTROL
(concordant eQTLs), Fisher exact — overall and non-European-only.

Inputs: results/phase0/p0_adqtl_set.tsv, p0_control_set.tsv, p0_calibration_set.tsv;
  data/external/gwas_catalog/{associations_full.tsv, ancestry.tsv}.
Outputs: results/phase1/p1_gwas_overlap.tsv, docs/phase1_gwas_enrichment_report.md
"""
from pathlib import Path
import bisect
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
P0 = BASE / "results" / "phase0"; P1 = BASE / "results" / "phase1"; DOCS = BASE / "docs"
GW = BASE / "data/external/gwas_catalog"
PVAL = 5e-8
W_STRICT = 1_000
W_LOCUS = 50_000
NONEUR_KEYS = ["African", "Asian", "Hispanic", "Latin", "Ocean", "Middle Eastern",
               "Native", "Sub-Saharan", "South Asian", "East Asian"]


def load_noneur_studies():
    anc = pd.read_csv(GW / "ancestry.tsv", sep="\t", usecols=["STUDY ACCESSION", "BROAD ANCESTRAL CATEGORY"],
                      dtype=str).dropna()
    noneur, allst = set(), set()
    for acc, cat in zip(anc["STUDY ACCESSION"], anc["BROAD ANCESTRAL CATEGORY"]):
        allst.add(acc)
        if any(k.lower() in cat.lower() for k in NONEUR_KEYS):
            noneur.add(acc)
    return noneur


def load_gwas():
    use = ["CHR_ID", "CHR_POS", "MAPPED_TRAIT", "STUDY ACCESSION", "P-VALUE", "DISEASE/TRAIT"]
    g = pd.read_csv(GW / "associations_full.tsv", sep="\t", usecols=use, dtype=str, low_memory=False)
    g["P"] = pd.to_numeric(g["P-VALUE"], errors="coerce")
    g["pos"] = pd.to_numeric(g["CHR_POS"], errors="coerce")
    g = g[(g.P < PVAL) & g.pos.notna() & g.CHR_ID.isin([str(i) for i in range(1, 23)])]
    g["chrom"] = "chr" + g.CHR_ID
    g["pos"] = g.pos.astype(int)
    return g


def build_index(g, noneur):
    """chrom -> (sorted pos array, parallel is_noneur array, parallel trait list)."""
    idx = {}
    for ch, sub in g.groupby("chrom"):
        sub = sub.sort_values("pos")
        pos = sub.pos.to_numpy()
        nev = sub["STUDY ACCESSION"].isin(noneur).to_numpy()
        tr = sub["MAPPED_TRAIT"].fillna(sub["DISEASE/TRAIT"]).to_numpy()
        idx[ch] = (pos, nev, tr)
    return idx


def overlap(idx, chrom, p, w):
    """return (any_hit, any_noneur_hit, traits[set]) within +/-w of position p."""
    if chrom not in idx:
        return False, False, set()
    pos, nev, tr = idx[chrom]
    lo = bisect.bisect_left(pos, p - w); hi = bisect.bisect_right(pos, p + w)
    if hi <= lo:
        return False, False, set()
    return True, bool(nev[lo:hi].any()), set(tr[lo:hi])


def main():
    noneur = load_noneur_studies()
    print(f"[p1_05] non-European-containing studies: {len(noneur)}")
    g = load_gwas()
    print(f"[p1_05] genome-wide-sig GWAS associations (autosomal): {len(g)}")
    idx = build_index(g, noneur)

    cal = pd.read_csv(P0 / "p0_calibration_set.tsv", sep="\t")[
        ["variant_kgpID", "variantChrom", "variantPosition"]]
    a = pd.read_csv(P0 / "p0_adqtl_set.tsv", sep="\t")[["variant_kgpID"]]; a["set"] = "adqtl"
    c = pd.read_csv(P0 / "p0_control_set.tsv", sep="\t")[["variant_kgpID"]]; c["set"] = "control"
    v = pd.concat([a, c]).drop_duplicates("variant_kgpID").merge(cal, on="variant_kgpID", how="left").dropna()

    rows = []
    for _, r in v.iterrows():
        ch, p = r.variantChrom, int(r.variantPosition)
        s_hit, s_ne, _ = overlap(idx, ch, p, W_STRICT)
        l_hit, l_ne, l_tr = overlap(idx, ch, p, W_LOCUS)
        rows.append({"variant_kgpID": r.variant_kgpID, "set": r.set,
                     "strict_hit": s_hit, "strict_noneur": s_ne,
                     "locus_hit": l_hit, "locus_noneur": l_ne,
                     "traits": ";".join(sorted(t for t in l_tr if isinstance(t, str))[:5])})
    df = pd.DataFrame(rows)
    df.to_csv(P1 / "p1_gwas_overlap.tsv", sep="\t", index=False)

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode())

    L("# Phase 1 — GWAS-Catalog overlap enrichment (adQTL vs matched control)\n")
    L(f"- variants: adQTL {int((df.set=='adqtl').sum())}, control {int((df.set=='control').sum())}")
    L(f"- GWAS hits: {len(g)} (P<5e-8); non-European-containing studies: {len(noneur)}")

    def fisher(col, label):
        ad = df[df.set == "adqtl"]; co = df[df.set == "control"]
        a1 = int(ad[col].sum()); b1 = len(ad) - a1
        c1 = int(co[col].sum()); d1 = len(co) - c1
        odds, p = stats.fisher_exact([[a1, b1], [c1, d1]], alternative="greater")
        L(f"- {label:<30} adQTL {a1/len(ad):.1%} vs control {c1/len(co):.1%}  OR={odds:.2f} p={p:.3f}"
          + ("  *" if p < 0.05 else ""))
        return {"test": label, "adqtl_rate": a1/len(ad), "control_rate": c1/len(co), "OR": odds, "p": p}

    L("\n## Overlap enrichment")
    res = [fisher("strict_hit", "any GWAS, +/-1kb (strict)"),
           fisher("locus_hit", "any GWAS, +/-50kb (locus)"),
           fisher("strict_noneur", "non-EUR GWAS, +/-1kb"),
           fisher("locus_noneur", "non-EUR GWAS, +/-50kb")]
    pd.DataFrame(res).to_csv(P1 / "p1_gwas_enrichment_stats.tsv", sep="\t", index=False)

    # example adQTL x GWAS overlaps (strict)
    ex = df[(df.set == "adqtl") & df.strict_hit & (df.traits != "")].head(15)
    L(f"\n## Example adQTL × GWAS overlaps (strict, ±1kb): {int((df[df.set=='adqtl']).strict_hit.sum())} adQTLs")
    for _, r in ex.iterrows():
        L(f"- {r.variant_kgpID:<22} {'[non-EUR]' if r.strict_noneur else '         '} {r.traits[:80]}")
    L("\n## Note")
    L("- Overlap (esp. ±50kb) is a proxy for colocalization (LD-confounded); a positive,")
    L("  ancestry-specific signal motivates formal coloc vs Pan-UKB (docs/NEXT_broader_coloc_plan.md).")
    (DOCS / "phase1_gwas_enrichment_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
