"""
Phase 1 — Step 02: Do ancestry-differential eQTLs (adQTLs) sit at multi-ancestry
disease/trait loci? Cross-reference adQTLs with MAGE-PAGE colocalizations.

PAGE = Population Architecture using Genomics and Epidemiology — a deliberately
multi-ancestry GWAS cohort (African American, Hispanic/Latino, Asian, Native
Hawaiian) across 25 complex traits. MAGE colocalized its cis-eQTLs with PAGE
(coloc.susie; PP.H4 = posterior of shared causal signal; moderate>=0.5, strong>=0.8).

Health-equity payoff: ancestry-differential regulatory effects AT disease loci are
exactly where differential risk / PRS-transfer failure would bite.

Analyses:
  - enrichment: among tested eGenes (EUR&AFR estimable), are adQTLs over-represented
    among genes that colocalize with a PAGE trait? (Fisher exact)
  - concrete list: each adQTL x trait colocalization, with which ancestry has the
    stronger effect (sign/magnitude of beta_EUR vs beta_AFR).

Inputs: results/phase1/p1_decomposition.tsv, MAGE PAGE coloc moderate_results.
Outputs: results/phase1/p1_adqtl_page_coloc.tsv, docs/phase1_page_coloc_report.md
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase1"
DOCS = BASE / "docs"
COLOC = BASE / "data/raw/MAGE/colocalization_results/PAGE.eQTL_colocalization.moderate_results.MAGE.v1.0.txt.gz"
PP_STRONG = 0.8


def main():
    dec = pd.read_csv(RES / "p1_decomposition.tsv", sep="\t")
    dec["gene0"] = dec["ensemblID"].str.split(".").str[0]
    col = pd.read_csv(COLOC, sep="\t")
    col["gene0"] = col["eQTL_gene_ensemblID"].str.split(".").str[0]

    coloc_genes = set(col["gene0"])
    strong_genes = set(col.loc[col["PP.H4"] >= PP_STRONG, "gene0"])
    dec["coloc"] = dec["gene0"].isin(coloc_genes)

    log = []
    def L(s):
        log.append(s)
        try: print(s)
        except UnicodeEncodeError: print(s.encode("ascii","replace").decode())

    L("# Phase 1 — adQTL x PAGE colocalization (disease / health-equity)\n")
    L(f"- PAGE coloc: {len(col)} colocalizations, {col.gene0.nunique()} eGenes, "
      f"{col.GWAS_catalog_accessionID.nunique()} traits (PP.H4>=0.5; strong>=0.8: {len(strong_genes)} genes)")
    L(f"- tested eGenes (EUR&AFR estimable): {dec.gene0.nunique()}; adQTLs: {int(dec.is_adqtl.sum())}")

    # gene-level table (one row per gene: is_adqtl, coloc)
    g = dec.groupby("gene0").agg(is_adqtl=("is_adqtl","max"), coloc=("coloc","max")).reset_index()
    a = int(((g.is_adqtl) & (g.coloc)).sum())
    b = int(((g.is_adqtl) & (~g.coloc)).sum())
    c = int(((~g.is_adqtl) & (g.coloc)).sum())
    d = int(((~g.is_adqtl) & (~g.coloc)).sum())
    odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    L(f"\n## Enrichment (gene-level Fisher exact)")
    L(f"-                 coloc   no-coloc")
    L(f"-   adQTL          {a:5d}   {b:6d}")
    L(f"-   concordant     {c:5d}   {d:6d}")
    rate_ad = a/max(1,a+b); rate_co = c/max(1,c+d)
    L(f"- adQTL coloc rate {rate_ad:.3%} vs concordant {rate_co:.3%}; OR={odds:.2f}, p={p:.3f}")

    # concrete adQTL colocalizations
    adq = dec[dec.is_adqtl][["gene0","geneSymbol","variant_kgpID","beta_EUR","beta_AFR",
                             "ObsDelta_EUR_AFR","p_het_EUR_AFR","I2"]]
    merged = col.merge(adq, on="gene0", how="inner")
    merged["stronger_in"] = np.where(merged.beta_EUR.abs() > merged.beta_AFR.abs(), "EUR", "AFR")
    merged["flip"] = np.sign(merged.beta_EUR) != np.sign(merged.beta_AFR)
    out = merged[["traitDescription","eQTL_gene_geneSymbol","PP.H4","beta_EUR","beta_AFR",
                  "ObsDelta_EUR_AFR","stronger_in","flip","p_het_EUR_AFR"]].sort_values("PP.H4",ascending=False)
    out.to_csv(RES / "p1_adqtl_page_coloc.tsv", sep="\t", index=False)
    L(f"\n## adQTL x PAGE-trait colocalizations: {len(out)}")
    for _, r in out.iterrows():
        L(f"- {r.eQTL_gene_geneSymbol:<10} x {r.traitDescription[:42]:<42} "
          f"PP.H4={r['PP.H4']:.2f} | bEUR={r.beta_EUR:+.2f} bAFR={r.beta_AFR:+.2f} "
          f"stronger={r.stronger_in}{' FLIP' if r.flip else ''}")

    L("\n## Interpretation")
    L("- adQTL x PAGE colocalizations are loci where an ancestry-differential cis-eQTL "
      "underlies a multi-ancestry GWAS signal: candidate contributors to ancestry-")
    L("  differential disease risk and PRS-transfer failure (health-equity).")
    (DOCS / "phase1_page_coloc_report.md").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
