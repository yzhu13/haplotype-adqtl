"""
Phase 2 -- Step 04: verdict on the Pan-UKB ancestry-matched coloc pilot
(`docs/NEXT_broader_coloc_plan.md`, Option B).

Key test: for each locus, compare PP.H4 (coloc.abf) across ancestry pairings
(EUR-eQTL x EUR-GWAS vs AFR-eQTL x AFR-GWAS, etc.) against which ancestry has
the STRONGER/sign-flipped eQTL effect (from `p0_adqtl_set.tsv`,
ObsDelta_EUR_AFR / beta_EUR / beta_AFR). The health-equity hypothesis: the
ancestry with the stronger eQTL effect should show higher PP.H4 with the
matched GWAS.

Output: docs/phase2_panukb_coloc_report.md
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parents[1]
COLOC = BASE / "results" / "phase2" / "p2_coloc.tsv"
ADQTL = BASE / "results" / "phase0" / "p0_adqtl_set.tsv"
DOCS = BASE / "docs"

PP4_HIT = 0.5


def main():
    coloc = pd.read_csv(COLOC, sep="\t")
    adq = pd.read_csv(ADQTL, sep="\t")

    lines = [
        "# Phase 2 -- Pan-UKB ancestry-matched colocalization pilot",
        "",
        f"Pilot of {coloc['geneSymbol'].nunique()} adQTL loci. coloc.abf (Wakefield ABF, "
        "p1=p2=1e-4, p12=1e-5, W=0.15^2) between MAGE per-ancestry regional cis-eQTL "
        "(+/-500kb TSS, EUR & AFR; this session's `p2_01`) and Pan-UKB per-ancestry "
        "GWAS (`p2_02`) for the best-matched anthropometric phenotype(s) per locus.",
        "",
        "## Results table",
        "",
        coloc.round(4).to_markdown(index=False),
        "",
    ]

    n_hits = (coloc["PP.H4"] >= PP4_HIT).sum()
    lines += [
        f"## Summary",
        "",
        f"- {len(coloc)} locus x ancestry x phenotype tests across "
        f"{coloc['geneSymbol'].nunique()} loci.",
        f"- {n_hits} tests with PP.H4 >= {PP4_HIT} (evidence of shared causal variant).",
        "",
    ]

    # Key ancestry-match test
    hits = coloc[coloc["PP.H4"] >= PP4_HIT]
    rows = []
    for gene in coloc["geneSymbol"].unique():
        adqr = adq[adq["geneSymbol"] == gene]
        if len(adqr) == 0:
            continue
        adqr = adqr.iloc[0]
        bE, bA = adqr.get("beta_EUR"), adqr.get("beta_AFR")
        stronger = None
        if pd.notna(bE) and pd.notna(bA):
            stronger = "EUR" if abs(bE) > abs(bA) else "AFR"
        sign_flip = (pd.notna(bE) and pd.notna(bA) and bE * bA < 0)
        g = coloc[coloc["geneSymbol"] == gene]
        for _, r in g.iterrows():
            rows.append({
                "geneSymbol": gene, "ancestry": r["ancestry"], "phenocode": r["phenocode"],
                "PP.H4": r["PP.H4"], "beta_EUR": bE, "beta_AFR": bA,
                "stronger_eqtl_ancestry": stronger, "sign_flip": sign_flip,
            })
    match_df = pd.DataFrame(rows)
    match_df.to_csv(BASE / "results" / "phase2" / "p2_ancestry_match.tsv", sep="\t", index=False)

    # Does the stronger-eQTL ancestry show higher PP.H4?
    matched_hits = []
    for gene, g in match_df.groupby("geneSymbol"):
        stronger = g["stronger_eqtl_ancestry"].iloc[0]
        if stronger is None:
            continue
        for pheno, gg in g.groupby("phenocode"):
            row_strong = gg[gg["ancestry"] == stronger]
            row_other = gg[gg["ancestry"] != stronger]
            if len(row_strong) == 0 or len(row_other) == 0:
                continue
            pp4_strong = row_strong["PP.H4"].iloc[0]
            pp4_other = row_other["PP.H4"].iloc[0]
            matched_hits.append({
                "geneSymbol": gene, "phenocode": pheno, "stronger_ancestry": stronger,
                "PP4_stronger": pp4_strong, "PP4_other": pp4_other,
                "sign_flip": gg["sign_flip"].iloc[0],
                "concordant_pattern": pp4_strong > pp4_other,
            })
    mh = pd.DataFrame(matched_hits)
    n_concordant = mh["concordant_pattern"].sum() if len(mh) else 0
    lines += [
        "## Ancestry-match test (health-equity hypothesis)",
        "",
        "For each locus x phenotype: does the ancestry with the STRONGER eQTL effect "
        "(|beta_EUR| vs |beta_AFR|, from `p0_adqtl_set.tsv`) also show higher PP.H4 "
        "with the matched Pan-UKB GWAS?",
        "",
        mh.round(4).to_markdown(index=False) if len(mh) else "(no comparable pairs)",
        "",
        f"- {n_concordant}/{len(mh)} locus x phenotype pairs show the concordant pattern "
        f"(stronger-eQTL ancestry has higher PP.H4)." if len(mh) else "",
        "",
    ]

    # Honest verdict
    lines += ["## Verdict", ""]
    if n_hits == 0:
        lines += [
            "**Null result.** No locus x ancestry x phenotype pair reaches "
            f"PP.H4 >= {PP4_HIT}. This is a THIRD consistent null for the disease/"
            "health-equity angle (after PAGE coloc p=0.49 and GWAS Catalog overlap "
            "p=0.78). Given AFR Pan-UKB GWAS power is inherently limited (n~6.5k vs "
            "EUR n~420k) and two prior broad approaches were also null, we recommend "
            "NOT scaling this up to the full ~85-locus set. Document as a consistent "
            "limitation: in this cohort/power regime, adQTL loci do not show detectable "
            "colocalization with anthropometric GWAS signals, regardless of method "
            "(locus overlap, PAGE coloc, or Pan-UKB ancestry-matched coloc.abf).",
        ]
    elif n_hits >= 1 and n_concordant >= max(1, len(mh) // 2 + 1):
        lines += [
            f"**Vignette-worthy.** {n_hits} locus x ancestry x phenotype pair(s) reach "
            f"PP.H4 >= {PP4_HIT}, and the ancestry-match pattern is concordant in "
            f"{n_concordant}/{len(mh)} cases -- i.e. the ancestry with the stronger "
            "eQTL effect tends to also show the stronger GWAS colocalization. This "
            "supports including 1-2 of these as a supplementary vignette/figure "
            "(Discussion or supplement), but the sample size is too small for a "
            "headline claim. Scale-up to the full ~85 loci could be considered if "
            "time permits, but is not essential.",
        ]
    else:
        lines += [
            f"**Mixed / weak.** {n_hits} locus x ancestry x phenotype pair(s) reach "
            f"PP.H4 >= {PP4_HIT}, but the ancestry-match pattern is not clearly "
            f"concordant ({n_concordant}/{len(mh)}). Treat any individual PP.H4>=0.5 "
            "hit as a candidate vignette (report honestly with caveats), but this "
            "does not support a broad health-equity claim. Recommend NOT scaling up "
            "to the full ~85-locus set; document as a third largely-null result "
            "alongside PAGE coloc and GWAS Catalog overlap.",
        ]

    out = DOCS / "phase2_panukb_coloc_report.md"
    out.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"[p2_04] -> {out}")
    print("\n".join(str(x) for x in lines[-8:]))


if __name__ == "__main__":
    main()
