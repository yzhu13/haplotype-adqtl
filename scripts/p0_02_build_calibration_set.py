"""
Phase 0 — Step 02: Build the eGene lead-variant CALIBRATION set.

This is the genotype-free FIRST GATE of Phase 0. Before
asking the hard question ("does AlphaGenome see ancestry-DIFFERENTIAL effects?")
we test the necessary precondition: can AlphaGenome predict the POOLED eQTL
effect (MAGE log2_aFC) at all, in the matched cell type (LCL / GM12878)?

If the engine can't predict pooled eQTL effect sign/magnitude here, the
ancestry-differential test is hopeless and we trigger the pivot narrative.

This step only PREPARES the prediction set (no API calls):
  - load MAGE fine-mapped lead variants + their pooled log2_aFC (aFCn),
  - map each eGene to its TSS (from the QTL phenotype BED),
  - QC filter (PIP, finite aFC, autosomes, cis-distance),
  - compute a 1 Mb AlphaGenome window centered on the variant<->TSS MIDPOINT
    so BOTH the variant and the TSS fall inside one 2^20 bp window,
  - emit relative offsets of variant & TSS within the window.

Input  (from p0_01):  data/raw/MAGE/...
Output: results/phase0/p0_calibration_set.tsv
        logs/p0_02_build_calibration_set.log
"""

import gzip
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MAGE = BASE / "data" / "raw" / "MAGE"
RES = BASE / "results" / "phase0"
LOGS = BASE / "logs"

AFCN = (MAGE / "QTL_results/eQTL_results/eQTL_finemapping_results/"
        "eQTL_finemapping.leadVariant_aFCn.MAGE.v1.0.txt.gz")
EXPR_BED = (MAGE / "QTL_results/eQTL_results/expression_quants/"
            "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz")

# AlphaGenome fixed context (2^20). Keep a margin so both anchors sit comfortably
# inside the window (avoid the extreme edges where the model is least reliable).
CONTEXT = 1_048_576
SAFE_HALF = 480_000          # max |variant - TSS|/2 + buffer we allow from center
MAX_VAR_TSS_DIST = 900_000   # drop pairs that cannot share one window with margin

# QC thresholds
MIN_PIP = 0.50               # lead variant must be a credible causal candidate
AUTOSOMES = {f"chr{i}" for i in range(1, 23)}  # aFC caveat: chrX non-PAR excluded


def load_tss_map():
    """ensemblID(+version) -> (chrom, tss_pos). BED is 1bp TSS per gene."""
    rows = []
    with gzip.open(EXPR_BED, "rt") as f:
        header = next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            rows.append((c[3], c[0], int(c[2])))  # ID, chrom, end(=TSS, 1-based)
    df = pd.DataFrame(rows, columns=["ensemblID", "tss_chrom", "tss_pos"])
    return df


def main():
    RES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("[p0_02] loading MAGE leadVariant_aFCn ...")
    afc = pd.read_csv(AFCN, sep="\t")
    n0 = len(afc)
    log(f"  lead variants (eGenes): {n0:,}")

    tss = load_tss_map()
    log(f"  TSS map genes: {len(tss):,}")

    # Merge TSS onto lead variants by ensemblID (exact version match).
    df = afc.merge(tss, on="ensemblID", how="left")
    n_no_tss = df["tss_pos"].isna().sum()
    df = df.dropna(subset=["tss_pos"]).copy()
    df["tss_pos"] = df["tss_pos"].astype(int)
    log(f"  dropped (no TSS match): {n_no_tss:,} -> {len(df):,}")

    # ---- QC filters (log each step) ----
    def step(mask, name):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        log(f"  filter {name}: {before:,} -> {len(df):,}")

    step(df["variantChrom"].isin(AUTOSOMES), "autosomes only")
    step(df["tss_chrom"] == df["variantChrom"], "variant & TSS same chrom")
    step(np.isfinite(df["log2_aFC"]) & np.isfinite(df["log2_aFC_error"]),
         "finite log2_aFC & error")
    step(df["variantPIP"] >= MIN_PIP, f"PIP >= {MIN_PIP}")

    df["var_tss_dist"] = (df["variantPosition"] - df["tss_pos"]).abs()
    step(df["var_tss_dist"] <= MAX_VAR_TSS_DIST,
         f"|variant-TSS| <= {MAX_VAR_TSS_DIST:,}")

    # ---- Window centered on variant<->TSS midpoint ----
    mid = ((df["variantPosition"] + df["tss_pos"]) // 2).astype(int)
    win_start = (mid - CONTEXT // 2).clip(lower=1)
    df["window_start"] = win_start.astype(int)
    df["window_end"] = (df["window_start"] + CONTEXT).astype(int)
    df["var_rel"] = (df["variantPosition"] - df["window_start"]).astype(int)
    df["tss_rel"] = (df["tss_pos"] - df["window_start"]).astype(int)

    # Sanity: both anchors inside window with margin
    ok = (df["var_rel"].between(CONTEXT // 2 - SAFE_HALF, CONTEXT // 2 + SAFE_HALF)
          & df["tss_rel"].between(CONTEXT // 2 - SAFE_HALF, CONTEXT // 2 + SAFE_HALF))
    step(ok, "both anchors within safe window margin")

    out_cols = [
        "ensemblID", "geneSymbol", "variantChrom", "variantPosition",
        "variantRef", "variantAlt", "variant_kgpID", "variant_rsID",
        "variantPIP", "log2_aFC", "log2_aFC_error",
        "tss_chrom", "tss_pos", "var_tss_dist",
        "window_start", "window_end", "var_rel", "tss_rel",
    ]
    df = df[out_cols].sort_values(["variantChrom", "variantPosition"])
    out = RES / "p0_calibration_set.tsv"
    df.to_csv(out, sep="\t", index=False)
    log(f"\n[p0_02] CALIBRATION SET: {len(df):,} eGene lead variants -> {out}")
    log(f"  log2_aFC range [{df.log2_aFC.min():.2f}, {df.log2_aFC.max():.2f}], "
        f"median |aFC|={df.log2_aFC.abs().median():.3f}")
    log(f"  var_tss_dist median={int(df.var_tss_dist.median()):,}, "
        f"max={int(df.var_tss_dist.max()):,}")

    (LOGS / "p0_02_build_calibration_set.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
