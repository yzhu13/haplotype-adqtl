"""
Phase 0 — Step 01: Acquire the MAGE cross-ancestry eQTL ground-truth subset.

MAGE (Taylor et al. 2024 Nature; Zenodo 10.5281/zenodo.10535719) is distributed
as ONE 59.5 GB zip. Downloading it whole is wasteful — for Phase 0 we only need
the cis-eQTL / aFC results, sample-to-population metadata, and genotype PCs.

Strategy: use HTTP range requests (`remotezip`) to read only the zip's central
directory, list all members, then stream-extract just the members we need. No
59.5 GB download, fully reproducible.

Outputs:
  data/raw/MAGE/<selected files>
  logs/mage_zenodo_manifest.tsv   (full member listing: name, size, compressed)
  logs/p0_01_fetch_mage.provenance.txt  (URL, date, caveats, what was pulled)

⚠️ KNOWN CAVEAT (recorded 2026-06-10): MAGE has a documented sample swap between
   libraries SRR19762530 and SRR19762653 (detected 2026-03-27); Zenodo not yet
   updated. Downstream analysis must drop/flag these two libraries.

Usage:
  # 1) List the archive contents only (no extraction) — do this first:
  python p0_01_fetch_mage.py --list-only

  # 2) Extract the default eQTL/metadata subset:
  python p0_01_fetch_mage.py

  # 3) Extract members matching a custom regex:
  python p0_01_fetch_mage.py --pattern "eQTL|aFC|metadata|genotypePC"
"""

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "MAGE"
LOGS = BASE / "logs"

ZENODO_RECORD = "10535719"
# Zenodo direct file-content endpoint (supports HTTP range requests).
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/MAGE.v1.0.data.zip"

# Curated Phase-0 subset (refined after inspecting the 111-member manifest on
# 2026-06-10). Each token is a distinctive path substring; we deliberately AVOID
# the 2-35 GB allAssociations / nominal_pass / sQTL files, which Phase 0 does not
# need. `.tbi` index files are skipped separately (we read whole small files).
PHASE0_TOKENS = [
    r"leadVariant_aFCn",                       # ★ per-eGene aFC effect sizes (core ground truth)
    r"eQTL_finemapping\.credibleSet_summary",  # fine-mapped credible sets (eQTL only)
    r"eQTL_finemapping\.significantAssociations",
    r"eQTL_summary\.MAGE",
    r"expression_filteredGenes",               # eGene list
    r"inverse_normal_TMM\.filtered\.TSS",      # QTL phenotype matrix (49 MB) for per-ancestry beta
    r"eQTL_covariates",
    r"global_expression_trends/contGroup_variance",  # per-continental-group expression variance
    r"Finemapped_Fst",                         # ★ Fst of fine-mapped variants (ancestry differentiation)
    r"chromHMM_annotation\.GM12878",           # lead-variant chromatin state (ATAC anchoring)
    r"PAGE\.eQTL_colocalization",              # Phase-3 multi-ancestry GWAS coloc
    r"/README$",                               # all dataset READMEs (tiny, document everything)
    r"sample\.metadata",                       # ★ individual -> population / continental group
    r"sequencing_library\.metadata",
]
DEFAULT_PATTERN = "(" + "|".join(PHASE0_TOKENS) + ")"

# Sample-swap caveat to record in provenance.
SAMPLE_SWAP_CAVEAT = (
    "Sample swap (detected 2026-03-27) between SRR19762530 and SRR19762653; "
    "Zenodo not yet corrected as of 2026-06-10. Drop/flag these libraries downstream."
)


def _require_remotezip():
    try:
        from remotezip import RemoteZip  # noqa: F401
        return
    except ImportError:
        sys.exit(
            "ERROR: this script needs `remotezip` for partial extraction.\n"
            "  pip install remotezip\n"
            "It reads only the zip central directory + the members you ask for\n"
            "via HTTP range requests (avoids the 59.5 GB full download)."
        )


def write_manifest(members):
    """members: list of zipfile.ZipInfo. Write a TSV manifest to logs/."""
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "mage_zenodo_manifest.tsv"
    lines = ["filename\tsize_bytes\tcompressed_bytes"]
    for m in sorted(members, key=lambda z: z.filename):
        lines.append(f"{m.filename}\t{m.file_size}\t{m.compress_size}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[manifest] {len(members)} members -> {out}")
    return out


def write_provenance(pattern, extracted):
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / "p0_01_fetch_mage.provenance.txt"
    now = dt.datetime.now().isoformat(timespec="seconds")
    body = [
        "MAGE Phase-0 acquisition provenance",
        f"date: {now}",
        f"source_zip: {ZIP_URL}",
        f"zenodo_record: {ZENODO_RECORD} (DOI 10.5281/zenodo.{ZENODO_RECORD})",
        f"citation: Taylor et al. 2024 Nature (s41586-024-07708-2)",
        f"method: partial extraction via HTTP range (remotezip)",
        f"select_pattern: {pattern}",
        f"CAVEAT: {SAMPLE_SWAP_CAVEAT}",
        "",
        f"extracted {len(extracted)} members:",
        *[f"  - {f}" for f in extracted],
    ]
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"[provenance] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true",
                    help="Only fetch the central directory and write the manifest; no extraction.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help="Case-insensitive regex selecting member paths to extract.")
    ap.add_argument("--max-size-gb", type=float, default=1.0,
                    help="Safety cap: refuse to extract any single member larger than this.")
    ap.add_argument("--keep-tbi", action="store_true",
                    help="Also extract .tbi tabix indices (skipped by default; not needed for whole-file reads).")
    args = ap.parse_args()

    _require_remotezip()
    from remotezip import RemoteZip

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"[open] reading central directory of remote zip...\n  {ZIP_URL}")

    with RemoteZip(ZIP_URL) as rz:
        members = rz.infolist()
        write_manifest(members)

        if args.list_only:
            # Print a compact preview so the user can refine --pattern.
            print("\n[preview] first 40 members:")
            for m in sorted(members, key=lambda z: z.filename)[:40]:
                print(f"  {m.file_size/1e6:9.2f} MB  {m.filename}")
            print("\n--list-only: nothing extracted. Inspect logs/mage_zenodo_manifest.tsv,"
                  " then re-run with a refined --pattern.")
            return

        rx = re.compile(args.pattern, re.IGNORECASE)
        cap = args.max_size_gb * 1e9
        selected = [m for m in members
                    if rx.search(m.filename)
                    and not m.filename.endswith("/")            # skip dir entries
                    and (args.keep_tbi or not m.filename.endswith(".tbi"))]
        if not selected:
            sys.exit(f"No members matched pattern: {args.pattern!r}. "
                     f"Inspect logs/mage_zenodo_manifest.tsv and refine.")

        print(f"\n[select] {len(selected)} members match {args.pattern!r}:")
        extracted = []
        for m in sorted(selected, key=lambda z: z.file_size):
            if m.file_size > cap:
                print(f"  SKIP (>{args.max_size_gb}GB): {m.filename} "
                      f"({m.file_size/1e9:.1f} GB)")
                continue
            # Preserve archive-relative path (minus the top "MAGE.v1.0.data/"),
            # so the many same-named README files don't collide.
            rel = re.sub(r"^MAGE\.v1\.0\.data/", "", m.filename)
            dest = RAW / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == m.file_size:
                print(f"  exists, skip: {dest.name}")
                extracted.append(m.filename)
                continue
            print(f"  extracting {m.file_size/1e6:.2f} MB: {m.filename}")
            with rz.open(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(m.filename)

        write_provenance(args.pattern, extracted)
        print(f"\n[done] extracted {len(extracted)} files to {RAW}")


if __name__ == "__main__":
    main()
