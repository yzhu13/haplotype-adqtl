"""
Phase 1 - Step 12: Haplotype Context Divergence (HCD) analysis.

For each locus in p1_cis_interaction.tsv, compute Jensen-Shannon divergence
of the background SNP dosage distribution between EUR and AFR individuals
who CARRY the focal alt allele (conditional HCD) and among all individuals
(unconditional HCD).

Output: results/phase1/p1_hcd.tsv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_remote_vcf import RemoteTabixVCF

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
RES1 = BASE / "results" / "phase1"
PROC = BASE / "data" / "processed"
MAGE = BASE / "data" / "raw" / "MAGE"
META = MAGE / "sample_library_info" / "sample.metadata.MAGE.v1.0.txt"

INT_TSV = RES1 / "p1_cis_interaction.tsv"
DOS_TSV = PROC / "lead_variant_dosages.tsv"
REF_TSV = RES1 / "p1_adqtl_refscore.tsv"
OUT = RES1 / "p1_hcd.tsv"

VCF_BASE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20201028_3202_phased/")
VCF_TMPL = VCF_BASE + "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"


# ---- helpers ----------------------------------------------------------------

def dosage_distribution(dosages):
    """Convert array of dosages to [freq(0), freq(1), freq(2)] with Laplace smoothing."""
    counts = np.array(
        [(dosages == 0).sum(), (dosages == 1).sum(), (dosages == 2).sum()],
        dtype=float
    )
    counts += 0.5  # Laplace smoothing
    return counts / counts.sum()


def compute_hcd(focal_dos_eur, focal_dos_afr, bg_dos_eur, bg_dos_afr):
    """HCD conditioned on carrying focal alt allele (dosage >= 1)."""
    eur_carriers = focal_dos_eur >= 1
    afr_carriers = focal_dos_afr >= 1
    if eur_carriers.sum() < 5 or afr_carriers.sum() < 5:
        return np.nan
    p_eur = dosage_distribution(bg_dos_eur[eur_carriers])
    p_afr = dosage_distribution(bg_dos_afr[afr_carriers])
    return float(jensenshannon(p_eur, p_afr))


def compute_hcd_unconditional(bg_dos_eur, bg_dos_afr):
    """HCD unconditional (all individuals)."""
    if len(bg_dos_eur) < 5 or len(bg_dos_afr) < 5:
        return np.nan
    p_eur = dosage_distribution(bg_dos_eur)
    p_afr = dosage_distribution(bg_dos_afr)
    return float(jensenshannon(p_eur, p_afr))


def gt_to_dosage(gt_field):
    """'a|b[:...]' -> alt-allele count (0/1/2), or np.nan if missing."""
    gt = gt_field.split(":", 1)[0]
    sep = "|" if "|" in gt else "/"
    parts = gt.split(sep)
    try:
        return sum(1 for p in parts if p not in (".", "0"))
    except Exception:
        return np.nan


def fetch_dosages_at(chrom, pos, variant_id, vcf, sample_order, sample_set):
    """
    Fetch dosages for `variant_id` at chrom:pos from remote VCF.
    variant_id format: "chrom:pos:REF:ALT" (no chr prefix in ID).
    Matches by position + REF/ALT alleles (VCF ID column is usually ".").
    Returns dict {sample_id: dosage} for samples in sample_set, or None.
    """
    vcf_samples = vcf.samples()
    col_pos = {s: i for i, s in enumerate(vcf_samples) if s in sample_set}
    if not col_pos:
        return None

    # Parse expected REF/ALT from variant_id
    parts = variant_id.split(":")
    expected_ref = parts[2] if len(parts) > 2 else None
    expected_alt = parts[3] if len(parts) > 3 else None

    for ln in vcf.query(chrom, pos, pos):
        f = ln.split("\t")
        if int(f[1]) != pos:
            continue
        # Match by alleles if available; otherwise match by position only
        if expected_ref and expected_alt:
            if f[3] != expected_ref or f[4] != expected_alt:
                continue
        gts = f[9:]
        dosages = {}
        for s, idx in col_pos.items():
            if idx < len(gts):
                d = gt_to_dosage(gts[idx])
                dosages[s] = d if d is not None else np.nan
        return dosages
    return None


# ---- VCF cache (one per chrom) ---------------------------------------------

_vcf_cache = {}

def get_vcf(chrom):
    if chrom not in _vcf_cache:
        url = VCF_TMPL.format(chrom=chrom)
        print(f"  Opening VCF for {chrom} ...", flush=True)
        _vcf_cache[chrom] = RemoteTabixVCF(url)
    return _vcf_cache[chrom]


# ---- main -------------------------------------------------------------------

def main():
    # Checkpoint
    if OUT.exists():
        existing = pd.read_csv(OUT, sep="\t")
        if len(existing) > 50:
            print(f"Checkpoint: {OUT} exists with {len(existing)} rows, skipping to figure.")
            return

    print("Loading interaction table ...", flush=True)
    int_df = pd.read_csv(INT_TSV, sep="\t")
    print(f"  Total rows: {len(int_df)}")
    print(f"  Columns: {list(int_df.columns)}")

    # Filter to loci with valid best_bg
    int_df = int_df[int_df["n_bg_tested"] > 0].copy()
    int_df = int_df[int_df["best_bg"].notna()].copy()
    print(f"  After filtering n_bg_tested>0 and best_bg valid: {len(int_df)}")

    # Load ObsDelta from refscore file
    print("Loading ObsDelta from refscore ...", flush=True)
    ref_df = pd.read_csv(REF_TSV, sep="\t")[
        ["variant_kgpID", "ensemblID", "set", "ObsDelta_EUR_AFR"]
    ]
    # Merge to get ObsDelta; use int_df set for rows not in ref_df
    int_df = int_df.merge(
        ref_df.rename(columns={"set": "set_ref", "ObsDelta_EUR_AFR": "ObsDelta_EUR_AFR"}),
        on=["variant_kgpID", "ensemblID"], how="left"
    )

    # Load sample metadata
    print("Loading sample metadata ...", flush=True)
    meta = pd.read_csv(META, sep="\t")
    eur_ids = set(meta.loc[meta["continentalGroup"] == "EUR", "sample_kgpID"])
    afr_ids = set(meta.loc[meta["continentalGroup"] == "AFR", "sample_kgpID"])
    print(f"  EUR samples: {len(eur_ids)}, AFR samples: {len(afr_ids)}")

    # Load focal dosages
    print("Loading focal dosages ...", flush=True)
    dos_df = pd.read_csv(DOS_TSV, sep="\t")
    dos_samples = list(dos_df.columns[3:])  # skip variant_kgpID, ensemblID, geneSymbol
    dos_eur_cols = [s for s in dos_samples if s in eur_ids]
    dos_afr_cols = [s for s in dos_samples if s in afr_ids]
    print(f"  Dosage EUR samples: {len(dos_eur_cols)}, AFR: {len(dos_afr_cols)}")

    all_samples = set(dos_samples)
    eur_in_dos = set(dos_eur_cols)
    afr_in_dos = set(dos_afr_cols)

    results = []
    n_loci = len(int_df)
    t0 = time.time()

    for i, row in enumerate(int_df.itertuples(index=False)):
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{n_loci}] elapsed={elapsed:.0f}s", flush=True)

        vid = row.variant_kgpID          # e.g. "1:248444054:A:G"
        eid = row.ensemblID
        grp = row.set
        best_bg = row.best_bg            # e.g. "1:248471564:A:C"
        bonf_p = row.bonf_p
        int_beta = row.int_beta
        obs_delta = getattr(row, "ObsDelta_EUR_AFR", np.nan)

        # Parse chrom/pos from variant_kgpID
        try:
            parts = vid.split(":")
            chrom_focal = "chr" + parts[0]
            pos_focal = int(parts[1])
        except Exception:
            print(f"  SKIP {vid}: cannot parse variant_kgpID", flush=True)
            continue

        # Get focal dosages from pre-computed file
        focal_row = dos_df[dos_df["variant_kgpID"] == vid]
        if focal_row.empty:
            print(f"  SKIP {vid}: not in dosage file", flush=True)
            continue
        focal_row = focal_row.iloc[0]

        focal_dos_eur = np.array(
            [focal_row[s] if s in focal_row.index else np.nan for s in dos_eur_cols],
            dtype=float
        )
        focal_dos_afr = np.array(
            [focal_row[s] if s in focal_row.index else np.nan for s in dos_afr_cols],
            dtype=float
        )
        # Drop NaN
        focal_dos_eur = focal_dos_eur[~np.isnan(focal_dos_eur)]
        focal_dos_afr = focal_dos_afr[~np.isnan(focal_dos_afr)]

        # Parse background SNP
        try:
            bg_parts = best_bg.split(":")
            chrom_bg = "chr" + bg_parts[0]
            pos_bg = int(bg_parts[1])
        except Exception:
            print(f"  SKIP {vid} bg={best_bg}: cannot parse best_bg", flush=True)
            continue

        # Fetch background SNP dosages from VCF (always use chr prefix)
        try:
            vcf_bg = get_vcf(chrom_bg)
            bg_dosages = fetch_dosages_at(
                chrom_bg, pos_bg, best_bg, vcf_bg,
                dos_samples, all_samples
            )
        except Exception as e:
            print(f"  SKIP {vid} bg={best_bg}: VCF error {e}", flush=True)
            continue

        if bg_dosages is None:
            print(f"  SKIP {vid}: best_bg {best_bg} not found in VCF at {chrom_bg}:{pos_bg}", flush=True)
            continue

        # Build dosage arrays for EUR and AFR
        bg_dos_eur = np.array(
            [bg_dosages.get(s, np.nan) for s in dos_eur_cols], dtype=float
        )
        bg_dos_afr = np.array(
            [bg_dosages.get(s, np.nan) for s in dos_afr_cols], dtype=float
        )

        # Drop NaN positions
        eur_valid = ~np.isnan(bg_dos_eur)
        afr_valid = ~np.isnan(bg_dos_afr)

        # Align focal and bg dosages for EUR
        n_eur = min(len(focal_dos_eur), eur_valid.sum())
        n_afr = min(len(focal_dos_afr), afr_valid.sum())

        # Use consistent sample ordering: iterate over columns
        fe_list, be_list = [], []
        for j, s in enumerate(dos_eur_cols):
            if j < len(focal_dos_eur) and not np.isnan(focal_dos_eur[j]) and not np.isnan(bg_dos_eur[j]):
                fe_list.append(focal_dos_eur[j])
                be_list.append(bg_dos_eur[j])

        fa_list, ba_list = [], []
        for j, s in enumerate(dos_afr_cols):
            if j < len(focal_dos_afr) and not np.isnan(focal_dos_afr[j]) and not np.isnan(bg_dos_afr[j]):
                fa_list.append(focal_dos_afr[j])
                ba_list.append(bg_dos_afr[j])

        fe_arr = np.array(fe_list)
        be_arr = np.array(be_list)
        fa_arr = np.array(fa_list)
        ba_arr = np.array(ba_list)

        hcd_cond = compute_hcd(fe_arr, fa_arr, be_arr, ba_arr)
        hcd_uncond = compute_hcd_unconditional(
            bg_dos_eur[~np.isnan(bg_dos_eur)],
            bg_dos_afr[~np.isnan(bg_dos_afr)]
        )

        results.append({
            "variant_kgpID": vid,
            "ensemblID": eid,
            "set": grp,
            "hcd_conditional": hcd_cond,
            "hcd_unconditional": hcd_uncond,
            "obs_delta": obs_delta,
            "bonf_p": bonf_p,
            "int_beta": int_beta,
            "n_eur_carriers": int((fe_arr >= 1).sum()) if len(fe_arr) > 0 else 0,
            "n_afr_carriers": int((fa_arr >= 1).sum()) if len(fa_arr) > 0 else 0,
        })

    print(f"Done. {len(results)} loci computed.", flush=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT, sep="\t", index=False)
    print(f"Saved: {OUT}", flush=True)

    # Quick summary
    for grp in ["adqtl", "control"]:
        sub = out_df[out_df["set"] == grp]["hcd_conditional"].dropna()
        print(f"  {grp}: n={len(sub)}, median_hcd={sub.median():.4f}", flush=True)


if __name__ == "__main__":
    main()
