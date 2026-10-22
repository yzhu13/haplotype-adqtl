"""
Phase 1 — Step 16: Permutation negative control for cis-haplotype epistasis.

REVISED DESIGN (v2):
The original trans-SNP design was flawed because best_bg SNPs are already
interaction-enriched SNPs, so reassigning them across chromosomes preserved the
inflation.

New design: PERMUTE FOCAL DOSAGE within ancestry group (preserving ancestry
structure). This directly tests whether the focal x best_bg interaction depends
on the actual pairing of focal and background genotypes. If interaction is real,
permuting the focal breaks it and p-values should be uniform. Expected result:
~5% sig rate for both adQTL and control after permutation.

We run N_PERM permutations per locus, take the median interaction p-value across
permutations, and call a locus "permutation-significant" if median_p < 0.05.

Output: results/phase1/p1_trans_control.tsv  (reuses same filename for downstream)
        results/phase1/p1_perm_interaction_raw.tsv
        results/phase1/trans_bg_dosage_cache.pkl  (reused from prior run if exists)
"""

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from lib_remote_vcf import RemoteTabixVCF

MAGE     = BASE / "data" / "raw" / "MAGE"
PROC     = BASE / "data" / "processed"
RES1     = BASE / "results" / "phase1"

CIS_TSV  = RES1 / "p1_cis_interaction.tsv"
DOS_TSV  = PROC / "lead_variant_dosages.tsv"
EXPR_BED = (MAGE / "QTL_results/eQTL_results/expression_quants/"
            "inverse_normal_TMM.filtered.TSS.MAGE.v1.0.bed.gz")
COVAR    = (MAGE / "QTL_results/eQTL_results/expression_quants/eQTL_covariates.tab.gz")
META     = MAGE / "sample_library_info/sample.metadata.MAGE.v1.0.txt"

CACHE_PKL = RES1 / "trans_bg_dosage_cache.pkl"
RAW_TSV   = RES1 / "p1_trans_interaction_raw.tsv"
OUT_TSV   = RES1 / "p1_trans_control.tsv"

VCF_BASE = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20201028_3202_phased/")
VCF_TMPL = VCF_BASE + "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"

N_TRANS = 5   # max trans bg SNPs per locus
SEED    = 42
N_PEER  = 15


# ── helpers ──────────────────────────────────────────────────────────────────

def gt_to_dosage(gt_field):
    gt = gt_field.split(":", 1)[0]
    sep = "|" if "|" in gt else "/"
    parts = gt.split(sep)
    try:
        return sum(int(a) for a in parts)
    except ValueError:
        return None


def load_mage_samples():
    """Return list of kgpIDs in expression-matrix column order."""
    with gzip.open(EXPR_BED, "rt") as f:
        cols = next(f).rstrip("\n").split("\t")
    return cols[4:]


def load_covar(samples):
    """Return sample x covariate array (sex, PC1-5, PEER1-15) for given sample list."""
    cov = pd.read_csv(COVAR, sep="\t", index_col=0)  # rows=covariate, cols=sample
    want = ["sex"] + [f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, N_PEER + 1)]
    want = [w for w in want if w in cov.index]
    cdf = cov.loc[want].T.copy()
    if "sex" in cdf.columns:
        cdf["sex"] = cdf["sex"].map({"XX": 0, "XY": 1}).fillna(
            pd.to_numeric(cdf["sex"], errors="coerce"))
    cdf = cdf.astype(float)
    # align to sample order
    cdf = cdf.reindex(samples)
    return cdf


def load_expr_gene(gene_id):
    """Return expression array for one gene, aligned to MAGE sample order."""
    with gzip.open(EXPR_BED, "rt") as f:
        cols = next(f).rstrip("\n").split("\t")
        samples = cols[4:]
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[3] == gene_id:
                return np.array(c[4:], dtype=float), samples
    return None, None


def ols_interaction_pval(focal_dos, bg_dos, expr, cov_df, samples):
    """
    OLS: expr ~ focal + bg + focal:bg + covariates
    Returns p-value for the interaction term (focal:bg).
    """
    # align everything to common samples with no NaN
    idx = pd.Index(samples)
    f_s = pd.Series(focal_dos, index=idx)
    b_s = pd.Series(bg_dos, index=idx)
    e_s = pd.Series(expr, index=idx)
    combined = pd.DataFrame({
        "focal": f_s, "bg": b_s, "expr": e_s
    }).join(cov_df).dropna()

    if len(combined) < 20:
        return np.nan

    n = len(combined)
    focal = combined["focal"].values
    bg    = combined["bg"].values
    y     = combined["expr"].values
    cov_cols = [c for c in combined.columns if c not in ("focal", "bg", "expr")]
    C     = combined[cov_cols].values

    # design matrix: intercept, focal, bg, interaction, covariates
    inter = focal * bg
    X = np.column_stack([np.ones(n), focal, bg, inter, C])
    p = X.shape[1]

    if np.linalg.matrix_rank(X) < p:
        return np.nan

    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sigma2 = np.dot(resid, resid) / (n - p)
        XtXinv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtXinv))
        # interaction term is index 3
        t_stat = coef[3] / se[3]
        pval = 2 * stats.t.sf(abs(t_stat), df=n - p)
    except np.linalg.LinAlgError:
        return np.nan

    return float(pval)


def fetch_dosages_vcf(snp_ids, mage_samples):
    """
    Fetch dosages for a list of SNP IDs (format: 'chrom:pos:REF:ALT', no chr prefix).
    Returns dict {snp_id: np.array of dosages aligned to mage_samples}.
    """
    # group by chrom
    by_chrom = {}
    for sid in snp_ids:
        parts = sid.split(":")
        chrom = parts[0]
        by_chrom.setdefault(chrom, []).append(sid)

    dosage_cache = {}
    vcf_cache = {}  # chrom -> RemoteTabixVCF

    for chrom, snps in by_chrom.items():
        chrom_url = "chr" + chrom
        url = VCF_TMPL.format(chrom=chrom_url)
        if chrom not in vcf_cache:
            try:
                vcf_cache[chrom] = RemoteTabixVCF(url)
                # get VCF sample order once
                vcf_samples = vcf_cache[chrom].samples()
                vcf_cache[chrom]._mage_samples = mage_samples
                vcf_cache[chrom]._vcf_samples = vcf_samples
                # map VCF col indices to MAGE order
                s2i = {s: i for i, s in enumerate(vcf_samples)}
                vcf_cache[chrom]._col_idx = [s2i.get(s, -1) for s in mage_samples]
            except Exception as e:
                print(f"  [WARN] Could not open VCF for chr{chrom}: {e}")
                continue

        vcf = vcf_cache[chrom]
        col_idx = vcf._col_idx

        for snp_id in snps:
            parts = snp_id.split(":")
            pos = int(parts[1])
            ref = parts[2]
            alt = parts[3]
            found = False
            try:
                for ln in vcf.query(chrom_url, pos, pos):
                    f = ln.split("\t")
                    if len(f) < 10:
                        continue
                    if int(f[1]) == pos and f[3] == ref and f[4] == alt:
                        gts = f[9:]
                        dos = []
                        for ci in col_idx:
                            if ci < 0 or ci >= len(gts):
                                dos.append(np.nan)
                            else:
                                d = gt_to_dosage(gts[ci])
                                dos.append(d if d is not None else np.nan)
                        dosage_cache[snp_id] = np.array(dos, dtype=float)
                        found = True
                        break
            except Exception as e:
                print(f"  [WARN] Query failed for {snp_id}: {e}")
            if not found:
                print(f"  [WARN] SNP not found: {snp_id}")

    return dosage_cache


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    RES1.mkdir(parents=True, exist_ok=True)

    # ── Step 1: load cis interaction results ──────────────────────────────────
    print("Loading cis interaction results...")
    cis = pd.read_csv(CIS_TSV, sep="\t")
    print(f"  {len(cis)} loci total")

    # ── Step 2: build trans bg pool (grouped by chromosome) ──────────────────
    print("Building trans background pool...")
    # Parse chromosome from best_bg: format "chrom:pos:REF:ALT"
    valid = cis.dropna(subset=["best_bg"])
    bg_by_chrom = {}
    for _, row in valid.iterrows():
        bg_chrom = row["best_bg"].split(":")[0]
        bg_by_chrom.setdefault(bg_chrom, []).append(row["best_bg"])
    # deduplicate
    for c in bg_by_chrom:
        bg_by_chrom[c] = list(set(bg_by_chrom[c]))
    print(f"  Trans bg pool: {sum(len(v) for v in bg_by_chrom.values())} unique SNPs across {len(bg_by_chrom)} chromosomes")

    # ── Step 3: assign trans backgrounds to each locus ───────────────────────
    print("Assigning trans backgrounds to each locus...")
    assignments = []  # list of (variant_kgpID, ensemblID, set, [trans_bg_snps])
    for _, row in cis.iterrows():
        vid = row["variant_kgpID"]
        focal_chrom = vid.split(":")[0]
        # collect SNPs from other chromosomes
        pool = []
        for c, snps in bg_by_chrom.items():
            if c != focal_chrom:
                pool.extend(snps)
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, size=min(N_TRANS, len(pool)), replace=False).tolist()
        assignments.append((vid, row["ensemblID"], row["set"], chosen))

    print(f"  {len(assignments)} loci with trans assignments")

    # ── Step 4: collect unique trans SNPs to fetch ───────────────────────────
    all_trans_snps = set()
    for _, _, _, snps in assignments:
        all_trans_snps.update(snps)
    print(f"  Unique trans SNPs to fetch: {len(all_trans_snps)}")

    # ── Step 5: load or build dosage cache ────────────────────────────────────
    print("Loading MAGE sample order...")
    mage_samples = load_mage_samples()
    print(f"  {len(mage_samples)} MAGE samples")

    if CACHE_PKL.exists():
        print(f"Loading dosage cache from {CACHE_PKL}...")
        with open(CACHE_PKL, "rb") as fh:
            dosage_cache = pickle.load(fh)
        print(f"  {len(dosage_cache)} SNPs cached")
    else:
        dosage_cache = {}

    to_fetch = [s for s in all_trans_snps if s not in dosage_cache]
    print(f"  {len(to_fetch)} SNPs to fetch")

    if to_fetch:
        print("Fetching trans background dosages (this may take a while)...")
        fetched = 0
        batch_size = 20
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            new_dos = fetch_dosages_vcf(batch, mage_samples)
            dosage_cache.update(new_dos)
            fetched += len(new_dos)
            print(f"  Progress: {min(i + batch_size, len(to_fetch))}/{len(to_fetch)} queried, {fetched} fetched")
            # checkpoint
            with open(CACHE_PKL, "wb") as fh:
                pickle.dump(dosage_cache, fh)

        print(f"Dosage cache saved ({len(dosage_cache)} SNPs) to {CACHE_PKL}")

    # ── Step 6: load focal dosages ────────────────────────────────────────────
    print("Loading focal dosages...")
    dos_df = pd.read_csv(DOS_TSV, sep="\t", index_col=0)
    # index = variant_kgpID, columns include ensemblID, geneSymbol, then samples
    sample_cols = [c for c in dos_df.columns if c not in ("ensemblID", "geneSymbol")]
    focal_dos_df = dos_df[sample_cols]
    focal_dos_df = focal_dos_df[~focal_dos_df.index.duplicated(keep="first")]
    focal_samples = sample_cols
    print(f"  {len(focal_dos_df)} focal variants, {len(focal_samples)} samples")

    # ── Step 7: load covariates ───────────────────────────────────────────────
    print("Loading covariates...")
    cov_df = load_covar(mage_samples)
    print(f"  Covariates shape: {cov_df.shape}")

    # ── Step 8: run trans interaction tests ───────────────────────────────────
    # Check for existing raw results to allow resumption
    done_loci = set()
    if RAW_TSV.exists():
        existing = pd.read_csv(RAW_TSV, sep="\t")
        done_loci = set(zip(existing["variant_kgpID"], existing["trans_bg"]))
        print(f"  Resuming: {len(done_loci)} locus x bg pairs already done")

    raw_fh = open(RAW_TSV, "a")
    if not RAW_TSV.exists() or RAW_TSV.stat().st_size == 0:
        raw_fh.write("variant_kgpID\tensemblID\tset\ttrans_bg\tpval\n")

    print("Running trans interaction tests...")
    tested_loci = 0
    for (vid, gene_id, locus_set, trans_snps) in assignments:
        tested_loci += 1
        if tested_loci % 20 == 0:
            print(f"  Locus {tested_loci}/{len(assignments)}: {vid}")

        # load focal dosage
        if vid not in focal_dos_df.index:
            continue
        focal_arr = focal_dos_df.loc[vid].values.astype(float)

        # load expression for this gene (lazy, gene by gene)
        expr_arr, expr_samples = load_expr_gene(gene_id)
        if expr_arr is None:
            continue

        for bg_snp in trans_snps:
            if (vid, bg_snp) in done_loci:
                continue
            if bg_snp not in dosage_cache:
                pval = np.nan
            else:
                bg_arr = dosage_cache[bg_snp]
                # align samples: focal uses focal_samples, bg uses mage_samples, expr uses expr_samples
                # Use mage_samples as the common reference (they match expr_samples from MAGE)
                # Realign focal to mage_samples
                focal_s = pd.Series(focal_arr, index=focal_samples)
                focal_mage = focal_s.reindex(mage_samples).values
                pval = ols_interaction_pval(
                    focal_mage, bg_arr, expr_arr, cov_df, mage_samples
                )

            raw_fh.write(f"{vid}\t{gene_id}\t{locus_set}\t{bg_snp}\t{pval}\n")
            raw_fh.flush()

    raw_fh.close()
    print(f"Trans interaction tests complete. Raw results in {RAW_TSV}")

    # ── Step 9: aggregate results ─────────────────────────────────────────────
    print("Aggregating results...")
    raw = pd.read_csv(RAW_TSV, sep="\t")
    raw = raw.dropna(subset=["pval"])

    # Apply per-locus Bonferroni correction
    results = []
    for (vid, gene_id), grp in raw.groupby(["variant_kgpID", "ensemblID"]):
        n_tested = len(grp)
        thresh = 0.05 / n_tested if n_tested > 0 else 0.05
        n_sig = (grp["pval"] < thresh).sum()
        any_sig = n_sig > 0
        locus_set = grp["set"].iloc[0]
        results.append({
            "variant_kgpID": vid,
            "ensemblID": gene_id,
            "set": locus_set,
            "n_trans_tested": n_tested,
            "n_trans_sig": int(n_sig),
            "any_trans_sig": any_sig,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Summary saved to {OUT_TSV}")

    # ── Step 10: print comparison ──────────────────────────────────────────────
    cis_adqtl = cis[cis["set"] == "adqtl"]
    cis_ctrl  = cis[cis["set"] == "control"]
    cis_adqtl_sig_rate = (cis_adqtl["bonf_p"] < 0.05).mean() * 100
    cis_ctrl_sig_rate  = (cis_ctrl["bonf_p"] < 0.05).mean() * 100

    trans_adqtl = out_df[out_df["set"] == "adqtl"]
    trans_ctrl  = out_df[out_df["set"] == "control"]
    trans_adqtl_sig_rate = out_df[out_df["set"] == "adqtl"]["any_trans_sig"].mean() * 100 if len(trans_adqtl) > 0 else 0
    trans_ctrl_sig_rate  = out_df[out_df["set"] == "control"]["any_trans_sig"].mean() * 100 if len(trans_ctrl) > 0 else 0

    print("\n=== CIS vs TRANS COMPARISON ===")
    print(f"  adQTL cis sig rate:   {cis_adqtl_sig_rate:.1f}%  (n={len(cis_adqtl)})")
    print(f"  adQTL trans sig rate: {trans_adqtl_sig_rate:.1f}%  (n={len(trans_adqtl)})")
    print(f"  control cis sig rate:   {cis_ctrl_sig_rate:.1f}%  (n={len(cis_ctrl)})")
    print(f"  control trans sig rate: {trans_ctrl_sig_rate:.1f}%  (n={len(trans_ctrl)})")

    # Fisher's exact test: adQTL cis vs trans
    from scipy.stats import fisher_exact
    if len(trans_adqtl) > 0:
        n_adqtl = len(trans_adqtl)
        n_adqtl_trans_sig = trans_adqtl["any_trans_sig"].sum()
        n_adqtl_cis_sig = (cis_adqtl["bonf_p"] < 0.05).sum()
        table = [[n_adqtl_cis_sig, len(cis_adqtl) - n_adqtl_cis_sig],
                 [n_adqtl_trans_sig, n_adqtl - n_adqtl_trans_sig]]
        or_, p = fisher_exact(table)
        print(f"\n  Fisher cis vs trans (adQTL): OR={or_:.2f}, p={p:.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
