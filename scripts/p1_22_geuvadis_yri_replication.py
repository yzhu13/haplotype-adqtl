"""
p1_22_geuvadis_yri_replication.py

Independent YRI replication of MAGE adQTL signals using Geuvadis GD462.

For each adQTL and matched control locus:
  1. Fetch focal variant dosages from NYGC 1000G phased VCF (Geuvadis samples)
  2. Estimate EUR and YRI per-ancestry eQTL effect sizes (OLS with PCA covariates)
  3. Compute EUR-YRI heterogeneity z-score (z_het) and p-value
  4. Compare adQTL vs control loci for enrichment of heterogeneity

Output: results/phase1/p1_geuvadis_yri_replication.tsv
        results/phase1/p1_geuvadis_yri_checkpoint.pkl
"""

import gzip
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.decomposition import PCA

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from lib_remote_vcf import RemoteTabixVCF

CHECKPOINT = BASE / "results" / "phase1" / "p1_geuvadis_yri_checkpoint.pkl"
OUT_TSV    = BASE / "results" / "phase1" / "p1_geuvadis_yri_replication.tsv"

GEUV_DIR   = BASE / "data" / "raw" / "Geuvadis"
EXPR_GZ    = GEUV_DIR / "GD462.GeneQuantRPKM.txt.gz"
PANEL_TSV  = GEUV_DIR / "1000G_panel.txt"
SEX_TSV    = GEUV_DIR / "geuv_sex.txt"

ADQTL_TSV  = BASE / "results" / "phase0" / "p0_adqtl_set.tsv"
CTRL_TSV   = BASE / "results" / "phase0" / "p0_control_set.tsv"

VCF_TMPL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)

CHECKPOINT_EVERY = 20
MIN_EUR = 20
MIN_YRI = 10

EUR_POPS = {"CEU", "FIN", "GBR", "TSI", "IBS"}
YRI_POPS = {"YRI"}


# ─── helpers ────────────────────────────────────────────────────────────────

def inv_normal(x):
    """Rank-based inverse normal transform."""
    r = rankdata(x) / (len(x) + 1)
    return stats.norm.ppf(r)


def ols_beta_se(expr_vals, dosage_vals, cov_matrix):
    """OLS: expr ~ dosage + covariates. Returns (beta, se, pval)."""
    X = np.column_stack([np.ones(len(expr_vals)), dosage_vals, cov_matrix])
    coef, _, _, _ = np.linalg.lstsq(X, expr_vals, rcond=None)
    resid = expr_vals - X @ coef
    n, p = X.shape
    if n <= p:
        return np.nan, np.nan, np.nan
    sigma2 = np.dot(resid, resid) / (n - p)
    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    se_arr = np.sqrt(sigma2 * np.diag(XtXinv))
    beta = coef[1]
    se   = se_arr[1]
    if se == 0 or not np.isfinite(se):
        return np.nan, np.nan, np.nan
    t    = beta / se
    pval = 2 * stats.t.sf(abs(t), df=n - p)
    return float(beta), float(se), float(pval)


def parse_variant(vkgp):
    """Parse 'chrN:pos:REF:ALT' or 'N:pos:REF:ALT'. Returns (chrom_str, pos_int, ref, alt)."""
    parts = vkgp.split(":")
    chrom = parts[0].lstrip("chr")  # keep bare number for VCF_TMPL
    pos   = int(parts[1])
    ref   = parts[2] if len(parts) > 2 else None
    alt   = parts[3] if len(parts) > 3 else None
    return chrom, pos, ref, alt


def get_dosage_from_vcf(vcf, sample_idx_map, chrom, pos, ref, alt):
    """
    Fetch dosage vector for the variant at chrom:pos REF/ALT.
    Returns dict {sample_id: dosage} or None if not found.
    """
    for line in vcf.query(chrom, pos, pos):
        fields = line.split("\t")
        if int(fields[1]) != pos:
            continue
        vcf_ref = fields[3]
        vcf_alts = fields[4].split(",")
        # Match alleles
        if ref and alt:
            if vcf_ref != ref:
                continue
            if alt not in vcf_alts:
                continue
            alt_idx = vcf_alts.index(alt) + 1  # 1-based in GT
        else:
            alt_idx = 1  # assume first alt

        fmt_fields = fields[8].split(":")
        gt_idx = fmt_fields.index("GT") if "GT" in fmt_fields else 0
        dosages = {}
        for samp_id, col_idx in sample_idx_map.items():
            gt_str = fields[col_idx + 9].split(":")[gt_idx]
            alleles = gt_str.replace("|", "/").split("/")
            try:
                dos = sum(1 for a in alleles if a != "." and int(a) == alt_idx)
                dosages[samp_id] = dos
            except ValueError:
                dosages[samp_id] = np.nan
        return dosages
    return None


# ─── Load expression ─────────────────────────────────────────────────────────

print("Loading Geuvadis expression...")
with gzip.open(EXPR_GZ, "rt") as f:
    expr_raw = pd.read_csv(f, sep="\t", index_col=0)

# Drop non-sample metadata columns
META_COLS = {"Gene_Symbol", "Chr", "Coord"}
sample_cols = [c for c in expr_raw.columns if c not in META_COLS]
expr_raw = expr_raw[sample_cols]
print(f"  Expression: {expr_raw.shape[0]} genes x {len(sample_cols)} samples")

# ─── Load population panel and sex ───────────────────────────────────────────

panel = pd.read_csv(PANEL_TSV, sep="\t")
panel = panel[["sample", "pop", "gender"]].set_index("sample")
# Also load sex supplement
sex_df = pd.read_csv(SEX_TSV, sep="\t").set_index("sample")

# Assign ancestry labels
eur_samples_all = [s for s in sample_cols if s in panel.index and panel.loc[s, "pop"] in EUR_POPS]
yri_samples_all = [s for s in sample_cols if s in panel.index and panel.loc[s, "pop"] in YRI_POPS]
print(f"  EUR samples in expression: {len(eur_samples_all)}")
print(f"  YRI samples in expression: {len(yri_samples_all)}")

all_geuv_samples = eur_samples_all + yri_samples_all

# ─── Expression preprocessing ────────────────────────────────────────────────

# Log2 transform on all samples together (for PCA)
expr_log = np.log2(expr_raw[all_geuv_samples] + 0.1)

# PCA on all samples (genes x samples -> transpose to samples x genes)
print("Running PCA on combined expression...")
pca_data = expr_log.T.values  # shape: (n_samples, n_genes)
# Remove genes with zero variance
variances = np.var(pca_data, axis=0)
pca_data_filt = pca_data[:, variances > 0]
pca = PCA(n_components=5)
pcs_all = pca.fit_transform(pca_data_filt)  # (n_samples, 5)
print(f"  PCA done. Variance explained: {pca.explained_variance_ratio_}")

# Map sample -> PC row index
sample_to_pcidx = {s: i for i, s in enumerate(all_geuv_samples)}

n_eur = len(eur_samples_all)
n_yri = len(yri_samples_all)
pcs_eur = pcs_all[:n_eur]  # (n_eur, 5)
pcs_yri = pcs_all[n_eur:]  # (n_yri, 5)

# Sex covariate
def get_sex_cov(slist):
    sex_arr = []
    for s in slist:
        # try sex_df first, then panel gender
        if s in sex_df.index:
            val = 1.0 if sex_df.loc[s, "sex"] == "female" else 0.0
        elif s in panel.index:
            val = 1.0 if panel.loc[s, "gender"] == "female" else 0.0
        else:
            val = 0.5
        sex_arr.append(val)
    return np.array(sex_arr)

sex_eur = get_sex_cov(eur_samples_all)
sex_yri = get_sex_cov(yri_samples_all)

# Full covariate matrix: PC1-5 + sex
cov_eur = np.column_stack([pcs_eur, sex_eur])
cov_yri = np.column_stack([pcs_yri, sex_yri])

# ─── Load adQTL and control sets ─────────────────────────────────────────────

adqtl_df = pd.read_csv(ADQTL_TSV, sep="\t")
ctrl_df  = pd.read_csv(CTRL_TSV,  sep="\t")
adqtl_df["set"] = "adqtl"
ctrl_df["set"]  = "control"
loci_df = pd.concat([adqtl_df, ctrl_df], ignore_index=True)
print(f"\nLoci to test: {len(adqtl_df)} adQTL + {len(ctrl_df)} control = {len(loci_df)} total")

# Strip gene version for matching
def strip_version(eid):
    return eid.split(".")[0] if isinstance(eid, str) else eid

# Geuvadis gene index (strip version)
gene_index = {strip_version(g): g for g in expr_raw.index}

# ─── Load checkpoint ─────────────────────────────────────────────────────────

if CHECKPOINT.exists():
    with open(CHECKPOINT, "rb") as f:
        done_results = pickle.load(f)
    done_keys = {(r["variant_kgpID"], r["ensemblID"]) for r in done_results}
    print(f"Resuming from checkpoint: {len(done_results)} loci already done")
else:
    done_results = []
    done_keys = set()

# ─── VCF cache ───────────────────────────────────────────────────────────────

vcf_cache = {}  # chrom -> RemoteTabixVCF
samp_idx_cache = {}  # chrom -> {sample: col_index}

def get_vcf(chrom):
    if chrom not in vcf_cache:
        url = VCF_TMPL.format(chrom=chrom)
        print(f"  Opening VCF for chr{chrom}...")
        vcf_cache[chrom] = RemoteTabixVCF(url)
        # Build sample index map for Geuvadis samples
        vcf_samples = vcf_cache[chrom].samples()
        samp_idx_cache[chrom] = {
            s: i for i, s in enumerate(vcf_samples) if s in set(all_geuv_samples)
        }
        print(f"    VCF has {len(vcf_samples)} samples, {len(samp_idx_cache[chrom])} overlap Geuvadis")
    return vcf_cache[chrom], samp_idx_cache[chrom]

# ─── Main loop ───────────────────────────────────────────────────────────────

results = list(done_results)
n_new = 0

for idx, row in loci_df.iterrows():
    vkgp    = row["variant_kgpID"]
    ens_ver = row["ensemblID"]
    ens_id  = strip_version(ens_ver)
    locus_set = row["set"]

    key = (vkgp, ens_ver)
    if key in done_keys:
        continue

    # Gene available in Geuvadis?
    if ens_id not in gene_index:
        print(f"  [{idx}] Gene {ens_id} not in Geuvadis -> skip")
        continue

    geuv_gene_id = gene_index[ens_id]

    # Fetch dosages
    try:
        chrom, pos, ref, alt = parse_variant(vkgp)
        vcf, samp_idx_map = get_vcf(chrom)
        dos_dict = get_dosage_from_vcf(vcf, samp_idx_map, chrom, pos, ref, alt)
    except Exception as e:
        print(f"  [{idx}] VCF error for {vkgp}: {e}")
        continue

    if dos_dict is None:
        print(f"  [{idx}] Variant {vkgp} not found in VCF -> skip")
        continue

    # Build dosage arrays for EUR and YRI
    def build_dos(slist):
        d = np.array([dos_dict.get(s, np.nan) for s in slist], dtype=float)
        return d

    dos_eur_raw = build_dos(eur_samples_all)
    dos_yri_raw = build_dos(yri_samples_all)

    # Filter out missing
    mask_e = np.isfinite(dos_eur_raw)
    mask_y = np.isfinite(dos_yri_raw)

    if mask_e.sum() < MIN_EUR or mask_y.sum() < MIN_YRI:
        print(f"  [{idx}] {vkgp}: EUR n={mask_e.sum()}, YRI n={mask_y.sum()} -> skip (min {MIN_EUR}/{MIN_YRI})")
        continue

    # Expression (log2 already done, now inv-normal within each ancestry)
    expr_row = expr_raw.loc[geuv_gene_id]

    expr_eur_raw = np.array([expr_row[s] for s in eur_samples_all], dtype=float)
    expr_yri_raw = np.array([expr_row[s] for s in yri_samples_all], dtype=float)

    # Filter by dosage availability
    expr_eur = expr_eur_raw[mask_e]
    dos_eur  = dos_eur_raw[mask_e]
    cov_e    = cov_eur[mask_e]

    expr_yri = expr_yri_raw[mask_y]
    dos_yri  = dos_yri_raw[mask_y]
    cov_y    = cov_yri[mask_y]

    # Median filter: gene expressed?
    log_e = np.log2(expr_eur + 0.1)
    log_y = np.log2(expr_yri + 0.1)
    if np.median(log_e) < np.log2(0.1 + 0.1) or np.median(log_y) < np.log2(0.1 + 0.1):
        # gene barely expressed; still include, just warn
        pass

    # Inverse normal transform
    expr_eur_int = inv_normal(log_e)
    expr_yri_int = inv_normal(log_y)

    # OLS
    beta_eur, se_eur, p_eur = ols_beta_se(expr_eur_int, dos_eur, cov_e)
    beta_yri, se_yri, p_yri = ols_beta_se(expr_yri_int, dos_yri, cov_y)

    if not (np.isfinite(beta_eur) and np.isfinite(beta_yri)):
        print(f"  [{idx}] {vkgp}: OLS failed -> skip")
        continue

    # Heterogeneity
    denom = np.sqrt(se_eur**2 + se_yri**2)
    if denom == 0:
        z_het = np.nan
        p_het = np.nan
    else:
        z_het = (beta_eur - beta_yri) / denom
        p_het = 2 * stats.norm.sf(abs(z_het))

    sign_concordant = bool(
        np.sign(beta_eur - beta_yri) == np.sign(row["ObsDelta_EUR_AFR"])
    )

    rec = {
        "variant_kgpID"        : vkgp,
        "ensemblID"            : ens_ver,
        "set"                  : locus_set,
        "beta_eur_geuvadis"    : beta_eur,
        "se_eur"               : se_eur,
        "p_eur"                : p_eur,
        "beta_yri_geuvadis"    : beta_yri,
        "se_yri"               : se_yri,
        "p_yri"                : p_yri,
        "z_het"                : z_het,
        "p_het"                : p_het,
        "beta_eur_mage"        : row["beta_EUR"],
        "beta_afr_mage"        : row["beta_AFR"],
        "ObsDelta_mage"        : row["ObsDelta_EUR_AFR"],
        "sign_concordant"      : sign_concordant,
        "n_eur"                : int(mask_e.sum()),
        "n_yri"                : int(mask_y.sum()),
    }
    results.append(rec)
    n_new += 1
    print(f"  [{idx}] {vkgp} ({locus_set}): "
          f"bEUR={beta_eur:.3f} bYRI={beta_yri:.3f} z_het={z_het:.2f} p={p_het:.3g} "
          f"sign={'OK' if sign_concordant else 'FLIP'}")

    # Checkpoint
    if n_new % CHECKPOINT_EVERY == 0:
        with open(CHECKPOINT, "wb") as f:
            pickle.dump(results, f)
        print(f"  ** Checkpoint saved ({len(results)} total)")

# Final save
with open(CHECKPOINT, "wb") as f:
    pickle.dump(results, f)

out_df = pd.DataFrame(results)
out_df.to_csv(OUT_TSV, sep="\t", index=False)
print(f"\nSaved {len(out_df)} rows -> {OUT_TSV}")

# ─── Summary stats ───────────────────────────────────────────────────────────

ad = out_df[out_df["set"] == "adqtl"].copy()
ct = out_df[out_df["set"] == "control"].copy()

print(f"\n=== Summary ===")
print(f"EUR samples: {len(eur_samples_all)}, YRI samples: {len(yri_samples_all)}")
print(f"adQTL loci tested: {len(ad)}, control loci tested: {len(ct)}")

if len(ad) > 0 and len(ct) > 0:
    ad_z = ad["z_het"].abs().dropna()
    ct_z = ct["z_het"].abs().dropna()
    print(f"\nMedian |z_het|: adQTL={ad_z.median():.3f}, control={ct_z.median():.3f}")

    ad_rate = (ad["p_het"] < 0.05).mean()
    ct_rate = (ct["p_het"] < 0.05).mean()
    print(f"p_het<0.05 rate: adQTL={ad_rate:.3f}, control={ct_rate:.3f}")

    ad_sign = ad["sign_concordant"].mean()
    ct_sign = ct["sign_concordant"].mean()
    print(f"Sign concordance: adQTL={ad_sign:.3f}, control={ct_sign:.3f}")

    # Spearman on ObsDelta vs Geuvadis EUR-YRI delta
    both = out_df.dropna(subset=["ObsDelta_mage", "beta_eur_geuvadis", "beta_yri_geuvadis"])
    both = both.copy()
    both["geuv_delta"] = both["beta_eur_geuvadis"] - both["beta_yri_geuvadis"]
    rho, rp = stats.spearmanr(both["ObsDelta_mage"], both["geuv_delta"])
    print(f"Spearman r (all loci ObsDelta): rho={rho:.3f}, p={rp:.3g}")

    ad_b = both[both["set"] == "adqtl"]
    ct_b = both[both["set"] == "control"]
    if len(ad_b) > 5:
        rho_a, rp_a = stats.spearmanr(ad_b["ObsDelta_mage"], ad_b["geuv_delta"])
        print(f"  adQTL only: rho={rho_a:.3f}, p={rp_a:.3g}")
    if len(ct_b) > 5:
        rho_c, rp_c = stats.spearmanr(ct_b["ObsDelta_mage"], ct_b["geuv_delta"])
        print(f"  control only: rho={rho_c:.3f}, p={rp_c:.3g}")

    # Fisher test: p_het < 0.05 enrichment
    from scipy.stats import fisher_exact
    a = (ad["p_het"] < 0.05).sum()
    b = len(ad) - a
    c = (ct["p_het"] < 0.05).sum()
    d = len(ct) - c
    oddsratio, fp = fisher_exact([[a, b], [c, d]], alternative="greater")
    print(f"\nFisher test (p_het<0.05 adQTL vs control): OR={oddsratio:.2f}, p={fp:.3g}")
    print(f"  adQTL: {a}/{len(ad)} ({100*a/len(ad):.1f}%)")
    print(f"  control: {c}/{len(ct)} ({100*c/len(ct):.1f}%)")
