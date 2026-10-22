"""
p1_18_generative_simulation.py
Generative simulation to prove cis-haplotype epistasis pipeline identifiability.
Four mechanistic models: A=Null, B=LD artifact, C=GxE, D=True cis-epistasis.
Outputs: results/phase1/p1_simulation_results.tsv
"""

import numpy as np
import pandas as pd
from scipy import stats
import os, sys

# ── reproducibility
np.random.seed(42)

# ── sample sizes (real data)
N_EUR = 142
N_AFR = 196
N_LOCI = 500

# ── load real parameter distributions
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cis = pd.read_csv(os.path.join(BASE, "results/phase1/p1_cis_interaction.tsv"), sep="\t")
adq = pd.read_csv(os.path.join(BASE, "results/phase0/p0_adqtl_set.tsv"), sep="\t")

# real gamma distribution
gamma_vals_real = cis["int_beta"].dropna().values
# real Delta_f distribution
cis["delta_f"] = (cis["bg_maf_EUR"] - cis["bg_maf_AFR"]).abs()
delta_f_real = cis["delta_f"].dropna().values
# real focal betas
beta_eur_real = adq["beta_EUR"].dropna().values
beta_afr_real = adq["beta_AFR"].dropna().values

print(f"Real gamma: mean={gamma_vals_real.mean():.3f}, std={gamma_vals_real.std():.3f}")
print(f"Real delta_f: mean={delta_f_real.mean():.3f}, std={delta_f_real.std():.3f}")
print(f"Real beta_EUR: mean={beta_eur_real.mean():.3f}, std={beta_eur_real.std():.3f}")

# ──────────────────────────────────────────────
# Core helper functions
# ──────────────────────────────────────────────

def simulate_genotype(n, maf):
    p = maf
    probs = [(1-p)**2, 2*p*(1-p), p**2]
    return np.random.choice([0,1,2], size=n, p=probs).astype(float)


def ols_beta_se(expr, dosage, covariates):
    n = len(expr)
    X = np.column_stack([np.ones(n), dosage, covariates])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, expr, rcond=None)
        resid = expr - X @ coef
        sigma2 = np.dot(resid, resid) / max(n - X.shape[1], 1)
        XtXinv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtXinv))
        return coef[1], se[1]
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def interaction_pval(expr, focal, bg, covariates):
    n = len(expr)
    inter = focal * bg
    X = np.column_stack([np.ones(n), focal, bg, inter, covariates])
    p = X.shape[1]
    if n <= p:
        return np.nan
    try:
        coef, _, _, _ = np.linalg.lstsq(X, expr, rcond=None)
        resid = expr - X @ coef
        sigma2 = np.dot(resid, resid) / max(n - p, 1)
        XtXinv = np.linalg.inv(X.T @ X)
        se = np.sqrt(max(sigma2 * XtXinv[3,3], 1e-30))
        t = coef[3] / se
        return float(2 * stats.t.sf(abs(t), df=n-p))
    except Exception:
        return np.nan


def interaction_pval_with_ancestry(expr, focal, bg, anc_label, covariates):
    """Interaction p-val after controlling for ancestry + focal:ancestry."""
    n = len(expr)
    inter_focal_bg = focal * bg
    inter_focal_anc = focal * anc_label
    X = np.column_stack([np.ones(n), focal, bg, inter_focal_bg, anc_label, inter_focal_anc, covariates])
    p = X.shape[1]
    if n <= p:
        return np.nan
    try:
        coef, _, _, _ = np.linalg.lstsq(X, expr, rcond=None)
        resid = expr - X @ coef
        sigma2 = np.dot(resid, resid) / max(n - p, 1)
        XtXinv = np.linalg.inv(X.T @ X)
        se = np.sqrt(max(sigma2 * XtXinv[3,3], 1e-30))
        t = coef[3] / se
        return float(2 * stats.t.sf(abs(t), df=n-p))
    except Exception:
        return np.nan


def het_pval(b_e, se_e, b_a, se_a):
    denom = np.sqrt(se_e**2 + se_a**2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (b_e - b_a) / denom
    return float(2 * stats.norm.sf(abs(z)))


def run_locus(focal_eur, focal_afr, bg_eur, bg_afr,
              expr_eur, expr_afr, cov_eur, cov_afr,
              compute_ancestry_control=False):
    """Run per-locus tests. Returns dict of stats."""
    # per-ancestry betas
    b_e, se_e = ols_beta_se(expr_eur, focal_eur, cov_eur)
    b_a, se_a = ols_beta_se(expr_afr, focal_afr, cov_afr)
    p_het = het_pval(b_e, se_e, b_a, se_a)

    # pooled interaction
    focal_pool = np.concatenate([focal_eur, focal_afr])
    bg_pool    = np.concatenate([bg_eur, bg_afr])
    expr_pool  = np.concatenate([expr_eur, expr_afr])
    cov_pool   = np.vstack([cov_eur, cov_afr])

    p_int = interaction_pval(expr_pool, focal_pool, bg_pool, cov_pool)

    p_int_anc = np.nan
    if compute_ancestry_control:
        anc_label = np.concatenate([np.ones(len(expr_eur)), np.zeros(len(expr_afr))])
        p_int_anc = interaction_pval_with_ancestry(
            expr_pool, focal_pool, bg_pool, anc_label, cov_pool)

    obs_delta = b_e - b_a

    return dict(b_e=b_e, se_e=se_e, b_a=b_a, se_a=se_a,
                p_het=p_het, p_int=p_int, p_int_anc=p_int_anc,
                obs_delta=obs_delta)


# ──────────────────────────────────────────────
# Model A: True null
# ──────────────────────────────────────────────
def run_model_A(n_loci=N_LOCI):
    records = []
    beta = 0.5   # same in both
    sigma = 1.0
    focal_maf = 0.3
    bg_maf_eur = 0.4
    bg_maf_afr = 0.4

    for i in range(n_loci):
        cov_e = np.random.randn(N_EUR, 5)
        cov_a = np.random.randn(N_AFR, 5)
        focal_e = simulate_genotype(N_EUR, focal_maf)
        focal_a = simulate_genotype(N_AFR, focal_maf)
        bg_e = simulate_genotype(N_EUR, bg_maf_eur)
        bg_a = simulate_genotype(N_AFR, bg_maf_afr)
        expr_e = beta * focal_e + np.random.randn(N_EUR) * sigma
        expr_a = beta * focal_a + np.random.randn(N_AFR) * sigma

        r = run_locus(focal_e, focal_a, bg_e, bg_a, expr_e, expr_a, cov_e, cov_a)
        r.update(dict(model="A", locus=i, gamma=0.0, delta_f=0.0, r2_focal_bg=np.nan))
        records.append(r)
    return records


# ──────────────────────────────────────────────
# Model B: LD / tagging artifact
# ──────────────────────────────────────────────
def simulate_tag(causal, r2, maf_tag):
    """Generate tag genotype correlated with causal at approximately r²."""
    n = len(causal)
    rho = np.sqrt(r2)
    # Generate a latent variable correlated with causal
    latent = rho * causal + np.sqrt(1 - r2) * np.random.randn(n)
    # Threshold to HWE genotype
    p = maf_tag
    t1 = stats.norm.ppf((1-p)**2 + 2*p*(1-p))
    t2 = stats.norm.ppf((1-p)**2)
    tag = np.zeros(n)
    tag[latent > t1] = 2
    tag[(latent > t2) & (latent <= t1)] = 1
    return tag.astype(float)


def run_model_B(n_loci=N_LOCI):
    records = []
    beta_true = 0.5
    sigma = 1.0
    causal_maf = 0.3
    tag_maf = 0.3
    bg_maf_eur = 0.4
    bg_maf_afr = 0.4

    # vary r2 to test PIP diagnostic
    r2_conditions = [(0.85, 0.45), (0.65, 0.35), (0.45, 0.25), (0.3, 0.15)]

    for i in range(n_loci):
        # cycle r2 conditions
        r2_eur, r2_afr = r2_conditions[i % len(r2_conditions)]
        cov_e = np.random.randn(N_EUR, 5)
        cov_a = np.random.randn(N_AFR, 5)
        # True causal
        causal_e = simulate_genotype(N_EUR, causal_maf)
        causal_a = simulate_genotype(N_AFR, causal_maf)
        # Tag (what we test)
        tag_e = simulate_tag(causal_e, r2_eur, tag_maf)
        tag_a = simulate_tag(causal_a, r2_afr, tag_maf)
        # Expression from true causal
        expr_e = beta_true * causal_e + np.random.randn(N_EUR) * sigma
        expr_a = beta_true * causal_a + np.random.randn(N_AFR) * sigma
        bg_e = simulate_genotype(N_EUR, bg_maf_eur)
        bg_a = simulate_genotype(N_AFR, bg_maf_afr)

        r = run_locus(tag_e, tag_a, bg_e, bg_a, expr_e, expr_a, cov_e, cov_a)
        r.update(dict(model="B", locus=i, gamma=0.0, delta_f=0.0,
                      r2_focal_bg=np.nan, r2_eur=r2_eur, r2_afr=r2_afr))
        records.append(r)
    return records


# ──────────────────────────────────────────────
# Model C: GxE / ancestry label effect
# ──────────────────────────────────────────────
def run_model_C(n_loci=N_LOCI):
    records = []
    beta_eur = 0.5
    beta_afr = -0.3
    sigma = 1.0
    focal_maf = 0.3
    bg_maf_eur = 0.7   # ancestry-differentiated but not functional
    bg_maf_afr = 0.3

    for i in range(n_loci):
        cov_e = np.random.randn(N_EUR, 5)
        cov_a = np.random.randn(N_AFR, 5)
        focal_e = simulate_genotype(N_EUR, focal_maf)
        focal_a = simulate_genotype(N_AFR, focal_maf)
        bg_e = simulate_genotype(N_EUR, bg_maf_eur)
        bg_a = simulate_genotype(N_AFR, bg_maf_afr)
        expr_e = beta_eur * focal_e + np.random.randn(N_EUR) * sigma
        expr_a = beta_afr * focal_a + np.random.randn(N_AFR) * sigma

        r = run_locus(focal_e, focal_a, bg_e, bg_a, expr_e, expr_a, cov_e, cov_a,
                      compute_ancestry_control=True)
        delta_f = abs(bg_maf_eur - bg_maf_afr)
        r.update(dict(model="C", locus=i, gamma=0.0, delta_f=delta_f, r2_focal_bg=np.nan))
        records.append(r)
    return records


# ──────────────────────────────────────────────
# Model D: True cis-haplotype epistasis
# ──────────────────────────────────────────────
def run_model_D_config(gamma, bg_maf_eur, bg_maf_afr, n_loci=N_LOCI,
                       focal_maf_eur=0.3, focal_maf_afr=0.35, beta_focal=0.3, sigma=1.0):
    records = []
    delta_f = abs(bg_maf_eur - bg_maf_afr)

    for i in range(n_loci):
        cov_e = np.random.randn(N_EUR, 5)
        cov_a = np.random.randn(N_AFR, 5)
        focal_e = simulate_genotype(N_EUR, focal_maf_eur)
        focal_a = simulate_genotype(N_AFR, focal_maf_afr)
        bg_e = simulate_genotype(N_EUR, bg_maf_eur)
        bg_a = simulate_genotype(N_AFR, bg_maf_afr)
        expr_e = beta_focal * focal_e + gamma * focal_e * bg_e + np.random.randn(N_EUR) * sigma
        expr_a = beta_focal * focal_a + gamma * focal_a * bg_a + np.random.randn(N_AFR) * sigma

        r = run_locus(focal_e, focal_a, bg_e, bg_a, expr_e, expr_a, cov_e, cov_a,
                      compute_ancestry_control=True)

        # Decomposition accuracy: half-sample split
        n_e = N_EUR // 2
        n_a = N_AFR // 2
        # train on first half
        Xtrain = np.column_stack([np.ones(n_e + n_a),
                                  np.concatenate([focal_e[:n_e], focal_a[:n_a]]),
                                  np.concatenate([bg_e[:n_e], bg_a[:n_a]]),
                                  np.concatenate([focal_e[:n_e]*bg_e[:n_e], focal_a[:n_a]*bg_a[:n_a]]),
                                  np.vstack([cov_e[:n_e], cov_a[:n_a]])])
        ytrain = np.concatenate([expr_e[:n_e], expr_a[:n_a]])
        try:
            coef_train, _, _, _ = np.linalg.lstsq(Xtrain, ytrain, rcond=None)
            gamma_hat = coef_train[3]
        except Exception:
            gamma_hat = np.nan

        # ObsDelta from test half
        b_e_test, se_e_test = ols_beta_se(expr_e[n_e:], focal_e[n_e:], cov_e[n_e:])
        b_a_test, se_a_test = ols_beta_se(expr_a[n_a:], focal_a[n_a:], cov_a[n_a:])
        obs_delta_test = b_e_test - b_a_test

        mean_bg_eur = np.mean(bg_e)
        mean_bg_afr = np.mean(bg_a)
        predicted_delta = gamma_hat * (mean_bg_eur - mean_bg_afr) * 2
        decomp_ratio = (predicted_delta / obs_delta_test) if obs_delta_test != 0 else np.nan

        r.update(dict(model="D", locus=i, gamma=gamma, delta_f=delta_f,
                      r2_focal_bg=np.nan, gamma_hat=gamma_hat,
                      obs_delta_test=obs_delta_test,
                      predicted_delta=predicted_delta,
                      decomp_ratio=decomp_ratio))
        records.append(r)
    return records


# ──────────────────────────────────────────────
# Main run
# ──────────────────────────────────────────────
all_records = []

print("Running Model A (Null)...")
all_records.extend(run_model_A())

print("Running Model B (LD artifact)...")
all_records.extend(run_model_B())

print("Running Model C (GxE)...")
all_records.extend(run_model_C())

# Model D: primary (gamma=0.5, delta_f=0.5)
print("Running Model D configs...")
for gamma in [0.1, 0.3, 0.5, 0.8]:
    for bg_maf_eur, bg_maf_afr in [(0.75, 0.25), (0.6, 0.4)]:
        delta_f = bg_maf_eur - bg_maf_afr
        label = f"D_g{gamma}_df{delta_f:.1f}"
        print(f"  {label}")
        recs = run_model_D_config(gamma=gamma, bg_maf_eur=bg_maf_eur, bg_maf_afr=bg_maf_afr)
        for r in recs:
            r["model_label"] = label
        all_records.extend(recs)

df_out = pd.DataFrame(all_records)
# fill model_label for A/B/C
if "model_label" not in df_out.columns:
    df_out["model_label"] = df_out["model"]
df_out["model_label"] = df_out["model_label"].fillna(df_out["model"])

out_path = os.path.join(BASE, "results/phase1/p1_simulation_results.tsv")
df_out.to_csv(out_path, sep="\t", index=False)
print(f"\nSaved {len(df_out)} rows to {out_path}")

# ── Summary statistics
print("\n=== SUMMARY STATISTICS ===")
for label, grp in df_out.groupby("model_label"):
    adqtl_rate = (grp["p_het"] < 0.05).mean()
    int_rate    = (grp["p_int"] < 0.05).mean()

    adqtl_mask = grp["p_het"] < 0.05
    nonadq_mask = grp["p_het"] > 0.3
    int_adqtl   = (grp.loc[adqtl_mask, "p_int"] < 0.05).mean() if adqtl_mask.sum() > 0 else np.nan
    int_nonadq  = (grp.loc[nonadq_mask, "p_int"] < 0.05).mean() if nonadq_mask.sum() > 0 else np.nan
    if int_nonadq and int_nonadq > 0:
        OR = (int_adqtl / (1 - int_adqtl + 1e-9)) / (int_nonadq / (1 - int_nonadq + 1e-9))
    else:
        OR = np.nan

    print(f"{label:30s}  adQTL%={adqtl_rate:.3f}  int%={int_rate:.3f}  OR={OR:.2f}")
