"""
Validation V8: split-sample decomposition estimate (corrects upward bias)
===================================================
The original p1_11 multi-bg decomposition selects bg SNPs (marginal p<0.01) and estimates γ in the same data,
so the "apparent 59%" is biased upward.

Correction: 2-fold cross-sample estimation
  - randomly split the EUR+AFR samples in half (preserving the EUR:AFR ratio)
  - half A: marginal interaction scan, select bg SNPs (p<0.01) + LD pruning
  - half B: estimate γ (joint model) in the held-out half and compute the decomposition fraction
  - swap A/B, repeat once, and average (2-fold)

Expected: the bias-corrected estimate is lower than 59% (same-data selection bias removed),
but direction and magnitude remain consistent (cis-haplotype contribution > 0).

Inputs: MAGE data + p1_multi_bg_decomp.tsv (reference for comparison)
Output: results/validation/v8_splitsample_decomp.tsv
     figures/validation/v8_splitsample_decomp.{png,pdf}
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
P0      = BASE / "results" / "phase0"
P1      = BASE / "results" / "phase1"
OUT_DIR = BASE / "results" / "validation"
FIG_DIR = BASE / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts"))
from lib_remote_vcf import RemoteTabixVCF
from p0_06_per_ancestry_betas import load_expr, load_covar, META

meta = pd.read_csv(META, sep="\t")
eur_ids = meta[meta.continentalGroup=="EUR"]["sample_kgpID"].tolist()
afr_ids = meta[meta.continentalGroup=="AFR"]["sample_kgpID"].tolist()
ea_ids  = eur_ids + afr_ids

from p0_05_fetch_genotypes import VCF_TMPL as _VCF_BASE
VCF_TMPL = _VCF_BASE  # NYGC 3202 hg38 phased; chrom format: chr1..chr22

COV_COLS    = None  # set in main after load_covar
MIN_N       = 40   # per half
MARGINAL_P  = 0.01
MAX_R2      = 0.5
MIN_MAF     = 0.1
MIN_DELTA_F = 0.1
WIN         = 50_000

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
AD = "#d62728"; CO = "#1f77b4"; BL = "#2166ac"


def gt_to_dosage(gt_str):
    g = gt_str.split(":")[0].replace("|", "/")
    if g in (".", "./.", ".|."):
        return np.nan
    try:
        a = g.split("/"); return int(a[0]) + int(a[1])
    except Exception:
        return np.nan


def adosage(gts, idx):
    return np.array([gt_to_dosage(gts[i]) if i < len(gts) else np.nan for i in idx], float)


def maf(d):
    d = d[~np.isnan(d)]
    return min(d.mean()/2, 1 - d.mean()/2) if len(d) > 0 else 0.0


def main():
    N_PEER = 15
    expr = load_expr()
    cov  = load_covar(N_PEER)
    COV_COLS = ["sex"] + [f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, N_PEER+1)]

    # Source from multi-bg decomp loci (already sig adQTL with cis-bg)
    mb = pd.read_csv(P1 / "p1_multi_bg_decomp.tsv", sep="\t")
    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    adq_sig = ci[(ci.set=="adqtl") & (ci.bonf_p < 0.05)]
    # Run on adQTL × gene pairs that have ObsDelta info
    adq_mb = adq_sig.merge(mb[["variant_kgpID","ensemblID","ObsDelta","frac_cis_multi"]],
                            on=["variant_kgpID","ensemblID"], how="inner", suffixes=("_ci",""))
    # resolve possible suffix collision
    if "ObsDelta" not in adq_mb.columns and "ObsDelta_" in " ".join(adq_mb.columns):
        adq_mb = adq_mb.rename(columns={c: "ObsDelta" for c in adq_mb.columns if c.startswith("ObsDelta") and not c.endswith("_ci")})
    print(f"adQTL loci with multi-bg result: {len(adq_mb)}")

    # Split samples: stratified by ancestry
    rng = np.random.default_rng(42)
    eur_idx = list(range(len(eur_ids)))
    afr_idx = list(range(len(eur_ids), len(ea_ids)))
    rng.shuffle(eur_idx); rng.shuffle(afr_idx)
    eur_a = [ea_ids[i] for i in eur_idx[:len(eur_idx)//2]]
    eur_b = [ea_ids[i] for i in eur_idx[len(eur_idx)//2:]]
    afr_a = [ea_ids[i] for i in afr_idx[:len(afr_idx)//2]]
    afr_b = [ea_ids[i] for i in afr_idx[len(afr_idx)//2:]]
    half_A = eur_a + afr_a; half_B = eur_b + afr_b
    print(f"  Half A: {len(half_A)} samples (EUR={len(eur_a)}, AFR={len(afr_a)})")
    print(f"  Half B: {len(half_B)} samples (EUR={len(eur_b)}, AFR={len(afr_b)})")

    CKPT = OUT_DIR / "v8_splitsample_ckpt.tsv"
    done = set(); results = []
    if CKPT.exists():
        prev = pd.read_csv(CKPT, sep="\t")
        done = set(zip(prev.variant_kgpID, prev.ensemblID))
        results = prev.to_dict("records"); print(f"  Resuming: {len(done)} done")

    vcf_cache = {}
    for _, r in adq_mb.iterrows():
        key = (r.variant_kgpID, r.ensemblID)
        if key in done: continue
        vid = r.variant_kgpID; gene = r.ensemblID
        chrom = vid.split(":")[0]; fpos = int(vid.split(":")[1])
        focal_ref = vid.split(":")[2]; focal_alt = vid.split(":")[3]
        obs_delta = r.ObsDelta

        try:
            chrom_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
            if chrom not in vcf_cache:
                vcf_cache[chrom] = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom_key))
            vcf = vcf_cache[chrom]
            vsamp = {s: i for i, s in enumerate(vcf.samples())}

            eur_cols = [vsamp[s] for s in eur_ids if s in vsamp]
            afr_cols = [vsamp[s] for s in afr_ids if s in vsamp]

            # Get gene expression and covariate arrays for full sample set
            if gene not in expr.index:
                gene_base = gene.split(".")[0]
                candidates = [g for g in expr.index if g.split(".")[0]==gene_base]
                if not candidates: continue
                gene = candidates[0]
            full_expr = expr.loc[gene, ea_ids].values.astype(float)

            # Focal dosages for all samples
            fd = {}
            for ln in vcf.query(chrom_key, fpos-1, fpos+1):
                f2 = ln.split("\t")
                if int(f2[1])==fpos and f2[3]==focal_ref and f2[4]==focal_alt:
                    gts = f2[9:]
                    for s in ea_ids:
                        if s in vsamp and vsamp[s] < len(gts):
                            fd[s] = gt_to_dosage(gts[vsamp[s]])
                    break
            if not fd: continue
            focal_all = np.array([fd.get(s, np.nan) for s in ea_ids], float)

            # Candidate bg SNPs: fetch all in window
            cands = []
            for ln in vcf.query(chrom_key, fpos-WIN, fpos+WIN):
                f2 = ln.split("\t")
                if len(f2)<10 or int(f2[1])==fpos or len(f2[3])!=1 or len(f2[4])!=1 or "," in f2[4]:
                    continue
                gts = f2[9:]
                fe = maf(adosage(gts, eur_cols)); fa = maf(adosage(gts, afr_cols))
                if fe < MIN_MAF or fa < MIN_MAF or abs(fe-fa) < MIN_DELTA_F:
                    continue
                bg_all = np.array([gt_to_dosage(gts[vsamp[s]]) if s in vsamp and vsamp[s] < len(gts) else np.nan
                                   for s in ea_ids], float)
                # r² with focal (full sample)
                ok0 = np.isfinite(focal_all) & np.isfinite(bg_all)
                if ok0.sum() < 20: continue
                r2 = np.corrcoef(focal_all[ok0], bg_all[ok0])[0,1]**2
                if r2 > MAX_R2: continue
                cands.append((f2[2], bg_all, fe-fa))

            if not cands: continue
            C_full = cov.loc[ea_ids, COV_COLS].astype(float).values

            def run_half(select_sids, estimate_sids, obs_d):
                """Select bg SNPs in select_sids, estimate γ in estimate_sids."""
                s_idx = [i for i,s in enumerate(ea_ids) if s in set(select_sids)]
                e_idx = [i for i,s in enumerate(ea_ids) if s in set(estimate_sids)]
                # MAF in estimation halves for Δf
                e_eur = [i for i in e_idx if ea_ids[i] in set(eur_ids)]
                e_afr = [i for i in e_idx if ea_ids[i] in set(afr_ids)]

                # Step 1: marginal scan on select half
                kept = []
                for bid, bg_all, delta_f_full in cands:
                    s_ok = (np.isfinite(focal_all[s_idx]) & np.isfinite(bg_all[s_idx])
                            & np.isfinite(full_expr[s_idx]) & np.isfinite(C_full[s_idx]).all(1))
                    if s_ok.sum() < MIN_N: continue
                    fo_s = focal_all[s_idx][s_ok]; bg_s = bg_all[s_idx][s_ok]
                    if np.std(fo_s)<0.01 or np.std(bg_s)<0.01: continue
                    X = np.column_stack([np.ones(s_ok.sum()), fo_s, bg_s, fo_s*bg_s, C_full[s_idx][s_ok]])
                    try:
                        p = sm.OLS(full_expr[s_idx][s_ok], X).fit().pvalues[3]
                    except Exception:
                        continue
                    if p < MARGINAL_P:
                        kept.append((bid, bg_all, delta_f_full, p))

                if not kept: return np.nan, 0

                # LD prune
                kept.sort(key=lambda t: t[3]); pruned = []
                for bid, bg_all_k, df_k, p_k in kept:
                    ok_e = np.isfinite(focal_all[e_idx]) & np.isfinite(bg_all_k[e_idx])
                    if not pruned:
                        pruned.append((bid, bg_all_k, df_k)); continue
                    r2s = []
                    for _, bg_p, _ in pruned:
                        ok2 = ok_e & np.isfinite(bg_p[e_idx])
                        if ok2.sum() < 20: r2s.append(1.0); continue
                        r2s.append(np.corrcoef(bg_all_k[e_idx][ok2], bg_p[e_idx][ok2])[0,1]**2)
                    if max(r2s) < 0.2:
                        pruned.append((bid, bg_all_k, df_k))

                if not pruned: return np.nan, 0

                # Step 2: joint model on estimate half
                e_ok = np.isfinite(focal_all[e_idx]) & np.isfinite(full_expr[e_idx]) & np.isfinite(C_full[e_idx]).all(1)
                for _, bg_all_k, _ in pruned:
                    e_ok &= np.isfinite(bg_all_k[e_idx])
                if e_ok.sum() < MIN_N: return np.nan, 0

                n_bg = len(pruned)
                fo_e = focal_all[e_idx][e_ok]
                bgs_e = [b[e_idx][e_ok] for _,b,_ in pruned]
                X_joint = np.column_stack([np.ones(e_ok.sum()), fo_e]
                                           + bgs_e
                                           + [fo_e * b for b in bgs_e]
                                           + [C_full[e_idx][e_ok]])
                try:
                    res = sm.OLS(full_expr[e_idx][e_ok], X_joint).fit()
                except Exception:
                    return np.nan, 0

                # γ_k = coefficient of focal:bg_k (columns 2+n_bg+k)
                gamma_start = 2 + n_bg
                obs_delta_cis = 0.0
                for k, (_, bg_all_k, _) in enumerate(pruned):
                    gamma_k = res.params[gamma_start + k]
                    # Δf_k in estimation half
                    df_e = np.nanmean(bg_all_k[[i for i in e_idx if ea_ids[i] in set(eur_ids)]]) / 2 - \
                           np.nanmean(bg_all_k[[i for i in e_idx if ea_ids[i] in set(afr_ids)]]) / 2
                    obs_delta_cis += 2 * gamma_k * df_e

                frac = obs_delta_cis / obs_d if abs(obs_d) > 0.01 else np.nan
                return frac, n_bg

            # 2-fold cross: A selects, B estimates; then B selects, A estimates
            frac_AB, n_AB = run_half(half_A, half_B, obs_delta)
            frac_BA, n_BA = run_half(half_B, half_A, obs_delta)
            frac_cv = np.nanmean([f for f in [frac_AB, frac_BA] if not np.isnan(f)]) \
                      if not (np.isnan(frac_AB) and np.isnan(frac_BA)) else np.nan

            row = {"variant_kgpID": vid, "ensemblID": gene,
                   "ObsDelta": obs_delta,
                   "frac_cis_multi_orig": r.frac_cis_multi,
                   "frac_AB": frac_AB, "frac_BA": frac_BA,
                   "frac_cv": frac_cv,
                   "n_bg_AB": n_AB, "n_bg_BA": n_BA}
            results.append(row); done.add(key)
            print(f"  {gene}  orig={r.frac_cis_multi:.3f}  cv={frac_cv:.3f}"
                  f" (AB={frac_AB:.3f}, BA={frac_BA:.3f})", flush=True)
            pd.DataFrame(results).to_csv(CKPT, sep="\t", index=False)

        except Exception as e:
            print(f"  ERROR {vid}: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "v8_splitsample_decomp.tsv", sep="\t", index=False)
    print(f"\nDone: {len(df)} loci")

    # ── Analysis ────────────────────────────────────────────────────
    df2 = df.dropna(subset=["frac_cv"])
    print(f"With CV estimate: {len(df2)}")
    print(f"  Original median:  {df.frac_cis_multi_orig.median():.3f}")
    if len(df2)>0:
        print(f"  CV (unbiased) median: {df2.frac_cv.median():.3f}")
        sign_conc = (np.sign(df2.frac_cv) == np.sign(df2.frac_cis_multi_orig)).mean()*100
        print(f"  Sign concordance orig vs CV: {sign_conc:.1f}%")

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    fracs_orig = df.frac_cis_multi_orig.clip(-0.5, 1.5)
    fracs_cv   = df2.frac_cv.clip(-0.5, 1.5) if len(df2)>0 else pd.Series([])
    bins = np.linspace(-0.5, 1.5, 25)
    ax.hist(fracs_orig, bins=bins, alpha=0.6, color=AD, label=f"original (same-data)\nmedian={fracs_orig.median():.2f}")
    if len(fracs_cv)>0:
        ax.hist(fracs_cv, bins=bins, alpha=0.6, color=BL,
                label=f"cross-sample (unbiased)\nmedian={fracs_cv.median():.2f}")
    ax.axvline(0, lw=1, color="k", ls="--")
    ax.set_xlabel("fraction of ObsDelta explained by cis-backgrounds")
    ax.set_ylabel("# adQTL loci")
    ax.set_title("A  Same-data vs cross-sample\ndecomposition estimate",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    if len(df2)>0:
        ax2.scatter(df2.frac_cis_multi_orig.clip(-0.5,1.5),
                    df2.frac_cv.clip(-0.5,1.5),
                    s=20, color=AD, alpha=0.6)
        lim = max(ax2.get_xlim()[1], ax2.get_ylim()[1]) if ax2.get_xlim()[1]>0 else 1.5
        ax2.plot([-0.5, lim], [-0.5, lim], "--k", lw=0.8)
        sp = stats.spearmanr(df2.frac_cis_multi_orig, df2.frac_cv)[0]
        ax2.text(0.05, 0.95, f"Spearman r = {sp:.2f}", transform=ax2.transAxes, va="top", fontsize=9)
    ax2.set_xlabel("original fraction (same-data)"); ax2.set_ylabel("cross-sample fraction")
    ax2.set_title("B  Per-locus concordance\n(x=y if no bias)", fontweight="bold", loc="left", fontsize=9)

    fig.suptitle("V8: Split-sample decomposition — bias-corrected cis-haplotype contribution",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "v8_splitsample_decomp.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"Figure -> {out}")
    print("[v8] DONE.")


if __name__ == "__main__":
    main()
