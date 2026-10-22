"""
Validation V7: interaction model with an ancestry covariate
=======================================================
Original p1_06 model: expr ~ focal + bg + focal:bg + sex + PC1-5 + PEER1-15
Added ancestry terms:  + ancestry_EUR (0/1) + focal:ancestry

Goal: test whether the focal × bg interaction remains significant after controlling for focal × ancestry,
i.e. distinguish "the bg SNP proxies the ancestry label" from "a true focal × bg genetic interaction".

Strategy:
  - in the pooled EUR+AFR sample, add ancestry_EUR (1=EUR, 0=AFR) as a covariate
  - also add a focal:ancestry term
  - test the partial P value of focal:bg
  - compare with the focal:bg P value from the original model

This is a more direct, model-level control than the within-ancestry test.

Inputs: adQTL list from p1_cis_interaction.tsv + MAGE data
Output: results/validation/v7_ancestry_model.tsv
     figures/validation/v7_ancestry_model.{png,pdf}
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

import pandas as pd2
meta = pd2.read_csv(META, sep="\t")
eur_ids = meta[meta.continentalGroup=="EUR"]["sample_kgpID"].tolist()
afr_ids = meta[meta.continentalGroup=="AFR"]["sample_kgpID"].tolist()
ea_ids  = eur_ids + afr_ids   # EUR+AFR combined

from p0_05_fetch_genotypes import VCF_TMPL as _VCF_BASE
VCF_TMPL = _VCF_BASE  # NYGC 3202 hg38 phased; chrom format: chr1..chr22

N_PEER   = 15
COV_COLS = ["sex"] + [f"PC{i}" for i in range(1, 6)] + [f"PEER{i}" for i in range(1, N_PEER+1)]
MIN_N    = 60

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
AD = "#d62728"; CO = "#1f77b4"


def gt_to_dosage(gt_str):
    g = gt_str.split(":")[0].replace("|", "/")
    if g in (".", "./.", ".|."):
        return np.nan
    try:
        a = g.split("/"); return int(a[0]) + int(a[1])
    except Exception:
        return np.nan


def main():
    # Load expression + covariates
    expr = load_expr()
    cov  = load_covar(N_PEER)
    ancestry_bin = np.array([1 if s in set(eur_ids) else 0 for s in ea_ids], float)

    ci   = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    ci["chrom"]     = ci.variant_kgpID.str.split(":").str[0]
    ci["focal_pos"] = ci.variant_kgpID.str.split(":").str[1].astype(int)
    ci["focal_ref"] = ci.variant_kgpID.str.split(":").str[2]
    ci["focal_alt"] = ci.variant_kgpID.str.split(":").str[3]
    ci["bg_pos"]    = ci.best_bg.str.split(":").str[1].apply(
        lambda x: int(x) if pd.notna(x) and str(x).isdigit() else np.nan)
    ci["bg_chrom"]  = ci.best_bg.str.split(":").str[0]
    ci["bg_ref"]    = ci.best_bg.str.split(":").str[2]
    ci["bg_alt"]    = ci.best_bg.str.split(":").str[3]
    ci_ok = ci.dropna(subset=["bg_pos"]).copy()
    print(f"Loci with bg info: {len(ci_ok)}")

    CKPT = OUT_DIR / "v7_ancestry_model_ckpt.tsv"
    done = set(); results = []
    if CKPT.exists():
        prev = pd.read_csv(CKPT, sep="\t")
        done = set(zip(prev.variant_kgpID, prev.ensemblID))
        results = prev.to_dict("records")
        print(f"  Resuming: {len(done)} done")

    vcf_cache = {}
    for _, r in ci_ok.iterrows():
        key = (r.variant_kgpID, r.ensemblID)
        if key in done: continue
        vid = r.variant_kgpID; gene = r.ensemblID
        chrom = r.chrom; bg_chrom = r.bg_chrom if str(r.bg_chrom) != "nan" else chrom
        bg_pos_int = int(r.bg_pos)

        try:
            chrom_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
            bg_chrom_key = bg_chrom if bg_chrom.startswith("chr") else f"chr{bg_chrom}"
            if chrom not in vcf_cache:
                vcf_cache[chrom] = RemoteTabixVCF(VCF_TMPL.format(chrom=chrom_key))
            vcf = vcf_cache[chrom]
            vsamp = {s: i for i, s in enumerate(vcf.samples())}

            if bg_chrom not in vcf_cache:
                vcf_cache[bg_chrom] = RemoteTabixVCF(VCF_TMPL.format(chrom=bg_chrom_key))
            vcf_bg = vcf_cache[bg_chrom]
            vsamp_bg = {s: i for i, s in enumerate(vcf_bg.samples())}

            if gene not in expr.index:
                # try stripping version
                gene_base = gene.split(".")[0]
                candidates = [g for g in expr.index if g.split(".")[0]==gene_base]
                if not candidates: print(f"  SKIP {vid}: gene not in expr"); continue
                gene = candidates[0]

            # focal dosages (EUR+AFR)
            fd_focal = None
            for ln in vcf.query(chrom_key, r.focal_pos-1, r.focal_pos+1):
                f2 = ln.split("\t")
                if int(f2[1])==r.focal_pos and f2[3]==r.focal_ref and f2[4]==r.focal_alt:
                    gts = f2[9:]
                    fd_focal = np.array([gt_to_dosage(gts[vsamp[s]]) if s in vsamp else np.nan
                                         for s in ea_ids], float)
                    break
            if fd_focal is None: print(f"  SKIP {vid}: focal not in VCF"); continue

            # bg dosages
            fd_bg = None
            for ln in vcf_bg.query(bg_chrom_key, bg_pos_int-1, bg_pos_int+1):
                f2 = ln.split("\t")
                if int(f2[1])==bg_pos_int and f2[3]==r.bg_ref and f2[4]==r.bg_alt:
                    gts = f2[9:]
                    fd_bg = np.array([gt_to_dosage(gts[vsamp_bg[s]]) if s in vsamp_bg else np.nan
                                      for s in ea_ids], float)
                    break
            if fd_bg is None: print(f"  SKIP {vid}: bg not in VCF"); continue

            gene_expr = expr.loc[gene, ea_ids].astype(float).values

            C = cov.loc[ea_ids, COV_COLS].astype(float).values
            fo = fd_focal; bgo = fd_bg; anc = ancestry_bin
            ok = (np.isfinite(fo) & np.isfinite(bgo) & np.isfinite(gene_expr)
                  & np.isfinite(C).all(1) & np.isfinite(anc))
            if ok.sum() < MIN_N:
                print(f"  SKIP {vid}: n={ok.sum()}<{MIN_N}"); continue

            fo_ok = fo[ok]; bgo_ok = bgo[ok]; anc_ok = anc[ok]; y_ok = gene_expr[ok]
            if np.std(fo_ok)<0.01 or np.std(bgo_ok)<0.01: continue

            # Model 1: original (no ancestry terms)
            X1 = np.column_stack([np.ones(ok.sum()), fo_ok, bgo_ok, fo_ok*bgo_ok, C[ok]])
            # Model 2: with ancestry + focal:ancestry
            X2 = np.column_stack([np.ones(ok.sum()), fo_ok, bgo_ok, fo_ok*bgo_ok,
                                   anc_ok, fo_ok*anc_ok, C[ok]])
            try:
                r1 = sm.OLS(y_ok, X1).fit()
                r2 = sm.OLS(y_ok, X2).fit()
            except Exception as e:
                print(f"  OLS fail {vid}: {e}"); continue

            row = {
                "variant_kgpID": vid, "ensemblID": gene, "set": r.set,
                "p_focal_bg_orig":     r1.pvalues[3],
                "beta_focal_bg_orig":  r1.params[3],
                "p_focal_bg_anc":      r2.pvalues[3],   # focal:bg in model with ancestry
                "beta_focal_bg_anc":   r2.params[3],
                "p_focal_anc":         r2.pvalues[5],   # focal:ancestry
                "n_samples":           int(ok.sum()),
                "mage_bonf_p":         r.bonf_p,
            }
            results.append(row); done.add(key)
            print(f"  [{r.set}] {gene}  p_orig={r1.pvalues[3]:.2e}  p_anc_ctrl={r2.pvalues[3]:.2e}",
                  flush=True)
            pd.DataFrame(results).to_csv(CKPT, sep="\t", index=False)

        except Exception as e:
            print(f"  ERROR {vid}: {e}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "v7_ancestry_model.tsv", sep="\t", index=False)
    print(f"\nTotal: {len(df)}")

    # ── Analysis ────────────────────────────────────────────────────
    adq  = df[df.set=="adqtl"].dropna(subset=["p_focal_bg_orig","p_focal_bg_anc"])
    ctrl = df[df.set=="control"].dropna(subset=["p_focal_bg_orig","p_focal_bg_anc"])
    thr  = 0.05

    # How many stay significant after ancestry control?
    if len(adq)>0:
        sig_orig = (adq.p_focal_bg_orig < thr/adq.n_samples).mean()*100  # bonf corrected
        sig_anc  = (adq.p_focal_bg_anc  < thr/adq.n_samples).mean()*100
        # Spearman correlation between original and ancestry-controlled p
        sp = stats.spearmanr(-np.log10(adq.p_focal_bg_orig.clip(1e-300)),
                              -np.log10(adq.p_focal_bg_anc.clip(1e-300)))[0]
        print(f"adQTL Bonf-sig orig: {sig_orig:.1f}%  ->  with ancestry ctrl: {sig_anc:.1f}%")
        print(f"Spearman correlation of -log10p: {sp:.3f}")

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    if len(adq)>0:
        ax.scatter(-np.log10(adq.p_focal_bg_orig.clip(1e-300)),
                   -np.log10(adq.p_focal_bg_anc.clip(1e-300)),
                   s=18, color=AD, alpha=0.65, label=f"adQTL (n={len(adq)})", zorder=3)
    if len(ctrl)>0:
        ax.scatter(-np.log10(ctrl.p_focal_bg_orig.clip(1e-300)),
                   -np.log10(ctrl.p_focal_bg_anc.clip(1e-300)),
                   s=10, color=CO, alpha=0.45, label=f"control (n={len(ctrl)})", zorder=2)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) if ax.get_xlim()[1]>0 else 5
    ax.plot([0, lim], [0, lim], "--k", lw=0.8)
    ax.set_xlabel("focal×bg -log10p (original model)")
    ax.set_ylabel("focal×bg -log10p\n(+ ancestry + focal:ancestry)")
    ax.set_title("A  focal×bg p-value before vs after\nancestry covariate control",
                 fontweight="bold", loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    # Distribution of p_focal_anc (focal:ancestry term)
    if len(adq)>0:
        lp_anc_adq  = -np.log10(adq.p_focal_anc.clip(1e-300))
        lp_anc_ctrl = -np.log10(ctrl.p_focal_anc.clip(1e-300)) if len(ctrl)>0 else np.array([])
        bins = np.linspace(0, max(lp_anc_adq.max(), lp_anc_ctrl.max() if len(lp_anc_ctrl)>0 else 0)+0.5, 30)
        ax2.hist(lp_anc_adq,  bins=bins, alpha=0.6, color=AD,  density=True, label="adQTL")
        if len(lp_anc_ctrl)>0:
            ax2.hist(lp_anc_ctrl, bins=bins, alpha=0.6, color=CO, density=True, label="control")
        ax2.axvline(-np.log10(0.05), color="k", lw=1, ls="--", label="p=0.05")
        ax2.set_xlabel("focal:ancestry -log10p"); ax2.set_ylabel("Density")
        ax2.set_title("B  focal×ancestry p-value distribution\n(confounding term)",
                      fontweight="bold", loc="left", fontsize=9)
        ax2.legend(fontsize=8)

    fig.suptitle("V7: focal×bg interaction robust to ancestry covariate control",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "v7_ancestry_model.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"Figure -> {out}")
    print("[v7] DONE.")


if __name__ == "__main__":
    main()
