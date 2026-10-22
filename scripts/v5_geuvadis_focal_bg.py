"""
Validation V5: individual-level focal × background-SNP interaction in Geuvadis
====================================================================
Using 276 Geuvadis EUR individuals with no overlap with MAGE, re-test the focal ×
background-SNP interaction at the top MAGE adQTL loci. This is an independent replication:
it tests the focal × bg interaction itself, not only the direction of the EUR effect.

Data:
  data/raw/Geuvadis/GD462.GeneQuantRPKM.txt.gz     — gene expression
  data/raw/Geuvadis/geuv_eur_indep.txt             — IDs of the 276 MAGE-independent EUR samples
  data/raw/Geuvadis/geuv_sex.txt                   — sex
  results/phase1/p1_cis_interaction.tsv             — MAGE interaction results (focal, best_bg positions)
  1000G VCF (HTTP range, via lib_remote_vcf.py)    — genotypes

Analysis:
  1. normalize expression (log2 RPKM → inverse-normal per gene)
  2. expression PC1-5 as covariates (in place of PEER factors)
  3. for the top 80 MAGE adQTL loci (bonf_p < 0.05) and an equal number of controls:
     fetch focal + best_bg genotypes from the 1000G VCF (276 Geuvadis EUR)
     fit expr ~ focal + bg + focal:bg + sex + PC1-5
     test the focal:bg interaction P value
  4. report:
     - distribution of Geuvadis interaction P values at MAGE-significant loci
     - sign concordance with the MAGE interaction direction (int_beta sign)
     - adQTL vs control enrichment

Outputs:
  results/validation/v5_geuvadis_focal_bg.tsv
  figures/validation/v5_geuvadis_focal_bg.{png,pdf}
"""

from pathlib import Path
import sys, gzip, io
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
GEUV    = BASE / "data" / "raw" / "Geuvadis"
P1      = BASE / "results" / "phase1"
OUT_DIR = BASE / "results" / "validation"
FIG_DIR = BASE / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / "scripts"))
from lib_remote_vcf import RemoteTabixVCF

import sys as _sys
_sys.path.insert(0, str(BASE / "scripts"))
from p0_05_fetch_genotypes import VCF_TMPL  # NYGC 3202 hg38 phased; chrom: chr1..chr22
POS_TOL = 15  # bp tolerance for position matching (MAGE vs VCF coords may differ slightly)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
AD  = "#d62728"; CO = "#1f77b4"

# ── helpers ──────────────────────────────────────────────────────────

def gt_to_dosage(gt_str):
    g = gt_str.split(":")[0].replace("|", "/")
    if g in (".", "./.", ".|."):
        return np.nan
    try:
        a = g.split("/")
        return int(a[0]) + int(a[1])
    except Exception:
        return np.nan


def inv_normal(x):
    """Per-gene inverse-normal transform (excluding NaN)."""
    r = stats.rankdata(x[~np.isnan(x)])
    n = len(r)
    p = (r - 0.5) / n
    z = stats.norm.ppf(p)
    out = np.full(len(x), np.nan)
    out[~np.isnan(x)] = z
    return out


# ── Load Geuvadis expression ─────────────────────────────────────────

def load_geuvadis_expr(sample_ids):
    """Load Geuvadis expression, return DataFrame (genes × samples)."""
    sid_set = set(sample_ids)
    rows = []
    with gzip.open(GEUV / "GD462.GeneQuantRPKM.txt.gz", "rt") as f:
        header = f.readline().strip().split("\t")
        sample_cols = header[4:]
        keep_idx = [i for i, s in enumerate(sample_cols) if s in sid_set]
        keep_sids = [sample_cols[i] for i in keep_idx]
        for line in f:
            p = line.strip().split("\t")
            gene = p[0].split(".")[0]  # strip version
            vals = np.array([float(p[4 + i]) if p[4 + i] != "NA" else np.nan
                             for i in keep_idx], float)
            rows.append((gene, vals))
    genes = [r[0] for r in rows]
    mat = np.array([r[1] for r in rows], float)  # genes × samples
    df = pd.DataFrame(mat, index=genes, columns=keep_sids)
    # log2 RPKM + 1 transform
    df = np.log2(df + 1)
    # inv-normal per gene
    for g in df.index:
        df.loc[g] = inv_normal(df.loc[g].values)
    return df


# ── Fetch single-position genotypes ──────────────────────────────────

def fetch_genotypes(chrom, pos, ref, alt, sample_ids, vsamp_map):
    """Return dosage array (len=len(sample_ids)) or None."""
    sid_idx = [vsamp_map[s] for s in sample_ids if s in vsamp_map]
    sid_ord = [s for s in sample_ids if s in vsamp_map]
    for ln in vcf_cache[chrom].query(chrom, pos - 1, pos + 1):
        f = ln.split("\t")
        if int(f[1]) != pos:
            continue
        if f[3] != ref or f[4] != alt:
            continue
        gts = f[9:]
        dos = np.array([gt_to_dosage(gts[vsamp_map[s]]) if s in vsamp_map else np.nan
                        for s in sample_ids], float)
        return dos
    return None


vcf_cache = {}


def get_vcf(chrom):
    if chrom not in vcf_cache:
        chr_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
        url = VCF_TMPL.format(chrom=chr_key)
        vcf_cache[chrom] = RemoteTabixVCF(url)
    return vcf_cache[chrom]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Load sample lists
    eur_indep = open(GEUV / "geuv_eur_indep.txt").read().strip().split("\n")
    sex_df    = pd.read_csv(GEUV / "geuv_sex.txt", sep="\t").set_index("sample")
    sex_bin   = {s: 1 if sex_df.loc[s, "sex"] == "male" else 0
                 for s in eur_indep if s in sex_df.index}
    samples   = [s for s in eur_indep if s in sex_bin]
    print(f"Geuvadis independent EUR: {len(samples)}")

    # Expression + PCs
    print("Loading Geuvadis expression…")
    expr = load_geuvadis_expr(samples)
    print(f"  Loaded {len(expr)} genes × {len(expr.columns)} samples")

    # Expression PCs (PC1-5 as covariates)
    ok_genes = expr.index[~expr.isnull().any(axis=1)]
    mat = expr.loc[ok_genes].values  # genes × samples
    mat -= mat.mean(axis=0); mat /= (mat.std(axis=0) + 1e-9)
    U, s_, Vt = np.linalg.svd(mat, full_matrices=False)
    pcs = Vt[:5].T  # samples × 5
    pc_df = pd.DataFrame(pcs, index=expr.columns, columns=[f"PC{i+1}" for i in range(5)])
    sex_arr = np.array([sex_bin[s] for s in samples], float)

    # MAGE cis-interaction results
    ci = pd.read_csv(P1 / "p1_cis_interaction.tsv", sep="\t")
    ci["chrom"] = ci["variant_kgpID"].str.split(":").str[0]
    ci["focal_pos"] = ci["variant_kgpID"].str.split(":").str[1].astype(int)
    ci["focal_ref"] = ci["variant_kgpID"].str.split(":").str[2]
    ci["focal_alt"] = ci["variant_kgpID"].str.split(":").str[3]
    # Best bg SNP position
    ci["bg_chrom"] = ci["best_bg"].str.split(":").str[0]
    ci["bg_pos"]   = ci["best_bg"].str.split(":").str[1].apply(
        lambda x: int(x) if pd.notna(x) and str(x).isdigit() else np.nan)
    ci["bg_ref"]   = ci["best_bg"].str.split(":").str[2]
    ci["bg_alt"]   = ci["best_bg"].str.split(":").str[3]
    # Keep rows with bg info + gene in Geuvadis
    ci["gene_base"] = ci["ensemblID"].str.split(".").str[0]
    ci_ok = ci.dropna(subset=["bg_pos","bg_chrom","bg_ref","bg_alt"]).copy()
    ci_ok = ci_ok[ci_ok["gene_base"].isin(expr.index)].copy()
    print(f"  MAGE loci with bg info + gene in Geuvadis: {len(ci_ok)}"
          f" (adQTL={( ci_ok.set=='adqtl').sum()}, ctrl={(ci_ok.set=='control').sum()})")

    # Get VCF sample map for each unique chromosome
    results = []
    CKPT    = OUT_DIR / "v5_geuvadis_focal_bg_ckpt.tsv"
    done    = set()
    if CKPT.exists():
        prev = pd.read_csv(CKPT, sep="\t")
        done = set(zip(prev.variant_kgpID, prev.ensemblID))
        results = prev.to_dict("records")
        print(f"  Resuming from checkpoint: {len(done)} done")

    first_write = not CKPT.exists()
    for _, r in ci_ok.iterrows():
        key = (r.variant_kgpID, r.ensemblID)
        if key in done:
            continue
        gene = r.gene_base
        chrom = r.chrom
        bg_chrom = r.bg_chrom if str(r.bg_chrom) != "nan" else chrom

        try:
            vcf = get_vcf(chrom)
            vsamp = {s: i for i, s in enumerate(vcf.samples())}
            chr_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
            # fetch focal dosages — tolerate ±POS_TOL bp position shift
            dos_focal = None
            for ln in vcf.query(chr_key, r.focal_pos - POS_TOL, r.focal_pos + POS_TOL):
                f2 = ln.split("\t")
                if abs(int(f2[1]) - r.focal_pos) <= POS_TOL and f2[3] == r.focal_ref and f2[4] == r.focal_alt:
                    dos_focal = np.array([gt_to_dosage(f2[9:][vsamp[s]])
                                          if s in vsamp else np.nan for s in samples], float)
                    break
            if dos_focal is None:
                print(f"  SKIP {r.variant_kgpID} focal not found")
                continue

            # fetch bg dosages
            vcf_bg = get_vcf(bg_chrom)
            vsamp_bg = {s: i for i, s in enumerate(vcf_bg.samples())}
            bg_chr_key = bg_chrom if bg_chrom.startswith("chr") else f"chr{bg_chrom}"
            bg_pos_int = int(r.bg_pos)
            dos_bg = None
            for ln in vcf_bg.query(bg_chr_key, bg_pos_int - POS_TOL, bg_pos_int + POS_TOL):
                f2 = ln.split("\t")
                if abs(int(f2[1]) - bg_pos_int) <= POS_TOL and f2[3] == r.bg_ref and f2[4] == r.bg_alt:
                    dos_bg = np.array([gt_to_dosage(f2[9:][vsamp_bg[s]])
                                       if s in vsamp_bg else np.nan for s in samples], float)
                    break
            if dos_bg is None:
                print(f"  SKIP {r.variant_kgpID} bg not found")
                continue

            # expression
            y = expr.loc[gene, samples].values.astype(float)
            C = np.column_stack([np.ones(len(samples)), sex_arr,
                                 pc_df.loc[samples, ["PC1","PC2","PC3","PC4","PC5"]].values])
            ok = (np.isfinite(dos_focal) & np.isfinite(dos_bg) &
                  np.isfinite(y) & np.isfinite(C).all(1))
            if ok.sum() < 40:
                print(f"  SKIP {r.variant_kgpID} n={ok.sum()}<40")
                continue
            fo = dos_focal[ok]; bgo = dos_bg[ok]
            if np.std(fo) < 0.01 or np.std(bgo) < 0.01:
                print(f"  SKIP {r.variant_kgpID} no variation")
                continue

            X = np.column_stack([C[ok], fo, bgo, fo * bgo])
            try:
                res = sm.OLS(y[ok], X).fit()
            except Exception as e:
                print(f"  OLS fail {r.variant_kgpID}: {e}")
                continue

            # interaction is last column
            p_int     = res.pvalues[-1]
            beta_int  = res.params[-1]
            row = {
                "variant_kgpID": r.variant_kgpID,
                "ensemblID": r.ensemblID,
                "set": r.set,
                "mage_int_p": r.min_int_p,
                "mage_bonf_p": r.bonf_p,
                "mage_int_beta": r.int_beta,
                "geuv_int_p": p_int,
                "geuv_int_beta": beta_int,
                "geuv_n": int(ok.sum()),
                "sign_conc": int(np.sign(r.int_beta) == np.sign(beta_int))
                             if pd.notna(r.int_beta) else np.nan,
            }
            results.append(row)
            done.add(key)
            print(f"  [{r.set}] {gene}  mage_p={r.min_int_p:.2e}  geuv_p={p_int:.2e}"
                  f"  sign={'Y' if row['sign_conc']==1 else 'N'}", flush=True)

            # checkpoint
            df_ckpt = pd.DataFrame(results)
            df_ckpt.to_csv(CKPT, sep="\t", index=False)

        except Exception as e:
            print(f"  ERROR {r.variant_kgpID}: {e}", flush=True)
            continue

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "v5_geuvadis_focal_bg.tsv", sep="\t", index=False)
    print(f"\nTotal tested: {len(df)} (adQTL={( df.set=='adqtl').sum()}, ctrl={(df.set=='control').sum()})")

    # ── Analysis ────────────────────────────────────────────────────
    adq  = df[df.set == "adqtl"].dropna(subset=["geuv_int_p"])
    ctrl = df[df.set == "control"].dropna(subset=["geuv_int_p"])
    thresh = 0.05

    rate_adq  = (adq.geuv_int_p  < thresh).mean() * 100
    rate_ctrl = (ctrl.geuv_int_p < thresh).mean() * 100
    if len(adq) > 0 and len(ctrl) > 0:
        fish = stats.fisher_exact([[int((adq.geuv_int_p  < thresh).sum()), len(adq)  - int((adq.geuv_int_p  < thresh).sum())],
                                   [int((ctrl.geuv_int_p < thresh).sum()), len(ctrl) - int((ctrl.geuv_int_p < thresh).sum())]],
                                  alternative="greater")
        or_v, fish_p = fish.statistic, fish.pvalue
    else:
        or_v, fish_p = np.nan, np.nan

    mage_sig = adq[adq.mage_bonf_p < 0.05]
    sc_pct = mage_sig.sign_conc.mean() * 100 if len(mage_sig) > 0 else np.nan
    mwu = stats.mannwhitneyu(-np.log10(adq.geuv_int_p.clip(1e-300)),
                              -np.log10(ctrl.geuv_int_p.clip(1e-300)),
                              alternative="greater") if len(adq)>0 and len(ctrl)>0 else None

    print(f"\nAdQTL p<{thresh}: {rate_adq:.1f}% ({(adq.geuv_int_p<thresh).sum()}/{len(adq)})")
    print(f"Ctrl  p<{thresh}: {rate_ctrl:.1f}% ({(ctrl.geuv_int_p<thresh).sum()}/{len(ctrl)})")
    print(f"Fisher OR={or_v:.2f} p={fish_p:.2e}")
    print(f"MAGE-sig adQTL sign concordance: {sc_pct:.1f}% (n={len(mage_sig)})")
    if mwu: print(f"MWU p (adQTL > ctrl in -log10p): {mwu.pvalue:.2e}")

    # Save summary
    summary = {
        "n_adqtl_tested": len(adq), "n_ctrl_tested": len(ctrl),
        "adqtl_sig_pct": round(rate_adq, 1), "ctrl_sig_pct": round(rate_ctrl, 1),
        "fisher_OR": round(float(or_v), 2), "fisher_p": float(fish_p),
        "mage_sig_adqtl_sign_conc_pct": round(float(sc_pct), 1) if not np.isnan(sc_pct) else np.nan,
        "mage_sig_n": len(mage_sig),
        "mwu_p": float(mwu.pvalue) if mwu else np.nan,
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "v5_geuvadis_focal_bg_summary.tsv", sep="\t", index=False)

    # ── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # A: bar chart sig rate
    ax = axes[0]
    ax.bar([0,1], [rate_ctrl, rate_adq], color=[CO, AD], width=0.5, edgecolor="black", lw=0.8)
    ax.text(0, rate_ctrl+0.5, f"{rate_ctrl:.1f}%\n(n={len(ctrl)})", ha="center", fontsize=9, fontweight="bold")
    ax.text(1, rate_adq+0.5,  f"{rate_adq:.1f}%\n(n={len(adq)})",  ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(["concordant\n(control)", "adQTL"])
    ax.set_ylabel(f"% with nominal p<{thresh}\n(Geuvadis focal×bg)"); ax.set_ylim(0, max(rate_adq,rate_ctrl)+12)
    ax.text(0.98, 0.97, f"OR={or_v:.1f}\np={fish_p:.1e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ax.set_title("A  focal×bg interaction in Geuvadis\n(independent EUR, n=276)", fontweight="bold", loc="left", fontsize=9)

    # B: MAGE vs Geuvadis int p scatter
    ax2 = axes[1]
    if len(adq)>0:
        ax2.scatter(-np.log10(adq.mage_int_p.clip(1e-300)),
                    -np.log10(adq.geuv_int_p.clip(1e-300)),
                    s=18, color=AD, alpha=0.65, label=f"adQTL (n={len(adq)})", zorder=3)
    if len(ctrl)>0:
        ax2.scatter(-np.log10(ctrl.mage_int_p.clip(1e-300)),
                    -np.log10(ctrl.geuv_int_p.clip(1e-300)),
                    s=10, color=CO, alpha=0.45, label=f"concordant (n={len(ctrl)})", zorder=2)
    if len(adq)>0:
        r_sp = stats.spearmanr(-np.log10(adq.mage_int_p.clip(1e-300)),
                               -np.log10(adq.geuv_int_p.clip(1e-300)))[0]
        ax2.text(0.05, 0.95, f"adQTL Spearman r={r_sp:.2f}", transform=ax2.transAxes,
                 va="top", fontsize=8.5)
    ax2.set_xlabel("MAGE -log10(int p)"); ax2.set_ylabel("Geuvadis -log10(int p)")
    ax2.set_title("B  MAGE vs Geuvadis interaction p", fontweight="bold", loc="left", fontsize=9)
    ax2.legend(fontsize=8, frameon=False)

    # C: sign concordance (MAGE sig adQTL)
    ax3 = axes[2]
    if len(mage_sig) > 0:
        conc = mage_sig.sign_conc.mean()*100
        disc = 100 - conc
        ax3.pie([conc, disc], labels=[f"concordant\n{conc:.1f}%", f"discordant\n{disc:.1f}%"],
                colors=[AD, "#aaaaaa"], startangle=90, autopct=None,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
        ax3.set_title(f"C  Interaction sign concordance\n(MAGE bonf_p<0.05, n={len(mage_sig)})",
                      fontweight="bold", loc="left", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "no MAGE-sig loci matched", ha="center"); ax3.axis("off")

    fig.suptitle("V5: Geuvadis independent replication of focal×background cis-interactions",
                 fontweight="bold", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "v5_geuvadis_focal_bg.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"\nFigure -> {out}")
    print("[v5] DONE.")


if __name__ == "__main__":
    main()
