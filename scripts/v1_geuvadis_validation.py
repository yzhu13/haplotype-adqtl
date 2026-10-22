"""
Validation V1: independent replication in Geuvadis
===================================
Data limitations:
  - Geuvadis EUR n=373 (adequate) -> 420k SNP-gene pairs
  - Geuvadis YRI n=89 (small) -> only ~20k SNP-gene pairs (few FDR 5% hits)
  - Overlap with the 396 MAGE genes: 181 genes in EUR, 33 in YRI

Analysis strategy (two tiers):
  A. Primary test (n=70+54): independent replication of the EUR effect
     - For matched MAGE lead variants, test whether the sign of MAGE beta_EUR agrees with
       Geuvadis r_EUR (sign concordance)
     - compare concordance between adQTL and control
     - test the correlation between MAGE |ObsDelta_EUR_AFR| and |r_EUR_geuv|
       (if the MAGE EUR effect of adQTLs is real, it should also be stronger in Geuvadis EUR)

  B. Secondary test (n=11+11): EUR-YRI direction
     - only the 22 variants that also have FDR 5% data in YRI
     - MAGE sign(ObsDelta_EUR_AFR) vs Geuvadis sign(r_EUR - r_YRI)
     - small sample; report the trend and state the limitation

Figure output: 3 panels
  A. EUR sign concordance (adQTL vs control) - main evidence
  B. Geuvadis |r_EUR| vs MAGE |ObsDelta_EUR_AFR| scatter (adQTL highlighted)
  C. EUR-YRI direction concordance for 11+11 (secondary; small n)
"""

import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from pyliftover import LiftOver
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "raw" / "Geuvadis"
OUT_DIR = BASE / "results" / "validation"
FIG_DIR = BASE / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

EUR_FILE   = DATA / "EUR373.gene.cis.FDR5.all.txt.gz"
YRI_FILE   = DATA / "YRI89.gene.cis.FDR5.all.txt.gz"
ADQTL_FILE = BASE / "results" / "phase0" / "p0_adqtl_set.tsv"
CTRL_FILE  = BASE / "results" / "phase0" / "p0_control_set.tsv"

POS_TOL = 5    # hg38 position-matching tolerance (bp)

# ── publication style ──────────────────────────────────────────
PLT_PARAMS = {
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}
plt.rcParams.update(PLT_PARAMS)
COL_ADQ  = "#d62728"
COL_CTRL = "#1f77b4"


def load_geuvadis():
    print("Loading Geuvadis EUR373 and YRI89...")
    eur = pd.read_csv(EUR_FILE, sep="\t")
    yri = pd.read_csv(YRI_FILE, sep="\t")
    for df in [eur, yri]:
        df["gene_base"] = df["GENE_ID"].str.split(".").str[0]
    lo = LiftOver("hg19", "hg38")
    def lift(chrom, pos):
        r = lo.convert_coordinate(f"chr{int(chrom)}", int(pos))
        return r[0][1] if r else None
    print("  Lifting EUR hg19->hg38 ...")
    eur["pos_hg38"] = [lift(c, p) for c, p in zip(eur["CHR_SNP"], eur["SNPpos"])]
    print("  Lifting YRI hg19->hg38 ...")
    yri["pos_hg38"] = [lift(c, p) for c, p in zip(yri["CHR_SNP"], yri["SNPpos"])]
    for df in [eur, yri]:
        df.dropna(subset=["pos_hg38"], inplace=True)
        df["pos_hg38"] = df["pos_hg38"].astype(int)
    print(f"  EUR: {len(eur)} pairs; YRI: {len(yri)} pairs")
    return eur, yri


def match_variants(mage_df, eur_df, yri_df, set_label):
    mage_df = mage_df.copy()
    mage_df["gene_base"] = mage_df["ensemblID"].str.split(".").str[0]
    mage_df["chrom"]     = mage_df["variant_kgpID"].str.split(":").str[0].astype(int)
    mage_df["pos_mage"]  = mage_df["variant_kgpID"].str.split(":").str[1].astype(int)

    eur_grp = {g: sub for g, sub in eur_df.groupby("gene_base")}
    yri_grp = {g: sub for g, sub in yri_df.groupby("gene_base")}

    rows = []
    for _, r in mage_df.iterrows():
        gb, chrom, pos = r["gene_base"], r["chrom"], r["pos_mage"]
        eur_sub = eur_grp.get(gb)
        if eur_sub is None:
            continue
        eur_chr = eur_sub[eur_sub["CHR_SNP"].astype(int) == chrom]
        if len(eur_chr) == 0:
            continue
        dist = (eur_chr["pos_hg38"] - pos).abs()
        bi = dist.idxmin()
        if dist[bi] > POS_TOL:
            continue
        er = eur_chr.loc[bi]
        r_yri = p_yri = np.nan
        yri_sub = yri_grp.get(gb)
        if yri_sub is not None:
            yri_chr = yri_sub[yri_sub["CHR_SNP"].astype(int) == chrom]
            if len(yri_chr) > 0:
                dy = (yri_chr["pos_hg38"] - pos).abs()
                bi2 = dy.idxmin()
                if dy[bi2] <= POS_TOL:
                    r_yri = yri_chr.loc[bi2, "rvalue"]
                    p_yri = yri_chr.loc[bi2, "pvalue"]
        rows.append({
            "variant_kgpID":    r["variant_kgpID"],
            "ensemblID":        r["ensemblID"],
            "geneSymbol":       r.get("geneSymbol", ""),
            "set":              set_label,
            "beta_EUR_mage":    r["beta_EUR"],
            "beta_AFR_mage":    r["beta_AFR"],
            "ObsDelta_EUR_AFR": r["ObsDelta_EUR_AFR"],
            "r_EUR_geuv":       er["rvalue"],
            "p_EUR_geuv":       er["pvalue"],
            "r_YRI_geuv":       r_yri,
            "p_YRI_geuv":       p_yri,
            "geuv_snp":         er["SNP_ID"],
            "pos_hg38_match":   int(er["pos_hg38"]),
        })
    df = pd.DataFrame(rows)
    print(f"  {set_label}: {len(mage_df)} -> {len(df)} EUR-matched "
          f"({df['r_YRI_geuv'].notna().sum()} also in YRI)")
    return df


def main():
    eur_df, yri_df = load_geuvadis()
    adqtl = pd.read_csv(ADQTL_FILE, sep="\t")
    ctrl  = pd.read_csv(CTRL_FILE,  sep="\t")

    print("\nMatching variants...")
    m_adq  = match_variants(adqtl, eur_df, yri_df, "adqtl")
    m_ctrl = match_variants(ctrl,  eur_df, yri_df, "control")
    all_m  = pd.concat([m_adq, m_ctrl], ignore_index=True)
    all_m.to_csv(OUT_DIR / "v1_geuvadis_match.tsv", sep="\t", index=False)

    # ── A. EUR sign concordance (primary test)──────────────────────────────
    all_m["sign_EUR_mage"]  = np.sign(all_m["beta_EUR_mage"])
    all_m["sign_EUR_geuv"]  = np.sign(all_m["r_EUR_geuv"])
    all_m["EUR_concordant"] = all_m["sign_EUR_mage"] == all_m["sign_EUR_geuv"]

    for label, sub in [("adqtl", m_adq), ("control", m_ctrl)]:
        sub2 = all_m[all_m["set"] == label]
        n_c = sub2["EUR_concordant"].sum()
        n   = len(sub2)
        bp  = stats.binomtest(n_c, n, 0.5, alternative="greater")
        print(f"EUR sign concordance [{label}]: {n_c}/{n} = {100*n_c/n:.1f}%  binom p={bp.pvalue:.3e}")

    all_m["abs_ObsDelta"] = all_m["ObsDelta_EUR_AFR"].abs()
    all_m["abs_r_EUR"]    = all_m["r_EUR_geuv"].abs()
    adq_sub  = all_m[all_m["set"] == "adqtl"]
    ctrl_sub = all_m[all_m["set"] == "control"]
    n_ca = adq_sub["EUR_concordant"].sum();  n_a = len(adq_sub)
    n_cc = ctrl_sub["EUR_concordant"].sum(); n_c2 = len(ctrl_sub)
    bp_a = stats.binomtest(n_ca, n_a, 0.5, alternative="greater")
    bp_c = stats.binomtest(n_cc, n_c2, 0.5, alternative="greater")

    # ── B. |ObsDelta| vs |r_EUR_geuv| correlation ───────────────────
    all_m["abs_ObsDelta"] = all_m["ObsDelta_EUR_AFR"].abs()
    all_m["abs_r_EUR"]    = all_m["r_EUR_geuv"].abs()
    rho_a, p_rho_a = stats.spearmanr(adq_sub["abs_ObsDelta"], adq_sub["abs_r_EUR"])
    rho_c, p_rho_c = stats.spearmanr(ctrl_sub["abs_ObsDelta"], ctrl_sub["abs_r_EUR"])
    print(f"\n|ObsDelta| vs |r_EUR_geuv| Spearman: adqtl rho={rho_a:.3f} p={p_rho_a:.3e}, ctrl rho={rho_c:.3f} p={p_rho_c:.3e}")

    # ── C. EUR-YRI direction concordance (secondary, small n)─────────────────────
    with_yri = all_m.dropna(subset=["r_YRI_geuv"]).copy()
    with_yri["delta_r_geuv"] = with_yri["r_EUR_geuv"] - with_yri["r_YRI_geuv"]
    with_yri["sign_mage"]    = np.sign(with_yri["ObsDelta_EUR_AFR"])
    with_yri["sign_geuv"]    = np.sign(with_yri["delta_r_geuv"])
    with_yri["concordant_EY"] = with_yri["sign_mage"] == with_yri["sign_geuv"]
    ey_adq  = with_yri[with_yri["set"] == "adqtl"]
    ey_ctrl = with_yri[with_yri["set"] == "control"]
    n_ey_a = ey_adq["concordant_EY"].sum();  n_ey_at = len(ey_adq)
    n_ey_c = ey_ctrl["concordant_EY"].sum(); n_ey_ct = len(ey_ctrl)
    bp_ey  = stats.binomtest(n_ey_a, n_ey_at, 0.5, alternative="greater") if n_ey_at > 0 else None
    print(f"\nEUR-YRI direction concordance (n={n_ey_at+n_ey_ct} total):")
    print(f"  adqtl:  {n_ey_a}/{n_ey_at} = {100*n_ey_a/n_ey_at:.0f}%  binom p={bp_ey.pvalue:.3f}" if n_ey_at>0 else "  adqtl: n=0")
    print(f"  control:{n_ey_c}/{n_ey_ct} = {100*n_ey_c/n_ey_ct:.0f}%" if n_ey_ct>0 else "  control: n=0")

    # save results
    result = {
        "n_adqtl_EUR_matched": n_a, "n_ctrl_EUR_matched": n_c2,
        "n_adqtl_YRI_also":    n_ey_at, "n_ctrl_YRI_also": n_ey_ct,
        "EUR_concordance_adqtl_pct": round(100*n_ca/n_a,1),
        "EUR_concordance_ctrl_pct":  round(100*n_cc/n_c2,1),
        "EUR_binom_p_adqtl": bp_a.pvalue, "EUR_binom_p_ctrl": bp_c.pvalue,
        "spearman_rho_adqtl": round(rho_a,3), "spearman_p_adqtl": p_rho_a,
        "spearman_rho_ctrl":  round(rho_c,3), "spearman_p_ctrl":  p_rho_c,
        "EY_concordance_adqtl_pct": round(100*n_ey_a/n_ey_at,1) if n_ey_at>0 else None,
        "EY_concordance_ctrl_pct":  round(100*n_ey_c/n_ey_ct,1) if n_ey_ct>0 else None,
        "EY_binom_p_adqtl": bp_ey.pvalue if bp_ey else None,
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / "v1_geuvadis_result.tsv", sep="\t", index=False)

    # ── FIGURE ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: EUR sign-concordance bar chart
    ax = axes[0]
    pcts = [result["EUR_concordance_ctrl_pct"], result["EUR_concordance_adqtl_pct"]]
    ns   = [n_c2, n_a]
    bars = ax.bar([0,1], pcts, color=[COL_CTRL, COL_ADQ], width=0.5,
                  edgecolor="black", linewidth=0.8)
    ax.axhline(50, color="gray", linestyle="--", lw=1.2, label="Chance (50%)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Concordant\neQTL\n(control)", "adQTL"], fontsize=10)
    ax.set_ylabel("MAGE beta_EUR sign == Geuvadis r_EUR sign (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("A  EUR effect replication\n(Geuvadis, n=373 EUR)", fontsize=10, fontweight="bold", loc="left")
    for bar, pct, n, bp in zip(bars, pcts, ns, [bp_c, bp_a]):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 1.5,
                f"{pct:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")
        pstr = f"p={bp.pvalue:.2e}" if bp.pvalue < 0.05 else f"p={bp.pvalue:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, 5, pstr,
                ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: |ObsDelta_MAGE| vs |r_EUR_Geuvadis| scatter
    ax2 = axes[1]
    for label, sub, col, mk in [("control", ctrl_sub, COL_CTRL, "o"), ("adqtl", adq_sub, COL_ADQ, "^")]:
        ax2.scatter(sub["abs_ObsDelta"], sub["abs_r_EUR"],
                    s=30, color=col, alpha=0.55, marker=mk, linewidths=0,
                    label=label.replace("adqtl","adQTL").replace("control","concordant eQTL"))
    # trend line for adQTL
    x, y = adq_sub["abs_ObsDelta"].values, adq_sub["abs_r_EUR"].values
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax2.plot(xr, m*xr+b, color=COL_ADQ, lw=1.5, ls="--", alpha=0.8)
    ax2.set_xlabel("|ObsDelta_EUR-AFR| (MAGE)", fontsize=9)
    ax2.set_ylabel("|r_EUR| (Geuvadis)", fontsize=9)
    ax2.set_title(f"B  MAGE adQTL effect size\nvs Geuvadis EUR replication", fontsize=10, fontweight="bold", loc="left")
    ax2.text(0.97, 0.05, f"adQTL Spearman rho={rho_a:.2f}\np={p_rho_a:.2e}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax2.legend(fontsize=8, loc="upper left")

    # Panel C: EUR-YRI direction (small-sample, secondary)
    ax3 = axes[2]
    if len(with_yri) > 0:
        for label, sub, col, mk in [
            ("control", ey_ctrl, COL_CTRL, "o"),
            ("adqtl",   ey_adq,  COL_ADQ,  "^"),
        ]:
            ax3.scatter(sub["ObsDelta_EUR_AFR"], sub["delta_r_geuv"],
                        s=45, color=col, alpha=0.7, marker=mk, linewidths=0.5,
                        edgecolors="white",
                        label=label.replace("adqtl","adQTL").replace("control","concordant eQTL"))
        ax3.axhline(0, color="gray", lw=0.8, ls="--")
        ax3.axvline(0, color="gray", lw=0.8, ls="--")
        # diagonal reference line
        lim = max(abs(with_yri["ObsDelta_EUR_AFR"]).max(), abs(with_yri["delta_r_geuv"]).max()) * 1.1
        ax3.plot([-lim, lim], [-lim*0.3, lim*0.3], color="gray", lw=0.7, ls=":", alpha=0.5)
        # annotate adQTL concordance
        if n_ey_at > 0:
            ax3.text(0.97, 0.97,
                     f"adQTL: {n_ey_a}/{n_ey_at} concordant\n({100*n_ey_a/n_ey_at:.0f}%, p={bp_ey.pvalue:.2f})\n"
                     f"[NOTE: n small, YRI n=89]",
                     transform=ax3.transAxes, ha="right", va="top", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax3.set_xlabel("MAGE ObsDelta (beta_EUR - beta_AFR)", fontsize=9)
    ax3.set_ylabel("Geuvadis r_EUR - r_YRI", fontsize=9)
    ax3.set_title(f"C  EUR-YRI direction (Geuvadis)\n[secondary; YRI n=89, only {len(with_yri)} matched]",
                  fontsize=9.5, fontweight="bold", loc="left")
    ax3.legend(fontsize=8, loc="lower right")

    fig.suptitle("Geuvadis independent validation of MAGE adQTL effects",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_png = FIG_DIR / "v1_geuvadis_panel.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_png).replace(".png",".pdf"), bbox_inches="tight")
    print(f"\nFigure -> {out_png}")
    print("[v1_geuvadis_validation] DONE.")


if __name__ == "__main__":
    main()
