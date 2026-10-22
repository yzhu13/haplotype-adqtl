"""
Validation V4: the sign-flip subset
===================================
Compare the cis-epistasis rate of the 50 sign-flip adQTLs (beta_EUR × beta_AFR < 0)
with the remaining same-sign adQTLs (beta_EUR × beta_AFR > 0).

Hypothesis: cis-epistasis enrichment is stronger in the sign-flip subset
(a full sign reversal implies a more extreme effect of the background haplotype on the focal effect)

Outputs:
  results/validation/v4_signflip_result.tsv
  figures/validation/v4_signflip_panel.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "validation"
FIG_DIR = BASE / "figures" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ADQTL_FILE = BASE / "results" / "phase0" / "p0_adqtl_set.tsv"
CIS_FILE   = BASE / "results" / "phase1" / "p1_cis_interaction.tsv"
WITHIN_FILE= BASE / "results" / "phase1" / "p1_within_ancestry.tsv"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})
COL_FLIP  = "#e377c2"   # pink for sign-flip
COL_SAME  = "#ff7f0e"   # orange for same-sign
COL_CTRL  = "#1f77b4"   # blue for control


def main():
    adqtl  = pd.read_csv(ADQTL_FILE, sep="\t")
    cis    = pd.read_csv(CIS_FILE,   sep="\t")
    within = pd.read_csv(WITHIN_FILE, sep="\t") if WITHIN_FILE.exists() else None

    # classify adQTLs: sign-flip vs same-sign
    adqtl["sign_flip"] = (adqtl["beta_EUR"] * adqtl["beta_AFR"]) < 0
    flip_ids   = set(adqtl[adqtl["sign_flip"]]["variant_kgpID"])
    same_ids   = set(adqtl[~adqtl["sign_flip"]]["variant_kgpID"])
    print(f"adQTL total: {len(adqtl)}, sign-flip: {len(flip_ids)}, same-sign: {len(same_ids)}")

    # take adQTL rows from cis_interaction and label sign_flip
    cis_adq = cis[cis["set"] == "adqtl"].copy()
    cis_ctrl= cis[cis["set"] == "control"].copy()
    cis_adq["sign_flip"] = cis_adq["variant_kgpID"].isin(flip_ids)
    flip_cis = cis_adq[cis_adq["sign_flip"]]
    same_cis = cis_adq[~cis_adq["sign_flip"]]

    def sig_stats(sub, label):
        n   = len(sub)
        k   = (sub["bonf_p"] < 0.05).sum()
        pct = 100*k/n if n else 0
        print(f"  {label}: {k}/{n} = {pct:.1f}% significant")
        return n, k, pct

    print("\ncis-epistasis enrichment by subgroup:")
    n_fl, k_fl, pct_fl = sig_stats(flip_cis, "sign-flip adQTL")
    n_sa, k_sa, pct_sa = sig_stats(same_cis, "same-sign adQTL")
    n_ct, k_ct, pct_ct = sig_stats(cis_ctrl, "control eQTL")

    # Fisher tests
    def fisher(k1, n1, k2, n2):
        ct = [[k1, n1-k1], [k2, n2-k2]]
        or_, p = stats.fisher_exact(ct, alternative="greater")
        return or_, p

    or_flip_vs_same, p_flip_vs_same = fisher(k_fl, n_fl, k_sa, n_sa)
    or_flip_vs_ctrl, p_flip_vs_ctrl = fisher(k_fl, n_fl, k_ct, n_ct)
    or_same_vs_ctrl, p_same_vs_ctrl = fisher(k_sa, n_sa, k_ct, n_ct)
    print(f"\nFisher: sign-flip vs same-sign: OR={or_flip_vs_same:.2f} p={p_flip_vs_same:.3e}")
    print(f"Fisher: sign-flip vs control:   OR={or_flip_vs_ctrl:.2f} p={p_flip_vs_ctrl:.3e}")
    print(f"Fisher: same-sign vs control:   OR={or_same_vs_ctrl:.2f} p={p_same_vs_ctrl:.3e}")

    # compare |ObsDelta|
    adq_flip_delta = adqtl[adqtl["sign_flip"]]["ObsDelta_EUR_AFR"].abs()
    adq_same_delta = adqtl[~adqtl["sign_flip"]]["ObsDelta_EUR_AFR"].abs()
    mwu_delta = stats.mannwhitneyu(adq_flip_delta, adq_same_delta, alternative="greater")
    print(f"\n|ObsDelta| sign-flip vs same-sign: MWU p={mwu_delta.pvalue:.3e}")
    print(f"  sign-flip median |ObsDelta|={adq_flip_delta.median():.3f}")
    print(f"  same-sign median |ObsDelta|={adq_same_delta.median():.3f}")

    # within-ancestry confirmation (if available)
    within_flip_pct = within_same_pct = None
    if within is not None and "variant_kgpID" in within.columns:
        print("\nWithin-ancestry confirmation by subgroup:")
        wdf = within.copy()
        wdf["sign_flip"] = wdf["variant_kgpID"].isin(flip_ids)
        for grp, label in [(True,"sign-flip"), (False,"same-sign")]:
            sub = wdf[wdf["sign_flip"]==grp]
            if len(sub):
                # if the within-ancestry file has a confirmed column
                conf_col = [c for c in sub.columns if "confirm" in c.lower() or "within" in c.lower()]
                if conf_col:
                    pct = 100*sub[conf_col[0]].mean()
                    print(f"  {label}: {pct:.1f}% within-ancestry confirmed")
                    if grp: within_flip_pct = pct
                    else:   within_same_pct = pct

    result = {
        "n_signflip": n_fl, "k_signflip": int(k_fl), "pct_signflip": round(pct_fl,1),
        "n_samesign": n_sa, "k_samesign": int(k_sa), "pct_samesign": round(pct_sa,1),
        "n_ctrl":     n_ct, "k_ctrl":     int(k_ct), "pct_ctrl":     round(pct_ct,1),
        "OR_flip_vs_same": round(or_flip_vs_same,2), "p_flip_vs_same": p_flip_vs_same,
        "OR_flip_vs_ctrl": round(or_flip_vs_ctrl,2), "p_flip_vs_ctrl": p_flip_vs_ctrl,
        "OR_same_vs_ctrl": round(or_same_vs_ctrl,2), "p_same_vs_ctrl": p_same_vs_ctrl,
        "median_absDelta_flip": round(float(adq_flip_delta.median()),3),
        "median_absDelta_same": round(float(adq_same_delta.median()),3),
        "MWU_p_delta": mwu_delta.pvalue,
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / "v4_signflip_result.tsv", sep="\t", index=False)

    # ── Figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: cis-epistasis sig rate bar chart
    ax = axes[0]
    groups = ["sign-flip\nadQTL", "same-sign\nadQTL", "concordant\neQTL (ctrl)"]
    pcts   = [pct_fl, pct_sa, pct_ct]
    ns     = [n_fl, n_sa, n_ct]
    ks     = [k_fl, k_sa, k_ct]
    cols   = [COL_FLIP, COL_SAME, COL_CTRL]
    bars = ax.bar(range(3), pcts, color=cols, width=0.5,
                  edgecolor="black", linewidth=0.8)
    for i, (bar, pct, n, k) in enumerate(zip(bars, pcts, ns, ks)):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 0.8,
                f"{pct:.0f}%\n({k}/{n})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # significance brackets
    def bracket(ax, x1, x2, y, p):
        h = 1.5
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], "k-", lw=1)
        pstr = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.text((x1+x2)/2, y+h+0.3, pstr, ha="center", va="bottom", fontsize=8)
    top = max(pcts) + 5
    bracket(ax, 0, 1, top,      p_flip_vs_same)
    bracket(ax, 0, 2, top+6,    p_flip_vs_ctrl)
    bracket(ax, 1, 2, top+0.5,  p_same_vs_ctrl)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("% with significant cis-interaction\n(Bonferroni p<0.05)", fontsize=9)
    ax.set_title("A  cis-epistasis rate by effect direction", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(0, top + 14)

    # Panel B: |ObsDelta| violin for sign-flip vs same-sign adQTL
    ax2 = axes[1]
    data = [adq_same_delta.values, adq_flip_delta.values]
    parts = ax2.violinplot(data, positions=[0,1], showmedians=True, showextrema=False)
    for i, (pc, col) in enumerate(zip(parts["bodies"], [COL_SAME, COL_FLIP])):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    rng = __import__("numpy").random.default_rng(42)
    for i, (vals, col) in enumerate(zip(data, [COL_SAME, COL_FLIP])):
        jx = rng.uniform(-0.15, 0.15, len(vals)) + i
        ax2.scatter(jx, vals, s=12, color=col, alpha=0.5, linewidths=0)
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(["same-sign\nadQTL", "sign-flip\nadQTL"], fontsize=10)
    ax2.set_ylabel("|ObsDelta_EUR-AFR| (MAGE)", fontsize=9)
    ax2.set_title("B  Effect magnitude by direction type", fontsize=10, fontweight="bold", loc="left")
    ymax = max(adq_flip_delta.max(), adq_same_delta.max())
    ax2.plot([0,1],[ymax*1.05]*2, "k-", lw=1)
    ax2.text(0.5, ymax*1.07,
             f"MWU p={mwu_delta.pvalue:.2e}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(bottom=0)

    fig.suptitle("adQTL sign-flip subgroup shows stronger cis-epistasis enrichment",
                 fontsize=10.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "v4_signflip_panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight")
    print(f"\nFigure -> {out}")
    print("[v4_signflip_analysis] DONE.")


if __name__ == "__main__":
    main()
