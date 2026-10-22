"""
Phase 0 — Step 04: Analyze the calibration run and render the GO/NO-GO verdict for
sub-gate A (can AlphaGenome predict POOLED eQTL effect in matched LCL cell type?).

Compares predicted cis-regulatory effect (p0_03) vs MAGE observed pooled log2_aFC.

PRE-REGISTERED thresholds (set before seeing results):
  Sub-gate A passes if the matched RNA readout (pred_rna_lfc) shows, on the
  near-TSS stratum (|var-TSS| <= 50 kb):
     (1) Spearman r vs observed log2_aFC > 0.20 AND p < 0.05, AND
     (2) sign concordance > 60%.
  Marginal: one of two met -> proceed with caution + larger n.
  Fail: neither met -> trigger pivot narrative.

Outputs:
  figures/phase0/p0_calibration_scatter.png
  results/phase0/p0_calibration_stats.tsv
  docs/phase0_subgateA_report.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "phase0"
FIG = BASE / "figures" / "phase0"
DOCS = BASE / "docs"
PRED = RES / "p0_alphagenome_predictions.tsv"

NEAR_TSS = 50_000
R_THRESH = 0.20
SIGN_THRESH = 0.60


def stats_block(df, pred_col, label):
    """Return dict of correlation + sign-concordance stats for a subset."""
    d = df[np.isfinite(df[pred_col]) & np.isfinite(df["log2_aFC"])]
    d = d[(d[pred_col] != 0) & (d["log2_aFC"] != 0)]
    if len(d) < 5:
        return {"label": label, "n": len(d), "spearman_r": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "sign_concordance": np.nan}
    sr, sp = stats.spearmanr(d[pred_col], d["log2_aFC"])
    pr, _ = stats.pearsonr(d[pred_col], d["log2_aFC"])
    sign = float((np.sign(d[pred_col]) == np.sign(d["log2_aFC"])).mean())
    return {"label": label, "n": len(d), "spearman_r": sr, "spearman_p": sp,
            "pearson_r": pr, "sign_concordance": sign}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED, sep="\t")
    print(f"[p0_04] loaded {len(df)} predictions")

    rows = []
    for pred_col in ["pred_rna_lfc", "pred_atac_lfc"]:
        rows.append(stats_block(df, pred_col, f"{pred_col} | all"))
        rows.append(stats_block(df[df.var_tss_dist <= NEAR_TSS], pred_col,
                                f"{pred_col} | near-TSS<=50kb"))
        rows.append(stats_block(df[df.var_tss_dist > NEAR_TSS], pred_col,
                                f"{pred_col} | distal>50kb"))
    sdf = pd.DataFrame(rows)
    sdf.to_csv(RES / "p0_calibration_stats.tsv", sep="\t", index=False)
    print(sdf.to_string(index=False))

    # Verdict from RNA near-TSS stratum
    key = sdf[sdf.label == "pred_rna_lfc | near-TSS<=50kb"].iloc[0]
    c1 = (key.spearman_r > R_THRESH) and (key.spearman_p < 0.05)
    c2 = key.sign_concordance > SIGN_THRESH
    verdict = "GO" if (c1 and c2) else ("MARGINAL" if (c1 or c2) else "NO-GO (pivot)")

    # Figure: scatter pred_rna vs obs aFC, colored by distance stratum
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pc, title in [(axes[0], "pred_rna_lfc", "RNA gene-LFC (matched aFC readout)"),
                          (axes[1], "pred_atac_lfc", "ATAC center-mask (accessibility)")]:
        near = df[df.var_tss_dist <= NEAR_TSS]
        dist = df[df.var_tss_dist > NEAR_TSS]
        ax.scatter(dist[pc], dist["log2_aFC"], s=14, alpha=0.4, label="distal>50kb", color="#bbb")
        ax.scatter(near[pc], near["log2_aFC"], s=18, alpha=0.7, label="near-TSS<=50kb", color="#c0392b")
        ax.axhline(0, lw=0.5, color="k"); ax.axvline(0, lw=0.5, color="k")
        st = stats_block(near, pc, "")
        ax.set_title(f"{title}\nnear-TSS Spearman r={st['spearman_r']:.2f} "
                     f"p={st['spearman_p']:.1e} sign={st['sign_concordance']:.0%} n={st['n']}")
        ax.set_xlabel(f"AlphaGenome {pc}"); ax.set_ylabel("MAGE observed log2_aFC")
        ax.legend(fontsize=8)
    fig.suptitle(f"Phase 0 sub-gate A — pooled eQTL prediction in LCL (GM12878)   "
                 f"VERDICT: {verdict}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p0_calibration_scatter.png", dpi=150)
    print(f"[fig] -> {FIG/'p0_calibration_scatter.png'}")

    # Report
    report = [
        "# Phase 0 — Sub-gate A report (pooled eQTL prediction, LCL)",
        f"\n**VERDICT: {verdict}**  (n predictions = {len(df)})",
        f"\nPre-registered thresholds: near-TSS Spearman r>{R_THRESH} & p<0.05 (c1={c1}); "
        f"sign concordance>{SIGN_THRESH:.0%} (c2={c2}).",
        "\n## Key stratum (RNA @TSS, near-TSS <=50kb)",
        f"- n = {int(key.n)}",
        f"- Spearman r = {key.spearman_r:.3f} (p = {key.spearman_p:.2e})",
        f"- Pearson r = {key.pearson_r:.3f}",
        f"- Sign concordance = {key.sign_concordance:.1%}",
        "\n## All strata\n",
        sdf.to_markdown(index=False),
        "\n## Interpretation",
        "- GO: proceed to per-ancestry effect estimation (needs genotypes) + ancestry-differential test.",
        "- MARGINAL: rerun with larger n; consider gene-level variant scorer instead of TSS-window aggregation.",
        "- NO-GO: trigger pivot narrative — sequence model cannot see LCL eQTL effects here.",
        "\n*Note:* ATAC@variant is a secondary local-accessibility readout; weak correlation with aFC "
        "(an expression effect) is expected and not part of the gate.",
    ]
    (DOCS / "phase0_subgateA_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[report] -> {DOCS/'phase0_subgateA_report.md'}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
