"""
Phase 1 — Step 17: Publication-quality figure for trans-chromosomal negative control.

Panel A: Cis vs Trans enrichment bar chart (4 bars)
Panel B: ECDF of trans interaction p-values (adQTL vs control)

Input:  results/phase1/p1_cis_interaction.tsv
        results/phase1/p1_trans_control.tsv
        results/phase1/p1_trans_interaction_raw.tsv (for ECDF)

Output: figures/paper/Figure_TransControl.png
        figures/paper/Figure_TransControl.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE  = Path(__file__).resolve().parents[1]
RES1  = BASE / "results" / "phase1"
from lib_figure_io import REGEN_DIR, suptitle, announce  # C3: no dead figures/paper, no baked-in numbers
FIGS  = REGEN_DIR

CIS_TSV   = RES1 / "p1_cis_interaction.tsv"
TRANS_TSV = RES1 / "p1_trans_control.tsv"
RAW_TSV   = RES1 / "p1_trans_interaction_raw.tsv"

FIGS.mkdir(parents=True, exist_ok=True)


def load_data():
    cis   = pd.read_csv(CIS_TSV, sep="\t")
    trans = pd.read_csv(TRANS_TSV, sep="\t")

    cis_adqtl = cis[cis["set"] == "adqtl"]
    cis_ctrl  = cis[cis["set"] == "control"]
    trans_adqtl = trans[trans["set"] == "adqtl"]
    trans_ctrl  = trans[trans["set"] == "control"]

    rates = {
        "adqtl_cis":   (cis_adqtl["bonf_p"] < 0.05).mean() * 100,
        "adqtl_trans": trans_adqtl["any_trans_sig"].mean() * 100 if len(trans_adqtl) > 0 else 0.0,
        "ctrl_cis":    (cis_ctrl["bonf_p"] < 0.05).mean() * 100,
        "ctrl_trans":  trans_ctrl["any_trans_sig"].mean() * 100 if len(trans_ctrl) > 0 else 0.0,
    }

    counts = {
        "adqtl_cis":   (len(cis_adqtl), (cis_adqtl["bonf_p"] < 0.05).sum()),
        "adqtl_trans": (len(trans_adqtl), trans_adqtl["any_trans_sig"].sum() if len(trans_adqtl) > 0 else 0),
        "ctrl_cis":    (len(cis_ctrl), (cis_ctrl["bonf_p"] < 0.05).sum()),
        "ctrl_trans":  (len(trans_ctrl), trans_ctrl["any_trans_sig"].sum() if len(trans_ctrl) > 0 else 0),
    }

    # Fisher: adQTL cis vs adQTL trans
    n_adqtl = len(trans_adqtl)
    n_adqtl_trans_sig = int(trans_adqtl["any_trans_sig"].sum()) if len(trans_adqtl) > 0 else 0
    n_adqtl_cis   = len(cis_adqtl)
    n_adqtl_cis_sig = int((cis_adqtl["bonf_p"] < 0.05).sum())
    table = [
        [n_adqtl_cis_sig,   n_adqtl_cis   - n_adqtl_cis_sig],
        [n_adqtl_trans_sig, n_adqtl       - n_adqtl_trans_sig],
    ]
    or_, pval = fisher_exact(table)

    return rates, counts, or_, pval


def load_raw_pvals():
    if not RAW_TSV.exists():
        return None, None
    raw = pd.read_csv(RAW_TSV, sep="\t").dropna(subset=["pval"])
    adqtl_p = raw[raw["set"] == "adqtl"]["pval"].values
    ctrl_p  = raw[raw["set"] == "control"]["pval"].values
    return adqtl_p, ctrl_p


def format_pval(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10**exp
        return f"p = {mant:.1f}x10$^{{{exp}}}$"
    else:
        return f"p = {p:.3f}"


def make_figure(rates, counts, or_val, fisher_p, adqtl_p, ctrl_p):
    # Colors
    COL = {
        "adqtl_cis":   "#c0392b",
        "adqtl_trans": "#e8a49a",
        "ctrl_cis":    "#555555",
        "ctrl_trans":  "#bbbbbb",
    }

    has_ecdf = adqtl_p is not None and len(adqtl_p) > 0

    if has_ecdf:
        fig, (ax_bar, ax_ecdf) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=(6, 5))

    # ── Panel A: bar chart ────────────────────────────────────────────────────
    keys   = ["adqtl_cis", "adqtl_trans", "ctrl_cis", "ctrl_trans"]
    labels = ["adQTL\n(cis)", "adQTL\n(trans)", "Control\n(cis)", "Control\n(trans)"]
    vals   = [rates[k] for k in keys]
    colors = [COL[k] for k in keys]

    x = np.arange(len(keys))
    bars = ax_bar.bar(x, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # null line at 5%
    ax_bar.axhline(5.0, color="black", linestyle="--", linewidth=1.2, label="5% null")

    # label bars with %
    for bar, v, k in zip(bars, vals, keys):
        n_tot, n_sig = counts[k]
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{v:.1f}%\n({int(n_sig)}/{n_tot})",
            ha="center", va="bottom", fontsize=8, color="black"
        )

    # Fisher p-value annotation (cis vs trans for adQTL)
    y_max = max(vals) * 1.25 + 5
    x0, x1 = 0, 1
    y_line = max(vals[0], vals[1]) + 3
    ax_bar.plot([x0, x0, x1, x1], [y_line, y_line + 1, y_line + 1, y_line],
                color="black", linewidth=0.8)
    ax_bar.text((x0 + x1) / 2, y_line + 1.5,
                format_pval(fisher_p),
                ha="center", va="bottom", fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_ylabel("% loci with significant interaction\n(Bonferroni-corrected)", fontsize=10)
    ax_bar.set_ylim(0, y_max)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(fontsize=9, frameon=False)
    ax_bar.set_title("A", loc="left", fontweight="bold", fontsize=12)

    # ── Panel B: ECDF ─────────────────────────────────────────────────────────
    if has_ecdf:
        for pvals, label, color in [
            (adqtl_p, "adQTL", "#c0392b"),
            (ctrl_p,  "Control", "#555555"),
        ]:
            if len(pvals) == 0:
                continue
            pv_sorted = np.sort(pvals)
            ecdf_y = np.arange(1, len(pv_sorted) + 1) / len(pv_sorted)
            ax_ecdf.plot(pv_sorted, ecdf_y, color=color, linewidth=1.5, label=label)

        # uniform diagonal (null)
        ax_ecdf.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Uniform (null)")

        ax_ecdf.set_xlabel("Trans interaction p-value", fontsize=10)
        ax_ecdf.set_ylabel("Cumulative fraction", fontsize=10)
        ax_ecdf.set_xlim(0, 1)
        ax_ecdf.set_ylim(0, 1)
        ax_ecdf.spines["top"].set_visible(False)
        ax_ecdf.spines["right"].set_visible(False)
        ax_ecdf.legend(fontsize=9, frameon=False)
        ax_ecdf.set_title("B", loc="left", fontweight="bold", fontsize=12)

    suptitle(fig,
        "Figure S_. Trans-chromosomal background SNPs show no interaction enrichment,\n"
        "confirming cis-specificity",
        fontsize=11, y=1.01
    )

    plt.tight_layout()

    png_path = FIGS / "Figure_TransControl.png"
    pdf_path = FIGS / "Figure_TransControl.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    announce(__file__, "Figure_TransControl")


def main():
    print("Loading summary data...")
    if not TRANS_TSV.exists():
        print(f"ERROR: {TRANS_TSV} not found. Run p1_16_trans_negative_control.py first.")
        return

    rates, counts, or_val, fisher_p = load_data()

    print("\nRates:")
    for k, v in rates.items():
        n_tot, n_sig = counts[k]
        print(f"  {k}: {v:.1f}% ({n_sig}/{n_tot})")
    print(f"\nFisher adQTL cis vs trans: OR={or_val:.2f}, p={fisher_p:.2e}")

    print("\nLoading raw p-values for ECDF...")
    adqtl_p, ctrl_p = load_raw_pvals()

    print("Generating figure...")
    make_figure(rates, counts, or_val, fisher_p, adqtl_p, ctrl_p)
    print("Done.")


if __name__ == "__main__":
    main()
