"""
p1_14_distance_decay.py
Compute distance-decay of cis-haplotype epistasis enrichment.
"""
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import os

# ── paths ──────────────────────────────────────────────────────────────────────
from pathlib import Path
BASE = str(Path(__file__).resolve().parents[1])
IN   = os.path.join(BASE, "results", "phase1", "p1_cis_interaction.tsv")
OUT  = os.path.join(BASE, "results", "phase1", "p1_distance_decay.tsv")

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN, sep="\t")
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# ── parse positions ────────────────────────────────────────────────────────────
def parse_pos(vid):
    """chrom:pos:REF:ALT -> int pos"""
    try:
        return int(str(vid).split(":")[1])
    except Exception:
        return np.nan

df["focal_pos"] = df["variant_kgpID"].apply(parse_pos)

# best_bg_pos column exists; use it directly (already numeric)
if "best_bg_pos" in df.columns:
    df["bg_pos"] = pd.to_numeric(df["best_bg_pos"], errors="coerce")
else:
    df["bg_pos"] = df["best_bg"].apply(parse_pos)

df["distance"] = (df["focal_pos"] - df["bg_pos"]).abs()

# ── filter ─────────────────────────────────────────────────────────────────────
mask = (df["n_bg_tested"] > 0) & df["best_bg"].notna() & df["distance"].notna()
df = df[mask].copy()
print(f"After filter: {len(df)} rows")
print(f"Distance summary (bp): median={df['distance'].median():.0f}  "
      f"min={df['distance'].min():.0f}  max={df['distance'].max():.0f}")

# ── bin ────────────────────────────────────────────────────────────────────────
bins   = [0, 5_000, 10_000, 20_000, 35_000, 50_000]
labels = ["0-5kb", "5-10kb", "10-20kb", "20-35kb", "35-50kb"]
df["bin"] = pd.cut(df["distance"], bins=bins, labels=labels, right=True, include_lowest=True)

df["sig"] = df["bonf_p"] < 0.05

# ── aggregate ──────────────────────────────────────────────────────────────────
rows = []
for lbl in labels:
    bdf = df[df["bin"] == lbl]
    for s in ["adqtl", "control"]:
        sub = bdf[bdf["set"] == s]
        n   = len(sub)
        ns  = sub["sig"].sum()
        rows.append({"bin_label": lbl, "set": s, "n": n, "n_sig": ns})

agg = pd.DataFrame(rows)
pivot = agg.pivot(index="bin_label", columns="set", values=["n", "n_sig"])
pivot.columns = ["_".join(c) for c in pivot.columns]
pivot = pivot.reindex(labels)  # preserve bin order

# ── OR + Fisher ───────────────────────────────────────────────────────────────
results = []
for lbl in labels:
    row = pivot.loc[lbl]
    n_a  = int(row.get("n_adqtl",   0))
    ns_a = int(row.get("n_sig_adqtl",   0))
    n_c  = int(row.get("n_control", 0))
    ns_c = int(row.get("n_sig_control", 0))

    rate_a = ns_a / n_a  if n_a  > 0 else np.nan
    rate_c = ns_c / n_c  if n_c  > 0 else np.nan

    # 2x2: [[sig_adqtl, nonsig_adqtl], [sig_ctrl, nonsig_ctrl]]
    a, b = ns_a, n_a  - ns_a
    c, d = ns_c, n_c  - ns_c

    if min(a, b, c, d) > 0:
        OR = (a * d) / (b * c)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        lo = np.exp(np.log(OR) - 1.96 * se)
        hi = np.exp(np.log(OR) + 1.96 * se)
    else:
        OR = np.nan; lo = np.nan; hi = np.nan

    _, fp = fisher_exact([[a, b], [c, d]])

    bin_idx = labels.index(lbl)
    bmin = bins[bin_idx]
    bmax = bins[bin_idx + 1]

    results.append({
        "bin_label":     lbl,
        "bin_min":       bmin,
        "bin_max":       bmax,
        "n_adqtl":       n_a,
        "n_sig_adqtl":   ns_a,
        "rate_adqtl":    rate_a,
        "n_control":     n_c,
        "n_sig_control": ns_c,
        "rate_control":  rate_c,
        "OR":            OR,
        "OR_lo":         lo,
        "OR_hi":         hi,
        "fisher_p":      fp,
    })

out_df = pd.DataFrame(results)
print("\nOR per bin:")
print(out_df[["bin_label","n_adqtl","rate_adqtl","n_control","rate_control","OR","OR_lo","OR_hi","fisher_p"]].to_string(index=False))

out_df.to_csv(OUT, sep="\t", index=False)
print(f"\nSaved -> {OUT}")
