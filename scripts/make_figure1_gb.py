"""
Figure 1 (study design) for the Genome Biology submission.

Redrawn 2026-11-11 in response to the co-PI comment "Can you increase font size on this
figure?". Same content and layout as the previous Figure 1, but drawn natively as a vector
figure at the BMC full-page width (170 mm) with every label at >= 7 pt.

Usage:  python scripts/make_figure1_gb.py [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BASE = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "figures" / "regenerated" / "make_figure1_gb"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

BLUE, BLUE_L, BLUE_D = "#1F5FA8", "#DDEAF6", "#153F73"
ORG, ORG_L, ORG_D = "#D9601A", "#FCE6D5", "#B24A10"
GRN, GRN_L, GRN_D = "#1F7F68", "#DDF0EA", "#155C4B"
GRAY, GRAY_L, INK = "#6E737A", "#F3F4F6", "#1C1F23"
GOLD = "#F5A300"

W, H = 17.0, 11.3   # cm
CM = 1 / 2.54
fig = plt.figure(figsize=(W * CM, H * CM), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")


def box(x, y, w, h, fc="white", ec=INK, lw=1.0, r=0.14, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def T(x, y, s, fs=7.5, w="normal", c=INK, ha="center", va="center", style="normal", ls=1.15, z=6):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, fontstyle=style,
            linespacing=ls, zorder=z)


def arrow(x0, y0, x1, y1, c=GRAY, lw=1.3, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, zorder=5, shrinkA=0, shrinkB=0))


def block_arrow(x, y, c=GRAY):
    ax.add_patch(Polygon([[x, y + 0.11], [x + 0.2, y + 0.11], [x + 0.2, y + 0.24], [x + 0.42, y],
                          [x + 0.2, y - 0.24], [x + 0.2, y - 0.11], [x, y - 0.11]],
                         closed=True, fc=c, ec="none", zorder=5))


def person(x, y, c, s=1.0):
    ax.add_patch(Circle((x, y + 0.13 * s), 0.085 * s, fc=c, ec="none", zorder=6))
    ax.add_patch(Ellipse((x, y - 0.07 * s), 0.27 * s, 0.22 * s, fc=c, ec="none", zorder=6))


def stage(x, y, w, h, label, sub, c, cl):
    box(x, y, w, h, fc="white", ec=c, lw=1.3)
    ax.add_patch(FancyBboxPatch((x + 0.02, y + h - 1.12), w - 0.04, 1.10,
                                boxstyle="round,pad=0,rounding_size=0.12", fc=cl, ec="none", zorder=3))
    T(x + 0.2, y + h - 0.33, label, fs=10, w="bold", c=c, ha="left")
    T(x + 0.2, y + h - 0.80, sub, fs=8.5, w="bold", ha="left")


def diamond(x, y, r=0.13):
    ax.add_patch(Polygon([[x, y + r], [x + r, y], [x, y - r], [x - r, y]], closed=True,
                         fc=GOLD, ec="none", zorder=7))


def haplo(x0, x1, y, focal_x, bg_x, bg_c):
    ax.plot([x0, x1], [y, y], color=INK, lw=1.0, zorder=4)
    diamond(focal_x, y)
    ax.add_patch(Circle((bg_x, y), 0.105, fc=bg_c, ec="none", zorder=6))
    mid, wd = (focal_x + bg_x) / 2, abs(bg_x - focal_x)
    ax.add_patch(Arc((mid, y + 0.06), wd, wd * 0.95, theta1=0, theta2=180, ls=(0, (3, 2)),
                     color=INK, lw=0.9, zorder=4))


# ------------------------------------------------------------------ top row
YT, HT = 4.55, 6.6            # stage boxes: bottom y, height
S1x, S1w = 0.10, 4.20
S2x, S2w = 4.75, 8.15
S3x, S3w = 13.35, 3.55
block_arrow(4.32, YT + HT / 2 - 0.4)
block_arrow(12.92, YT + HT / 2 - 0.4)

# ---- Stage 1
stage(S1x, YT, S1w, HT, "STAGE 1", "adQTL discovery", BLUE, BLUE_L)
cx = S1x + S1w / 2
T(cx, YT + HT - 1.45, "Per-ancestry expression (LCLs)", fs=7.5, w="bold", c=BLUE)
groups = [("EUR", 142), ("AFR", 196), ("EAS", 141), ("SAS", 139), ("AMR", 113)]
for x, (g, n) in zip(np.linspace(S1x + 0.5, S1x + S1w - 0.5, 5), groups):
    ax.add_patch(Circle((x, YT + HT - 2.05), 0.25, fc="white", ec=BLUE, lw=1.0, zorder=5))
    person(x, YT + HT - 2.06, BLUE, s=0.95)
    T(x, YT + HT - 2.50, g, fs=7.5, w="bold")
    T(x, YT + HT - 2.80, f"n={n}", fs=7)
arrow(cx, YT + HT - 3.0, cx, YT + HT - 3.3)
T(cx, YT + HT - 3.72, "EUR–AFR effect-size\nheterogeneity test", fs=8, w="bold", c=BLUE)
by = YT + 1.85
ax.add_patch(Rectangle((cx - 0.85, by), 0.5, 0.62, fc=BLUE, ec="none", zorder=5))
ax.add_patch(Rectangle((cx + 0.35, by), 0.5, 0.32, fc=GRAY, ec="none", zorder=5))
ax.plot([cx - 1.15, cx + 1.15], [by, by], color=INK, lw=1.0, zorder=5)
T(cx - 0.6, by - 0.2, "β EUR", fs=7.5, c=BLUE, style="italic")
T(cx + 0.6, by - 0.2, "β AFR", fs=7.5, style="italic")
arrow(cx, by - 0.38, cx, by - 0.64)
box(S1x + 0.45, YT + 0.2, S1w - 0.9, 0.95, fc=BLUE_D, ec=BLUE_D)
T(cx, YT + 0.675, "200 adQTLs identified\n(FDR < 0.1)", fs=7.5, w="bold", c="white", ls=1.25)

# ---- Stage 2
stage(S2x, YT, S2w, HT, "STAGE 2", "Mechanistic testing (within MAGE)", ORG, ORG_L)
pw, gap = 1.87, 0.12
px0 = S2x + (S2w - 4 * pw - 3 * gap) / 2
py, ph = YT + 0.2, HT - 1.75
titles = ["Interaction\nenrichment", "Within-\nancestry\nconfirmation", "Generative\nsimulation",
          "Context\ndivergence\n(HCD)"]
for k, t in enumerate(titles):
    x = px0 + k * (pw + gap)
    box(x, py, pw, ph, fc="white", ec=ORG, lw=1.1, r=0.1)
    ax.add_patch(Circle((x + pw / 2, py + ph), 0.27, fc=ORG, ec="white", lw=1.0, zorder=7))
    T(x + pw / 2, py + ph, f"P{k + 1}", fs=8, w="bold", c="white", z=8)
    T(x + pw / 2, py + ph - 0.4, t, fs=7.5, w="bold", va="top")

# P1
x = px0
haplo(x + 0.2, x + pw - 0.2, py + 2.25, x + 0.6, x + 1.32, BLUE_D)
diamond(x + 0.24, py + 1.22, r=0.11)
T(x + 0.42, py + 1.22, "Focal", fs=7, ha="left")
ax.add_patch(Circle((x + 0.24, py + 0.75), 0.1, fc=BLUE_D, ec="none", zorder=6))
T(x + 0.42, py + 0.75, "Background", fs=7, ha="left")
# P2
x = px0 + (pw + gap)
for j, (lab, bgc) in enumerate([("EUR", BLUE_D), ("AFR", GRAY)]):
    yy = py + 1.72 - j * 1.55
    box(x + 0.14, yy, pw - 0.28, 1.4, fc="white", ec=BLUE, lw=1.0, r=0.08)
    T(x + 0.28, yy + 1.16, lab, fs=7.5, w="bold", c=BLUE, ha="left")
    haplo(x + 0.3, x + pw - 0.3, yy + 0.36, x + 0.62, x + 1.25, bgc)
# P3
x = px0 + 2 * (pw + gap)
for j, (l, fc, ec) in enumerate([("A", BLUE_L, BLUE), ("B", BLUE_L, BLUE), ("C", ORG_L, ORG), ("D", ORG_L, ORG)]):
    bx = x + 0.17 + j * 0.4
    box(bx, py + 2.55, 0.34, 0.42, fc=fc, ec=ec, lw=0.9, r=0.05)
    T(bx + 0.17, py + 2.76, l, fs=7.5, w="bold")
T(x + 0.11, py + 1.45, "A  null\nB  LD artifact\nC  G×E\nD  cis-epistasis", fs=7, ls=1.4, ha="left")
# P4
x = px0 + 3 * (pw + gap)
xx = np.linspace(-0.55, 0.55, 60)
yc = py + 2.1
ax.plot(x + pw / 2 + xx[:30], yc + 1.9 * xx[:30] ** 2, color=BLUE, lw=1.4, zorder=5)
ax.plot(x + pw / 2 + xx[30:], yc + 1.9 * xx[30:] ** 2, color=ORG, lw=1.4, zorder=5)
ax.plot([x + 0.3, x + pw - 0.3], [yc, yc], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.add_patch(Circle((x + pw / 2, yc), 0.07, fc=INK, ec="none", zorder=6))
T(x + pw / 2, py + 1.0, "Haplotype\ncontexts differ\nby ancestry", fs=7, ls=1.25)

# ---- Stage 3
stage(S3x, YT, S3w, HT, "STAGE 3", "Replication", GRN, GRN_L)
cx = S3x + S3w / 2
T(cx, YT + HT - 1.5, "Geuvadis", fs=9, w="bold")
for j, x in enumerate(np.linspace(cx - 1.2, cx + 1.2, 9)):
    person(x, YT + HT - 2.0, GRN if j in (0, 1, 2, 7, 8) else GRAY, s=0.95)
T(cx, YT + HT - 2.62, "n = 445\n(358 EUR + 87 YRI)", fs=7.5, ls=1.25)
arrow(cx, YT + HT - 3.05, cx, YT + HT - 3.4)
T(cx, YT + HT - 3.82, "Directional concordance\nwith MAGE adQTLs", fs=7.5, ls=1.25)
for dy in (0.0, 0.25):
    arrow(cx - 0.55, YT + 2.25 - dy, cx + 0.55, YT + 2.25 - dy, c=GRN, lw=1.4, ms=9)
arrow(cx, YT + 1.72, cx, YT + 1.38)
box(S3x + 0.25, YT + 0.2, S3w - 0.5, 1.02, fc=GRN_D, ec=GRN_D)
T(cx, YT + 0.71, "Independent cohort:\nno overlap with MAGE", fs=7.5, w="bold", c="white", ls=1.25)

# ------------------------------------------------------------------ bottom: mechanism
box(0.10, 0.15, W - 0.2, 4.1, fc=GRAY_L, ec="#9AA0A6", lw=1.0)
T(0.4, 3.88, "CENTRAL MECHANISM", fs=9.5, w="bold", ha="left")
ya, yb = 3.05, 1.55
T(0.4, ya, "Ancestry A", fs=8.5, w="bold", c=BLUE, ha="left")
T(0.4, yb, "Ancestry B", fs=8.5, w="bold", c=ORG, ha="left")
hx0, hx1, fx = 2.3, 6.2, 4.1
for y, col, pattern in ((ya, BLUE_D, (1, 0, 1)), (yb, ORG_D, (0, 1, 1))):
    ax.plot([hx0, hx1], [y, y], color=INK, lw=1.0, zorder=4)
    for bx, filled in zip((2.9, 4.9, 5.6), pattern):
        ax.add_patch(Circle((bx, y), 0.115, fc=col if filled else "white", ec=col, lw=1.0, zorder=6))
    diamond(fx, y, r=0.15)
ax.plot([fx, fx], [yb + 0.17, ya - 0.17], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
T(fx + 0.2, (ya + yb) / 2, "same causal\nvariant", fs=7.5, style="italic", ha="left")
T((hx0 + hx1) / 2, 0.82, "different background haplotype", fs=7.5, style="italic")

arrow(6.5, ya, 7.0, ya); arrow(6.5, yb, 7.0, yb)
for (dx, dy), a in zip([(-0.2, -0.08), (0.12, 0.2), (0.03, -0.2), (0.3, -0.08)], (0.35, 1.0, 0.6, 0.8)):
    ax.add_patch(Circle((7.65 + dx, ya + dy), 0.115, fc=BLUE, ec=BLUE_D, lw=0.5, alpha=a, zorder=6))
    ax.add_patch(Circle((7.65 + dx, yb + dy), 0.115, fc=ORG, ec=ORG_D, lw=0.5, alpha=a, zorder=6))
T(7.7, 0.72, "different regulatory\ncontext", fs=7.5, style="italic", ls=1.2)
arrow(8.4, ya, 8.9, ya); arrow(8.4, yb, 8.9, yb)

T(10.1, 3.72, "Expression effect", fs=8, w="bold")
ax.plot([9.3, 9.3], [1.05, 3.45], color=INK, lw=1.0, zorder=5)
ax.plot([9.15, 11.0], [1.05, 1.05], color=INK, lw=1.0, zorder=5)
ax.add_patch(Rectangle((9.6, 1.05), 0.5, 2.15, fc=BLUE, ec="none", zorder=5))
ax.add_patch(Rectangle((10.3, 1.05), 0.5, 1.05, fc=ORG, ec="none", zorder=5))

T(11.45, 2.95, "Same causal variant, different haplotype\ncontext → different expression effect.",
  fs=7.5, ha="left", ls=1.35)
ax.plot([11.45, 16.5], [2.2, 2.2], color="#9AA0A6", lw=0.8, zorder=4)
T(11.45, 1.45, "Shared LCL culture conditions\nminimize environmental confounding.", fs=7.5, ha="left", ls=1.35)

fig.savefig(OUT / "Fig1.pdf", facecolor="white")
fig.savefig(OUT / "Fig1.png", dpi=600, facecolor="white")
print("wrote", OUT / "Fig1.pdf")
