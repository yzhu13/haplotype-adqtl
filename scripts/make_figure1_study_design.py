"""
Figure 1 -- Study design overview (schematic, not a data plot).

Redesign per co-author review (Rebecca): the previous hand-made Figure 1 led with a
large "data resources" panel (MAGE / SuSiE / 1000 Genomes specs) that competed for
space with the actual three-stage narrative, and it printed specific statistical
results (p-values, ORs) directly in the schematic. This version drops the top data
panel to a single unobtrusive input line, removes every statistical result from the
image, and focuses entirely on the three-stage pipeline:

    Stage 1  adQTL discovery (MAGE, 5 ancestry groups)
        -->  Stage 2  mechanistic testing of 4 predictions, in the paper's current
                       (post-revision) order: P1 interaction enrichment,
                       P2 within-ancestry confirmation, P3 generative simulation /
                       identifiability, P4 haplotype context divergence (HCD)
        -->  Stage 3  independent replication (Geuvadis, zero sample overlap)

A bottom "central mechanism" strip illustrates, in the abstract, why the same causal
variant produces different expression effects across ancestry groups: it sits on a
different local haplotype background, which changes the combinatorial regulatory
(TF-binding) context, which changes the expression effect -- and because all LCLs are
cultured under identical laboratory conditions, that difference is attributable to
genetics rather than environment.

Counts shown (731 LCLs, per-ancestry n's, 200 adQTLs at FDR < 0.1, n = 445 replication
donors, zero MAGE overlap) describe study design, not results, per the review comment
("remove any specific results, like p-values -- keep focused on the big picture").

Renders into figures/regenerated/make_figure1_study_design/ (project convention; see
lib_figure_io.py). Nothing under figures/ root is touched.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc

BASE = Path(__file__).resolve().parents[1]
from lib_figure_io import REGEN_DIR, suptitle, announce  # script dir is on sys.path[0]

FIGS = REGEN_DIR

# ---------------------------------------------------------------------------
# Palette -- one restrained hue per stage + neutral gray. No saturated primaries,
# no red/green as the sole distinguishing pair (Cell Press colorblind guidance).
# ---------------------------------------------------------------------------
C1, C1_L, C1_D = "#3D6C93", "#DCE7F0", "#284B69"   # Stage 1 -- slate blue
C2, C2_L, C2_D = "#B9772E", "#F1E3D0", "#8A5A20"   # Stage 2 -- muted amber/ochre
C3, C3_L, C3_D = "#3F8F82", "#DCEEEB", "#2A6259"   # Stage 3 -- muted teal
GRAY, GRAY_L, GRAY_D = "#8B8F94", "#EFF1F2", "#4A4E54"
INK = "#1F2328"
ARROW_GRAY = "#6B6F76"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

# ---------------------------------------------------------------------------
# Canvas -- Cell Press full-width column, 17.4 cm. Coordinates are in cm.
# ---------------------------------------------------------------------------
W_CM, H_CM = 17.4, 10.15
CM = 1 / 2.54


def new_axes():
    fig = plt.figure(figsize=(W_CM * CM, H_CM * CM), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_CM)
    ax.set_ylim(0, H_CM)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def rrect(ax, x, y, w, h, fc, ec, lw=1.0, r=0.09, zorder=2, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          facecolor=fc, edgecolor=ec, linewidth=lw,
                          alpha=alpha, zorder=zorder, joinstyle="round")
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, fs=7, weight="normal", color=INK, ha="center", va="center",
        style="normal", zorder=5, linespacing=1.15):
    return ax.text(x, y, s, fontsize=fs, fontweight=weight, color=color, ha=ha, va=va,
                    fontstyle=style, zorder=zorder, linespacing=linespacing,
                    family="sans-serif")


def arrow(ax, x0, y0, x1, y1, color=ARROW_GRAY, lw=1.2, ms=6.5, zorder=4,
          connectionstyle=None, shrinkA=0, shrinkB=0):
    fa = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle=f"-|>,head_width={ms*0.06},head_length={ms*0.09}",
                          color=color, linewidth=lw, zorder=zorder, mutation_scale=ms,
                          connectionstyle=connectionstyle,
                          shrinkA=shrinkA, shrinkB=shrinkB, capstyle="round")
    ax.add_patch(fa)
    return fa


def stage_header(ax, x0, y_top, w, label, subtitle, color, dark):
    txt(ax, x0 + 0.12, y_top - 0.08, label, fs=8.3, weight="bold", color=color,
        ha="left", va="top")
    txt(ax, x0 + 0.12, y_top - 0.50, subtitle, fs=8.0, weight="bold", color=INK,
        ha="left", va="top")


def badge(ax, x0, y0, w, h, lines, fc, ec, text_color="white", fs=7.2, weight="bold",
          linespacing=1.35):
    """Rounded, filled callout box. Lines are drawn as ONE multi-line text object so
    matplotlib centers the whole block correctly (two separately-positioned Text
    objects is what caused the earlier overflow bug)."""
    rrect(ax, x0, y0, w, h, fc=fc, ec=ec, lw=1.0, r=0.08, zorder=3)
    txt(ax, x0 + w / 2, y0 + h / 2, "\n".join(lines), fs=fs, weight=weight,
        color=text_color, linespacing=linespacing)


def divider_arrow_h(ax, x0, x1, y, color=ARROW_GRAY):
    arrow(ax, x0, y, x1, y, color=color, lw=1.3, ms=8)


# ---------------------------------------------------------------------------
# Small conceptual icons
# ---------------------------------------------------------------------------
def ancestry_dots(ax, x_center, y, labels, ns, color=GRAY, r=0.155, spacing=0.70):
    n = len(labels)
    xs = x_center + (np.arange(n) - (n - 1) / 2) * spacing
    for x, lab, ni in zip(xs, labels, ns):
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                             linewidth=0.6, zorder=3))
        txt(ax, x, y - r - 0.20, lab, fs=5.3, weight="bold", color=GRAY_D)
        txt(ax, x, y - r - 0.42, f"n={ni}", fs=4.6, color=GRAY_D)
    return xs


def diff_bars_icon(ax, x_center, y0, color_a, color_b, h_a=0.55, h_b=0.30, w=0.30, gap=0.16):
    """Two bars of different height -- abstract 'effect differs by ancestry'."""
    xa = x_center - gap / 2 - w
    xb = x_center + gap / 2
    ax.add_patch(Rectangle((xa, y0), w, h_a, facecolor=color_a, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((xb, y0), w, h_b, facecolor=color_b, edgecolor="none", zorder=3))
    ax.plot([xa - 0.12, x_center + gap / 2 + w + 0.12], [y0, y0], color=GRAY_D, lw=0.6, zorder=2)
    return xa, xb


def interaction_icon(ax, x_center, y, color, scale=1.0, hap_color=GRAY):
    """Focal (diamond) + background (circle) variant on a haplotype line, joined by
    a curved interaction arc -- the visual shorthand for cis focal x background."""
    hw = 0.42 * scale
    ax.plot([x_center - hw, x_center + hw], [y, y], color=hap_color, lw=1.0, zorder=2,
            solid_capstyle="round")
    xf, xb = x_center - hw * 0.55, x_center + hw * 0.55
    # background variant (circle)
    ax.add_patch(Circle((xb, y), 0.075 * scale, facecolor=C2, edgecolor="white",
                         linewidth=0.4, zorder=4))
    # focal variant (diamond)
    d = 0.10 * scale
    diamond = plt.Polygon([[xf, y + d], [xf + d, y], [xf, y - d], [xf - d, y]],
                           closed=True, facecolor="#D9A441", edgecolor="white",
                           linewidth=0.4, zorder=5)
    ax.add_patch(diamond)
    # interaction arc above
    arc_h = 0.20 * scale
    arc = Arc((x_center, y), hw * 1.1, arc_h * 2.4, theta1=15, theta2=165,
              color=C2_D, lw=0.9, zorder=3)
    ax.add_patch(arc)


def model_row_icon(ax, x_center, y, scale=1.0):
    """Four tiny model boxes A-D with D (true mechanism) picked out -- echoes the
    paper's own model-letter convention (Fig. 5: null/LD-artifact/GxE/cis-epistasis)."""
    labs = ["A", "B", "C", "D"]
    sp = 0.30 * scale
    xs = x_center + (np.arange(4) - 1.5) * sp
    s = 0.115 * scale
    for x, lab in zip(xs, labs):
        is_true = lab == "D"
        fc = C2 if is_true else "white"
        ec = C2_D if is_true else GRAY
        tc = "white" if is_true else GRAY_D
        rrect(ax, x - s / 2, y - s / 2, s, s, fc=fc, ec=ec, lw=0.8, r=0.02, zorder=3)
        txt(ax, x, y, lab, fs=4.6, weight="bold", color=tc, zorder=4)
    return xs


def divergence_icon(ax, x_center, y, color_a, color_b, scale=1.0):
    """Two offset arcs ('distributions') with a gap arrow -- population divergence."""
    w = 0.55 * scale
    a1 = Arc((x_center - 0.10 * scale, y - 0.06 * scale), w, w * 0.62, theta1=0, theta2=180,
             color=color_a, lw=1.1, zorder=3)
    a2 = Arc((x_center + 0.10 * scale, y - 0.06 * scale), w, w * 0.62, theta1=0, theta2=180,
             color=color_b, lw=1.1, zorder=3)
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.plot([x_center - 0.10 * scale, x_center - 0.10 * scale], [y - 0.06 * scale, y - 0.20 * scale],
             color=color_a, lw=0.0)


def check_concordance_icon(ax, x_center, y, color, scale=1.0):
    """Two parallel same-direction arrows -- directional concordance between cohorts."""
    w = 0.55 * scale
    arrow(ax, x_center - w / 2, y + 0.10 * scale, x_center + w / 2, y + 0.10 * scale,
          color=color, lw=1.1, ms=5.5)
    arrow(ax, x_center - w / 2, y - 0.10 * scale, x_center + w / 2, y - 0.10 * scale,
          color=color, lw=1.1, ms=5.5)


def cohort_dots_icon(ax, x_center, y, color_a, color_b, n_a=6, n_b=3, r=0.06, spacing=0.155):
    n = n_a + n_b
    xs = x_center + (np.arange(n) - (n - 1) / 2) * spacing
    colors = [color_a] * n_a + [color_b] * n_b
    rng = np.random.default_rng(7)
    order = np.arange(n)
    rng.shuffle(order)
    colors = [colors[i] for i in np.argsort(order)]
    for x, c in zip(xs, colors):
        ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor="white", linewidth=0.4, zorder=3))


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def main():
    fig, ax = new_axes()

    # optional baked-in title, only drawn with --title (project convention)
    suptitle(fig, "Figure 1. Study design and analysis overview.", fontsize=9, y=0.995)

    # ===================================================================
    # Stage row geometry
    # ===================================================================
    y0_stage, y1_stage = 3.30, 9.85
    h_stage = y1_stage - y0_stage

    x0_s1, w_s1 = 0.30, 3.90
    x1_s1 = x0_s1 + w_s1
    x0_s2 = x1_s1 + 0.55
    w_s2 = 7.85
    x1_s2 = x0_s2 + w_s2
    x0_s3 = x1_s2 + 0.55
    w_s3 = 3.95
    x1_s3 = x0_s3 + w_s3  # == W_CM - 0.3

    mid_y = y0_stage + h_stage / 2
    header_h = 0.92

    # panel frames
    rrect(ax, x0_s1, y0_stage, w_s1, h_stage, fc="white", ec=C1, lw=1.3, r=0.12, zorder=1)
    rrect(ax, x0_s2, y0_stage, w_s2, h_stage, fc="white", ec=C2, lw=1.3, r=0.12, zorder=1)
    rrect(ax, x0_s3, y0_stage, w_s3, h_stage, fc="white", ec=C3, lw=1.3, r=0.12, zorder=1)
    # faint header wash
    rrect(ax, x0_s1, y1_stage - header_h, w_s1, header_h, fc=C1_L, ec="none", r=0.10, zorder=1)
    rrect(ax, x0_s2, y1_stage - header_h, w_s2, header_h, fc=C2_L, ec="none", r=0.10, zorder=1)
    rrect(ax, x0_s3, y1_stage - header_h, w_s3, header_h, fc=C3_L, ec="none", r=0.10, zorder=1)

    # inter-stage arrows
    divider_arrow_h(ax, x1_s1 + 0.06, x0_s2 - 0.06, mid_y, color=ARROW_GRAY)
    divider_arrow_h(ax, x1_s2 + 0.06, x0_s3 - 0.06, mid_y, color=ARROW_GRAY)

    content_top = y1_stage - header_h - 0.15   # just below the header wash
    badge_h = 0.85
    badge_bottom = y0_stage + 0.20
    badge_top = badge_bottom + badge_h

    # -------------------------------------------------------------
    # STAGE 1 -- adQTL discovery
    # -------------------------------------------------------------
    stage_header(ax, x0_s1, y1_stage, w_s1, "STAGE 1", "adQTL discovery", C1, C1_D)

    cx1 = x0_s1 + w_s1 / 2
    y_cap1 = content_top - 0.15
    txt(ax, cx1, y_cap1, "Per-ancestry expression (LCLs)", fs=6.0, color=INK)

    y_dots1 = y_cap1 - 0.35
    ancestry_dots(ax, cx1, y_dots1,
                  labels=["EUR", "AFR", "EAS", "SAS", "AMR"],
                  ns=[142, 196, 141, 139, 113],
                  color=C1, spacing=0.68, r=0.155)
    block1_bottom = y_dots1 - 0.155 - 0.42 - 0.10

    y_arrow1_top = block1_bottom - 0.10
    y_arrow1_bot = y_arrow1_top - 0.95
    arrow(ax, cx1, y_arrow1_top, cx1, y_arrow1_bot, color=ARROW_GRAY, lw=1.0, ms=5)

    y_cap2 = y_arrow1_bot - 0.28
    txt(ax, cx1, y_cap2, "EUR–AFR effect-size\nheterogeneity test", fs=6.0, color=INK,
        linespacing=1.3)

    y_bars_base = y_cap2 - 0.55
    diff_bars_icon(ax, cx1, y_bars_base, C1, GRAY, h_a=0.50, h_b=0.24, w=0.28, gap=0.18)
    y_beta = y_bars_base - 0.20
    txt(ax, cx1 - 0.35 - 0.28 / 2, y_beta, "β EUR", fs=4.8, color=C1_D)
    txt(ax, cx1 + 0.18 + 0.28 / 2, y_beta, "β AFR", fs=4.8, color=GRAY_D)
    block2_bottom = y_beta - 0.08

    y_arrow2_top = block2_bottom - 0.10
    y_arrow2_bot = max(badge_top + 0.18, y_arrow2_top - 1.10)
    arrow(ax, cx1, y_arrow2_top, cx1, y_arrow2_bot, color=ARROW_GRAY, lw=1.0, ms=5)

    badge(ax, x0_s1 + 0.35, badge_bottom, w_s1 - 0.70, badge_h,
          ["200 adQTLs identified", "(FDR < 0.1)"], fc=C1, ec=C1_D, fs=7.0)

    # -------------------------------------------------------------
    # STAGE 2 -- mechanistic testing (4 predictions, current post-revision order)
    # -------------------------------------------------------------
    stage_header(ax, x0_s2, y1_stage, w_s2, "STAGE 2",
                 "Mechanistic testing (within MAGE)", C2, C2_D)

    n_p = 4
    pad_lr = 0.30
    gap_p = 0.22
    card_w = (w_s2 - 2 * pad_lr - (n_p - 1) * gap_p) / n_p
    card_h = 2.55
    # center the card row in the space between the header and the box floor,
    # leaving headroom above card_top for the number badges to straddle the edge
    zone_top = content_top - 0.30
    zone_bot = y0_stage + 0.30
    card_y0 = zone_bot + (zone_top - zone_bot - card_h) / 2
    card_top = card_y0 + card_h

    preds = [
        ("P1", "Interaction\nenrichment"),
        ("P2", "Within-ancestry\nconfirmation"),
        ("P3", "Generative\nsimulation"),
        ("P4", "Context\ndivergence\n(HCD)"),
    ]

    card_xs = []
    for i in range(n_p):
        cx = x0_s2 + pad_lr + i * (card_w + gap_p)
        card_xs.append(cx)
        rrect(ax, cx, card_y0, card_w, card_h, fc="white", ec=C2, lw=0.9, r=0.08, zorder=2)
        cxc = cx + card_w / 2

        # a thin connector line ties the number tab to the icon to the caption
        ax.plot([cxc, cxc], [card_top - 0.05, card_y0 + 0.62], color=C2_L, lw=3.0,
                 solid_capstyle="round", zorder=2, alpha=0.6)

        # number tab, straddling the top edge of the card
        ax.add_patch(Circle((cxc, card_top), 0.17, facecolor=C2, edgecolor="white",
                             linewidth=0.8, zorder=4))
        txt(ax, cxc, card_top, preds[i][0], fs=6.8, weight="bold", color="white", zorder=5)

        icon_y = card_top - 0.68
        if i == 0:
            interaction_icon(ax, cxc, icon_y, C2, scale=1.15)
            # legend: focal (diamond) vs background (circle) variant markers.
            # Card width is only ~1.65cm, so labels must be short ("Background variant"
            # in full does not fit and overflows the card border at any usable font size).
            leg_x0 = cx + 0.26
            leg_y1 = icon_y - 0.62
            leg_y2 = leg_y1 - 0.24
            d = 0.07
            diamond = plt.Polygon(
                [[leg_x0, leg_y1 + d], [leg_x0 + d, leg_y1],
                 [leg_x0, leg_y1 - d], [leg_x0 - d, leg_y1]],
                closed=True, facecolor="#D9A441", edgecolor="white", linewidth=0.3, zorder=5)
            ax.add_patch(diamond)
            txt(ax, leg_x0 + 0.17, leg_y1, "Focal", fs=4.4, color=GRAY_D, ha="left")
            ax.add_patch(Circle((leg_x0, leg_y2), 0.055, facecolor=C2, edgecolor="white",
                                 linewidth=0.3, zorder=5))
            txt(ax, leg_x0 + 0.17, leg_y2, "Background", fs=4.4, color=GRAY_D, ha="left")
        elif i == 1:
            for dx, lab in [(-0.44, "EUR"), (0.44, "AFR")]:
                bx = cxc + dx
                rrect(ax, bx - 0.38, icon_y - 0.22, 0.76, 0.40, fc=GRAY_L, ec=GRAY, lw=0.5,
                      r=0.03, zorder=3)
                interaction_icon(ax, bx, icon_y - 0.02, C2, scale=0.62)
                txt(ax, bx, icon_y - 0.32, lab, fs=4.6, weight="bold", color=GRAY_D)
        elif i == 2:
            model_row_icon(ax, cxc, icon_y, scale=1.25)
        elif i == 3:
            divergence_icon(ax, cxc, icon_y, C1, C2, scale=1.25)

        txt(ax, cxc, card_y0 + 0.38, preds[i][1], fs=5.1, weight="bold", color=INK,
            linespacing=1.4)

    # thin sequence connectors P1->P2->P3->P4, between the number tabs
    for i in range(n_p - 1):
        xL = card_xs[i] + card_w
        xR = card_xs[i + 1]
        arrow(ax, xL + 0.05, card_top, xR - 0.05, card_top,
              color=C2_D, lw=0.8, ms=4)

    # -------------------------------------------------------------
    # STAGE 3 -- independent replication
    # -------------------------------------------------------------
    stage_header(ax, x0_s3, y1_stage, w_s3, "STAGE 3",
                 "Independent replication", C3, C3_D)

    cx3 = x0_s3 + w_s3 / 2
    y_title3 = content_top - 0.15
    txt(ax, cx3, y_title3, "Geuvadis", fs=6.6, weight="bold", color=INK)

    y_dots3 = y_title3 - 0.36
    cohort_dots_icon(ax, cx3, y_dots3, C3, GRAY, n_a=7, n_b=2, r=0.09, spacing=0.22)
    y_n3 = y_dots3 - 0.36
    txt(ax, cx3, y_n3, "n = 445 (358 EUR + 87 YRI)", fs=5.4, color=GRAY_D)
    block1_bottom3 = y_n3 - 0.10

    y_arrow3a_top = block1_bottom3 - 0.10
    y_arrow3a_bot = y_arrow3a_top - 0.95
    arrow(ax, cx3, y_arrow3a_top, cx3, y_arrow3a_bot, color=ARROW_GRAY, lw=1.0, ms=5)

    y_cap3 = y_arrow3a_bot - 0.28
    txt(ax, cx3, y_cap3, "Directional concordance\nwith MAGE adQTLs", fs=6.0, color=INK,
        linespacing=1.3)

    y_icon3 = y_cap3 - 0.55
    check_concordance_icon(ax, cx3, y_icon3, C3, scale=1.3)
    block2_bottom3 = y_icon3 - 0.20

    y_arrow3b_top = block2_bottom3 - 0.10
    y_arrow3b_bot = max(badge_top + 0.18, y_arrow3b_top - 1.10)
    arrow(ax, cx3, y_arrow3b_top, cx3, y_arrow3b_bot, color=ARROW_GRAY, lw=1.0, ms=5)

    badge(ax, x0_s3 + 0.30, badge_bottom, w_s3 - 0.60, badge_h,
          ["Independent cohort:", "zero overlap with MAGE"], fc=C3, ec=C3_D, fs=6.6)

    # ===================================================================
    # Bottom strip -- central mechanism (abstract, no statistics)
    # ===================================================================
    y0_m, y1_m = 0.30, y0_stage - 0.30
    h_m = y1_m - y0_m
    x0_m, x1_m = 0.30, W_CM - 0.30
    w_m = x1_m - x0_m

    rrect(ax, x0_m, y0_m, w_m, h_m, fc=GRAY_L, ec="#CBCED2", lw=1.0, r=0.10, zorder=1)
    txt(ax, x0_m + 0.20, y1_m - 0.16, "CENTRAL MECHANISM", fs=7.0, weight="bold",
        color=INK, ha="left", va="top")

    # --- two stacked haplotype tracks, same focal variant, different background ---
    label_x = x0_m + 0.20
    track_x0 = x0_m + 2.05
    track_w = 3.10
    yA = y1_m - 0.55
    yB = y0_m + 0.75
    mid_y_mech = (yA + yB) / 2
    focal_x = track_x0 + track_w * 0.55
    bg_xs_rel = [0.15, 0.65, 0.88]

    for y_track, lab, present, col, col_d in [
        (yA, "Ancestry A", [True, False, True], C1, C1_D),
        (yB, "Ancestry B", [False, True, True], C2, C2_D),
    ]:
        ax.plot([track_x0, track_x0 + track_w], [y_track, y_track], color=GRAY, lw=1.1,
                 solid_capstyle="round", zorder=2)
        txt(ax, label_x, y_track, lab, fs=5.4, weight="bold", color=col_d, ha="left")
        for frac, present_here in zip(bg_xs_rel, present):
            bx = track_x0 + track_w * frac
            fc = col if present_here else "white"
            ax.add_patch(Circle((bx, y_track), 0.075, facecolor=fc, edgecolor=col_d,
                                 linewidth=0.6, zorder=4))
        d = 0.10
        diamond = plt.Polygon(
            [[focal_x, y_track + d], [focal_x + d, y_track], [focal_x, y_track - d],
             [focal_x - d, y_track]],
            closed=True, facecolor="#D9A441", edgecolor="white", linewidth=0.5, zorder=5)
        ax.add_patch(diamond)

    footnote_y = y0_m + 0.24
    txt(ax, track_x0 + track_w / 2, mid_y_mech, "same causal\nvariant", fs=4.7,
        style="italic", color=GRAY_D, va="center", linespacing=1.2)
    txt(ax, track_x0 + track_w / 2, footnote_y, "background allele frequencies differ",
        fs=4.7, style="italic", color=GRAY_D)

    # arrow 1: haplotype tracks -> regulatory context
    x_ctx = track_x0 + track_w + 1.00
    arrow(ax, track_x0 + track_w + 0.18, mid_y_mech, x_ctx - 0.42, mid_y_mech,
          color=ARROW_GRAY, lw=1.1, ms=6)

    # regulatory context icon: two small TF-complex clusters (tight vs. spread)
    for y_track, col, spread in [(yA, C1, 0.09), (yB, C2, 0.20)]:
        offsets = [(-spread, 0.05), (spread * 0.9, -0.06), (0.02, spread * 0.8)]
        for ox, oy in offsets:
            ax.add_patch(Circle((x_ctx + ox, y_track + oy), 0.11, facecolor=col,
                                 edgecolor="white", linewidth=0.4, alpha=0.75, zorder=4))
    txt(ax, x_ctx, footnote_y, "combinatorial regulatory\ncontext differs",
        fs=4.7, style="italic", color=GRAY_D, linespacing=1.2)

    # arrow 2: regulatory context -> expression effect
    x_expr = x_ctx + 1.55
    arrow(ax, x_ctx + 0.55, mid_y_mech, x_expr - 0.55, mid_y_mech,
          color=ARROW_GRAY, lw=1.1, ms=6)

    # expression effect bars (abstract, unlabeled magnitude)
    bars_base = y0_m + 0.65
    diff_bars_icon(ax, x_expr, bars_base, C1, C2, h_a=1.05, h_b=0.55, w=0.34, gap=0.22)
    txt(ax, x_expr, y1_m - 0.24, "Expression\neffect", fs=5.4, weight="bold", color=INK,
        linespacing=1.2)

    # closing caption -- generous, dedicated right-hand column so text never runs
    # off the panel (the earlier draft anchored this off the right edge of the canvas)
    x_cap0 = x_expr + 0.85
    txt(ax, x_cap0, mid_y_mech,
        "Same causal variant, different haplotype\n"
        "context → different expression effect.\n\n"
        "Shared LCL culture conditions minimize\n"
        "environmental confounding.",
        fs=6.1, color=GRAY_D, ha="left", va="center", linespacing=1.4)

    # -------------------------------------------------------------
    fig.savefig(FIGS / "Figure1.png", dpi=600, facecolor="white")
    fig.savefig(FIGS / "Figure1.pdf", facecolor="white")
    announce(__file__, "Figure1")
    plt.close(fig)
    print(f"Saved: {FIGS / 'Figure1.png'}")
    print(f"Saved: {FIGS / 'Figure1.pdf'}")


if __name__ == "__main__":
    main()
