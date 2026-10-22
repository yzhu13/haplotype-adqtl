"""
Shared figure output policy.

- Scripts render into `figures/regenerated/<script>/` under their historical file name.
  Nothing in `figures/` root is touched until a render is deliberately promoted with
  `finalize_figures.py --apply`.
- `suptitle()` is a no-op unless `--title` is passed (Cell Press typesets legends
  separately, so figure numbers are not burned into the images).
- `DRAFT_FIGURE` maps each (script stem, legacy file stem) to its manuscript figure.
  It is keyed by script because legacy names collide: `make_paper_figures` saves the
  sequence-model panel as "Figure5" while `make_updated_figures` saves the
  cis-epistasis panel under the same name.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "figures"                  # final, manuscript-embedded figures
REGEN_ROOT = FIG_DIR / "regenerated"


def regen_dir_for(script_stem):
    """Output directory for one script's renders."""
    d = REGEN_ROOT / script_stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _calling_script_stem():
    """Stem of the script being run, so each script gets its own output directory.

    Without this the legacy file names collide: make_paper_figures saves the
    sequence-model panel as "Figure5" and make_updated_figures saves the cis-epistasis
    panel under the same name, so whichever ran last would silently win.
    """
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    return Path(argv0).stem or "misc"


# Per-script output directory. Scripts just use REGEN_DIR; the split is automatic.
REGEN_DIR = regen_dir_for(_calling_script_stem())

# (script stem, legacy figure stem) -> draft figure stem, or None if the figure did not
# make it into the current draft.
DRAFT_FIGURE = {
    ("make_paper_figures", "Figure1"): "Figure2",       # adQTL atlas
    ("make_paper_figures", "Figure2"): "FigureS1",      # not an AF/LD/power artifact
    ("make_paper_figures", "Figure4"): "Figure7",       # chromHMM + reference scoring
    ("make_paper_figures", "Figure5"): None,            # sub-gate B readout ladder — dropped
    ("make_updated_figures", "Figure2"): "FigureS1",    # supersedes make_paper_figures
    ("make_updated_figures", "Figure5"): "Figure3",     # supersedes p1_10 (adds ECDF panel)
    ("p1_10_figure_cis_epistasis", "Figure3"): "Figure3",
    ("p1_13_figure_hcd", "Figure_HCD"): "Figure5",
    ("p1_15_figure_distance_decay", "Figure_DistanceDecay"): "FigureS3",
    ("p1_17_figure_trans_control", "Figure_TransControl"): None,   # not in the draft
    ("p1_17b_figure_perm_control", "Figure_PermControl"): "FigureS2",
    ("p1_19_figure_simulation", "Figure_Simulation"): "Figure4",
    ("p1_21_figure_prs_portability", "Figure_PRS_Portability"): "FigureS4",
    ("p1_23_figure_geuvadis_yri", "Figure_GeuvadisYRI"): "Figure6",
    ("v2_r2_robustness", "v2_r2_robustness"): "FigureS6",
    ("v10_figure_s5_robustness", "FigureS5"): "FigureS5",
    ("make_figure1_study_design", "Figure1"): "Figure1",  # replaces the hand-edited version
}


# Two scripts render the same draft figure in two cases. This records which one wins,
# so promotion never depends on which script happened to run last.
PREFERRED_SOURCE = {
    "Figure3": "make_updated_figures",   # adds the P-value ECDF panel that p1_10 lacks
    "FigureS1": "make_updated_figures",  # adds the Geuvadis replication panel
}

# Draft figures produced outside this pipeline, for completeness.
# Figure1 was moved into this pipeline 2026-09-01 (scripts/make_figure1_study_design.py);
# the old docs/make_fig1_studydesign.py is stale reference material only (dead output
# path, superseded Stage 2 prediction order) and is no longer the source of Figure1.
UNMANAGED = {}


def titles_enabled():
    """True only when the caller was run with --title."""
    return "--title" in sys.argv


def suptitle(fig, *args, **kwargs):
    """fig.suptitle(...) that draws nothing unless --title was passed.

    Keeps the call sites intact so the intended wording stays in the source, while
    keeping a stale figure number out of the rendered image.
    """
    if titles_enabled():
        return fig.suptitle(*args, **kwargs)
    return None


def draft_name(source, legacy_stem):
    """Which draft figure this render corresponds to ('Figure6', or None if dropped)."""
    return DRAFT_FIGURE.get((Path(source).stem, legacy_stem), "?")


def announce(source, legacy_stem):
    """Print where a render landed and which draft figure it feeds."""
    target = draft_name(source, legacy_stem)
    if target is None:
        note = "not used in the current draft"
    elif target == "?":
        note = "no draft mapping recorded — add one to lib_figure_io.DRAFT_FIGURE"
    else:
        note = f"-> draft {target}"
    rel = (regen_dir_for(Path(source).stem) / legacy_stem).relative_to(BASE)
    print(f"  {rel}.png/.pdf   [{note}]")
