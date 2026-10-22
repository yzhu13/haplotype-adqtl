"""
Promote regenerated renders to the draft's final figure files.

Figure scripts write into `figures/regenerated/` under their historical file names
(Figure_HCD, Figure_GeuvadisYRI, ...). The manuscript embeds `figures/FigureN.png` under
the current draft numbering. This script maps one to the other using
`lib_figure_io.DRAFT_FIGURE` and copies, so a rename can never be done by hand against
the wrong number -- which is how FigureS5.png ended up titled "Figure S2".

Dry run by default: it lists what would be copied and whether the target would change.
Nothing is written without --apply.

Usage:
    python scripts/finalize_figures.py                  # show the plan
    python scripts/finalize_figures.py --apply          # copy, backing up replaced files
    python scripts/finalize_figures.py --apply --only Figure6 FigureS5
"""
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

from lib_figure_io import (FIG_DIR, REGEN_ROOT, DRAFT_FIGURE, PREFERRED_SOURCE,
                           UNMANAGED, regen_dir_for)

BACKUP_DIR = FIG_DIR / "superseded"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    apply = "--apply" in sys.argv
    only = []
    if "--only" in sys.argv:
        only = [a for a in sys.argv[sys.argv.index("--only") + 1:] if not a.startswith("--")]

    # draft figure -> list of (legacy stem, source script) that claim it
    claims = {}
    for (script, legacy), draft in DRAFT_FIGURE.items():
        if draft:
            claims.setdefault(draft, []).append((legacy, script))

    plan, skipped = [], []
    for draft, sources in sorted(claims.items()):
        if only and draft not in only:
            continue
        available = [(lg, sc) for lg, sc in sources
                     if (regen_dir_for(sc) / f"{lg}.png").exists()]
        if not available:
            skipped.append((draft, "no regenerated render on disk", sources))
            continue
        if len(available) > 1:
            winner = PREFERRED_SOURCE.get(draft)
            preferred = [(lg, sc) for lg, sc in available if sc == winner]
            if not preferred:
                skipped.append((draft, f"ambiguous: {len(available)} renders claim it "
                                       f"and no PREFERRED_SOURCE entry", available))
                continue
            available = preferred
        legacy, script = available[0]
        src = regen_dir_for(script) / f"{legacy}.png"
        dst = FIG_DIR / f"{draft}.png"
        if dst.exists() and md5(src) == md5(dst):
            state = "identical, nothing to do"
        elif dst.exists():
            state = "WOULD REPLACE existing"
        else:
            state = "new file"
        plan.append((draft, legacy, script, state))

    print(f"regenerated: {REGEN_ROOT}")
    print(f"final:       {FIG_DIR}\n")
    print(f"{'draft':<12} {'from render':<26} {'source script':<30} status")
    print("-" * 96)
    for draft, legacy, script, state in plan:
        print(f"{draft:<12} {legacy:<26} {script:<30} {state}")
    for draft, why, detail in skipped:
        print(f"{draft:<12} {'--':<26} {'--':<30} SKIPPED: {why}")
    for draft, note in sorted(UNMANAGED.items()):
        print(f"{draft:<12} {'--':<26} {'--':<30} not managed here: {note}")

    if not apply:
        print("\ndry run. re-run with --apply to copy "
              "(replaced files are backed up to figures/superseded/).")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    n = 0
    for draft, legacy, _script, state in plan:
        if state == "identical, nothing to do":
            continue
        for ext in ("png", "pdf"):
            src = regen_dir_for(_script) / f"{legacy}.{ext}"
            dst = FIG_DIR / f"{draft}.{ext}"
            if not src.exists():
                continue
            if dst.exists():
                BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, BACKUP_DIR / f"{draft}.{stamp}.{ext}")
            shutil.copy2(src, dst)
            n += 1
        print(f"  {legacy} -> {draft}")
    print(f"\ncopied {n} file(s). replaced originals are in {BACKUP_DIR}")


if __name__ == "__main__":
    main()
