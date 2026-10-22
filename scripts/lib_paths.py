"""
Locations of inputs that are not produced by this pipeline.

Only the AlphaGenome steps need these (Phase 0 calibration / background scoring,
p1_04b reference scoring, p1_09 TF vignette). Everything else uses paths relative to
the repository root.

Resolution order, first hit wins:
  ALPHAGENOME_ENV : environment variable, else `<repo>/.env`
                    (the file must define ALPHAGENOME_API_KEY; see `.env.example`)
  HG38_FASTA      : environment variable, else `<repo>/data/external/hg38/hg38.fa`
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

ALPHAGENOME_ENV = Path(os.environ.get("ALPHAGENOME_ENV", BASE / ".env"))
HG38_FASTA = Path(os.environ.get("HG38_FASTA", BASE / "data" / "external" / "hg38" / "hg38.fa"))


if __name__ == "__main__":
    print(f"repository root  : {BASE}")
    print(f"AlphaGenome .env : {ALPHAGENOME_ENV}   exists={ALPHAGENOME_ENV.exists()}")
    print(f"hg38 FASTA       : {HG38_FASTA}   exists={HG38_FASTA.exists()}")
