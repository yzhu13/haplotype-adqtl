"""
Pure-Python remote tabix slicer for Pan-UKB per-phenotype summary-stat files.

Pan-UKB sumstats (https://pan.ukbb.broadinstitute.org/) are bgzipped TSVs with a
.tbi index, columns chr/pos/ref/alt (chrom names are bare numbers '1'..'22'/'X',
no 'chr' prefix) followed by per-ancestry-group beta/se/pval/af columns
(suffixes _AFR/_AMR/_CSA/_EAS/_EUR/_MID, plus *_meta).

The .tbi has col_seq=1, col_beg=2, col_end=2 -- same convention as VCF
(chrom=f[0], pos=f[1]), so RemoteTabixVCF.query() works unchanged. We just add
a header-aware row parser on top.

- Data files:  https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/<filename>
- Tabix index: https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files_tabix/<filename>.tbi
"""

import pandas as pd

from lib_remote_vcf import RemoteTabixVCF, http_range, bgzf_decompress

DATA_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files/"
TABIX_BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files_tabix/"


class RemotePanUKB:
    """Region queries against a remote Pan-UKB per-phenotype sumstat file."""

    def __init__(self, filename):
        self.filename = filename
        self.url = DATA_BASE + filename
        self.tbi_url = TABIX_BASE + filename + ".tbi"
        self._vcf = RemoteTabixVCF(self.url, tbi_url=self.tbi_url)
        self._header = None

    def header(self):
        """Fetch and cache the column-name header (first line of the file)."""
        if self._header is not None:
            return self._header
        raw = http_range(self.url, 0, 65536)
        text = bgzf_decompress(raw).decode("utf-8", "replace")
        self._header = text.split("\n", 1)[0].split("\t")
        return self._header

    def query_region(self, chrom, beg, end):
        """Return a DataFrame of rows overlapping 1-based [beg, end] on `chrom`
        (chrom as '1'..'22'/'X', no 'chr' prefix)."""
        cols = self.header()
        rows = []
        for ln in self._vcf.query(chrom, beg, end):
            f = ln.rstrip("\n").split("\t")
            if len(f) != len(cols):
                continue
            rows.append(f)
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)


def variant_id(row):
    """chr:pos:ref:alt id matching the 1000G/MAGE kgpID convention's coordinate part."""
    return f"{row['chr']}:{row['pos']}:{row['ref']}:{row['alt']}"


def _smoke_test():
    """AMZ1 region (chr1:~110.5Mb hg19!) -- Pan-UKB is hg19. Just check the
    plumbing returns rows with expected columns for Standing height (phenocode 50)."""
    pu = RemotePanUKB("continuous-50-both_sexes-irnt.tsv.bgz")
    cols = pu.header()
    print("n_cols =", len(cols))
    print("cols[:8] =", cols[:8])
    assert "beta_EUR" in cols and "beta_AFR" in cols and "se_EUR" in cols
    df = pu.query_region("1", 11000, 20000)
    print("rows in chr1:11000-20000:", len(df))
    print(df.head(3))
    assert len(df) > 0
    print("\nlib_remote_panukb smoke test PASSED.")


if __name__ == "__main__":
    _smoke_test()
