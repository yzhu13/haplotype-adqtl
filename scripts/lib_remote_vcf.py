"""
Pure-Python remote tabix slicer for bgzipped + .tbi-indexed VCFs over HTTP.

Why: this box is MINGW64 + native Windows Python 3.13 with no htslib wheels
(pysam/cyvcf2 fail to build) and no bcftools/tabix. We only need genotypes at a
few thousand sites (and later SNP windows for haplotype backgrounds), so we read
the tiny .tbi index, compute BGZF byte offsets, and HTTP-range-fetch + decompress
only the blocks we need — no multi-GB whole-chromosome download.

Implements: .tbi parse, UCSC reg2bins, BGZF block decompression, region query.
Reference: HTSlib tabix/BGZF spec (samtools.github.io/hts-specs/tabix.pdf).

Tested against the 1000G high-coverage 3202-sample phased VCFs at EBI.
"""

import gzip
import struct
import urllib.request
import zlib
from dataclasses import dataclass, field


# ── UCSC binning ────────────────────────────────────────────────────────────
def reg2bins(beg, end):
    """Bins overlapping 0-based half-open [beg, end). Yields ints."""
    end -= 1
    yield 0
    for first, shift in ((1, 26), (9, 23), (73, 20), (585, 17), (4681, 14)):
        k0 = first + (beg >> shift)
        k1 = first + (end >> shift)
        for k in range(k0, k1 + 1):
            yield k


# ── BGZF ────────────────────────────────────────────────────────────────────
def bgzf_decompress(data):
    """Decompress concatenated BGZF (gzip) blocks. `data` must start at a block
    boundary. Returns the full decompressed bytes."""
    out = bytearray()
    mv = memoryview(data)
    while len(mv) > 0:
        d = zlib.decompressobj(wbits=31)  # gzip member (BGZF block is one member)
        out += d.decompress(mv)
        out += d.flush()
        rem = d.unused_data
        if not rem or len(rem) == len(mv):
            break
        mv = memoryview(rem)
    return bytes(out)


# ── HTTP range ──────────────────────────────────────────────────────────────
def http_range(url, start, end=None):
    """Fetch bytes [start, end] inclusive (end=None -> to EOF). Returns bytes."""
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    req = urllib.request.Request(url, headers={"Range": rng})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ── Tabix index ─────────────────────────────────────────────────────────────
@dataclass
class TabixRef:
    bins: dict = field(default_factory=dict)   # bin -> list[(cbeg, cend)] virtual offsets
    intervals: list = field(default_factory=list)  # linear index virtual offsets


@dataclass
class Tabix:
    names: list
    refs: list           # list[TabixRef], parallel to names
    fmt: int
    col_seq: int
    col_beg: int
    col_end: int
    meta: int
    skip: int

    def name_index(self, chrom):
        if chrom in self.names:
            return self.names.index(chrom)
        # try toggling 'chr' prefix
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        return self.names.index(alt) if alt in self.names else None


def parse_tbi(raw_gz):
    """Parse a .tbi (gzip-compressed). Returns Tabix."""
    buf = gzip.decompress(raw_gz)
    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from("<i", buf, off)[0]; off += 4; return v

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", buf, off)[0]; off += 8; return v

    magic = buf[:4]; off = 4
    assert magic == b"TBI\x01", f"bad tabix magic {magic!r}"
    n_ref = u32(); fmt = u32(); col_seq = u32(); col_beg = u32()
    col_end = u32(); meta = u32(); skip = u32(); l_nm = u32()
    names_blob = buf[off:off + l_nm]; off += l_nm
    names = [n.decode() for n in names_blob.split(b"\x00") if n]

    refs = []
    for _ in range(n_ref):
        ref = TabixRef()
        n_bin = u32()
        for _ in range(n_bin):
            bin_id = struct.unpack_from("<I", buf, off)[0]; off += 4
            n_chunk = u32()
            chunks = []
            for _ in range(n_chunk):
                cbeg = u64(); cend = u64()
                chunks.append((cbeg, cend))
            ref.bins[bin_id] = chunks
        n_intv = u32()
        ref.intervals = [u64() for _ in range(n_intv)]
        refs.append(ref)

    return Tabix(names, refs, fmt, col_seq, col_beg, col_end, meta, skip)


class RemoteTabixVCF:
    """Region queries against a remote bgzipped+tabix VCF."""

    def __init__(self, vcf_url, tbi_url=None):
        self.url = vcf_url
        self.tbi = parse_tbi(http_get(tbi_url or vcf_url + ".tbi"))
        self._header = None

    @staticmethod
    def coff(vo):
        return vo >> 16

    @staticmethod
    def uoff(vo):
        return vo & 0xFFFF

    def header_lines(self):
        """Fetch and cache the VCF header (lines starting with '#')."""
        if self._header is not None:
            return self._header
        # Header sits at file start; grab enough bytes to include #CHROM line.
        raw = http_range(self.url, 0, 5_000_000)
        text = bgzf_decompress(raw).decode("utf-8", "replace")
        lines = []
        for ln in text.split("\n"):
            if ln.startswith("#"):
                lines.append(ln)
                if ln.startswith("#CHROM"):
                    break
            elif lines:
                break
        self._header = lines
        return lines

    def samples(self):
        for ln in self.header_lines():
            if ln.startswith("#CHROM"):
                return ln.rstrip("\n").split("\t")[9:]
        return []

    def query(self, chrom, beg, end):
        """Yield VCF record lines (str) overlapping 1-based [beg, end]."""
        rid = self.tbi.name_index(chrom)
        if rid is None:
            return
        ref = self.tbi.refs[rid]
        # candidate chunks from overlapping bins (use 0-based coords for binning)
        b0, e0 = beg - 1, end
        chunks = []
        for b in reg2bins(b0, e0):
            chunks.extend(ref.bins.get(b, []))
        if not chunks:
            return
        # linear index: minimum offset for the 16kb window of beg
        min_vo = 0
        li = b0 >> 14
        if li < len(ref.intervals):
            min_vo = ref.intervals[li]
        chunks = [(cb, ce) for (cb, ce) in chunks if ce > min_vo]
        if not chunks:
            return
        chunks.sort()
        # contig name as it appears in the file
        file_chrom = self.tbi.names[rid]

        for (cb, ce) in chunks:
            c_start = self.coff(cb)
            c_end = self.coff(ce)
            # fetch from block start of cb to block start of ce + one max block (64KB)
            raw = http_range(self.url, c_start, c_end + 65536)
            data = bgzf_decompress(raw)
            text = data[self.uoff(cb):].decode("utf-8", "replace")
            for ln in text.split("\n"):
                if not ln or ln.startswith("#"):
                    continue
                f = ln.split("\t", 2)
                if len(f) < 2 or f[0] != file_chrom:
                    continue
                try:
                    pos = int(f[1])
                except ValueError:
                    continue
                if pos < beg:
                    continue
                if pos > end:
                    return
                yield ln


def vcf_chrom_name(tbi_url, want):
    """Return the contig name used in the file matching `want` (handles chr prefix)."""
    tbi = parse_tbi(http_get(tbi_url))
    idx = tbi.name_index(want)
    return tbi.names[idx] if idx is not None else None
