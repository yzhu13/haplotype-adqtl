# Population differences in regulatory haplotype backgrounds explain ancestry-differential eQTL effects

Analysis code for:

> Zhu Y, et al. *Population differences in regulatory haplotype backgrounds explain
> ancestry-differential eQTL effects.* (manuscript in preparation)

The same fine-mapped regulatory variant can change gene expression by different amounts,
or even in opposite directions, in European- and African-ancestry individuals. This
pipeline tests whether these **ancestry-differential eQTLs (adQTLs)** are explained by
**cis-haplotype epistasis**: the focal variant's effect depends on nearby background
variants on the same haplotype, and those background variants differ in frequency
between populations (Δobs ≈ γ × Δf).

All inputs are public: MAGE (731 LCLs, five continental groups), 1000 Genomes NYGC
high-coverage phased genotypes, and Geuvadis (EUR + YRI LCLs).

---

## Pipeline overview

| Stage | What it does | Main scripts | Manuscript |
|---|---|---|---|
| **Stage 1: adQTL discovery** | Per-ancestry eQTL effects of fine-mapped lead variants; EUR–AFR heterogeneity test (BH FDR < 0.1) | `p0_01`, `p0_02`, `p0_05`, `p0_06`, `p0_07` | Figure 2 |
| Artifact exclusion | AF / power / LD decomposition, split-half reproducibility | `p1_01`, `p1_03`, `v1` | Figure 2D, Figure S1 |
| **P1: cis focal × background enrichment** | Interaction scan within ±50 kb; permutation and distance controls; r² robustness | `p1_06`, `p1_16b`, `p1_14`, `v2`, `v3` | Figure 3A/E, S2, S3, S6 |
| **P2: within-ancestry confirmation** | Interaction tested within EUR / AFR separately and with ancestry covariates | `p1_07`, `v7` | Figure 3B, S5 |
| Effect decomposition | Fraction of Δobs explained by cis background (same-sample and split-sample) | `p1_11`, `v8` | Figure 3C, S5 |
| **P3: generative simulation** | Null / LD artifact / GxE / cis-epistasis models run through the full pipeline | `p1_18` | Figure 4 |
| **P4: haplotype context divergence (HCD)** | Jensen–Shannon divergence of background haplotypes in focal-allele carriers | `p1_12`, `v4` | Figure 5 |
| **Stage 3: independent replication** | Geuvadis EUR + YRI effect-direction and heterogeneity replication; focal × bg replication in Geuvadis EUR | `p1_22`, `v5` | Figure 6, S5A |
| Annotation and sequence models | ChromHMM enrichment, AlphaGenome reference-haplotype scoring | `p1_04a`, `p0_03`, `p0_04`, `p1_04b`, `p1_04c` | Figure 7 |
| PGS portability simulation | Consequences of cis-epistasis for cross-ancestry polygenic scores | `p1_20` | Figure S4 |
| Disease colocalization (null) | PAGE coloc, GWAS Catalog overlap, Pan-UKB per-ancestry coloc | `p1_02`, `p1_05`, `p2_01`–`p2_04` | Discussion |

Supplementary / exploratory scripts (not central to the main text):

- `p0_08a`–`p0_09`: feasibility test for scoring custom ancestry backgrounds with
  AlphaGenome. The public API scores reference-genome variants only, so this is reported
  as a tooling limitation.
- `p1_08`, `p1_16`, `p1_17`: single-background and trans-background controls.
- `p1_09`, `p1_09b`, `v9`: predicted TF-binding effects of background SNPs (Figure S5F).

---

## Repository layout

```
scripts/        all analysis and figure scripts (flat; they import each other by name)
results/        summary-level result tables produced by the scripts (included)
data/           inputs; only three small Geuvadis sample lists are included (see below)
docs/           per-step text reports are written here
figures/        figure renders are written here (not tracked)
```

All paths are relative to the repository root, so scripts can be run from anywhere:
`python scripts/<name>.py`.

---

## Installation

Python ≥ 3.10 (tested with 3.13). No htslib, bcftools or R is needed: genotypes are read
directly from the remote 1000 Genomes VCFs with a pure-Python tabix reader
(`scripts/lib_remote_vcf.py`).

```bash
pip install -r requirements.txt
```

The AlphaGenome steps additionally need `pip install alphagenome` and an API key
(copy `.env.example` to `.env` and fill in `ALPHAGENOME_API_KEY`).

---

## Input data

| Dataset | Used for | How to obtain |
|---|---|---|
| MAGE v1.0 (Taylor et al. 2024, *Nature*) | expression, covariates, SuSiE fine-mapping, ChromHMM, PAGE coloc | `python scripts/p0_01_fetch_mage.py` downloads only the needed members of the Zenodo archive ([10.5281/zenodo.10535719](https://doi.org/10.5281/zenodo.10535719)) into `data/raw/MAGE/` |
| 1000 Genomes NYGC high-coverage phased VCFs (GRCh38) | genotypes | read remotely from the EBI FTP; nothing to download |
| Geuvadis (Lappalainen et al. 2013) | Stage 3 replication | from ArrayExpress E-GEUV-1, place in `data/raw/Geuvadis/`: gene RPKM matrix as `GD462.GeneQuantRPKM.txt.gz`; EUR / YRI cis-eQTL tables as `EUR373.gene.cis.FDR5.all.txt.gz` and `YRI89.gene.cis.FDR5.all.txt.gz`; the 1000 Genomes sample panel (`integrated_call_samples_v3.20130502.ALL.panel`) as `1000G_panel.txt` |
| Geuvadis sample lists (included) | Stage 3 replication | `geuv_eur_indep.txt` / `geuv_yri_indep.txt`: Geuvadis EUR / YRI donors not in MAGE; `geuv_sex.txt`: donor sex from the 1000 Genomes panel |
| GENCODE v44 basic annotation | exon masks (AlphaGenome readout) | `data/external/gencode.v44.basic.annotation.gtf.gz` |
| GWAS Catalog | locus overlap | full associations (`gwas-catalog-download-associations-alt-full.tsv`) and `ancestry.tsv` in `data/external/gwas_catalog/` |
| Pan-UKB | per-ancestry coloc | `phenotype_manifest.tsv.bgz` in `data/external/panukb/`; summary statistics are streamed from the public S3 bucket |
| hg38 reference FASTA | AlphaGenome background sequences only | set `HG38_FASTA`, or place at `data/external/hg38/hg38.fa` |

Note: MAGE libraries SRR19762530 and SRR19762653 have a documented sample swap and are
excluded.

---

## Running the pipeline

The `results/` tables are included, so every figure script can be run directly without
recomputing the analysis. The one exception is the Figure 3D vignette, which needs the
MAGE expression data (`p0_01_fetch_mage.py`) and reads genotypes remotely. To reproduce
from raw data, run in this order.

```bash
# Stage 1: adQTL discovery
python scripts/p0_01_fetch_mage.py
python scripts/p0_02_build_calibration_set.py
python scripts/p0_05_fetch_genotypes.py              # remote 1000G genotypes
python scripts/p0_06_per_ancestry_betas.py
python scripts/p0_07_characterize_adqtl.py
python scripts/p1_01_decompose_ancestry_effect.py
python scripts/p1_03_robustness_splithalf.py

# Stage 2: mechanistic predictions
python scripts/p1_06_cis_interaction.py              # P1 (long; checkpointed)
python scripts/p1_07_within_ancestry_confirm.py      # P2
python scripts/p1_11_multi_bg_decomposition.py
python scripts/p1_14_distance_decay.py
python scripts/p1_16b_perm_negative_control.py
python scripts/p1_18_generative_simulation.py        # P3
python scripts/p1_12_haplotype_context_divergence.py # P4
python scripts/p1_20_prs_simulation.py

# Stage 3: independent replication
python scripts/p1_22_geuvadis_yri_replication.py

# Robustness and validation
python scripts/v1_geuvadis_validation.py
python scripts/v2_r2_robustness.py
python scripts/v3_interaction_pval_dist.py
python scripts/v4_signflip_analysis.py
python scripts/v5_geuvadis_focal_bg.py               # checkpointed
python scripts/v7_ancestry_covariate_model.py        # checkpointed
python scripts/v8_splitsample_decomp.py              # checkpointed

# Annotation and sequence models
python scripts/p1_04a_chromhmm_enrichment.py
python scripts/p0_03_run_alphagenome_calibration.py  # needs ALPHAGENOME_API_KEY
python scripts/p0_04_calibration_analysis.py
python scripts/p1_04b_score_adqtl_reference.py       # needs ALPHAGENOME_API_KEY
python scripts/p1_04c_refscore_analysis.py

# Disease colocalization (null results)
python scripts/p1_02_page_coloc.py
python scripts/p1_05_gwas_catalog_enrichment.py
python scripts/p2_01_regional_eqtl_nominal.py
python scripts/p2_02_fetch_panukb.py
python scripts/p2_03_coloc_run.py
python scripts/p2_04_coloc_verdict.py
```

Long-running steps write checkpoints and can be interrupted and resumed. AlphaGenome
gRPC calls occasionally hang; the scoring scripts include a watchdog and resume from
their checkpoint when restarted.

### Figures

Each figure script renders into `figures/regenerated/<script>/`.
`python scripts/finalize_figures.py --apply` copies the renders to `figures/FigureN.png`
using the mapping in `scripts/lib_figure_io.py`.

| Figure | Script |
|---|---|
| Figure 1 (study design) | `make_figure1_study_design.py` |
| Figure 2 (adQTL atlas) | `make_paper_figures.py` |
| Figure 3 (cis-haplotype epistasis) | `make_updated_figures.py` (based on `p1_10_figure_cis_epistasis.py`) |
| Figure 4 (generative simulation) | `p1_19_figure_simulation.py` |
| Figure 5 (haplotype context divergence) | `p1_13_figure_hcd.py` |
| Figure 6 (Geuvadis EUR + YRI replication) | `p1_23_figure_geuvadis_yri.py` |
| Figure 7 (regulatory annotation and sequence models) | `make_paper_figures.py` |
| Figure S1 (artifact exclusion) | `make_updated_figures.py` |
| Figure S2 (permutation control) | `p1_17b_figure_perm_control.py` |
| Figure S3 (distance) | `p1_15_figure_distance_decay.py` |
| Figure S4 (PGS portability simulation) | `p1_21_figure_prs_portability.py` |
| Figure S5 (robustness) | `v10_figure_s5_robustness.py` |
| Figure S6 (r² threshold) | `v2_r2_robustness.py` |

---

## Key result tables

| File | Content |
|---|---|
| `results/phase1/p1_decomposition.tsv` | all 3,790 tested variant–gene pairs; `is_adqtl` marks the 200 adQTLs |
| `results/phase1/p1_cis_interaction.tsv` | focal × background interaction scan (adQTL and control loci) |
| `results/phase1/p1_within_ancestry.tsv` | within-ancestry interaction tests |
| `results/validation/v8_splitsample_decomp.tsv` | cross-validated cis-background fraction (median 35.9%) |
| `results/phase1/p1_simulation_results.tsv` | generative simulation |
| `results/phase1/p1_hcd.tsv` | haplotype context divergence |
| `results/phase1/p1_geuvadis_yri_replication.tsv` | Geuvadis EUR + YRI replication |

---

## Citation

If you use this code, please cite the manuscript above (citation will be updated on
publication).

## License

MIT; see `LICENSE`.
