# ZOLA — reproducibility archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22133432.svg)](https://doi.org/10.5281/zenodo.22133432)

**Author:** Yizhe Liu (刘翊哲) · **License:** MIT (see LICENSE) ·
**Cite:** CITATION.cff — Zenodo DOI
[10.5281/zenodo.22133432](https://doi.org/10.5281/zenodo.22133432)
(all versions; v1.0.0:
[10.5281/zenodo.22133433](https://doi.org/10.5281/zenodo.22133433))

Code, frozen specifications, and archived results behind the ZOLA
manuscript (*Which zero is which? Identified functionals,
channel-separated testing, and abstention for microbiome differential
abundance*). **Data policy:** raw cohort data and metadata are NOT
redistributed — `data/ACCESSIONS.md` gives the public accessions and the
derivation scripts that rebuild every analysis input.

Repo-style organization of every archived script and result CSV behind
the ZOLA manuscript (v4.12). Scripts are copied **verbatim**
from their phase archives — nothing was refactored or edited; this package
organizes, it does not modify. `PROVENANCE.md` lists every file with a
sha256 prefix; `MISSING.md` lists what is NOT here and why;
`REPRODUCING.md` is the runner guide, and `tests/test_regression_bridges.py`
replays the archived verification bridges (run it green before trusting any
new number in this environment).

## Layout

- `code/platform/` — the ORIGINAL platform code tree, rescued 2026-08-23
  (`zola_code_all.tar.gz` export of 2026-08-12; 181 files, original relative
  layout preserved verbatim; the export's Chinese index file stays in the
  private project archive). Contains the estimation
  layer (`code/estimation{,_v2,_v3}`: model.py, composite_likelihood.py,
  posterior.py, validators), both simulation generations
  (`code/simulation{,_v3}`: generators.py, design.py, metrics.py,
  weighting.py, abs_glm.py, tests/), the predecessor development chain
  (`analysis/method_fix` v3.1-v3.8, ej_criterion, estimated_weighting),
  realdata/ scripts (byte-identical to the device copies), fixes/,
  paper_draft figure scripts, and r_setup/. Authenticity verified by two
  exact end-to-end bridges (MISSING.md item 0).
- Pre-registration protocol: every batch froze its specification before
  execution (nine SPECs, maintained with the project's internal records —
  MISSING.md item 5). The frozen seeds, cell tables, and acceptance rules
  they fixed are embedded in the archived runners; deviations are archived
  as amendments, and those relevant to reproduction are disclosed in the
  manuscript.
- `code/testing/` — the two-channel test core: `twochannel.py` (channel
  statistics, detection-curve fitting, BH), `run_rich.py` (permutation
  layer, cohort loaders, richness adjustment). `model_ORIGINAL.py` is the
  rescued original (authoritative; Gamma-form log_g); place it at
  `ch_smoke/code/estimation_v3/model.py` for the imports to resolve.
  `model_RECONSTRUCTED.py` is the 2026-08-23 stand-in reconstruction used
  while the original was missing — algebraically identical (betaln form;
  max |Delta| 3.5e-9 vs the original on a verification grid) and
  end-to-end validated by 72/72 archived p-value reproduction
  (recorded in the project's internal archive); kept for the session record.
- `code/generation2/` — battery generator (`generators_ext.py`; its gen-1
  `generators` dependency now resolves from `code/platform/code/simulation_v3/`) and the frozen
  cell table + battery runners (`run_pilot_v4.py` = authoritative CELLS).
- `code/wrap/` — WRAP-01/02 comparison runners and the R method wrappers
  (`wrap_methods.R` from WRAP-01; `wrap_methods_wrap2.R` is WRAP-02's
  variant — the two differ and both are kept verbatim).
- `code/real/` — real-cohort pipelines: REAL-CH, EXT (external methods +
  spikes), K=9999 rerun (A4), richness K=9999, SHAPE A/B, full-spectrum.
- `code/rev/` — referee compute batch (ANCOM-BC2 install/battery/cohorts,
  cluster-mode externals, DESeq2 completion, gen-1 bridge).
- `code/exhibit/` — batch-2 exhibit scripts (attribution accuracy, e-BH
  two arms incl. Amendment EX-A1, CIs, AGP effect sizes).
- `code/edits/` — manuscript edit scripts (v4.8, v4.9, v4.10): each edit
  asserted unique-match in both tex files before replacement.
- `code/estimation_device/` — estimation-layer analysis scripts that DO
  exist on the device (`realdata/*.py`, S1 Prop-1 numerical verification).
  They import `composite_likelihood` and the full `model` from the
  original platform tree (see MISSING.md) and are included as the honest
  partial record.
- `results/<phase>/` — every archived CSV. Key files:
  `gen2_battery/fix3_v4_*` (Table 2), `wrap/*` + `rev_compute/wrap_deseq2_*`
  (Table 3), `real10k/*` (Table 4 official), `rev_compute/*` (ANCOM-BC2
  battery + full-spectrum), `exhibit/*` (attribution, e-BH, effect sizes,
  CIs), `gen1_baselines/scores_pooled_detectable_dualtruth.csv` (the §4
  dual-truth table).

## Reproduction notes

- Data: public cohorts (IBDMDB, MBQC, AGP). Genus npz matrices and
  metadata CSVs are NOT redistributed here; loaders expect
  `zola_project/realdata/data/*.npz` and `zola_project 2/datasets/...`
  under `/mnt/user-data/uploads` (see `run_rich.load_cohorts`). Frozen
  subsample seeds live in the code (e.g., 20260304 cohort subsamples).
- Environment: see `env/ENVIRONMENT.md` (R rebuild recipe incl. ANCOMBC
  2.6.1 branch install; python numpy 2.4.4/scipy 1.17.1 bridge-validated;
  set `PYTHONHASHSEED=0` for spike-arm reruns and disclose — the archived
  spike runs used an unseeded interpreter hash, so their permutation
  stream is pinned by the archived CSVs, not by replay).
- Every headline number in the manuscript maps to a CSV here; the code map
  in REPRODUCING.md gives the exhibit-to-file mapping. Bridge discipline:
  reruns must first reproduce an archived slice exactly (executable form:
  `tests/test_regression_bridges.py`) before any new number is trusted.
