# ENVIRONMENT — rebuild recipe (verbatim from the project engineering record)

## Python (testing layer, battery, exhibits)
- Verified stack: numpy 2.4.4, scipy 1.17.1, pandas 3.0.2 (2026-08-23 session;
  archived p-values reproduced exactly — replayed by
  tests/test_regression_bridges.py).
  pydeseq2 0.5.4 for the DESeq2 arms (pip).
- `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`; multiprocessing
  uses spawn; reference container: 2 CPUs.
- Spike-arm reruns: `PYTHONHASHSEED=0` (and disclose). The original
  run_real_k10k spike seeds included `hash(armname)` from an unseeded
  interpreter, so the archived spike permutation stream is pinned by the
  archived CSVs, not replayable; native-arm seeds contain no hash and replay
  exactly.
- Module path: scripts expect `/home/claude/ch_smoke/` on sys.path with
  `twochannel.py`, `run_rich.py`, and `code/estimation_v3/model.py`
  (use `code/testing/model_RECONSTRUCTED.py` until the original is rescued).

## R (external methods: LinDA, LOCOM, LDM, ANCOM-BC2)
Sandboxed networks block CRAN/Bioconductor domains. Working recipe:
- Base packages via apt (`r-cran-*`, `r-bioc-*`); missing CRAN sources via
  `github.com/cran/<pkg>` mirror clones + `R CMD INSTALL`.
- **ANCOMBC 2.6.1** from the RELEASE_3_19 GitHub branch (R >= 4.3): strip the
  5 CVXR `importFrom` lines from NAMESPACE before install (CVXR is used only
  by trend tests, not by any call path in our scripts; precedent: LDM built
  with castor stripped). `ancombc2` needs `meta_data` with >= 2 columns
  (works around its single-column drop bug); configuration used:
  fix_formula two-group, `struc_zero=TRUE` + group (the one declared
  deviation, predecessor convention), everything else shipped defaults.
- **LOCOM**: `rm src/*.o` then Makevars `PKG_LIBS=$(LAPACK_LIBS) $(BLAS_LIBS)
  $(FLIBS)`.
- **DESeq2**: pydeseq2 (pip) or apt r-bioc-deseq2.
- Rebuild time ~15 min.

## LaTeX
- `pdflatex` x2 (a third pass after adding a new cross-referenced float);
  microtype with `expansion=false` (twocol).
- Two-manuscript discipline: every edit applied to `main_condensed.tex` and
  `main_twocol.tex` with unique-match assertions (see code/edits/ scripts;
  twocol variants: `table*` floats, different abstract tail wrap).
