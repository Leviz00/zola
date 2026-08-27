# MISSING — items not in this package, and where they live
# (updated 2026-08-23 evening, after the platform-code rescue)

0. **RESOLVED — original code tree rescued.** The full platform code export
   (`zola_code_all.tar.gz`, 181 files + CODE_INDEX, generated 2026-08-12 on
   the original platform) was recovered by the author on 2026-08-23 and is
   absorbed verbatim under `code/platform/` (original relative layout:
   `code/estimation{,_v2,_v3}` with `model.py`/`composite_likelihood.py`,
   `code/simulation{,_v3}` with `generators.py`, `analysis/` method-fix
   v3.1–v3.8 chain, `realdata/`, `fixes/`, `paper_draft/figures/_work/`,
   `r_setup/`). Verification (code-rescue record, project archive): (i) original
   `model.log_g` (Gamma form) vs the reconstruction (betaln form) —
   algebraically identical, max |Δ| = 3.5e-9 over a 168-point grid;
   (ii) end-to-end regeneration bridges, battery cells 2001 and 2005
   (rep 0, frozen SeedSequence([20260819,cell])) through original
   generators.py -> generators_ext -> twochannel with original model.py:
   every archived field reproduced EXACTLY (lib_p to 16 digits, nrej/fdp/
   tpr/typeI/oracle identical) against fix3_v4_detail.csv. Full-stack
   reproducibility (estimation layer + both simulation generations +
   testing layer) is now packageable.

Still not included (pointers only):
1. **Gen-1 per-replicate baseline intermediates** (`r_grid_exchange/`,
   ~101 MB, 3,999 files) and v3_baselines `exchange/` + `raw/` trees —
   needed only for re-scoring gen-1 baselines from raw; scored CSVs and the
   dual-truth table are under `results/gen1_baselines/`.
2. **R library snapshot** (`r_env/`, 693 MB) — rebuild via
   `env/ENVIRONMENT.md` (~15 min) or `code/platform/r_setup/`.
3. **Raw cohort data** — public; not redistributed (loaders and frozen
   subsample seeds in code; accessions in README).
4. **Deck/report assets** of the original workspace (PPT trees, screenshots)
   — not code; remain in the author's archives.
5. **Internal working documents (Chinese)** — the per-phase execution
   memos, the platform tree's index file, and the nine frozen
   pre-registration SPECs are working documents of the project and are not
   distributed. The frozen seeds, cell tables, and acceptance rules they
   fixed are embedded verbatim in the archived runners (e.g. the
   `run_pilot_v4.py` cell table and the `SeedSequence` conventions), the
   archived result CSVs, and the code map in `REPRODUCING.md`; amendments
   that affect reproduction are disclosed in the manuscript.
