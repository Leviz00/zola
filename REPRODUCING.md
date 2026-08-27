# REPRODUCING — how to run this package (added 2026-08-23)

**Bridge-first principle.** Before trusting any newly computed number, replay
an archived slice and require exact agreement. The executable form of that
rule is `tests/test_regression_bridges.py`:

    pip install -r env/requirements-verified.txt
    python tests/test_regression_bridges.py            # ~35 s: T1–T3
    ZOLA_FAST=1 python tests/...                       # T1 only (~1 s)
    ZOLA_DATA_ROOT=/path/to/uploads python tests/...   # + cohort bridge T4

Green = your environment reproduces the archived record; only then run
anything new.

## Path layout the frozen scripts expect

The frozen runners hard-code the original container layout
`/home/claude/ch_smoke/`. To rerun them verbatim, recreate it:

    mkdir -p /home/claude/ch_smoke/code
    cp code/testing/twochannel.py code/testing/run_rich.py \
       code/generation2/*.py /home/claude/ch_smoke/
    cp -r code/platform/code/simulation_v3 \
          code/platform/code/estimation_v3 /home/claude/ch_smoke/code/

(The tests do NOT need this — they import through the package layout.)
Cohort loaders read `$UP/zola_project/realdata/data/*.npz` and
`$UP/zola_project 2/datasets/...` with `UP=/mnt/user-data/uploads`
(`run_rich.py`); public-data accessions in README.md. Spike-arm reruns:
`PYTHONHASHSEED=0`, and disclose (ENVIRONMENT.md explains why).

## Code map — which script produced which exhibit

| Paper exhibit | Runner | Test core | Archived CSV |
|---|---|---|---|
| Table 1 (gen-1 grid) | code/generation2/run_confirm_v4.py | twochannel.two_channel_test | gen-1 archives (results/gen1_baselines) |
| Table 2 (battery) | code/generation2/run_fix3_v4.py | twochannel.two_channel_test | results/gen2_battery/fix3_v4_detail.csv |
| Table 3 (wrap) | code/wrap/run_wrap.py (+run_wrap2, rev DESeq2/ANC) | same + weighted_bh | results/wrap*, results/rev_compute |
| Table 4 / Fig. sources (cohorts, K=9999) | code/real/run_real_k10k.py | run_rich.two_channel_m | results/real10k/ |
| Full-spectrum | code/real/run_fullspec.py | run_rich.two_channel_m | results/rev_compute/fullspec_* |
| Shape A/B | code/real/run_shape_ab.py | (Wald + perm arms) | results/shape/ |
| ANCOM-BC2 battery/cohorts | code/rev/run_rev_ancombc.py + rev_ancombc.R | — | results/rev_compute/ |
| Attribution, e-BH, CIs, effect sizes (v4.10) | code/exhibit/*.py | run_rich.two_channel_m (+ mirror) | results/exhibit/ |
| Identifiability figure (v4.11) | make_fig_identifiability.py (in paper src zip) | model.log_g render | realdata/results fits (device) |

Note there are TWO implementations of the test statistic layer, both frozen:
`twochannel.two_channel_test` (gen-1/battery/wrap era; within-set ranks via
ordinal argsort) and `run_rich.two_channel_m` (cohort era; ranks via
scipy `rankdata(method="max")`, propensity passed as `m`). They were each
bridge-validated in their own domain; do not mix them across eras, and treat
any consolidation as a numbers-changing refactor requiring a new SPEC +
bridge (code-review register item C4, project archive).
