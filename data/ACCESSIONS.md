# Data access — derivation scripts + accessions only

Per the archive's data policy, **no raw cohort data or metadata tables are
redistributed here**. Every analysis input is derived from public releases:

| Cohort | Source (public) | Identifiers | What the paper used |
|---|---|---|---|
| IBDMDB | IBDMDB / HMP2 portal, https://ibdmdb.org (Lloyd-Price et al. 2019, Nature 569:655) | portal 16S products + sample metadata | 178 16S samples / 81 subjects; genus table -> `ibdmdb_genus.npz` |
| MBQC | MBQC baseline, http://www.mbqc.org/baseline-data (Sinha et al. 2017, Nat Biotechnol 35:1077); sequence deposition BioProject PRJNA260846 (acquisition record) | mothur OTU table + handling metadata + nbt.3981 supplements | genus aggregation -> `mbqc_genus.npz`; two-pipeline contrast, 350/arm seeded subsample |
| AGP | American Gut / Qiita study 10317; ENA PRJEB11419 (McDonald et al. 2018, mSystems 3:e00031-18) | fecal 16S table + metadata | 9,511-sample fecal table -> `agp_genus.npz`; IBD cases/controls, 350/arm seeded subsample |

**Derivation**: `code/platform/realdata/aggregate.py` builds the
`{cohort}_genus.npz` matrices (fields: `Y` counts, `depths`, `samples`,
`taxa`); `code/testing/run_rich.py::load_cohorts` documents the expected
file layout and applies the frozen cohort subsamples (seed 20260304) and
exchangeability units. Metadata columns consumed: IBDMDB `diagnosis`,
`subject_id`; AGP `ibd` self-report field. Fetch raw inputs from the portals
above and re-run the derivation; every downstream number is then covered by
`tests/test_regression_bridges.py`.
