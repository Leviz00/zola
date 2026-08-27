"""apply_v48_edits.py -- v4.8: compute-batch results into both manuscripts.
DESeq2 completion numbers are appended by a second pair-set (PAIRS_DESEQ2)
once wrap_deseq2b lands -- run with --with-deseq2 then."""
import sys

PAIRS = [

("V1-abstract-seventh",
"""generations the test holds level, leads six wrapped external
methods where cells separate them, and is paired-superior to the""",
"""generations the test holds level, leads six error-controlled
external methods where cells separate them (a seventh, ANCOM-BC2,
is anticonservative throughout), and is paired-superior to the"""),

("V2-intro-seven",
"""executed consumer program: six popular methods wrapped, implanted
with the framework's components, and run head-to-head on the same
simulations and cohorts.""",
"""executed consumer program: seven popular methods wrapped, implanted
with the framework's components, and run head-to-head on the same
simulations and cohorts.""",),

("V3-gen1-dualtruth",
"""masked workflow's declared abstention held FDR at $0.014$ at the
price of union-truth power $0.196$.

On this unchanged grid""",
"""masked workflow's declared abstention held FDR at $0.014$ at the
price of union-truth power $0.196$. The archived dual-truth
rescoring of the same baseline runs completes the separation: under
the union truth the pooled detectable-layer FDR falls to $0.053$
(LinDA), $0.279$ (DESeq2), and $0.081$ (Wilcoxon), while
ANCOM-BC2's $0.201$ is invariant to the truth choice---its
structural-zero screen already absorbs the absences---so the truth
switch dissolves the estimand-mismatch component and leaves genuine
miscalibration exposed (S3).

On this unchanged grid"""),

("V4-ancombc-battery",
"""a property of its regime: the rank reversal of
Section~\\ref{sec:intro}, reproduced under control. Head-to-head, the two-channel""",
"""a property of its regime: the rank reversal of
Section~\\ref{sec:intro}, reproduced under control. ANCOM-BC2,
executed on the battery in revision (release 2.6.1,
structural-zero screen on; environment and configuration in S3), is
the counterexample this two-generation design predicts: the one
generation-1 baseline whose inflation was truth-invariant is also
the one external that inflates on realistic data---shipped-call FDP
$0.26$--$0.83$ across all eight cells (type-I $0.19$--$0.55$;
$0.29$ on pure-null replicates of the intensity cell, so the
miscalibration is the method's, not the truth set's), its
prevalence screen discarding about $46$ of $100$ taxa, and its
structural-zero detector---built for all-or-none absence---catching
none of the battery's graded presence effects ($0$ of $10$ per
replicate; presence-cell FDP $0.833$). Upstream weights neither
help nor hurt it (pooled $\\Delta$TPR $-0.002$, $2$$+$/$2$$-$) and
cannot rescue it: the weighted arm inherits FDP $0.640$,
Theorem~\\ref{thm:whbh}'s license failing wholesale where the foil
failed at the margin (implementation cross-checked against the
archived grid numbers: $0.060/0.840$ reproduced exactly on cell
1005, S3). Head-to-head, the two-channel"""),

("V5-tab3-row",
"""DESeq2 & raw & --- & 0.11$\\to$0.12 & --- & $+0.010$ (1$+$/0$-$) &
0.020 \\\\
\\bottomrule""",
"""DESeq2 & raw & --- & 0.11$\\to$0.12 & --- & $+0.010$ (1$+$/0$-$) &
0.020 \\\\
ANCOM-BC2 & raw & 0.22$\\to$0.22 & 0.60$\\to$0.60 & 0.90$\\to$0.89 &
$-0.002$ (2$+$/2$-$) & 0.640 \\\\
\\bottomrule"""),

("V5b-tab3-caption",
"""Raw-$p$ weighted arms
(ZINQ, DESeq2) sit outside Theorem~\\ref{thm:whbh}'s license and are
empirically controlled only.""",
"""Raw-$p$ weighted arms
(ZINQ, DESeq2, ANCOM-BC2) sit outside Theorem~\\ref{thm:whbh}'s
license; ZINQ's and DESeq2's are empirically controlled, while
ANCOM-BC2's (added in revision) is anticonservative throughout---its
row completes the family, not a controlled comparison (Section
text).""",),

("V6-externals-cluster",
"""Six external methods then ran on
the identical matrices and spike-ins (shipped defaults, sample-level;
LinDA's mixed-effects formula and LDM's matched-set mode can encode
clusters and are not yet part of the executed comparison): on
longitudinal IBDMDB, where
cluster-respecting analyses return $0$--$3$ rejections,
sample-level LinDA/ZINQ/DESeq2 return $25/26/45$---a gap that prices
ignoring the exchangeability unit, though the cluster-capable modes
above must run before it reads as a method comparison; on the design-clean cohorts the official lists
are broadly corroborated (Jaccard to $0.71$) and the
\\emph{Akkermansia}--\\emph{Campylobacter} core is independently hit
by all three usable externals; and the spike yardstick cuts both
ways---ZINQ matches our presence-arm recovery and exceeds our
intensity arm ($11/15$, $10/15$ vs $8/15$), recorded as a direction
for the intensity channel, so the framework's real-data
differentiators are design-valid inference, attribution, and
diagnostics rather than raw power.""",
"""Seven external methods then ran on
the identical matrices and spike-ins (shipped defaults; the
cluster-capable modes executed in revision): on longitudinal
IBDMDB, where cluster-respecting analyses return $0$--$3$
rejections, sample-level LinDA/ZINQ/DESeq2 return
$25/26/45$---and the pseudo-replication is now demonstrated
\\emph{within} the methods themselves: LinDA under its own
mixed-effects formula ($\\sim$grp$+(1|\\mathrm{subject})$) collapses
from $25$ to $4$ (all four literature-coherent
\\emph{Lachnospiraceae}-family depletions), ANCOM-BC2 in its mixed
mode returns $0$, and LDM's matched-set permutation refuses the
design outright, its machinery requiring equal-size clusters where
IBDMDB subjects contribute one to five samples. The exchangeability
unit is not a technicality but the difference between a method's
own two modes. On the design-clean cohorts the official lists
are broadly corroborated (Jaccard to $0.71$) and the
\\emph{Akkermansia}--\\emph{Campylobacter} core is independently hit
by all four usable externals (ANCOM-BC2's revision run included:
$15$ AGP rejections, four shared, both core genera among them---to
be read against its battery calibration); and the spike yardstick
cuts both
ways---ZINQ matches our presence-arm recovery and exceeds our
intensity arm ($11/15$, $10/15$ vs $8/15$; ANCOM-BC2 likewise
$11/15$ on AGP intensity), recorded as a direction
for the intensity channel, so the framework's real-data
differentiators are design-valid inference, attribution, and
diagnostics rather than raw power.""",),

("V7-fullspec",
"""diagnostics rather than raw power. These analyses hold to the same
discipline as the
simulations:""",
"""diagnostics rather than raw power. Finally, a full-spectrum
analysis removes the top-$100$ filter and lets the framework's own
gates do its work: every nonzero genus enters ($282/218/200$ per
cohort), the channels' activity gates abstain on $47/6/1$ of them,
and the official $K=9999$ resolution is what makes the enlarged
family readable at all (at $m=200$ the rank-one BH threshold falls
below the $K=999$ permutation floor; the coarse-resolution rerun
indeed returns zero on AGP). The AGP core survives in every
arm---the richness-adjusted column still returns exactly
\\emph{Akkermansia} and \\emph{Campylobacter}---and one discovery
the filter had hidden appears: \\emph{Mobiluncus} (prevalence
$0.036$, detection channel, $p=6\\times10^{-4}$), reported at
candidate level. MBQC grows from thirty to $58$ discoveries; the
twenty-nine additions sit at prevalences $0.014$--$0.12$, all but
one detection-channel, extending the pipeline-effect reading into
exactly the rare tail the filter excluded. The honest costs are on
the same record: the doubled BH family drops five AGP boundary
members (the testability weight recovers three while zero-weighting
one near-saturated intensity discovery---the naive proxy's known
blind spot), MBQC loses \\emph{Bifidobacterium} at the boundary,
and IBDMDB stays empty in every arm. These analyses hold to the same
discipline as the
simulations:""",),

("V8-disc-gaps",
"""The comparison set now spans both generations, the two-part family,
and the real cohorts with spike-in scoring, but remains
defaults-only and covariate-unadjusted, and four named gaps stand:
ANCOM-BC2 on the battery, the cluster-capable modes of LinDA
(mixed effects) and LDM (matched sets) on IBDMDB, completion of
DESeq2's cost-capped cells, and both-truth rescoring of the
generation-one baselines---so no superiority beyond
the executed comparisons is claimed, and ZINQ's stronger real-data
intensity recovery is on the record. The real-data deployment also
filters to the top-$100$ prevalence genera, so it does not yet
exercise the framework where its rationale lives---the rare-taxon
ridge; replacing the filter with testability weights and declared
abstention is the designed next analysis.""",
"""The comparison set now spans both generations, the two-part family,
ANCOM-BC2 on battery and cohorts, the cluster-capable modes on
IBDMDB, completed DESeq2 cells, and the archived dual-truth
rescoring of the generation-one baselines, but remains
defaults-only and covariate-unadjusted---so no superiority beyond
the executed comparisons is claimed, and ZINQ's stronger real-data
intensity recovery is on the record. The full-spectrum analysis
replaces the top-$100$ filter with the framework's own gates and
weights; what still bounds it sits upstream of the framework---the
construction of the genus tables themselves---and the naive
information weight's intensity-channel blind spot keeps real-data
weight design the concrete open problem.""",),
]

# Filled after wrap_deseq2b lands (run with --with-deseq2):
PAIRS_DESEQ2 = [
("D1-tab3-deseq2-row",
"""DESeq2 & raw & --- & 0.11$\\to$0.12 & --- & $+0.010$ (1$+$/0$-$) &
0.020 \\\\""",
"""DESeq2 & raw & 0.06$\\to$0.12 & 0.14$\\to$0.19 & 0.03$\\to$0.04 &
$+0.037$ (12$+$/0$-$) & 0.003 \\\\"""),
("D2-tab3-caption-caps",
"""arm's average. LDM $R{=}10$; DESeq2 REAL-MIX only, $R{=}10$
(pre-declared cost caps).}""",
"""arm's average. LDM $R{=}10$; DESeq2 completed at $R{=}20$ on all
three cells in revision (the earlier pre-declared cap priced
pydeseq2's iterative size-factor mode at realistic zero fractions,
about six minutes per dataset; the capped run's ten replicates
reproduce exactly).}"""),
]

FILES = ["/home/claude/paper_v2/main_condensed.tex",
         "/home/claude/paper_v2/main_twocol.tex"]

pairs = list(PAIRS)
if "--with-deseq2" in sys.argv:
    pairs += PAIRS_DESEQ2

ok = True
for path in FILES:
    with open(path) as f:
        text = f.read()
    applied, missed, multi = [], [], []
    for name, old, new in pairs:
        c = text.count(old)
        if c == 1:
            text = text.replace(old, new)
            applied.append(name)
        elif c == 0:
            missed.append(name)
        else:
            multi.append((name, c))
    with open(path, "w") as f:
        f.write(text)
    print(f"== {path.split('/')[-1]}: {len(applied)} applied")
    if missed:
        ok = False
        print("   MISSED:", missed)
    if multi:
        ok = False
        print("   MULTI:", multi)
print("ALL OK" if ok else "FIX NEEDED")
