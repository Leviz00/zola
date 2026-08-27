#!/usr/bin/env python3
# Batch-2 write-in (v4.9 -> v4.10): exhibits E1-E7 into the manuscript.
# Discipline: each old string matches EXACTLY ONCE per file; abort otherwise.
import sys

FILES = ["main_condensed.tex", "main_twocol.tex"]

EDITS = [
# W1 - Sec 5 shape A/B: unstratified de-confounding numbers (roadmap 1)
("W1-unstratified-numbers",
"""($0.048$--$0.053$ vs $0.056$--$0.071$ type-I) while unadjusted
comparison rejects up to $55\\%$. Seven external methods then ran on""",
"""($0.048$--$0.053$ vs $0.056$--$0.071$ type-I) while unadjusted
comparison rejects up to $55\\%$; and the de-confounding is
structural rather than stratification-borne---\\emph{unstratified}
permutation with occupancy residualization alone holds type-I at
$0.056$--$0.077$ (against $0.343$--$0.550$ without it, $40$ BH
false rejections per hundred taxa), stratification completing the
tail to $0.042$--$0.051$. Seven external methods then ran on"""),

# W2 - Table 1 caption: truth counts
("W2-tab1-truthcounts",
"""and 1006--1008, which are power-limited by design (undetectable
layer).}
\\label{tab:one}""",
"""and 1006--1008, which are power-limited by design (undetectable
layer). Each cell plants ten intensity-truth taxa per replicate
among $m=100$; cell 1009 adds thirty group-aligned structural
absences (the union truth's presence layer).}
\\label{tab:one}"""),

# W3 - Table 2 caption: truth counts
("W3-tab2-truthcounts",
"""$\\le0.034$ adversarial, TPR $\\le0.046$, type-I $\\le0.011$).}
\\label{tab:two}""",
"""$\\le0.034$ adversarial, TPR $\\le0.046$, type-I $\\le0.011$. Each
cell plants ten truths per replicate among $m=100$ taxa: ten
intensity effects, except REAL-PRES (ten presence effects, OR
$0.25$) and REAL-MIX (five plus five).}
\\label{tab:two}"""),

# W4 - Table 3 caption: paired bootstrap CIs (roadmap 6)
("W4-tab3-ci",
"""reproduce exactly).}\\label{tab:three}""",
"""reproduce exactly). Pooled paired $95\\%$ bootstrap CIs for
$\\Delta$TPR: two-channel $[.012,.038]$, ZINQ $[.008,.033]$, LDM
$[.010,.047]$, LOCOM $[-.003,.007]$, Wilcoxon $[.005,.027]$, LinDA
$[.000,.013]$, DESeq2 $[.017,.060]$, ANCOM-BC2
$[-.012,.007]$.}\\label{tab:three}"""),

# W5 - Table 4 caption: spike Clopper-Pearson CIs (roadmap 6)
("W5-tab4-spikeci",
"""exactly at the BH boundary and are not counted as
discoveries.}\\label{tab:four}""",
"""exactly at the BH boundary and are not counted as
discoveries. Clopper--Pearson $95\\%$ intervals for the spike
recoveries: presence $.73\\,[.45,.92]$ / $.93\\,[.68,1.00]$ /
$.87\\,[.60,.98]$; intensity $0\\,[0,.22]$ /
$.53\\,[.27,.79]$ (both).}\\label{tab:four}"""),

# W6 - Sec 5: attribution accuracy exhibit (roadmap 4)
("W6-attribution",
"""$9$--$14/15$---the diagnostic's sensitivity cost, disclosed. At the coarse resolution, a naive upstream-weighted variant""",
"""$9$--$14/15$---the diagnostic's sensitivity cost, disclosed.
The same runs measure the attribution labels themselves: all $38$
recovered presence spike-ins (across cohorts) are attributed to the
detection channel, and $14$ of $16$ recovered intensity spike-ins
to the intensity channel---the two cross-attributions land on taxa
carrying \\emph{native} detection-channel signal (one an official
AGP discovery, one an MBQC boundary candidate), collisions of
random spike placement with real signal rather than channel
confusion. At the coarse resolution, a naive upstream-weighted variant""",),

# W7 - Sec 3.2: BH dependence-condition sentence (roadmap 7, text half)
("W7-dependence-sentence",
"""FDR control under arbitrary cross-taxon dependence
\\cite{wang2022ebh} as a secondary regime, usable at $K\\gtrsim100$.""",
"""FDR control under arbitrary cross-taxon dependence
\\cite{wang2022ebh} as a secondary regime, usable at $K\\gtrsim100$.
BH itself is applied to the permutation $p$-values under the usual
positive-dependence conditions \\cite{benjamini1995}, which
cross-taxon coupling (Remark~\\ref{rem:leak}) is not guaranteed to
satisfy; Section~\\ref{sec:real} therefore carries the e-BH column
on the cohort lists."""),

# W8 - Sec 5: e-BH two arms + Granulicatella veto (roadmap 7 + new finding)
("W8-ebh-granulicatella",
"""handling design---not to pipeline effects at large.""",
"""handling design---not to pipeline effects at large.
An e-BH column prices the dependence conditions beneath these BH
lists. Under the identity transform on the $\\chi^2$-sum statistic,
permutation e-values are capped by null mass (largest observed
$282$, against e-BH thresholds $m/(\\alpha k)$) and reject nothing;
under a fixed tail-indicator transform matched to the permutation
resolution ($f=\\mathbf{1}\\{C\\ge-2\\ln10^{-4}\\}$, an archived
amendment), the dependence-agnostic regime retains exactly
\\emph{Akkermansia} on AGP ($e=10^{4}$, zero permutation
exceedances) and eighteen of MBQC's thirty, dropping boundary
members---arbitrary-dependence robustness keeps the floor-level
core, and the transform, fixed in advance, is load-bearing. The
same runs surface one discovery the Cauchy primary vetoes:
\\emph{Granulicatella}, detected in $198$ versus $67$ of $350$
samples (detection $z^{2}=120$, rank one among $10^{4}$
permutations), draws combined $p=0.9999$ because its anti-extreme
intensity component ($P_{\\mathrm{int}}\\to1$, Cauchy weight at the
clipped endpoint) cancels the detection component in the
average; the $\\chi^{2}$-sum face recovers it ($e=5000$)---a
measured instance of the Cauchy combination's known sensitivity to
anti-extreme components, and the standing reason the sensitivity
column accompanies every list."""),

# W9 - Sec 5 AGP: effect-size table pointer (roadmap 6)
("W9-agpeff-pointer",
"""intensity entries barely move and drop only via the BH threshold.""",
"""intensity entries barely move and drop only via the BH threshold.
Table~\\ref{tab:agpeff} tabulates the fourteen: the twelve
detection-channel members deplete coherently (detection OR
$0.37$--$0.61$, case rates uniformly below control) with amplitude
ratios near one, while the two intensity members are amplitude
effects---\\emph{Streptococcus} at $8.0\\times$ with essentially no
detection contrast---so the channel split is visible in the raw
rates themselves."""),
]

TABLE_BODY = """\\centering\\small
\\caption{Effect sizes for the fourteen official AGP discoveries
(official analysis set; archived per-taxon CSV in the exhibit
record). Detection rate = share of samples with the genus detected;
OR = empirical case-vs-control detection odds ratio; amplitude
ratio = case/control mean of anchor-normalized counts over detected
cells. unclassified\\_unknown's detection OR reflects near-saturated
detection ($0.97$ vs $0.99$), where odds are boundary-driven; its
signal is the intensity channel's.}\\label{tab:agpeff}
\\begin{tabular}{lccccc}
\\toprule
taxon & ch. & det.\\ case & det.\\ ctrl & det.\\ OR & amp.\\ ratio \\\\
\\midrule
unclassified\\_[Cerasicoccaceae] & det & 0.034 & 0.089 & 0.37 & 0.36 \\\\
\\emph{Campylobacter} & det & 0.074 & 0.177 & 0.37 & 1.13 \\\\
\\emph{Akkermansia} & det & 0.506 & 0.723 & 0.39 & 0.59 \\\\
1-68 & det & 0.149 & 0.266 & 0.48 & 0.98 \\\\
unclassified\\_[Mogibacteriaceae] & det & 0.683 & 0.806 & 0.52 & 1.05 \\\\
rc4-4 & det & 0.177 & 0.283 & 0.55 & 1.26 \\\\
\\emph{Lachnobacterium} & det & 0.211 & 0.329 & 0.55 & 1.34 \\\\
WAL\\_1855D & det & 0.269 & 0.394 & 0.56 & 1.03 \\\\
unclassified\\_Victivallaceae & det & 0.137 & 0.220 & 0.56 & 0.76 \\\\
\\emph{Butyricimonas} & det & 0.397 & 0.531 & 0.58 & 1.08 \\\\
\\emph{Prevotella} & det & 0.646 & 0.751 & 0.60 & 0.56 \\\\
unclassified\\_Oxalobacteraceae & det & 0.257 & 0.363 & 0.61 & 1.07 \\\\
\\emph{Streptococcus} & int & 0.800 & 0.817 & 0.90 & 8.05 \\\\
unclassified\\_unknown & int & 0.971 & 0.989 & 0.39 & 0.74 \\\\
\\bottomrule
\\end{tabular}"""

ANCHOR = """13/15 & 8/15 \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""

PER_FILE = {
"main_condensed.tex": [
  ("W10-agpeff-table", ANCHOR,
   ANCHOR + "\n\n\\begin{table}[t]\n" + TABLE_BODY + "\n\\end{table}"),
  ("W11-header",
   "% v4.8.2 + text batch 1 of the panel roadmap: increment-(i) re-scope,",
   "% v4.8.2 + batch 1 (text) + batch 2 exhibits (v4.10): attribution\n"
   "% accuracy, e-BH two-arm column + Granulicatella veto, CIs, AGP\n"
   "% effect-size table, truth counts, unstratified A/B numbers\n"
   "% (SPEC_EXHIBIT + EXHIBIT_MEMO). Batch 1: increment-(i) re-scope,"),
],
"main_twocol.tex": [
  ("W10-agpeff-table", ANCHOR,
   ANCHOR + "\n\n\\begin{table*}[t]\n" + TABLE_BODY + "\n\\end{table*}"),
  ("W11-header",
   "% main_condensed.tex v4.9. Build: pdflatex main_twocol (twice).",
   "% main_condensed.tex v4.10. Build: pdflatex main_twocol (twice)."),
],
}

failed = False
for fname in FILES:
    with open(fname) as f:
        text = f.read()
    fail_here = False
    for eid, old, new in EDITS + PER_FILE[fname]:
        n = text.count(old)
        if n != 1:
            print(f"FAIL {fname} {eid}: {n} matches")
            fail_here = failed = True
            continue
        text = text.replace(old, new)
    if not fail_here:
        with open(fname, "w") as f:
            f.write(text)
        print(f"OK {fname}: {len(EDITS)+2} edits applied")

sys.exit(1 if failed else 0)
