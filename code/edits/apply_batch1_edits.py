#!/usr/bin/env python3
# Batch 1 (v4.9) text edits — panel roadmap 1(text)/2/5 + ethics + citations.
# Discipline: each old string must match EXACTLY ONCE in EACH file; abort otherwise.
import sys

FILES = ["main_condensed.tex", "main_twocol.tex"]

# (id, old, new) — applied to BOTH files.
EDITS = [
# R1 — abstract: tier qualifier + predecessor one-liner (roadmap 2a, 2d)
("R1-abstract-tier-predecessor",
"""is anticonservative throughout), and is paired-superior to the
masked workflow it replaces, recovering group-aligned absences that
workflow renounced;""",
"""is anticonservative throughout), and is paired-superior---on its
archived asymptotic tier---to the masked predecessor it replaces (a
workflow that masked zeros judged structural before an
intensity-only test), recovering group-aligned absences that
workflow renounced;"""),

# R2 — abstract: "robust to" -> "surviving" (roadmap 2b / DA-8)
("R2-abstract-surviving",
"""discoveries (an \\emph{Akkermansia} core robust to richness
conditioning on AGP;""",
"""discoveries (an \\emph{Akkermansia} core surviving richness
conditioning on AGP;"""),

# R3 — abstract: in-silico modifier (roadmap 2c)
("R3-abstract-insilico",
"""and calibrates both channels by spike-in.
\\end{abstract}""",
"""and calibrates both channels by in-silico spike-in.
\\end{abstract}"""),

# R4 — Sec 1 (iii): weight-design-open clause (roadmap 2e / DA-9)
("R4-contrib-weight-open",
"""per-taxon test, with abstention recovered as $0/1$ weighting
(Theorem~\\ref{thm:whbh}); and (iv)""",
"""per-taxon test, with abstention recovered as $0/1$ weighting and
real-data weight design reported as an open problem
(Theorem~\\ref{thm:whbh}); and (iv)"""),

# R5 — Sec 1 (iv): in-silico at first use (roadmap 2c)
("R5-contrib-insilico",
"""cohorts with spike-in calibration of \\emph{both} channels (to our""",
"""cohorts with in-silico spike-in calibration of \\emph{both}
channels (to our"""),

# R6 — Sec 1 increment (i): re-scope to Sec 3.1 verdict (roadmap 1, DA-1)
("R6-increment-rescope",
"""The increments here are three: the
occupancy-shaped detection adjustment, supplied by the model as a
label-blind offset and stress-tested against alternatives on
six-decade real depths; finite-sample calibration holding jointly""",
"""The increments here are three: the
occupancy-shaped detection adjustment, supplied by the model as a
label-blind offset---priced honestly on six-decade real depths,
where free-slope $\\log N$ adjustment calibrates equally well, so
the distinctive value is structural: occupancy residualization
de-confounds even unstratified permutation
(Section~\\ref{sec:test}); finite-sample calibration holding jointly"""),

# R7 — Thm whbh: add BH1995 to the FDR-control citation
("R7-whbh-bh1995",
"""controls FDR under the standard conditions \\cite{genovese2006},""",
"""controls FDR under the standard conditions
\\cite{benjamini1995,genovese2006},"""),

# R8 — after Thm perm: add-one convention citation (Phipson-Smyth)
("R8-phipson-sentence",
"""designed guarantee with named conditions. Score-form statistics""",
"""designed guarantee with named conditions. Monte Carlo $p$-values
throughout use the add-one convention, exact for randomly drawn
permutations \\cite{phipson2010}. Score-form statistics"""),

# R9 — Sec 4: reconcile six-external vs seven-procedures counting
("R9-counting",
"""every one of the twenty-one method-by-cell combinations of the seven
error-controlled methods, with a single""",
"""every one of the twenty-one method-by-cell combinations of the
seven error-controlled procedures (six external, plus the
two-channel reference), with a single"""),

# R10 — Sec 5 MBQC: shared-specimen + redraw robustness + scope (roadmap 5a, 5b)
("R10-mbqc-disclosures",
"""follows: harmonizing what is detected matters more for
cross-laboratory comparability than harmonizing normalization.""",
"""follows: harmonizing what is detected matters more for
cross-laboratory comparability than harmonizing normalization.
Two qualifications delimit this reading. The two arms process
largely the same physical specimens ($1{,}578$ of $1{,}606$
specimens appear in both arms; $89$ cross-arm pairs in the official
subsample), so the contrast is paired-specimen rather than
independent-sample; a disjoint-specimen redraw at the official
resolution returns $38$ discoveries ($29$ detection, $9$
intensity), with $23$ of the official thirty persisting---the
shared-specimen sampling is, if anything, conservative, and the
channel verdict is unchanged. And the count comparison speaks to
this technical contrast---two bioinformatics pipelines under MBQC's
handling design---not to pipeline effects at large."""),

# R11 — Sec 2: rho-constancy interpretive license (roadmap 5c)
("R11-rho-license",
"""($\\rho\\equiv1$ in all fitted results; its blank-anchored estimation
is future work, and its confounding with abundance absent anchors is
part of the theory below).""",
"""($\\rho\\equiv1$ in all fitted results; its blank-anchored estimation
is future work, and its confounding with abundance absent anchors is
part of the theory below). That constancy is also the interpretive
license of Section~\\ref{sec:real}: with detection efficiency shared
across groups, a detection-channel contrast reads as ecological,
while where the groups \\emph{are} pipelines---MBQC's engineered
contrast---the same channel correctly absorbs the technical effect."""),

# R12 — Sec 5: ethics line (roadmap 6, text part)
("R12-ethics",
"""subsample---$700$ distinct participants, no repeats---in a single
unadjusted contrast.""",
"""subsample---$700$ distinct participants, no repeats---in a single
unadjusted contrast. All three are public, de-identified cohorts
analyzed under their released terms; no new human-subjects data
were collected."""),

# R13 — bibliography: Benjamini-Hochberg 1995 (alphabetical: after aitchison)
("R13-bib-bh1995",
"""\\bibitem{davis2018}""",
"""\\bibitem{benjamini1995}
Benjamini Y, Hochberg Y.
\\newblock Controlling the false discovery rate: a practical and powerful approach to multiple testing.
\\newblock \\emph{Journal of the Royal Statistical Society, Series B}, 1995, 57(1): 289--300.
\\newblock doi:10.1111/j.2517-6161.1995.tb02031.x.

\\bibitem{davis2018}"""),

# R14 — bibliography: Phipson-Smyth 2010 (alphabetical: before sarkar)
("R14-bib-phipson",
"""\\bibitem{sarkar2021}""",
"""\\bibitem{phipson2010}
Phipson B, Smyth G K.
\\newblock Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn.
\\newblock \\emph{Statistical Applications in Genetics and Molecular Biology}, 2010, 9(1): Article 39.
\\newblock doi:10.2202/1544-6115.1585.

\\bibitem{sarkar2021}"""),
]

# Per-file header version-note edits.
HEADER_EDITS = {
"main_condensed.tex": (
"% main_condensed.tex -- ZOLA, condensed journal manuscript (v4.8:",
"% main_condensed.tex -- ZOLA, condensed journal manuscript (v4.9 =\n"
"% v4.8.2 + text batch 1 of the panel roadmap: increment-(i) re-scope,\n"
"% abstract wording set, MBQC shared-specimen/scope + rho-constancy\n"
"% disclosures, ethics line, BH1995 + Phipson-Smyth citations; record\n"
"% in V49_TEXT_BATCH_MEMO.md. Prior (v4.8:"),
"main_twocol.tex": (
"% main_condensed.tex v4.8. Build: pdflatex main_twocol (twice).",
"% main_condensed.tex v4.9. Build: pdflatex main_twocol (twice)."),
}

failed = False
for fname in FILES:
    with open(fname) as f:
        text = f.read()
    for eid, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"FAIL {fname} {eid}: {n} matches")
            failed = True
            continue
        text = text.replace(old, new)
    hold, hnew = HEADER_EDITS[fname]
    n = text.count(hold)
    if n != 1:
        print(f"FAIL {fname} header: {n} matches")
        failed = True
    else:
        text = text.replace(hold, hnew)
    if not failed:
        with open(fname, "w") as f:
            f.write(text)
        print(f"OK {fname}: {len(EDITS)} edits + header applied")

sys.exit(1 if failed else 0)
