# E2S editorial preview — 2026-10-05

Independent page: /blog/e2s-finetuning-preview/.
The existing /blog/learned-on-policy-sampler/ and its assets remain unchanged.

Applies e2s_blog_detailed_editorial_feedback.md: eight-section narrative,
three-bullet TL;DR, constrained-KL derivation, constant-log-gap explanation,
collapsed implementation, result-first experiments, compact aggregate table,
and restrained research-note styling. References are renumbered by first use.

Diagrams: run python assets/blog/e2s-preview/source/render_figures.py.
SVGs have separate desktop and mobile compositions. Colors encode expert,
student, sampler, and generated data roles consistently.

Results and trajectory reuse the existing learned-sampler/source records.
The user confirmed the scaling results on 2026-10-05 (“你的scaling的结果没问题 我check过了”).
The earlier scaling record was originally created as an illustration; its history
has not been rewritten. Following the detailed feedback, the new page omits
that plot and discusses repeated sampling briefly in Looking ahead.

Validation: Jekyll build, numerical identity checks in the existing verify_math.py,
MathJax rendering, anchor/reference resolution, and 320/390/768/1440 px layouts.

Update: the author subsequently requested restoring Diversity scaling. The preview now includes a separate scaling section and source/scaling-data.json records both the original estimate history and the later author confirmation.
