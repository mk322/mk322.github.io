Current article figures are original SVG diagrams with no measured performance data.

Regenerate the MCMC/amortization cost comparison:
    python3 assets/blog/learned-sampler/source/render_amortization.py

Regenerate offline and online process diagrams:
    python3 assets/blog/learned-sampler/source/render_clear_figures.py

Regenerate the shared-backbone LoRA diagram:
    python3 assets/blog/learned-sampler/source/render_lora.py

Displayed assets: amortized-comparison, offline-stages, lora-sampler, online-cycle, each with a
separate mobile composition. The two schedules remain distinct: only online has
student-to-sampler feedback after SFT. The comparison separates repeated per-response MCMC search from sampler training
and parameter reuse at data-generation time, without claiming a measured speedup.

Earlier categorical/smooth distribution illustrations and source files are kept
as unused exploratory assets. The current article no longer depends on A/B ratios.
Published baseline scores come from Finetuning with Sampling, arXiv:2610.02140v1,
Table 1 math panel. On 2026-10-04 the author confirmed all current offline and online
scores as real results. The table now displays them without draft badges or footnote
markers. This is author confirmation, not an independent audit of evaluation logs.

Offline Math avg. is 55.5% (+2.1pp versus MCMC + SFT); online is 59.4% (+6.0pp).
Prior avg. is 42.1% offline and 42.2% online, slightly above plain MCMC + SFT and
at or below the base model. Scores and averages are unchanged by confirmation.
results-data.json records the current confirmation separately from earlier drafts.
Integer reconstruction counts originated during drafting; they provide the existing
arithmetic precision but are not claimed to be observed correct-count logs.
Published baseline task scores and prior averages are preserved. Math averages use
the published scores, with decimal half-up display. Prose gains use displayed
averages. GPQA Diamond remains the reconstruction basis; the reference paper’s
variant and seed aggregation are unspecified.
MCMC + SFT + RL is excluded again at the author’s latest request. Its historical
published scores remain in the source record only; no displayed comparison uses them. The author confirmed online reward is the existing GFlowNet target density
(student probability × verification); it trains the sampler, while the student uses SFT.
No extra student RL stage is inferred. Both algorithms now specify group rollouts,
centered log-gap loss, frozen parameters, and separate student SFT updates.

Regenerate the aggregate comparison and optional per-task breakdown and their source-data record:
    python3 assets/blog/learned-sampler/source/render_tables.py

Verify mathematical identities and sampled gradient relations (requires numpy):
    python3 assets/blog/learned-sampler/source/verify_math.py

The amortization figure distinguishes response-state updates in MCMC from shared-parameter
updates in sampler training. The learned parameters pass into a separate data-generation
stage that appends tokens to construct responses; MCMC revises response states. Both
routes explicitly terminate in verified SFT data and student SFT. No MCMC trajectories
are used as sampler training targets. There are no performance numbers in the diagram.
Conceptual references: Finetuning with Sampling (https://aakaran.github.io/finetuning_with_sampling/)
and Bengio & Hu on amortized inference (https://yoshuabengio.org/en/blog/scaling-service-reasoning-model-based-ml). Also consulted Bengio’s GFN explanation (https://yoshuabengio.org/en/blog/generative-flow-networks).
Offline and online pseudocode are separate.


Opening revision (2026-10-04): SFT efficiency → off-policy mismatch → distribution
shift and forgetting → GRPO / OPD → transform the SFT distribution. New references:
- RL’s Razor, arXiv:2509.04259v1: §4 reports the empirical KL–forgetting association;
  §5 / Appendix A analyze projections under restricted policy-family assumptions.
  Our Eq. (2–4) corresponds to its rejection-sampling lemma; we do not claim its
  idealized convergence theorem holds for arbitrary neural-network / GRPO updates.
- Retaining by Doing, ICML 2026 (arXiv:2510.18874v3): §3 gives a Gaussian-mixture
  explanation, §4.2 tests approximately on-policy SFT, and Appendix A.5 shows
  that training-task KL does not universally predict forgetting.
- DeepSeekMath, arXiv:2402.03300: §4.1 describes group rollouts and relative-reward
  advantages. GRPO has no learned critic; the opening does not imply otherwise.
The added p_fit = q_data identity is an ideal population-fit explanation, not a
forgetting theorem. The KL inequality after Eq. (4) assumes the original data
satisfy the same validity constraint. Neither proves retention on unseen tasks.

Table layout (2026-10-04): one shared comparison for offline and online. The main
view shows Math avg. and Prior avg. Explicit comparisons are in the results prose. A disclosure preserves
all seven task scores, with New tasks and Prior tasks grouped and separated by a dashed rule. OPSD, GRPO, and UFT are retained and explained in Setup.
New sources: OPSD (arXiv:2601.18734) and UFT (arXiv:2505.16984, §3).

Regenerate the illustrative online curve (requires numpy and matplotlib):
    python3 assets/blog/learned-sampler/source/render_online_curve.py

online-training-curve.svg / -mobile.svg and the PNG share one synthetic trajectory.
The plot is labeled illustrative, with normalized progress rather than fabricated
step counts. online-curve-data.json records construction, seed, and points. Its
endpoint matches the author-confirmed online Math average. There are no measured
rollout logs, variance estimates, speedups, or convergence claims in this curve.


Reference-image replacements (2026-10-04):
Four figures now use built-in imagegen edits of the author's supplied reference PNGs:
- amortized-expert-to-onpolicy.png: repeated MCMC response search vs learned sampler reuse;
- offline-expert-to-onpolicy.png: freeze student, fit sampler, transform expert data, SFT;
- lora-sampler-expert-to-onpolicy.png: shared backbone with adapter enabled / disabled;
- online-expert-to-onpolicy.png: fixed expert input and updated-student feedback loop.
Each has a dedicated -mobile.png portrait composition selected below 760px.
Exact source filenames and all editing prompts are in reference-image-edit-prompts.json.
The older SVGs remain as source explorations; their renderers do not update these PNGs.
The figures emphasize off-policy expert traces → student-aligned training data.
Outputs approximate the verified, expert-constrained student target; "on-policy"
does not claim exact sampling from the unconstrained student. Offline alignment is
with the frozen reference student; online refits the adapter as the student evolves.
The comparison separates response updates in MCMC from parameter learning in the
sampler, and claims no measured speedup. Both data sources are expert-guided.
