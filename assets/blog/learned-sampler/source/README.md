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
Table 1 math panel. On 2026-10-04 the author confirmed all displayed result values;
the article now presents these as results, with the baseline rows still attributed
to the paper. Displayed scores are unchanged. No raw run logs, seed-level results,
runtime measurements, or causal ablation findings were supplied to the editor.
results-data.json records this confirmation and the previous draft history.
Its display_derivation_counts retain the old draft's decimal arithmetic and must
not be mistaken for independently verified experimental counts. Published baseline
rounding is untouched. The prior-task average uses unrounded task percentages;
displayed deltas use displayed rounded scores. The baseline GPQA variant remains
unspecified, and its match to the Diamond basis is not inferred from the scores.
The MCMC + SFT + RL baseline is included as the stronger online comparator.

Regenerate the two combined result tables and their source-data record:
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

Table layout (2026-10-04): one comparison per schedule, with four math columns
and Chemistry / MMLU / GPQA / their unweighted average together. Red marks
prior-task scores below the displayed base; method names remain sticky on mobile.
Experimental prose follows Setup → Results → Conclusion; reference settings and
proposed causal ablations are available in optional disclosures.
