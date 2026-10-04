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
Table 1 math panel. Sampler values were revised again at the author's request on
2026-10-04. The new rows are estimates, not the previously author-confirmed scores.
Both rows target an equal-weight Math average gain over MCMC + SFT: offline
1.5–2.5pp, online 3–5pp. Prior-task average estimates exceed the MCMC pipelines.
Integer planning counts define sample-compatible decimals; they are not observed
outcomes. results-data.json records the earlier confirmation separately from the
new estimates. Published per-task baseline values and prior averages are preserved;
Math averages are derived from their published scores, with decimal half-up display.
All sampler values and arithmetic are computed before display rounding. Δ uses the
rounded average scores. GPQA Diamond remains an explicit basis; the baseline's
variant and seed aggregation are not inferred from its scores.
MCMC + SFT + RL is excluded from the displayed comparison at the author’s request.
Its historical published row remains in the source record only.

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
view shows Math avg., Δ Math versus MCMC + SFT, and Prior avg. A disclosure preserves
all seven task scores. OPSD, GRPO, and UFT are retained and explained in Setup.
New sources: OPSD (arXiv:2601.18734) and UFT (arXiv:2505.16984, §3).

Regenerate the illustrative online curve (requires numpy and matplotlib):
    python3 assets/blog/learned-sampler/source/render_online_curve.py

online-training-curve.svg / -mobile.svg and the PNG share one synthetic trajectory.
The plot is labeled illustrative, with normalized progress rather than fabricated
step counts. online-curve-data.json records construction, seed, and points. Its
endpoint follows the current online Math-average estimate. There are no measured
rollout logs, variance estimates, speedups, or convergence claims in this curve.
