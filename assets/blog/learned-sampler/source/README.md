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
Table 1 math panel. Our displayed numbers are draft estimates, not measurements:
+2 percentage points offline, +4 online (the midpoint of the requested +3–5).
Prior-task values copy the base checkpoint as explicit retention targets.
The MCMC + SFT + RL baseline is included as the stronger online comparator.

Regenerate all four result tables and their source-data record:
    python3 assets/blog/learned-sampler/source/render_tables.py

Verify mathematical identities and sampled gradient relations (requires numpy):
    python3 assets/blog/learned-sampler/source/verify_math.py

The amortization figure distinguishes response-state updates in MCMC from shared-parameter
updates in sampler training. The learned parameters pass into a separate data-generation
stage; no MCMC trajectories are used as training targets. There are no performance numbers.
Conceptual references: Finetuning with Sampling (https://aakaran.github.io/finetuning_with_sampling/)
and Bengio & Hu on amortized inference (https://yoshuabengio.org/en/blog/scaling-service-reasoning-model-based-ml). Offline and online pseudocode are separate.
