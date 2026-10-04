# Uni-LaDiR blog figures

All sources now follow [arXiv:2609.19878v3](https://arxiv.org/abs/2609.19878v3), September 29, 2026.

## Original vector figures

The PDFs were copied from the public v3 arXiv source. `results.json` records original filenames and SHA-256 checksums.

- `fig1.pdf`: paper Figure 1, overview.
- `fig2.pdf`: paper Figure 2, `method-overview-v45-reference.pdf`. Replaces the older blog diagram.
- `paper-sharing.pdf`: paper Figure 3, teacher-sharing study.
- `joint-training.pdf`: paper Figure 4, staged versus joint training.
- `objective.pdf`: paper Figure 5, encoding and latent-target objectives.
- `diffusion.pdf`: paper Figure 6, thought-token generation objectives.

`render.py` exports these PDFs to SVG and PNG. Raster previews are produced before SVG export because the installed PyMuPDF version can change subsequent rendering state on the page. No measured marks or values are changed in the original figure exports. Paper sharing and objective plots retain the original truncated axes.

## Replots and tables

- `main.svg` and its mobile version use displayed scores from Tables 1–2. The VLM main comparisons use the same backbone and evaluation examples, with method-specific training data and recipes.
- `sharing-mobile.svg` replots v3 Figure 3 values as horizontal bars with axes starting at zero. The desktop post uses the paper's original `paper-sharing.svg`.
- Blog interventions reproduce Table 7. The three ablation summary gains come directly from Section 4.4 and Figures 4–6, rather than being reconstructed from rounded scores.

Reported relative gains use the paper's unrounded aggregation; plotted scores follow displayed rounding. Sharing studies keep backbone, data, teachers, and thought-token dimensions fixed, but do not match encoder parameter counts.

Run with Python, PyMuPDF, and Matplotlib installed:

```sh
python3 assets/blog/uni-ladir/source/render.py
```

The method walkthrough highlights the panels of the current diagram. It explains training and inference paths; it is not a simulation of a learned diffusion trajectory. Teacher traces are training-only. The scene in Figure 1 is illustrative.
