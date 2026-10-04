# Uni-LaDiR blog figures

- `fig1.pdf`: original manuscript Figure 1, `fig1-0912-motion-detail-v76.pdf`.
- `fig2.pdf`: original manuscript Figure 2, `method-sequence-v23.pdf`.
- `render.py`: renders the original vectors to SVG/PNG and replots measured results using Matplotlib. Run with Python, PyMuPDF, and Matplotlib installed.
- `results.json`: scores used by the plots. Results updated to [arXiv:2609.19878v3](https://arxiv.org/abs/2609.19878v3), September 29, 2026. Main panels use Tables 1–2; sharing panels use Figure 3. Values follow the displayed paper rounding; relative gains are those reported from unrounded scores. All are percentages. Relative gains use the baseline denominator.

Diagram sources remain the manuscript supplied by the author on September 14, 2026. Experimental values now follow v3. The figures' teacher paths are training views; they do not imply access to future teacher observations at inference. Scene images illustrate the concept. The method walkthrough highlights existing diagram regions and is not a simulation of learned diffusion trajectories.

Desktop result charts show all comparisons side by side. Mobile charts reflow the same values as horizontal bars. All axes start at zero; no uncertainty has been invented. VLM main results compare complete methods with different training recipes. Sharing studies fix backbone, data, teachers, and thought-token dimensions, but do not match encoder parameter counts; see the blog captions.
