# Uni-LaDiR blog figures

- `fig1.pdf`: original manuscript Figure 1, `fig1-0912-motion-detail-v76.pdf`.
- `fig2.pdf`: original manuscript Figure 2, `method-sequence-v23.pdf`.
- `render.py`: renders the original vectors to SVG/PNG and replots measured results using Matplotlib. Run with Python, PyMuPDF, and Matplotlib installed.
- `results.json`: scores used by the plots. Main panels use manuscript Tables 1 and 3; sharing panels use Figure 3 and supplementary sharing tables. All are percentages. Relative gains use the baseline denominator.

Source manuscript supplied by the author on September 14, 2026. The figures' teacher paths are training views; they do not imply access to future teacher observations at inference. Scene images illustrate the concept. The method walkthrough highlights existing diagram regions and is not a simulation of learned diffusion trajectories.

Desktop result charts show all comparisons side by side. Mobile charts reflow the same values as horizontal bars. All axes start at zero; no uncertainty has been invented. The paper distinguishes matched experiments from reported baseline comparisons; see the blog captions.
