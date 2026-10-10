# Report

`report.tex` — *Two-Tier Reinforcement Learning with an Oracle-Free Top-K
Search for Deep Learning Job Scheduling on Heterogeneous GPU Clusters*.

The project report; it covers both the two-tier scheduler and the oracle-free
search. The built PDF is `report.pdf`.

## Build

```bash
pdflatex report && bibtex report && pdflatex report && pdflatex report
```

## Figures

`images/` holds one PDF per figure. All of them are regenerated from the
measurement JSONs in `results/` by, from the repository root:

```bash
python -m scripts.figure_data     # the two measurements not already on disk
python -m scripts.figures         # -> report/images/report_*.pdf
```

The architecture diagram is TikZ rather than a plot; its source is
`images/workflow.tex`, built with `pdflatex workflow`. `images/workflow.png` is a
200-dpi render of it for the repository README, since GitHub cannot display PDFs
inline:

```bash
cd images && pdflatex workflow && pdftoppm -png -r 200 -singlefile workflow.pdf workflow
```

`images/policy_comparison.png` is the same kind of render of
`report_policy_comparison.pdf`, also for the README.
