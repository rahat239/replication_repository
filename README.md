# Does Naive Cross-Validation Inflate Software Energy-Regression Prediction?

Replication package for the manuscript submitted to *Information and Software Technology*:
**"Does Naive Cross-Validation Inflate Software Energy-Regression Prediction? A
Leakage-Aware, Statistically-Gated, Cross-Project Empirical Study"**
Author: Rahat Ahmed Jobu, Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh.

## What's in this repository

```
.
├── manuscript.tex              # Full LaTeX source (elsarticle, elsarticle-num citation style)
├── references.bib              # Bibliography (all entries independently verified)
├── highlights.txt              # Elsevier-required Highlights file
├── cover_letter.txt            # Cover letter to the Information and Software Technology editor
├── data/
│   ├── energy_regression_dataset.csv     # Full benchmark: 3,684 labeled commit transitions
│   ├── threshold_sensitivity.csv         # 5-threshold sensitivity sweep results
│   ├── per_project_breakdown.csv         # Per-project/per-language composition
│   └── algorithm_comparison.csv          # RF/XGBoost/LogReg comparison results
├── code/
│   ├── 01_build_dataset.py     # Clones EnergyTrackr data + 9 source repos, builds the benchmark
│   ├── 02_ablation_test.py     # log_prev_energy feature ablation
│   ├── 03_algorithm_comparison.py  # Multi-algorithm robustness check
│   ├── 04_make_figures.py      # Generates all 7 manuscript figures from data/
│   └── 05_heuristic_baseline.py    # Training-free churn-magnitude baseline (Table 7)
├── figures/                    # All 7 figures as vector PDFs (as used in the manuscript)
└── notebook/
    └── energy_regression_prediction_v2.ipynb  # Interactive Colab notebook, runs end-to-end
```

## Data source

All raw energy measurements originate from the public **EnergyTrackr** data release:
https://github.com/flipflop133/energytrackr-data (Bechet et al., 2026). This repository
does not redistribute those raw measurements beyond what is already public; `data/energy_regression_dataset.csv`
is our derived benchmark (features + statistically-gated labels), not a copy of EnergyTrackr's
raw per-commit joule readings.

## Reproducing the results

**Fastest path**: open `notebook/energy_regression_prediction_v2.ipynb` in Google Colab and
run all cells. Data collection (cloning EnergyTrackr's data repo plus the 9 source project
repos) takes roughly 10-20 minutes; all modeling, statistics, and figure generation after that
completes in under a minute.

**From source**, in order:
```bash
pip install -r requirements.txt

python code/01_build_dataset.py        # builds data/energy_regression_dataset.csv from scratch
python code/02_ablation_test.py        # reproduces Table 5 (feature ablation)
python code/03_algorithm_comparison.py # reproduces Table 6 (algorithm comparison)
python code/04_make_figures.py         # reproduces all 7 figures
python code/05_heuristic_baseline.py   # reproduces Table 7 (heuristic baseline)
```

All scripts use a fixed random seed (`RANDOM_SEED = 42`) throughout, as stated in the
manuscript's Methodology section.

## Environment

- Python 3.12
- Exact package versions pinned in `requirements.txt` (scikit-learn 1.8.0, xgboost 3.4.0,
  scipy 1.17.1, pandas 3.0.2, numpy 2.4.4, shap 0.52.0, statsmodels 0.14.6), matching the
  Computational Environment paragraph in the manuscript's Methodology section (Section 3.6).
  No GPU required; all experiments run on CPU in a few minutes total.

## What this repository does NOT contain

- Raw per-commit energy joule measurements (these remain EnergyTrackr's original data; fetch
  them via `code/01_build_dataset.py`, which clones the public release directly).
- Any manually re-labeled or filtered version of the benchmark following the construct-validity
  concerns raised in Section 4.11 of the manuscript. The released dataset contains the original,
  automated statistical labels exactly as evaluated in the paper, including the ~50% of sampled
  labels later found via manual inspection to plausibly reflect measurement artifacts rather than
  genuine code-efficiency regressions. Anyone reusing this benchmark should read Section 4.11 and
  Section 6 (Threats to Validity) before treating the positive class as ground truth, and is
  encouraged to apply the recommended release-metadata/test-only-commit filter first.

## Citation

If you use this benchmark or code, please cite the manuscript (citation details to be added upon
publication) and the underlying EnergyTrackr data release.

## License

Code in this repository is released under the MIT License. See the EnergyTrackr repository for
the license governing the underlying raw measurement data.

## Contact

Rahat Ahmed Jobu — rahatahmed537@gmail.com
