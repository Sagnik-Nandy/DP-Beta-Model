# Private β-Models

Implementation and experiments for
[Mandal & Nandy, "Privacy-Utility Trade-offs for Parameter Estimation in
Degree-Heterogeneous Higher-Order Networks"](https://arxiv.org/abs/2602.03948):
finite-sample minimax rates for differentially private parameter estimation
in the r-uniform hypergraph β-model, under both local and central edge
differential privacy. A local-DP estimator (discrete Laplace noise on the
degree sequence) and a central-DP estimator (noisy gradient descent) are each
compared against non-private (MLE and ridge) baselines, on synthetic data
(`Numerical experiments/`) and a real communication network — the Enron email
hypergraph (`Data analysis/`) — matching the paper's own experiments.

## Repository layout

| Folder | Contents |
|---|---|
| [`Numerical experiments/`](Numerical%20experiments/README.md) | Synthetic-data experiments: Slurm-launched parameter-recovery and link-prediction trials, aggregated into the paper's plots. |
| [`Data analysis/`](Data%20analysis/README.md) | The same estimators applied to the real Enron email hypergraph for link prediction. |

Each folder's README has the detailed layout, run order, and per-file
breakdown.

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `numpy`, `scipy` | Core linear algebra; hypergraph log-likelihood/gradient computations, discrete Laplace and Gaussian noise mechanisms |
| `pandas` | Loading/aggregating per-task result CSVs |
| `scikit-learn` | Link-prediction metrics (ROC-AUC, F1, average precision, Brier score) |
| `matplotlib` | Result plots |
| `jupyter` | Running the aggregation and analysis notebooks |
| `tqdm`, `joblib` | Progress bars and parallel replicate sweeps in the prototype notebook (`Numerical experiments/vary_eps.ipynb`) |
| `numba` | Accelerates the 4-uniform hypergraph likelihood in `Data analysis/codes/link_prediction_order4.ipynb` |
