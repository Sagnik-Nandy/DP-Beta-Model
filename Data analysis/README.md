# Data Analysis

Real-data companion to the synthetic experiments in `../Numerical experiments/`,
matching Section 6 (and Appendix E) of the paper: link prediction on the Enron
email hypergraph, comparing the same four estimators (non-private MLE,
non-private ridge, local-DP, central-DP) from the r-uniform hypergraph
beta-model.

## Data

`data/email-Enron/` contains the public Enron email hypergraph dataset
(Benson et al., 2018; originally from https://www.cs.cmu.edu/~./enron/, this
copy via https://www.cs.cornell.edu/~arb/data/). Each simplex is an email;
its nodes are the Enron employees who sent or received it.

- `email-Enron-nverts.txt`, `email-Enron-simplices.txt`, `email-Enron-times.txt`:
  the raw timestamped simplex data (see `DATA-DESCRIPTION.txt` for the exact
  encoding).
- `enron_order{2,3,4}_edges.csv`: deduplicated 2-, 3-, and 4-uniform hyperedge
  lists (each kept at its earliest occurrence), extracted by the loader
  notebooks below.

Note: the original dataset also ships a node-labels file mapping node ids to
employees' real names and email addresses. It is not needed by any script
here (only `nverts`/`simplices`/`times` and the derived edge CSVs are used)
and is intentionally omitted from this repository.

## Codes

```
enron_email_data_loader_order2.ipynb   Extract 2-uniform hyperedges -> enron_order2_edges.csv
enron_email_data_loader_order3.ipynb   Extract 3-uniform hyperedges -> enron_order3_edges.csv
enron_email_data_loader_order4.ipynb   Extract 4-uniform hyperedges -> enron_order4_edges.csv
enron_email.ipynb                      Early prototype of the loader (superseded by the three above)

link_prediction_order2.ipynb   Link prediction on the 2-uniform (ordinary graph) hypergraph
link_prediction_order3.ipynb   Link prediction on the 3-uniform hypergraph (paper's main setting)
link_prediction_order4.ipynb   Link prediction on the 4-uniform hypergraph (Numba-accelerated)
```

Each `link_prediction_order*.ipynb` holds out 10% of observed hyperedges as
test positives, fits the four estimators on the remaining training-set
degrees, and evaluates ROC-AUC / average precision / Brier score / log-loss /
ECE / F1 against an equal number of sampled negative hyperedges, sweeping
`eps in {0.001, 0.01, 0.1, 1.0}`. Since the true `M` (the box constraint on
beta) is unknown for real data, these notebooks use the conservative choice
`M = 2*sqrt(log n)` and a fixed, empirically-stable step size `eta = 0.005`
rather than the theoretical worst-case rate used in `../Numerical experiments/`.

Each notebook writes its results table to a CSV in `codes/` (e.g.
`link_prediction_results_order2.csv`); these are gitignored and regenerated
by running the notebook.
