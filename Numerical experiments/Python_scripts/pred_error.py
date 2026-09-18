#!/usr/bin/env python3
"""Link-prediction experiment for differentially private estimation in the
3-uniform hypergraph beta-model.

Protocol: sample beta_true and a hypergraph from p_ijk = sigmoid(beta_i+beta_j+beta_k);
hold out 20% of hyperedges as test positives (remaining 80% form the training
graph); fit four estimators from the training degrees (non-private ridge,
non-private MLE, local-DP ridge, central-DP gradient descent); and evaluate
each on held-out positive/negative hyperedges using ROC-AUC, F1 (at 0.5 and
best-over-threshold), Brier score, and expected calibration error (ECE).

Run as a CLI; see `main()` for arguments. Designed to be sharded across a
Slurm job array via --rep_offset.
"""

import os
import argparse
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import brier_score_loss
from scipy.special import comb

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_thread_env(threads=1):
    """Cap BLAS/OpenMP thread counts to avoid oversubscription on shared compute nodes."""
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))


def sample_truncnorm(mean, sd, low, high, size, rng):
    """Draw `size` iid N(mean, sd^2) samples truncated to [low, high] via rejection sampling."""
    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        m = int(1.5 * (size - filled) + 32)
        x = rng.normal(loc=mean, scale=sd, size=m)
        x = x[(x >= low) & (x <= high)]
        take = min(x.size, size - filled)
        if take:
            out[filled:filled + take] = x[:take]
            filled += take
    return out


def sample_beta_sparse_truncnorm(n, M, p0=0.3, mean=-1.0, sd=0.5, rng=None):
    """Sample a sparse parameter vector: each coordinate is 0 with probability p0,
    otherwise drawn from N(mean, sd^2) truncated to [-M, M]."""
    rng = np.random.default_rng() if rng is None else rng
    beta = np.zeros(n, dtype=float)
    mask = rng.random(n) > p0
    k = int(mask.sum())
    if k:
        beta[mask] = sample_truncnorm(mean, sd, -M, M, k, rng)
    return beta


def discrete_laplace_noise(eps_per_coordinate, size, rng=None):
    """Sample iid discrete Laplace noise with parameter a = exp(-eps_per_coordinate).

    For an r-uniform hypergraph degree sequence, the caller typically sets
    eps_per_coordinate = eps / r so that the released degrees satisfy eps-edge
    local differential privacy.
    """
    rng = np.random.default_rng() if rng is None else rng
    a = np.exp(-eps_per_coordinate)
    mag = rng.geometric(p=1 - a, size=size) - 1
    sign = rng.choice([-1, 1], size=size)
    return mag * sign


def triu_pairs(n):
    """Return the index pairs (j, k) with j < k for n vertices."""
    j_all, k_all = np.triu_indices(n, k=1)
    return j_all.astype(np.int32), k_all.astype(np.int32)


def masks_excluding_vertex(n, j_all, k_all):
    """For each vertex i, return a boolean mask selecting pairs (j, k) not containing i."""
    masks = []
    for i in range(n):
        masks.append((j_all != i) & (k_all != i))
    return masks


def sample_edges_and_degrees_3uniform(beta, rng=None, chunk_size_pairs=250_000):
    """Exactly sample a 3-uniform hypergraph from p_ijk = sigmoid(beta_i+beta_j+beta_k).

    Returns
    -------
    edges : (m, 3) int32 array of realized hyperedges, each row sorted i<j<k.
    d : (n,) float array, the resulting degree sequence.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = beta.size
    j_all, k_all = triu_pairs(n)

    d = np.zeros(n, dtype=np.int64)
    edges_list = []

    for i in range(n - 2):
        mask = (j_all > i)
        jj_all = j_all[mask]
        kk_all = k_all[mask]

        m = jj_all.size
        start = 0
        while start < m:
            end = min(start + chunk_size_pairs, m)
            jj = jj_all[start:end]
            kk = kk_all[start:end]

            logits = beta[i] + beta[jj] + beta[kk]
            p = expit(logits)
            chosen = rng.random(p.size) < p

            if chosen.any():
                j_sel = jj[chosen]
                k_sel = kk[chosen]

                d[i] += chosen.sum()
                np.add.at(d, j_sel, 1)
                np.add.at(d, k_sel, 1)

                edges_list.extend(zip([i]*len(j_sel), j_sel.tolist(), k_sel.tolist()))

            start = end

    edges = np.array(edges_list, dtype=np.int32) if edges_list else np.zeros((0, 3), dtype=np.int32)
    return edges, d.astype(float)


def degrees_from_edges(n, edges):
    """Recompute the degree sequence implied by an (m, 3) array of hyperedges."""
    if edges.shape[0] == 0:
        return np.zeros(n, dtype=float)
    flat = edges.reshape(-1)
    return np.bincount(flat, minlength=n).astype(float)


def edge_set_from_edges(edges):
    """Build a Python set of sorted-triple tuples for O(1) hyperedge membership tests."""
    return set(map(tuple, edges.tolist()))


def sample_negative_edges(n, forbidden_set, m, rng):
    """Uniformly sample m vertex triples not present in forbidden_set, for use
    as negative (non-edge) examples."""
    neg = []
    seen = set()
    tries = 0
    max_tries = 50 * m + 10_000
    while len(neg) < m and tries < max_tries:
        tries += 1
        tri = rng.choice(n, size=3, replace=False)
        tri.sort()
        t = (int(tri[0]), int(tri[1]), int(tri[2]))
        if t in forbidden_set or t in seen:
            continue
        seen.add(t)
        neg.append(t)
    if len(neg) < m:
        raise RuntimeError(
            f"Could not sample enough negative edges: got {len(neg)} of {m}. "
            f"Graph may be too dense for n={n}."
        )
    return np.array(neg, dtype=np.int32)


def ece_score(y_true, y_prob, n_bins=15):
    """Compute the expected calibration error (ECE) of predicted probabilities
    y_prob against binary outcomes y_true, using n_bins equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.size
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        frac = mask.mean()
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += frac * abs(acc - conf)
    return float(ece)


class HypergraphBeta3:
    """Log-partition function, gradients, and degree-based estimators for the
    3-uniform hypergraph beta-model.
    """

    def __init__(self, n):
        self.n = n
        self.r = 3
        self.j_all, self.k_all = triu_pairs(n)
        self.exclude_masks = masks_excluding_vertex(n, self.j_all, self.k_all)
        self.C = comb(self.n, self.r)  # binom(n, 3)

    def A_and_grad(self, beta, chunk_size_pairs=250_000):
        """Compute the log-partition function A(beta) = sum_{i<j<k} log(1+exp(beta_i+beta_j+beta_k))
        and its gradient (the vector of expected degrees), iterating over all
        n-choose-3 triples in chunks for memory efficiency.
        """
        n = self.n
        j_all, k_all = self.j_all, self.k_all

        A = 0.0
        grad = np.zeros(n, dtype=float)

        for i in range(n - 2):
            mask = (j_all > i)
            jj = j_all[mask]
            kk = k_all[mask]

            ps = beta[jj] + beta[kk]
            m = ps.size

            start = 0
            while start < m:
                end = min(start + chunk_size_pairs, m)
                s = beta[i] + ps[start:end]

                A += np.logaddexp(0.0, s).sum()

                p = expit(s)

                # Each triple contributes its edge probability to all 3 participating
                # vertices' expected degrees directly; no extra normalization needed.
                grad[i] += p.sum()
                np.add.at(grad, jj[start:end], p)
                np.add.at(grad, kk[start:end], p)

                start = end

        return A, grad

    def ridge_box_fit_from_degrees(self, d_obs, M, lam, beta_init=None,
                                    maxiter=2000, chunk_size_pairs=250_000, verbose=False):
        """Fit beta via projected gradient descent on the ridge-regularized negative
        log-likelihood given observed (possibly noisy) degrees d_obs, then clip the
        result to [-M, M]. Setting lam=0 recovers the box-constrained MLE.
        """
        n = self.n
        if beta_init is None:
            beta_init = np.zeros(n, dtype=float)

        beta = beta_init.copy()

        # Same step size as the central DP-GD estimator (see central_dp_gd).
        eta = 0.25 * n * np.exp(-2.0 * self.r * M)

        if verbose:
            print(f"Starting GD: eta={eta}, iterations={maxiter}")

        for t in range(maxiter):
            _, gA = self.A_and_grad(beta, chunk_size_pairs=chunk_size_pairs)
            grad_obj = (gA - d_obs + lam * beta) / self.C

            beta = beta - eta * grad_obj

            if verbose and t % 500 == 0:
                print(f"Iter {t}: grad_norm={np.linalg.norm(grad_obj)}")

        beta_truncated = np.clip(beta, -M, M)

        return beta_truncated, None

    def box_mle_fit_from_degrees(self, d_obs, M, beta_init=None,
                                maxiter=10000, chunk_size_pairs=250_000, verbose=False):
        """Box-constrained MLE: special case of ridge_box_fit_from_degrees with lam=0."""
        return self.ridge_box_fit_from_degrees(
            d_obs=d_obs, M=M, lam=0.0, beta_init=beta_init,
            maxiter=maxiter, chunk_size_pairs=chunk_size_pairs, verbose=verbose
        )

    def grad_ell(self, beta, d_true, chunk_size_pairs=250_000):
        """Gradient of the (unregularized) negative log-likelihood at beta,
        given the observed degree sequence d_true."""
        _, gA = self.A_and_grad(beta, chunk_size_pairs=chunk_size_pairs)
        return (gA - d_true) / self.C


def central_dp_gd(model, d_true, n, r, M, eps, delta,
                   chunk_size_pairs=250_000, seed=None, beta0=None, T_cap=1000, verbose=False):
    """Differentially private gradient descent estimator for beta under central
    (eps, delta)-edge differential privacy: adds Gaussian noise to the gradient
    at each step, with step size, iteration count, and noise scale calibrated
    so the final iterate is (eps, delta)-DP.
    """
    rng = np.random.default_rng(seed)
    beta = np.zeros(n, dtype=float) if beta0 is None else beta0.astype(float).copy()

    eta = 0.25 * n * np.exp(-2.0 * r * M)
    T_init = int(np.ceil(32.0 * (r - 1) * np.exp(4.0 * r * M) * ((r - 1) * np.log(n) + 2.0 * np.log(M))))
    T = min(T_init, T_cap)

    sigma2 = 4.0 * r * T * (n ** (-2.0 * r)) * (eps ** (-2.0)) * np.log(1.0 / delta)
    sigma = float(np.sqrt(sigma2))

    for t in range(T):
        if verbose and ((t + 1) % 100 == 0 or (t + 1) == T):
            print(f"Iter {t+1}/{T} eps={eps}", flush=True)
        g = model.grad_ell(beta, d_true, chunk_size_pairs=chunk_size_pairs)
        z = rng.normal(0.0, sigma, size=n)
        beta = beta - eta * (g + z)

    beta = np.clip(beta, -M, M)

    return beta, dict(eta=float(eta), T=int(T), sigma=float(sigma))


def score_edges(beta_hat, edges):
    """Score candidate hyperedges under a fitted beta_hat via
    p_hat(e) = sigmoid(sum_{v in e} beta_hat[v])."""
    s = beta_hat[edges[:, 0]] + beta_hat[edges[:, 1]] + beta_hat[edges[:, 2]]
    return expit(s)


def linkpred_metrics(y_true, y_prob, ece_bins=15):
    """Compute ROC-AUC, F1 at threshold 0.5, best F1 over thresholds, Brier
    score, and ECE for a set of scored candidate hyperedges."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    out = {}
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    y_pred_05 = (y_prob >= 0.5).astype(int)
    out["f1_05"] = float(f1_score(y_true, y_pred_05))

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    out["f1_best"] = float(np.nanmax(f1s)) if f1s.size else float(out["f1_05"])

    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["ece"] = float(ece_score(y_true, y_prob, n_bins=ece_bins))

    return out


def one_replicate(rep_seed, n, r, M, eps_list, delta,
                  p0, mean, sd, c_lam,
                  holdout_frac,
                  chunk_size_pairs, maxiter_mle, T_cap,
                  neg_ratio=1.0, ece_bins=15, verbose_dp=False):
    """Run one Monte Carlo replicate: sample beta_true and a hypergraph, hold
    out holdout_frac of the hyperedges as test positives, fit the four
    estimators from the training-set degrees, and evaluate link prediction on
    the held-out positives plus an equal number of sampled negatives. Returns
    one result row per epsilon.
    """
    rng = np.random.default_rng(rep_seed)
    assert r == 3, "This script currently implements r=3 only."

    beta_true = sample_beta_sparse_truncnorm(n, M, p0=p0, mean=mean, sd=sd, rng=rng)
    edges_all, _d_all = sample_edges_and_degrees_3uniform(beta_true, rng=rng, chunk_size_pairs=chunk_size_pairs)

    m_all = edges_all.shape[0]
    if m_all < 10:
        raise RuntimeError(f"Too few edges sampled (m={m_all}). Adjust beta distribution to avoid ultra-sparse graphs.")

    perm = rng.permutation(m_all)
    m_test = max(1, int(np.floor(holdout_frac * m_all)))
    test_idx = perm[:m_test]
    train_idx = perm[m_test:]

    edges_test = edges_all[test_idx]
    edges_train = edges_all[train_idx]

    d_train = degrees_from_edges(n, edges_train)

    forbidden = edge_set_from_edges(edges_all)

    m_neg = int(np.ceil(neg_ratio * edges_test.shape[0]))
    edges_neg = sample_negative_edges(n, forbidden, m_neg, rng)

    y_pos = np.ones(edges_test.shape[0], dtype=int)
    y_neg = np.zeros(edges_neg.shape[0], dtype=int)
    y_true = np.concatenate([y_pos, y_neg], axis=0)

    lam = c_lam * (n ** ((r - 1) / 2.0))

    model = HypergraphBeta3(n)

    beta_ridge, _ = model.ridge_box_fit_from_degrees(
        d_obs=d_train, M=M, lam=lam, beta_init=np.zeros(n),
        maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
    )

    beta_np_mle, _ = model.box_mle_fit_from_degrees(
        d_obs=d_train, M=M, beta_init=np.zeros(n),
        maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
    )

    rows = []
    for eps in eps_list:
        z = discrete_laplace_noise(eps_per_coordinate=eps / r, size=n, rng=rng)
        d_loc = d_train + z
        beta_loc, _ = model.ridge_box_fit_from_degrees(
            d_obs=d_loc, M=M, lam=lam, beta_init=np.zeros(n),
            maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
        )

        beta_cen, meta = central_dp_gd(
            model=model, d_true=d_train, n=n, r=r, M=M,
            eps=eps, delta=delta, chunk_size_pairs=chunk_size_pairs,
            seed=int(rng.integers(1_000_000_000)), beta0=np.zeros(n),
            T_cap=T_cap, verbose=verbose_dp
        )

        X_test = np.vstack([edges_test, edges_neg])

        prob_ridge = score_edges(beta_ridge, X_test)
        prob_mle = score_edges(beta_np_mle, X_test)
        prob_loc = score_edges(beta_loc, X_test)
        prob_cen = score_edges(beta_cen, X_test)

        met_ridge = linkpred_metrics(y_true, prob_ridge, ece_bins=ece_bins)
        met_mle = linkpred_metrics(y_true, prob_mle, ece_bins=ece_bins)
        met_loc = linkpred_metrics(y_true, prob_loc, ece_bins=ece_bins)
        met_cen = linkpred_metrics(y_true, prob_cen, ece_bins=ece_bins)

        row = {
            "rep": int(rep_seed),
            "n": int(n), "r": int(r), "M": float(M),
            "p0": float(p0), "mean": float(mean), "sd": float(sd),
            "c_lam": float(c_lam), "lam": float(lam),
            "eps": float(eps), "delta": float(delta),
            "holdout_frac": float(holdout_frac),
            "m_edges_all": int(m_all),
            "m_edges_train": int(edges_train.shape[0]),
            "m_edges_test": int(edges_test.shape[0]),
            "m_neg_test": int(edges_neg.shape[0]),
            "T": int(meta["T"]),
            "eta": float(meta["eta"]),
            "sigma": float(meta["sigma"]),
        }

        for prefix, met in [("ridge_np", met_ridge), ("mle_np", met_mle), ("loc", met_loc), ("cen", met_cen)]:
            row[f"{prefix}_roc_auc"] = met["roc_auc"]
            row[f"{prefix}_f1_05"] = met["f1_05"]
            row[f"{prefix}_f1_best"] = met["f1_best"]
            row[f"{prefix}_brier"] = met["brier"]
            row[f"{prefix}_ece"] = met["ece"]

        rows.append(row)

    return rows


def main():
    """CLI entry point: partitions Monte Carlo replicates across a Slurm array
    (via --rep_offset / SLURM_ARRAY_TASK_ID) and writes one CSV of results per task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="run1")

    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--M", type=float, default=2.0)

    parser.add_argument("--p0", type=float, default=0.3)
    parser.add_argument("--mean", type=float, default=-1.0)
    parser.add_argument("--sd", type=float, default=0.5)
    parser.add_argument("--c_lam", type=float, default=0.1)

    parser.add_argument("--eps_list", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--delta", type=float, default=None)

    parser.add_argument("--holdout_frac", type=float, default=0.20)
    parser.add_argument("--neg_ratio", type=float, default=1.0)  # #negatives = neg_ratio * #positives
    parser.add_argument("--ece_bins", type=int, default=15)

    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--reps_per_task", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=12345)
    parser.add_argument("--rep_offset", type=int, default=0,
                        help="Starting replicate index for this Slurm task (use with Slurm array mapping).")

    parser.add_argument("--chunk_size_pairs", type=int, default=250000)
    parser.add_argument("--maxiter_mle", type=int, default=10000)

    parser.add_argument("--T_cap", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose_dp", action="store_true")

    args = parser.parse_args()

    set_thread_env(args.threads)
    os.makedirs(args.outdir, exist_ok=True)

    eps_list = [float(x) for x in args.eps_list.split(",") if x.strip()]

    if args.delta is None:
        delta = args.n ** (-2)
    else:
        delta = float(args.delta)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    start_rep = int(args.rep_offset)
    end_rep = min(start_rep + args.reps_per_task, args.R)

    if start_rep >= args.R:
        print(f"Task {task_id}: nothing to do (start_rep={start_rep} >= R={args.R})")
        return

    rng = np.random.default_rng(args.base_seed)
    all_seeds = rng.integers(1_000_000_000, size=args.R)

    rows_all = []
    for rep_idx in range(start_rep, end_rep):
        rep_seed = int(all_seeds[rep_idx])
        print(f"\n=== Replicate {rep_idx+1}/{args.R} (seed={rep_seed}) ===", flush=True)
        rows = one_replicate(
            rep_seed=rep_seed,
            n=args.n, r=args.r, M=args.M,
            eps_list=eps_list, delta=delta,
            p0=args.p0, mean=args.mean, sd=args.sd,
            c_lam=args.c_lam,
            holdout_frac=args.holdout_frac,
            chunk_size_pairs=args.chunk_size_pairs,
            maxiter_mle=args.maxiter_mle,
            T_cap=args.T_cap,
            neg_ratio=args.neg_ratio,
            ece_bins=args.ece_bins,
            verbose_dp=args.verbose_dp
        )
        rows_all.extend(rows)

    outfile = os.path.join(args.outdir, f"{args.tag}_task{task_id:03d}.csv")
    header = list(rows_all[0].keys())
    with open(outfile, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows_all:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    print(f"\nWrote {len(rows_all)} rows to {outfile}", flush=True)


if __name__ == "__main__":
    main()
