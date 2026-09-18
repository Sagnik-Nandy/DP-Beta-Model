#!/usr/bin/env python3
"""Parameter-recovery experiment for differentially private estimation in the
r-uniform hypergraph beta-model.

Compares four estimators of the node parameter vector beta from a (possibly
privatized) degree sequence: non-private ridge, non-private MLE, a local-DP
estimator (discrete Laplace noise on the degrees), and a central-DP estimator
(noisy gradient descent). For each replicate and privacy budget, writes the
normalized MSE of each estimator against the true beta to a CSV.

Run as a CLI; see `main()` for arguments. Designed to be sharded across a
Slurm job array via --rep_offset.
"""

import os
import argparse
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
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
            out[filled:filled+take] = x[:take]
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


def sample_degrees_3uniform(beta, rng=None, chunk_size_pairs=250_000):
    """Exactly sample the degree sequence of a 3-uniform hypergraph beta-model,
    where each triple (i, j, k) is an independent hyperedge with probability
    sigmoid(beta_i + beta_j + beta_k). Each triple is sampled exactly once.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = beta.size
    j_all, k_all = triu_pairs(n)

    d = np.zeros(n, dtype=float)

    for i in range(n - 2):
        mask = (j_all > i)
        jj = j_all[mask]
        kk = k_all[mask]

        logits = beta[i] + beta[jj] + beta[kk]
        p = expit(logits)

        chosen = rng.random(p.size) < p

        if chosen.any():
            j_sel = jj[chosen]
            k_sel = kk[chosen]

            d[i] += j_sel.size
            np.add.at(d, j_sel, 1)
            np.add.at(d, k_sel, 1)

    return d


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


def one_replicate(rep_seed, n, r, M, eps_list, delta,
                  p0, mean, sd, c_lam,
                  chunk_size_pairs, maxiter_mle, T_cap, verbose_dp=False):
    """Run one Monte Carlo replicate: sample beta_true and its degree sequence,
    fit the non-private (ridge, MLE) and private (local-DP, central-DP)
    estimators across eps_list, and return one result row per epsilon.
    """
    rng = np.random.default_rng(rep_seed)

    beta_true = sample_beta_sparse_truncnorm(n, M, p0=p0, mean=mean, sd=sd, rng=rng)
    d_true = sample_degrees_3uniform(beta_true, rng=rng, chunk_size_pairs=chunk_size_pairs)

    lam = c_lam * (n ** ((r - 1) / 2.0))

    model = HypergraphBeta3(n)

    print("Computing non-private ridge baseline (box + ridge) ...", flush=True)
    beta_ridge, _ = model.ridge_box_fit_from_degrees(
        d_obs=d_true, M=M, lam=lam, beta_init=np.zeros(n),
        maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
    )
    nmse_ridge = float(np.mean((beta_ridge - beta_true) ** 2))

    print("Computing non-private (unregularized) MLE baseline (box only) ...", flush=True)
    beta_np_mle, _ = model.box_mle_fit_from_degrees(
        d_obs=d_true, M=M, beta_init=np.zeros(n),
        maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
    )
    nmse_np_mle = float(np.mean((beta_np_mle - beta_true) ** 2))

    rows = []
    for eps in eps_list:
        print(f"Running local dp for eps={eps} rep={rep_seed} \n", flush=True)
        z = discrete_laplace_noise(eps_per_coordinate=eps / r, size=n, rng=rng)
        d_loc = d_true + z

        beta_loc, _ = model.ridge_box_fit_from_degrees(
            d_obs=d_loc, M=M, lam=lam, beta_init=np.zeros(n),
            maxiter=maxiter_mle, chunk_size_pairs=chunk_size_pairs
        )

        print(f"Running central dp for eps={eps} rep={rep_seed} \n", flush=True)
        beta_cen, meta = central_dp_gd(
            model=model, d_true=d_true, n=n, r=r, M=M,
            eps=eps, delta=delta, chunk_size_pairs=chunk_size_pairs,
            seed=int(rng.integers(1_000_000_000)), beta0=np.zeros(n),
            T_cap=T_cap, verbose=verbose_dp
        )

        nmse_loc = float(np.mean((beta_loc - beta_true) ** 2))
        nmse_cen = float(np.mean((beta_cen - beta_true) ** 2))

        rows.append({
            "rep": int(rep_seed),
            "n": int(n), "r": int(r), "M": float(M),
            "p0": float(p0), "mean": float(mean), "sd": float(sd),
            "c_lam": float(c_lam), "lam": float(lam),
            "eps": float(eps), "delta": float(delta),
            "nmse_ridge_np": nmse_ridge,
            "nmse_mle_np": nmse_np_mle,
            "nmse_loc": nmse_loc,
            "nmse_cen": nmse_cen,
            "T": int(meta["T"]),
            "eta": float(meta["eta"]),
            "sigma": float(meta["sigma"]),
        })
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

    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--reps_per_task", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=12345)

    parser.add_argument("--chunk_size_pairs", type=int, default=250000)
    parser.add_argument("--maxiter_mle", type=int, default=80)
    parser.add_argument("--tol_mle", type=float, default=1e-5)

    parser.add_argument("--T_cap", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose_dp", action="store_true")

    parser.add_argument("--rep_offset", type=int, default=0,
                    help="Starting replicate index for this Slurm task (overrides SLURM_ARRAY_TASK_ID mapping).")

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
        rows = one_replicate(
            rep_seed=rep_seed,
            n=args.n, r=args.r, M=args.M,
            eps_list=eps_list, delta=delta,
            p0=args.p0, mean=args.mean, sd=args.sd,
            c_lam=args.c_lam,
            chunk_size_pairs=args.chunk_size_pairs,
            maxiter_mle=args.maxiter_mle,
            T_cap=args.T_cap,
            verbose_dp=args.verbose_dp
        )
        rows_all.extend(rows)

    outfile = os.path.join(args.outdir, f"{args.tag}_task{task_id:03d}.csv")
    header = list(rows_all[0].keys())
    with open(outfile, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows_all:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    print(f"Wrote {len(rows_all)} rows to {outfile}")


if __name__ == "__main__":
    main()
