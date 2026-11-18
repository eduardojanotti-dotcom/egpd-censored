#!/usr/bin/env python3
"""
Monte Carlo study of the paper (Section 4).

Event times ~ GPD(xi, sigma=1); censoring times ~ GPD(xi_c, sigma_c), independent.
1000 replications for every combination of
    n in {50, 100, 500, 1000}, xi in {0.5, 1, 1.5, 2}, xi_c in {-1, 0, 1},
    sigma_c in SIGMA_C[(xi, xi_c)]   (about 75%, 50% and 25% censoring),
and m in {10, 20, 80, 140} with m < n for the Bernstein and histogram models.

Output: one row per (replication, scenario, m) with columns
    rep, n_samples, hazard, tail, hazard_censoring, tail_censoring,
    Censoring proportion  (NB: stored as the proportion of EVENTS),
    m, p1, p2, p3 (Beta: xi, mu, sigma), b1, b2, b3 (Bernstein), h1, h2, h3 (histogram)

Usage:  python run_simulation.py [--out FILE] [--workers N] [--reps R]
The full study (R = 1000) takes several hours on a 32-thread machine.
"""
import argparse
import csv
import os
import sys
import warnings
from itertools import product
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HEADER = ["rep", "n_samples", "hazard", "tail", "hazard_censoring", "tail_censoring",
          "Censoring proportion", "m", "p1", "p2", "p3", "b1", "b2", "b3", "h1", "h2", "h3"]

# censoring scale sigma_c for each (xi, xi_c): approx. 75%, 50%, 25% censoring
SIGMA_C = {
    (0.5, -1): [0.7, 2, 6],   (0.5, 0): [0.4, 1.3, 4.5], (0.5, 1): [0.2, 0.8, 7.5],
    (1.0, -1): [0.7, 2.5, 9.5], (1.0, 0): [0.4, 1.7, 7],  (1.0, 1): [0.2, 1, 5],
    (1.5, -1): [0.8, 3, 15],  (1.5, 0): [0.5, 2, 11],    (1.5, 1): [0.2, 1.3, 8],
    (2.0, -1): [0.9, 4, 24],  (2.0, 0): [0.5, 2.5, 18],  (2.0, 1): [0.3, 1.6, 12.5],
}
M_VALUES = [10, 20, 80, 140]


def build_jobs(reps):
    return [(rep, n, 1, xi, sc, xc)
            for rep, n, xi, xc in product(range(reps), [50, 100, 500, 1000],
                                          [0.5, 1.0, 1.5, 2.0], [-1, 0, 1])
            for sc in SIGMA_C[(xi, xc)]]


def run_one_simulation(job):
    warnings.filterwarnings("ignore")
    import numpy as np
    from scipy.stats import CensoredData, genpareto
    from egpd import Semi_Ext_Gen_Pareto

    rep, n, scale, xi, sigma_c, xi_c = job
    x = genpareto.rvs(xi, scale=scale, size=n, random_state=rep)
    c = genpareto.rvs(xi_c, scale=sigma_c, size=n, random_state=rep + 12345)
    t = np.minimum(x, c)
    event = x <= c
    prop_events = np.sum(event) / n
    data = CensoredData(uncensored=t[event], right=t[~event])

    def fit(**kw):
        try:
            model = Semi_Ext_Gen_Pareto()
            model.fit(data, **kw)
            return list(model.Ѳ[-1])
        except Exception:
            return [None] * 3

    beta_fit = fit(kernel="Beta")
    rows = []
    for m in M_VALUES:
        if m >= n:
            continue
        rows.append([rep, n, scale, xi, sigma_c, xi_c, prop_events, m, *beta_fit,
                     *fit(m=m, kernel="Bernstein"), *fit(m=m, kernel="Histogram")])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="simulation_results.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()

    jobs = build_jobs(a.reps)
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, rows in enumerate(pool.imap_unordered(run_one_simulation, jobs, chunksize=4), 1):
            w.writerows(rows)
            if i % 500 == 0:
                f.flush()
                print(f"{i}/{len(jobs)} scenarios done", flush=True)


if __name__ == "__main__":
    main()
