#!/usr/bin/env python3
"""
Summary tables of the simulation study (Tables S1-S11 of the Supplementary Material).

  S1-S4   proportion of replications in which each model has the lowest absolute
          error, overall and by xi, n and xi_c
  S5-S8   proportion of configurations (n, xi, xi_c, m; sigma_c pooled, 156 in total)
          in which each model has the lowest absolute bias, overall and by xi, n and xi_c
  S9-S11  per-scenario censoring proportion, median, MAE and bias of sigma and xi
          for the Beta, Bernstein and histogram models

Ties are broken in the order Beta, Bernstein, histogram; a failed fit never wins.

Usage:  python summary_tables.py RESULTS.csv[.gz] [--outdir tables]
"""
import argparse
import os

import numpy as np
import pandas as pd

MODELS = ["Beta", "Bernstein", "Histogram"]
PREFIX = {"Beta": "beta", "Bernstein": "bern", "Histogram": "hist"}


def load(path):
    sim = pd.read_csv(path).rename(columns={
        "p1": "beta_xi", "p3": "beta_sigma", "b1": "bern_xi", "b3": "bern_sigma",
        "h1": "hist_xi", "h3": "hist_sigma"})
    # the stored column is the proportion of events
    sim["censoring"] = 1 - sim["Censoring proportion"]
    return sim


def winner(a, b, c):
    a, b, c = (np.nan_to_num(np.asarray(x, float), nan=np.inf) for x in (a, b, c))
    return np.where((a <= b) & (a <= c), "Beta",
                    np.where((b <= a) & (b <= c), "Bernstein", "Histogram"))


def proportions(df, col, by=None):
    if by is None:
        return df[col].value_counts(normalize=True).reindex(MODELS, fill_value=0).to_frame("all").T
    return (df.groupby(by)[col].value_counts(normalize=True)
              .unstack(fill_value=0).reindex(columns=MODELS, fill_value=0))


def winners(sim):
    ae = pd.DataFrame({
        "n": sim["n_samples"], "xi": sim["tail"], "xi_c": sim["tail_censoring"],
        "xi_winner": winner(*[(sim[f"{PREFIX[m]}_xi"] - sim["tail"]).abs() for m in MODELS]),
        "sigma_winner": winner(*[(sim[f"{PREFIX[m]}_sigma"] - 1).abs() for m in MODELS]),
    })
    g = sim.groupby(["n_samples", "tail", "tail_censoring", "m"])
    bias = pd.DataFrame({
        f"{PREFIX[m]}_{q}": g[f"{PREFIX[m]}_{q}"].mean() - (g["tail"].first() if q == "xi" else 1)
        for m in MODELS for q in ["xi", "sigma"]}).reset_index()
    bias["xi_winner"] = winner(*[bias[f"{PREFIX[m]}_xi"].abs() for m in MODELS])
    bias["sigma_winner"] = winner(*[bias[f"{PREFIX[m]}_sigma"].abs() for m in MODELS])
    bias = bias.rename(columns={"n_samples": "n", "tail": "xi", "tail_censoring": "xi_c"})
    return ae, bias


def detailed(sim, model):
    p = PREFIX[model]
    keys = ["n_samples", "tail", "tail_censoring", "hazard_censoring"] + ([] if model == "Beta" else ["m"])
    d = sim.assign(e_s=(sim[f"{p}_sigma"] - 1), e_x=(sim[f"{p}_xi"] - sim["tail"]))
    g = d.groupby(keys)
    out = pd.DataFrame({
        "cp": g["censoring"].mean(),
        "median_sigma": g[f"{p}_sigma"].median(),
        "mae_sigma": g["e_s"].apply(lambda v: v.abs().mean()),
        "bias_sigma": g["e_s"].mean(),
        "median_xi": g[f"{p}_xi"].median(),
        "mae_xi": g["e_x"].apply(lambda v: v.abs().mean()),
        "bias_xi": g["e_x"].mean(),
    }).reset_index().rename(columns={"n_samples": "n", "tail": "xi", "tail_censoring": "xi_c",
                                     "hazard_censoring": "sigma_c"})
    return out.sort_values(["n", "xi", "xi_c"] + ([] if model == "Beta" else ["m"]) + ["sigma_c"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--outdir", default="tables")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sim = load(a.results)
    ae, bias = winners(sim)
    tabs = {}
    for name, df, d in [("S1-S4_lowest_abs_error", ae, 4), ("S5-S8_lowest_abs_bias", bias, 3)]:
        for q in ["xi", "sigma"]:
            for by in [None, "xi", "n", "xi_c"]:
                key = f"{name}_{q}_{'overall' if by is None else 'by_' + by}"
                tabs[key] = proportions(df, f"{q}_winner", by).round(d)
    for model, s in zip(MODELS, ["S9", "S10", "S11"]):
        tabs[f"{s}_{model.lower()}_detailed"] = detailed(sim, model).round(2)

    for key, t in tabs.items():
        t.to_csv(os.path.join(a.outdir, key + ".csv"), index=not key.startswith(("S9", "S10", "S11")))
    print(f"{len(bias)} bias configurations; wrote {len(tabs)} tables to {a.outdir}/")
    for key in [k for k in tabs if "overall" in k]:
        print(f"\n{key}\n{tabs[key].to_string()}")


if __name__ == "__main__":
    main()
