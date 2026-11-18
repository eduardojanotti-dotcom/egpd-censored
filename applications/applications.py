#!/usr/bin/env python3
"""
Real-data applications of the paper (Section 5).

Bladder cancer recurrence (data/bladder.csv, from the R package `survival`):
    first recurrence, placebo (rx=1) and thiotepa (rx=2), time = stop / 30.
INCOR-FMUSP heart failure data: NOT distributed with this repository (the authors
    are authorized to use them but not to share them). If you have access, pass
    the file with --incor; otherwise only the bladder analysis is run.

For each dataset the Beta, Bernstein (m=10) and histogram (m=10) EGP models are
fitted, estimates are printed, and two figures are written to --outdir:
survival curves (with Kaplan-Meier and an exponential mixture cure model) and
the fitted perturbation densities h(u).

Usage:  python applications.py [--incor PATH] [--outdir figures]
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import ExponentialFitter, KaplanMeierFitter, MixtureCureFitter
from scipy.stats import CensoredData

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from egpd import Semi_Ext_Gen_Pareto  # noqa: E402

plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})


def first_episode(df, id_col, order_col, time_col):
    df = df.sort_values([id_col, order_col]).copy()
    df["enum_"] = df.groupby(id_col).cumcount() + 1
    df["prev_stop"] = df.groupby(id_col)[time_col].shift(fill_value=0)
    df["duration"] = df[time_col] - df["prev_stop"]
    zero_seen = df["duration"].eq(0).groupby(df[id_col]).cumsum()
    df = df[zero_seen == 0]
    return df[df["enum_"] == 1]


def bladder(rx):
    df = pd.read_csv(os.path.join(HERE, "..", "data", "bladder.csv"))
    df = first_episode(df, "id", "enum", "stop")
    df = df[df["rx"] == rx].dropna(subset=["event", "duration"])
    return df["duration"].to_numpy() / 30, df["event"].to_numpy()


def incor(path):
    df = pd.read_csv(path)
    df = first_episode(df, "CASO_NUMER", "DAT_ENT_PR", "TEMPO")
    df = df.dropna(subset=["DELTA", "duration"])
    return df["duration"].to_numpy(), df["DELTA"].to_numpy()


def fit_and_plot(name, time, event, lim_sup, surv_png, pert_png, outdir, m=10):
    data = CensoredData(uncensored=time[event == 1], right=time[event == 0])
    mb = Semi_Ext_Gen_Pareto(); mb.fit(data, m=m, kernel="Bernstein")
    mh = Semi_Ext_Gen_Pareto(); mh.fit(data, m=m, kernel="Histogram")
    mp = Semi_Ext_Gen_Pareto(); mp.fit(data, kernel="Beta")

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(0, lim_sup, 1000)
    ax.plot(x, [1 - mp.cdf(t) for t in x], color="#0072B2", linestyle=":", lw=2.0, label="Beta")
    ax.plot(x, [1 - mb.cdf(t) for t in x], color="#009E73", linestyle=":", lw=1.5, label="Bernstein")
    ax.plot(x, [1 - mh.cdf(t) for t in x], color="#E69F00", linestyle="--", lw=1.5, label="Histogram")
    kmf = KaplanMeierFitter().fit(durations=time, event_observed=event)
    km_t = kmf.survival_function_.index.values
    ax.step(km_t, kmf.survival_function_["KM_estimate"].values, where="post",
            color="#D55E00", lw=0.5, label="Kaplan--Meier")
    ax.fill_between(km_t, kmf.confidence_interval_.iloc[:, 0].values,
                    kmf.confidence_interval_.iloc[:, 1].values, step="post", alpha=0.2, color="#D55E00")
    cure = MixtureCureFitter(base_fitter=ExponentialFitter()).fit(time, event_observed=event)
    ax.plot(x, cure.survival_function_at_times(x).values, linestyle="-.",
            color="#CC79A7", lw=1.5, label="Mixture cure model")
    ax.set_xlabel("Month"); ax.set_ylabel("Survival probability")
    ax.set_xlim(-0.1, lim_sup + 0.1); ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="best", frameon=False)
    fig.savefig(os.path.join(outdir, surv_png), bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    u = np.linspace(0, 1, 1000)
    ax.plot(u, [mp.model.pdf(t) for t in u], color="#0072B2", linestyle=":", lw=2.0, label="Beta")
    ax.plot(u, [mb.model.pdf(t) for t in u], color="#009E73", linestyle=":", lw=1.5, label="Bernstein")
    ax.plot(u, [mh.model.pdf(t) for t in u], color="#E69F00", linestyle="--", lw=1.5, label="Histogram")
    ax.set_xlabel("u"); ax.set_ylabel("Perturbation density h(u)")
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 4.1)
    ax.legend(loc="best", frameon=False)
    fig.savefig(os.path.join(outdir, pert_png), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n{name}: n={len(time)}, censored={np.mean(event == 0):.3f}")
    for lab, mod in [("Beta", mp), ("Bernstein", mb), ("Histogram", mh)]:
        xi, _, sigma = mod.Ѳ[-1]
        extra = f"  k={mod.model.a:.4f}" if lab == "Beta" else ""
        print(f"  {lab:9s} xi={xi:.4f} sigma={sigma:.4f} iterations={len(mod.Ѳ) - 1}{extra}")
    print(f"  Cure model: cured fraction={cure.cured_fraction_:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incor", help="path to the INCOR-FMUSP csv (not distributed)")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t, e = bladder(1); fit_and_plot("Bladder, placebo", t, e, 5, "b4.png", "b5.png", a.outdir)
    t, e = bladder(2); fit_and_plot("Bladder, thiotepa", t, e, 5, "a4.png", "a5.png", a.outdir)
    if a.incor:
        t, e = incor(a.incor); fit_and_plot("INCOR heart failure", t, e, 300, "c4.png", "c5.png", a.outdir)
    else:
        print("\nINCOR data not provided (--incor); skipped.")


if __name__ == "__main__":
    main()
