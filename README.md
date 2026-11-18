# Extended generalized Pareto models for right-censored survival data

Code for the paper

> Janotti, E., Henriques-Rodrigues, L. and Pedroso-de-Lima, A. C.
> *Long-Term Tail Modeling in Survival Analysis via Extended Generalized Pareto Distributions.*
> Submitted to the *Journal of Applied Statistics*.

If you use this code, please cite the article (see `CITATION.cff`, or GitHub's "Cite this repository" button).

The survival function is modelled as F(x) = H(G_θ(x)), where G_θ is a generalized
Pareto distribution (GPD) with scale σ and extreme value index ξ, and H is a perturbation
distribution on [0, 1]. Three perturbations are implemented, all fitted to right-censored
data through Kaplan–Meier-based pseudo-observations within the same iterative procedure:

| Perturbation | Class | Estimation of H |
|---|---|---|
| Beta, H(u) = u^k | `Beta_one_parameter` | censored maximum likelihood |
| Bernstein polynomial of degree m | `Bernstein_interval_Censored` | Kaplan–Meier weights |
| Histogram with m bins | `Histogram_interval_Censored` | Kaplan–Meier weights |

## Repository layout

```
egpd/models.py                   models and the iterative estimation procedure
simulation/run_simulation.py     Monte Carlo study (Section 4)
simulation/summary_tables.py     Tables S1–S11 of the Supplementary Material
applications/applications.py     real-data applications (Section 5) and Figures
data/bladder.csv                 bladder cancer recurrence data (R package survival)
results/simulation_results.csv.gz  simulation output used in the paper
```

## Installation

Tested with Python 3.13.

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from scipy.stats import CensoredData
from egpd import Semi_Ext_Gen_Pareto

data = CensoredData(uncensored=event_times, right=censoring_times)

model = Semi_Ext_Gen_Pareto()
model.fit(data, kernel="Beta")                 # or kernel="Bernstein" / "Histogram" with m=10
xi, _, sigma = model.Ѳ[-1]                     # final estimates (location is fixed at 0)
survival = 1 - model.cdf(np.array([1.0, 5.0]))
```

## Reproducing the paper

```bash
# Section 5: estimates and figures (bladder data only; see "Data" below)
python applications/applications.py --outdir figures

# Section 4: summary tables from the stored simulation output
python simulation/summary_tables.py results/simulation_results.csv.gz --outdir tables

# Section 4: re-run the full simulation (several hours on a 32-thread machine)
python simulation/run_simulation.py --out simulation_results.csv
```

In the simulation output, the column `Censoring proportion` stores the proportion of
**events**; `summary_tables.py` reports the censoring proportion as one minus this value.
Columns `p*`, `b*` and `h*` hold (ξ, μ, σ) for the Beta, Bernstein and histogram models.

**Note on reproducibility.** The stored results were produced before a correction to the
numerical safeguard used when the GPD quantile function is not finite (pseudo-observations
are now truncated at 1 − 10⁻¹⁰ first, as described in the paper). Running the current code
reproduces the stored Beta and histogram estimates to machine precision; the Bernstein
estimates differ only in the fewer than 0.5% of fits (all with n ≤ 100) in which this
safeguard is triggered.

## Data

* **Bladder cancer recurrence** (Byar, 1980): distributed with the R package
  [`survival`](https://cran.r-project.org/package=survival) and included in `data/`.
* **Heart failure, INCOR–FMUSP**: the authors are authorized to use these data but not to
  share them, so they are **not** included. Users with access can pass the file with
  `python applications/applications.py --incor PATH`.

## License

MIT; see [`LICENSE`](LICENSE).
