# Learned-CFA

Code and trained policies for *Bridging Deterministic and Stochastic Scheduling: Learning
Risk-aware Forecast Adjustments for PV-BESS Operation*.

The method keeps the deterministic day-ahead MILP of a PV-BESS plant unchanged and learns,
with an evolutionary strategy, how to reshape the forecasts it is fed. A small neural policy
(626 parameters) reads the daily uncertainty context and outputs one multiplier per hour for
the PV and demand forecasts; the MILP then plans as usual. Risk awareness comes from the
training objective, which combines the day-ahead cost with the CVaR of the imbalance cost,
and not from a stochastic program solved online.

> **Reference** — *(authors)*, "Bridging Deterministic and Stochastic Scheduling: Learning
> Risk-aware Forecast Adjustments for PV-BESS Operation", *Applied Energy*, 2026.
> <https://doi.org/10.1016/j.apenergy.2026.128975>

## Repository layout

```
MainScript.py              simulation and training entry point: edit the control inputs at the top and run it
learned_cfa_src/           the method
    milp/                  day-ahead MILP, battery and energy-flux models
    evolution_strategy/    policy networks, ES training, hyperparameter search
    data_pipeline/         construction of the input data
    Optimize_Risk.py       the reward: day-ahead cost and CVaR of the imbalance cost
    directory_names.py     every folder of the repository, defined once
config/                    hand-editable parameters: plant, training hyperparameters
scripts/                   the chain that builds the dataset (see docs/dataset_pipeline.md)
postprocessing/            figures of the paper
saved_ANN/                 the published policies
docs/                      dataset format and pipeline
```

Output folders (`results/`, `data/`) are created on demand and are not versioned.

## Installation

Python 3.13 and the pinned dependencies:

```bash
pip install -r requirements.txt
```

No solver to install separately: PuLP ships the CBC solver it calls. No GPU is used — the run
time is dominated by the MILP solves, not by the network.

## Data

The repository ships the three test months of the paper — **February, July and October
2024** — so a clone is enough to reproduce its results (about 90 MB):

```
data/json_repository_for_simulation/
├── SystemData/          91 files, data_2024-02-01.json to data_2024-10-31.json
├── ScenarioData/        91 files, Scenarios_Results_2024-02-01.json to ...-10-31.json
└── normalization_stats.json
```

One `SystemData` file holds the inputs of a day — plant, forecasts with their uncertainty
description, prices — and the matching `ScenarioData` file holds the 1 000 pre-computed
scenario MILP solutions used by the risk term. `docs/data_format.md` describes both.

The **complete dataset** — all 366 days of 2024, plus the raw sources the JSON files are built
from — is archived separately: *(add the archive link and its DOI.)* It is needed to train a
policy, which uses the nine months outside the test set, and to rebuild the pipeline. Unpack
it over `data/`: the folder names are the same, and its test-month files are byte-identical to
the ones already versioned here, so nothing is lost by overwriting them.

That archive also carries the raw data the chain is built from — PV scenarios and forecasts,
owned by INESC TEC and redistributed with their permission, plus the measured values and the
two price series, included so that no personal ENTSO-E token is needed. The dataset can
therefore be rebuilt end to end; see `docs/dataset_pipeline.md`.

## Reproducing the results of the paper

Everything runs through `MainScript.py`. Edit the control inputs at the top of the file and
run it; there are no command-line arguments.

```bash
python MainScript.py
```

Set `start_date` and `end_date` to the test month you want — February, July or October — and
then one of these combinations:

| Policy | `Opt_Type` | `Data_Type` | `Model_Name` | Result file in `results/` |
|---|---|---|---|---|
| Deterministic baseline | `'Deterministic'` | `"forecast"` | ignored | `Results_<from>_to_<to>_Deterministic.json` |
| Perfect foresight | `'Deterministic'` | `"Observed"` | ignored | `Results_<from>_to_<to>_Deterministic_perf.json` |
| Static-CFA | `'Static_CFA'` | `"forecast"` | ignored | `Results_<from>_to_<to>_Static_CFA.json` |
| Learned-CFA | `'Learned_CFA'` | `"forecast"` | `"toff0.5_sid101"` | `Results_<from>_to_<to>_Learned_CFA_toff0.5_sid101.json` |

Keep `NN_Training = 'n'` and `NN_Optimization = 'n'`, and set `workers` to the number of
physical cores of the machine.

These four runs per month produce the monthly cost and risk aggregates of *(Table …)*, the
delivery and storage statistics of *(Table …)* and the execution times of *(Table …)*.

Then the figures:

```bash
python postprocessing/Plotting.py
```

with `Opt_Type` and the dates set the same way. It writes five figures, in PDF and PNG, into
`results/figure/`. The cumulative-profit panel compares several λ values and only draws the
ones whose result files are present.

**October.** Two days of the test period, 9 and 10 October, have corrupted measured demand —
more than 40 % zeros, a sensor data loss — and are excluded from the October aggregates.
`scripts/Data_cleaning.py` identifies them from the data. The remaining differences you may
see when regenerating are a few euros on the imbalance columns, caused by degenerate MILPs
reaching different optimal vertices with the same objective value.

## Training a policy from scratch

Set `NN_Training = 'y'`, choose a **new** `Model_Name` so that the published policies are not
overwritten, and select the policy with `Opt_Type`:

| | `Opt_Type` | Parameters | Artefact produced |
|---|---|---|---|
| Learned-CFA | `'Learned_CFA'` | 626 network weights | `saved_ANN/<Model_Name>.pth` |
| Static-CFA | `'Static_CFA'` | 48 hourly multipliers | `saved_ANN/<Model_Name>.pth` and `.json` |

Both are trained by the same evolutionary strategy on the same reward and the same nine
training months, so the only difference between them is whether the policy sees the daily
context. Hyperparameters live in `config/training_hyperparameters.json`.

Expect about **7 hours on 32 cores** for the full 1 000 iterations. A log with one row per
iteration is written to `results/TrainingCheck/training_log_<Model_Name>.csv`.

To re-run the hyperparameter search instead, set `NN_Optimization = 'y'`: 50 Optuna trials,
with the outcome written to `results/NN_Size/best_hyperparameters.json`.

## Further documentation

- `docs/data_format.md` — the structure of the two dataset files, field by field, with units.
  Read this to apply the method to another plant.
- `docs/dataset_pipeline.md` — how the dataset is built, what each script costs, and the known
  quirks of the published data.

## Citation

*(add `CITATION.cff` and the citation text.)*

## License

MIT — see `LICENSE`. The dataset is distributed separately and carries its own license.
