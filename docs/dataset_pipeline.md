# How the dataset is built

This document describes the chain that produces the dataset in
`data/json_repository_for_simulation/`, and records the quirks of that chain that a reader
would otherwise discover by regenerating it.

**Reproducing the results of the paper does not require any of this.** The dataset is
published as an archive, and everything in `MainScript.py` and `postprocessing/` starts from
it. This chain matters if you want to extend the period, apply the method to another plant,
or verify how the inputs were made.

```
ENTSO-E API ──▶ 1. DownloadFromEntsoEAPI ──▶ data/prices/, data/imbalance/
                                                        │
PV data (INESC TEC) ─────────────────────▶ 2. Data_Handling_Script ──▶ SystemData/
                                                        │
                                            3. Data_cleaning        ──▶ anomaly logs
                                            4. generate_global_stats ─▶ normalization_stats.json
                                            5. Scenario_MILPS_Generations ─▶ ScenarioData/
                                            6. trade_off_computation ─▶ the λ analysis
```

Steps 3 to 6 need only `SystemData`, so anyone with the published archive can run them.
Steps 1 and 2 need the raw data, which the archive also carries: the prices of step 1 are
included, so no personal ENTSO-E token is needed to rebuild them, and the PV data of step 2 is
owned by INESC TEC and redistributed with their permission.

---

## 1. Prices — `scripts/DownloadFromEntsoEAPI.py`

Downloads the day-ahead and imbalance prices of the Portuguese bidding zone, one CSV per day,
into `data/prices/` and `data/imbalance/`.

Requires a token from the ENTSO-E Transparency Platform — free, from *Account Settings → Web
Api Security Token* — pasted into `API_TOKEN` at the top of the script. Days already on disk
are skipped, so a run interrupted by a network error resumes where it stopped instead of
querying the whole year again.

## 2. System data — `scripts/Data_Handling_Script.py`

Combines the PV scenarios, the forecasts, the observed values and the prices into one
`SystemData/data_YYYY-MM-DD.json` per day. Plant parameters come from
`config/system_parameters.json`; `quantile_probability` and `scaling_PV`, which shape the
generation rather than describe the plant, stay in the script.

Needs three raw files in `data/raw/`:

| File | Size | Content |
|---|---|---|
| `scenarios_full.pkl` | 77 MB | 500 probabilistic PV scenarios per day |
| `forecasts_full.pkl` | 11 MB | q50 forecasts |
| `observed_values.csv` | 0.4 MB | measured PV and demand |

Roughly half an hour for the full year, almost all of it spent re-reading the same 77 MB
pickle: the `Data_Handling` object is constructed inside the day loop.

## 3. Quality check — `scripts/Data_cleaning.py`

Scans `SystemData` for corrupted observed demand — more than 40 % zeros, or negative values —
and writes two logs into `results/`. On the published dataset it finds **13 corrupted days**,
of which **two fall in the test months: 2024-10-09 and 2024-10-10**, both flagged for data
loss on the measured load. This is the basis for excluding those two days when reproducing
the October aggregates.

## 4. Normalisation statistics — `scripts/generate_global_stats.py`

Computes the mean and standard deviation of the four statistical features over the training
months and writes `normalization_stats.json`. Seconds. **See the quirk below.**

**This step needs the complete dataset.** The script keeps only the days whose month is in
`train_months`, and February, July and October are excluded by design — which are exactly the
three months this repository ships. Run on the repository alone it therefore finds no eligible
day, writes `NaN` into every entry of `normalization_stats.json`, overwrites the file that
ships with the repository, and still reports success; a policy trained afterwards would see
`NaN` inputs. Download the full dataset before running it. If you have already overwritten the
file, restore it with

```bash
git checkout data/json_repository_for_simulation/normalization_stats.json
```

## 5. Scenario pre-computation — `scripts/Scenario_MILPS_Generations.py`

The expensive step: for each day it solves 1 000 scenario MILPs plus three reference ones, and
writes `ScenarioData/Scenarios_Results_YYYY-MM-DD.json`. About **88 seconds per day**, so
roughly **9 hours** for the year on one core.

`scripts/Scenario_MILPS_Generation_HPC.py` is the parallel counterpart: same seed, same
results, the 1 000 solves of each day dispatched to a process pool. Measured at 17.5 s per day
with 12 workers, about 1.8 hours for the year; with 32 workers it drops below the hour. Set
`WRITE_JSON = True` to generate the dataset; left at `False` it only measures the timings and
writes nothing. The two generators have been verified to produce **byte-identical** files.

Both must be run **from 1 January**: see the quirk below.

## 6. Trade-off analysis — `scripts/trade_off_computation.py`

Computes the λ that makes the day-ahead cost and the imbalance risk indifferent, day by day
over the training months, and prints the average. On the published dataset: **λ = 0.4945**
over 256 days, with 19 days discarded because the geometry of the trade-off degenerates and
λ comes out negative. This is the number behind the choice λ = γ = 0.5.

---

## Known quirks

These are properties of the published dataset, not defects introduced later. They are
documented because regenerating the data reproduces them, and because anyone applying the
chain to different data should know they are there.

### The scenario generation must start from 1 January

Two mechanisms make a single day impossible to regenerate on its own.

The random draw of the scenario indices comes from a stream seeded once (`sid = 101`) and
advanced day by day, so starting on a different date draws different scenarios. And the
initial state of charge of each day is inherited from the previous one. Regenerating day 200
alone produces a valid file, but not the one in the published dataset.

### The state of charge is chained only up to the second day

`SoC_End` accumulates with `append` but the loop always reads positions 0…999. Day 2
therefore correctly inherits the states of charge of day 1, while **from day 3 onwards every
day re-reads the values produced on day 1**. The day-to-day chaining is effectively frozen
after the second day.

### A quarter of the normalisation sample is three repeated days

In the version of `generate_global_stats.py` that produced the shipped
`normalization_stats.json`, the file-reading statement sat outside the training-month test, so
each test-month iteration re-read the last training file seen. The consequence: of the 8 784
hourly samples, **2 184 are copies of just three days** — 2024-01-31 counted 29 times,
2024-06-30 and 2024-09-30 counted 31 times each.

The script has since been corrected and now reads one file per training day, 6 600 samples.
Re-running it therefore produces values that differ from the shipped file, by up to 2.21 on
`mu` and 1.02 on `sigma`.

This does not affect the published results. A trained policy stores its own normalisation
inside its checkpoint, so the file is read when training a new policy and ignored when
evaluating an existing one. The measured difference in the normalised inputs between the two
versions is between 0.004 and 0.028 standard deviations, uniformly across all months — the
three repeated days happen to be ordinary ones whose deviations largely cancel.

### Regenerating gives tiny numerical differences

Several MILPs in the chain are degenerate: the tracking problem is blind to prices, and the
day-ahead problem is degenerate whenever consecutive hours share the same price. Different
solver versions, or the same solver reaching a different optimal vertex, produce a different
solution with the same objective value.

Regenerating the first days of `ScenarioData` with the current code reproduces the published
files to within 1e-14 — the grid exchanges are bit-identical, the costs differ in the last bit
of a double. Regenerating `SystemData/data_2024-07-02.json` reproduces it exactly, with zero
differences.

### Library versions

The chain was originally run with a pandas older than 2.2. `DataFrame.fillna(method=...)` was
removed in pandas 3 and the code has been updated to `ffill()`/`bfill()`, which behave
identically and work on both. This is why `requirements.txt` pins exact versions.
