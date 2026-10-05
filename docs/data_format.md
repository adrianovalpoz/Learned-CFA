# Dataset format

The dataset is distributed separately from the code (see the README for the archive and its
DOI) and unpacks into `data/json_repository_for_simulation/`:

```
data/json_repository_for_simulation/
├── SystemData/          data_YYYY-MM-DD.json              366 files, one per day of 2024
├── ScenarioData/        Scenarios_Results_YYYY-MM-DD.json  366 files, one per day of 2024
└── normalization_stats.json
```

`SystemData` holds the inputs of one day: plant parameters, PV and demand forecasts with
their uncertainty description, and the three price series. `ScenarioData` holds the outcome
of the offline scenario pre-computation for that same day: the 1 000 MILP solutions used to
build the risk term of the reward. The two are always used together, one pair per day.

This document describes both files so that the method can be applied to another plant: the
script that builds `SystemData` (`scripts/Data_Handling_Script.py`) is tied to the format of
this plant's raw PV data, so producing input files of the right shape is the practical route.

**Units.** Power and energy are in **kW** and **kWh** at hourly resolution, so the two are
numerically interchangeable over one time step. Prices are in **€/kWh** — the sources publish
€/MWh and the pipeline divides by 1 000 — so a price multiplied by an hourly energy gives
euros directly. All arrays are ordered by hour, index 0 to 23, local time.

---

## SystemData — `data_YYYY-MM-DD.json`

Six top-level keys.

### `Battery`

Physical parameters of the storage. Identical in every file: they come from
`config/system_parameters.json` and describe the plant, not the day.

| Field | Type | Value in the case study | Meaning |
|---|---|---|---|
| `SoC_in` | float | 0.5 | state of charge at the start of the day, as a fraction |
| `Storage_Capacity` | float | 1258.33 | usable capacity [kWh] |
| `eff_charge` | float | 0.9 | charging efficiency |
| `eff_discharge` | float | 0.9 | discharging efficiency |
| `Max_Charge` | float | 880.83 | charging power limit [kW] |
| `Max_Discharge` | float | 880.83 | discharging power limit [kW] |

The upper state-of-charge bound of 0.9 is not stored here: it is written into the MILP
constraint (`learned_cfa_src/milp/Battery.py`, `Overcharge_Limit`).

### `Simulation`

| Field | Type | Value | Meaning |
|---|---|---|---|
| `time_resolution` | int | 1 | steps per hour |
| `time_horizon` | int | 24 | hours in the optimisation horizon |
| `sin_enc` | list[24] | −1…1 | sine encoding of the hour, an input feature of the policy |
| `cos_enc` | list[24] | −1…1 | cosine encoding of the hour, an input feature of the policy |

### `Fluxes`

Names of the decision variables of the MILP, and which battery fluxes are inflows.

| Field | Type | Content |
|---|---|---|
| `Battery Fluxes` | list[4] of str | `x_Market_Storage`, `x_Energy_Storage`, `x_Storage_Market`, `x_Storage_Demand` |
| `Battery Fluxes Type` | dict | 1 for a flux entering the battery, 0 for one leaving it |
| `Other Fluxes` | list[3] of str | `x_Energy_Market`, `x_Energy_Demand`, `x_Market_Demand` |
| `Total Fluxes` | list[7] of str | the two lists concatenated |

### `Energy` and `Demand`

Same seven keys in both: PV generation and site demand, in kW. The ranges below are from
2 July 2024 and are indicative.

| Field | Type | Meaning |
|---|---|---|
| `forecast` | list[24] | the q50 forecast — the deterministic baseline the CFA multipliers act on |
| `mean` | list[24] | mean across the 500 scenarios, an input feature of the policy |
| `deviation` | list[24] | standard deviation across the scenarios, an input feature of the policy |
| `quantile` | list[24] | quantile of the scenario distribution: probability 0.3 for `Energy`, 0.7 for `Demand` |
| `bands` | dict | `min` and `max`, list[24] each: Chebyshev uncertainty envelope, which bounds the multipliers |
| `scenario` | str | **500 probabilistic scenarios, see below** |
| `Observed` | list[24] | what actually happened, used only for the ex-post evaluation, never in the day-ahead decision |

`bands` is what turns into the per-hour bounds `theta_max` and `theta_min` of the policy
(`Theta_Bounds` in `learned_cfa_src/evolution_strategy/es_Policy.py`), which is why the
Learned-CFA multipliers are day-dependent while the Static-CFA benchmark ignores them.

**`scenario` is a string, not an array.** It contains a nested JSON document, the
serialisation of a pandas DataFrame in `columns` orientation:

```json
"scenario": "{\"0\":{\"0\":0.00073,\"1\":0.00073, ... ,\"23\":0.0008}, \"1\":{...}, ... , \"499\":{...}}"
```

The outer keys `"0"`…`"499"` are the 500 scenarios, each mapping the hours `"0"`…`"23"` to a
value in kW. Read it with `json.loads` on the string first, or with `pandas.read_json`.

### `Prices`

Three hourly series in €/kWh.

| Field | Type | Meaning |
|---|---|---|
| `Market` | list[24] | day-ahead price |
| `Imbalance_positive` | list[24] | price applied when the delivery exceeds the commitment |
| `Imbalance_negative` | list[24] | price applied when the delivery falls short |

Two processing decisions are baked into these series and matter when comparing with the raw
source data (`learned_cfa_src/data_pipeline/Data_Handling.py`):

- **non-positive day-ahead prices are floored to 0.001 €/kWh**, i.e. 1 €/MWh, instead of
  being kept negative;
- **missing imbalance prices are filled forward, then backward**, so a gap in the source data
  is replaced by the neighbouring hour rather than left empty.

Both imbalance series can be negative. The dual-pricing rule of the settlement — a deviation
is charged only when it hurts the system, and is never rewarded when it helps — is applied in
the code (`imbalance_costs` in `learned_cfa_src/Optimize_Risk.py`), not in the data.

---

## ScenarioData — `Scenarios_Results_YYYY-MM-DD.json`

Twelve top-level keys, the outcome of solving 1 000 scenario MILPs plus three reference MILPs
for that day. Costs are in €, grid exchanges in kWh per hour.

| Field | Type | Meaning |
|---|---|---|
| `Costs` | list[1000] | day-ahead cost of each scenario solution |
| `Grid_Exchange` | list[1000] × list[24] | hourly grid exchange of each scenario solution |
| `C_baseline` | float | day-ahead cost of the deterministic q50 plan |
| `Grid_Exchange_Baseline` | list[24] | hourly grid exchange of the deterministic q50 plan |
| `Best_DA` | float | lowest cost among the scenarios, `min(Costs)` |
| `Worst_DA` | float | highest cost among the scenarios, `max(Costs)` |
| `Best_ICVaR` | float | imbalance CVaR of the plan that attains `Worst_DA` |
| `Worst_ICVaR` | float | imbalance CVaR of the plan that attains `Best_DA` |
| `C_best_cheby` | float | day-ahead cost at the aggressive Chebyshev bound |
| `C_worst_cheby` | float | day-ahead cost at the conservative Chebyshev bound |
| `ICVaR_worst_Cheby` | float | imbalance CVaR at the aggressive bound |
| `ICVaR_best_cheby` | float | imbalance CVaR at the conservative bound |

**`Best_ICVaR` and `Worst_ICVaR` are crossed over on purpose**: the plan with the best
day-ahead cost is the one exposed to the worst imbalance risk, and vice versa. They are the
two extremes that normalise the risk term of the reward, so swapping them changes the
objective.

The four Chebyshev values come from two extra MILPs solved at the edges of the uncertainty
envelope, and are the inputs of the λ trade-off analysis in
`scripts/trade_off_computation.py`.

---

## `normalization_stats.json`

The mean and standard deviation used to normalise the four statistical inputs of the policy
network, in the order `[Energy mean, Energy deviation, Demand mean, Demand deviation]`.

```json
{"mu": [71.176, 19.402, 58.103, 26.663], "sigma": [100.536, 24.329, 41.132, 15.247], "info": "..."}
```

A trained policy stores these values inside its own checkpoint, so the file is read when
training a new policy and ignored when evaluating an existing one. See
`docs/dataset_pipeline.md` for how it is computed and for a known quirk of that computation.
