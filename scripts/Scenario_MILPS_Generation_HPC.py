"""
Parallel counterpart of Scenario_MILPS_Generations.py, for the offline scenario-precomputation
stage. It also produces the timing figures asked for by Reviewer 1, comment 9 (the
"scenario-precomputation time" behind the computational-efficiency claim).

Same folder as the sequential script, same folders from directory_names, same output: the two
generators write byte-comparable ScenarioData files, and either one can rebuild the dataset.

  * the 1,000 scenario MILPs of each day are dispatched to a process pool (N_WORKERS)
    instead of being solved sequentially;
  * elapsed time is measured with time.perf_counter(), reported separately for the
    scenario solves and for the per-day baseline/bounds/CVaR block;
  * WRITE_JSON decides what the run is for. False (the default) measures the times and
    writes nothing, so it cannot touch the existing ScenarioData files; True regenerates
    the dataset. The JSON is written outside the timers, so the measured times are the
    same either way.
  * either way the run must START FROM THE FIRST DAY: the random stream advances day by
    day, so a single day re-generated on its own would draw different scenarios.

WHY THE INNER LOOP CAN BE PARALLELISED. In the original, scenario i of a given day reads
SoC_End[i] - a value produced on a PREVIOUS day - and appends its own result to the end of
the same list. Iteration i never consumes the result of iteration i-1, so within a day the
1,000 solves are independent given the incoming SoC list; only the day-to-day chaining is
sequential. That structure is preserved exactly.

ONE PROPERTY OF THE ORIGINAL REPRODUCED HERE DELIBERATELY:
     SoC_End is never reset between days: it grows (1,000 entries after day 1, 2,000 after
     day 2, ...) while the loop always indexes positions 0..999, so from the third day
     onwards every day re-reads the SoC values produced on day one. Reproduced verbatim,
     so that both generators rebuild the published dataset - but worth knowing when
     interpreting it.

Run:
    py Scenario_MILPS_Generation_HPC.py
"""

import os
import json
import time
import random
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np

# ----------------------------------------------------------------------------------
WRITE_JSON = False      # True = write the dataset into ScenarioData; False = timing only
N_WORKERS = 32          # <-- set by hand: 32 on the workstation, ~6 on a laptop
N_SAMPLES = 1000        # joint scenarios per day, as in the original
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
SEED = 101
# ----------------------------------------------------------------------------------

# _path puts the repository root on sys.path. Every spawned worker re-applies it, because
# under 'spawn' each worker re-imports this module, whatever the working directory was.
import _path                                                                         # noqa: F401,E402
from learned_cfa_src.data_pipeline.Problem_Data import Problem_Data                   # noqa: E402
from learned_cfa_src.data_pipeline.optimize_energy_flux import optimize_energy_flux   # noqa: E402
from learned_cfa_src.directory_names import data_json_dir, scenario_json_dir          # noqa: E402

# ---------------------------------------------------------------------------------
# PuLP/CBC temp-file cleanup race (Windows). After a solve, silent_remove() deletes the
# temp files but only catches FileNotFoundError; under many concurrent CBC subprocesses
# os.remove() can transiently raise PermissionError, which kills the worker and with it
# the whole pool - even though the solve itself already succeeded. Patched in-process,
# at module level, so that every spawned worker re-applies it on import.
# ---------------------------------------------------------------------------------
import pulp  # noqa: E402


def _patched_silent_remove(self, file):
    for attempt in range(5):
        try:
            os.remove(file)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 4:
                return          # give up silently: cleanup must never abort a solve
            time.sleep(0.05 * (attempt + 1))


if not getattr(pulp.apis.core.LpSolver_CMD, "_windows_race_patched", False):
    pulp.apis.core.LpSolver_CMD.silent_remove = _patched_silent_remove
    pulp.apis.core.LpSolver_CMD._windows_race_patched = True


def _make_problem_data(D):
    """Problem_Data with the 8 placeholders this tree's signature requires."""
    return Problem_Data(D["Battery"], D["Simulation"], D["Fluxes"], D["Prices"],
                        D["Energy"], D["Demand"], "", "", "", "", "", "", "", "")


# --- per-worker cache: a worker loads a given day once, not once per scenario ----------
_CACHE = {"day": None, "data": None}


def _get_day(day):
    if _CACHE["day"] != day:
        with open(os.path.join(data_json_dir, f"data_{day}.json"), "r") as f:
            _CACHE["data"] = _make_problem_data(json.load(f))
        _CACHE["day"] = day
    return _CACHE["data"]


def solve_scenario(args):
    """One scenario MILP. Module-level and picklable, as 'spawn' requires."""
    day, e_idx, d_idx, soc_in = args
    DATA = _get_day(day)
    if soc_in is not None:
        DATA.Battery_Data["SoC_in"] = soc_in
    costs, _, soc_day, _, _, _, _, _, _, _, _, _, grid = optimize_energy_flux(
        DATA, e_idx, d_idx, "scenario")
    return costs, soc_day[-1], grid


def theta_bounds(DATA):
    """Verbatim from the original."""
    e_forecast = DATA.Energy_Data["forecast"]
    e_bound_max = DATA.Energy_Data["bands"]["max"]
    e_bound_min = DATA.Energy_Data["bands"]["min"]
    d_forecast = DATA.Demand_Data["forecast"]
    d_bound_max = DATA.Demand_Data["bands"]["max"]
    d_bound_min = DATA.Demand_Data["bands"]["min"]

    arrays = [e_forecast, e_bound_max, e_bound_min, d_forecast, d_bound_max, d_bound_min]
    for i in range(len(arrays)):
        if not isinstance(arrays[i], np.ndarray):
            arrays[i] = np.array(arrays[i])

    theta_max_e = arrays[1] / arrays[0]
    theta_min_e = arrays[2] / arrays[0]
    theta_max_d = arrays[4] / arrays[3]
    theta_min_d = arrays[5] / arrays[3]
    for t in range(len(arrays[0])):
        if arrays[0][t] <= 1:
            theta_max_e[t] = 0
            theta_min_e[t] = 0
        if arrays[3][t] <= 1:
            theta_max_d[t] = 0
            theta_min_d[t] = 0

    r = 1
    theta_max_e = np.minimum(1 + r, theta_max_e)
    theta_min_e = np.maximum(1 - r, theta_min_e)
    theta_max_d = np.minimum(1 + r, theta_max_d)
    theta_min_d = np.maximum(1 - r, theta_min_d)

    e_forecast = np.array(e_forecast)
    d_forecast = np.array(d_forecast)
    return ((e_forecast * theta_max_e).tolist(), (e_forecast * theta_min_e).tolist(),
            (d_forecast * theta_max_d).tolist(), (d_forecast * theta_min_d).tolist())


def RightCVaR(Data, alpha):
    Data = np.asarray(Data)
    if Data.size == 0:
        return 0, 0, 0
    VaR = np.percentile(Data, alpha * 100)
    tail = Data[Data >= VaR]
    return tail.mean(), VaR, tail.std() + 1e-6


def ImbalanceCostsCVaR(data, Grid_Exchange_Milps, Grid_Exchange):
    N = len(Grid_Exchange_Milps)
    timesteps = data.Simulation_Data["time_horizon"] * data.Simulation_Data["time_resolution"]
    pp = data.Prices["Imbalance_positive"]
    nn = data.Prices["Imbalance_negative"]
    tot = []
    for s in range(N):
        c = []
        for t in range(timesteps):
            imb = Grid_Exchange_Milps[s][t] - Grid_Exchange[t]
            if imb >= 0:
                c.append(-pp[t] * imb if pp[t] < 0 else 0)
            else:
                c.append(-nn[t] * imb if nn[t] > 0 else 0)
        tot.append(sum(c))
    return RightCVaR(tot, alpha=0.95)[0]


if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)

    days = []
    d = START_DATE
    while d <= END_DATE:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    print("=" * 96)
    print("SCENARIO PRECOMPUTATION - TIMING RUN (nothing is written to disk)")
    print("=" * 96)
    print(f"  data folder     : {os.path.normpath(data_json_dir)}")
    print(f"  days            : {len(days)}")
    print(f"  scenarios/day   : {N_SAMPLES}")
    print(f"  scenario MILPs  : {len(days) * N_SAMPLES:,}")
    print(f"  extra MILPs/day : 3 (baseline + 2 Chebyshev bounds)")
    print(f"  workers         : {N_WORKERS}\n")

    t_scen = 0.0        # time inside the parallel scenario solves
    t_extra = 0.0       # time in the per-day baseline/bounds MILPs and CVaR loops
    n_milp = 0
    SoC_End = []

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for k, day in enumerate(days):
            with open(os.path.join(data_json_dir, f"data_{day}.json"), "r") as f:
                DATA = _make_problem_data(json.load(f))

            # indices drawn from the same seeded stream, in the same order as the original
            idx_e = np.random.choice(range(DATA.Energy_scenarios.shape[1]),
                                     size=N_SAMPLES, replace=True)
            idx_d = np.random.choice(range(DATA.Demand_scenarios.shape[1]),
                                     size=N_SAMPLES, replace=True)

            first = (k == 0)
            tasks = [(day, int(idx_e[i]), int(idx_d[i]),
                      None if first else SoC_End[i]) for i in range(N_SAMPLES)]

            ts = time.perf_counter()
            results = list(pool.map(solve_scenario, tasks, chunksize=25))
            t_scen += time.perf_counter() - ts
            n_milp += N_SAMPLES

            costs = np.array([r[0] for r in results])
            grids = [r[2] for r in results]
            SoC_End.extend(r[1] for r in results)      # grows across days, as in the original

            te = time.perf_counter()
            i_best, i_worst = int(np.argmin(costs)), int(np.argmax(costs))
            ICVaR_at_Best_DA = ImbalanceCostsCVaR(DATA, grids, grids[i_best])
            ICVaR_at_Worst_DA = ImbalanceCostsCVaR(DATA, grids, grids[i_worst])

            base = optimize_energy_flux(DATA, "", "", "forecast")
            C_baseline, g_base = base[0], base[12]
            ImbalanceCostsCVaR(DATA, grids, g_base)    # computed by the sequential script too, never stored

            e_max, e_min, d_max, d_min = theta_bounds(DATA)
            maxrisk = optimize_energy_flux(DATA, e_max, d_min, "bounds")
            C_best_cheby, g_maxrisk = maxrisk[0], maxrisk[12]
            ICVaR_worst_Cheby = ImbalanceCostsCVaR(DATA, grids, g_maxrisk)

            minrisk = optimize_energy_flux(DATA, e_min, d_max, "bounds")
            C_worst_cheby, g_minrisk = minrisk[0], minrisk[12]
            ICVaR_best_cheby = ImbalanceCostsCVaR(DATA, grids, g_minrisk)
            t_extra += time.perf_counter() - te
            n_milp += 3

            # Written outside the timers, so the measured times stay comparable.
            if WRITE_JSON:
                Scenario_MILP_Results = {
                    "Costs": costs.tolist(),
                    "Grid_Exchange": grids,
                    "C_baseline": C_baseline,
                    "Grid_Exchange_Baseline": g_base,
                    "Best_DA": float(costs[i_best]),
                    "Worst_DA": float(costs[i_worst]),
                    "Best_ICVaR": float(ICVaR_at_Worst_DA),   # crossed over, as in the sequential script
                    "Worst_ICVaR": float(ICVaR_at_Best_DA),
                    "C_best_cheby": C_best_cheby,
                    "C_worst_cheby": C_worst_cheby,
                    "ICVaR_worst_Cheby": float(ICVaR_worst_Cheby),
                    "ICVaR_best_cheby": float(ICVaR_best_cheby)}
                os.makedirs(scenario_json_dir, exist_ok=True)
                with open(os.path.join(scenario_json_dir, f"Scenarios_Results_{day}.json"), "w") as fjson:
                    json.dump(Scenario_MILP_Results, fjson, indent=4)

            if k == 0 or (k + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  {day}  ({k+1:3d}/{len(days)})  elapsed {el/60:6.2f} min"
                      f"  | projected total {el/(k+1)*len(days)/3600:5.2f} h")

    total = time.perf_counter() - t0

    print("\n" + "=" * 96)
    print("RESULT")
    print("=" * 96)
    print(f"  Scenario MILPs solved          : {len(days)*N_SAMPLES:,}")
    print(f"  Total MILPs solved             : {n_milp:,}")
    print(f"  Time in scenario solves        : {t_scen:10.1f} s = {t_scen/3600:.2f} h")
    print(f"  Time in baseline/bounds + CVaR : {t_extra:10.1f} s = {t_extra/3600:.2f} h")
    print(f"  TOTAL WALL-CLOCK ({N_WORKERS} workers) : {total:10.1f} s = {total/3600:.2f} h")
    print(f"  Equivalent core-hours          : {total/3600*N_WORKERS:.1f}")
    print(f"  Throughput                     : {n_milp/total:.1f} MILP/s"
          f"  ({n_milp/total/N_WORKERS:.2f} per worker)")
