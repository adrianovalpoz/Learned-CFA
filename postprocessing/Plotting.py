import json
import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
import _path
from validationplot import plot_validation_results, comulative_results
from learned_cfa_src.directory_names import results_dir, fig_dir

start_date_plot = datetime(2024,7,1) # Choice how many days you want to load as Data and to train
end_date_plot = datetime(2024,7,31)
Opt_Type = "Static_CFA"  # Type of Optimization # ['Deterministic', 'Static_CFA', 'Learned_CFA']
#"gamma_1_cheby"
models = ["toff0.3_sid101","toff0.4_sid101","toff0.5_sid101","toff0.6_sid101","toff0.7_sid101"]
Model_Name = models[2]

# --- CONFIGURAZIONE ---


start_str = start_date_plot.strftime('%d-%m-%Y')
end_str   = end_date_plot.strftime('%d-%m-%Y')


if Opt_Type == "Learned_CFA":
    JSON_FILE = f"Results_{start_str}_to_{end_str}_{Opt_Type}_{Model_Name}.json"
else:
    JSON_FILE = f"Results_{start_str}_to_{end_str}_{Opt_Type}.json"

json_path = os.path.join(results_dir, JSON_FILE)


os.makedirs(fig_dir, exist_ok=True)

def save_article_figure(fig, filename):
    """Save cropped figures for LaTeX/Overleaf in vector and raster formats."""
    fig.savefig(os.path.join(fig_dir, f"{filename}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(fig_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0.02, dpi=300)

# --- CARICAMENTO DATI ---
if not os.path.exists(json_path):
    raise FileNotFoundError(f"❌ Impossibile trovare il file JSON in: {json_path}")

with open(json_path, 'r') as f:
    d = json.load(f)

# Scorciatoie per i dati
meta = d["metadata"]
opt_type = meta["opt_type"]
daily = d["daily_data"]
hourly = d["hourly_data"]
dist = d["scenarios_data"]

# --- 1. PROFITTO CUMULATO ---

fig = comulative_results(opt_type,models,start_str,end_str,results_dir)
save_article_figure(fig, f"{opt_type}_Cumulative_Profit")

# --- 2. DAILY RISK TIMELINE ---
def ticks_including_zero(y_min, y_max, step=40):
    start = step * np.floor(y_min / step)
    end = step * np.ceil(y_max / step)
    ticks = np.arange(start, end + step, step)
    return ticks[(ticks >= y_min) & (ticks <= y_max)]

RISK_Y_LIMITS = (-140, 145)  # stesso asse per tutti i run
RISK_Y_TICK_STEP = 40
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
days = np.arange(1, len(daily["costs_planned"]) + 1)
planned = np.array(daily["costs_planned"])
realized = np.array(daily["costs_realized_total"])
cvar = np.array(daily["imbalance_costs_cvar"])

totrisk = planned +cvar
if max(totrisk) >= max(realized):
    print(f"max = {max(totrisk)}")
else:
    print(f"max = {max(realized)}")

print(f"min = {min(planned)}")
plt.fill_between(days, planned, planned + cvar, color='orange', alpha=0.3, label='Risk Boundary (DA + CVaR)')
plt.plot(days, planned, 'b--', marker='o', label='DA Plan')
plt.plot(days, realized, 'r-', marker='x', linewidth=2, label='Realized')
#plt.title(f"Timeline: Planned vs Realized ({opt_type}= {labels[Model_Name]})")# = {labels[Model_Name]}
plt.legend(loc="upper left")
plt.xlabel("Days")
plt.ylabel("Costs [€]")
plt.ylim(*RISK_Y_LIMITS)
plt.yticks(ticks_including_zero(*RISK_Y_LIMITS, step=RISK_Y_TICK_STEP))

plt.grid(True, alpha=0.3)
save_article_figure(fig, f"{opt_type}_Risk_Timeline")

#%% --- 3. PROFILI ORARI (Energy/Demand) ---
fig, axes = plt.subplots(4, 1, figsize=(7, 7.5), sharex=True, constrained_layout=True)
x_hours = range(len(hourly["energy_forecast"]))

# Nota: le chiavi nel JSON sono minuscole

# Copia dei dati per non modificare il dizionario originale (best practice)
theta_e_plot = np.array(hourly["theta_e"]).copy()
theta_d_plot = np.array(hourly["theta_d"]).copy()
e_forecast = np.array(hourly["energy_forecast"])
d_forecast = np.array(hourly["demand_forecast"])
e_theta = np.array(hourly["energy_modified"])
d_theta = np.array(hourly["demand_modified"])
e_obs = np.array(hourly["energy_observed"])
d_obs = np.array(hourly["demand_observed"])

# Sovrascrittura: se il forecast è vicino a zero, forza theta a 1
threshold = 20
theta_e_plot[e_forecast <= threshold] = 1.0
theta_d_plot[d_forecast <= threshold] = 1.0
print(f"mean_e_q50 = {np.sum(e_forecast)}")
print(f"mean_d_q50 = {np.sum(d_forecast)}")



print(f"mean_e_theta = {np.sum(e_theta)}")
print(f"mean_d_theta= {np.sum(d_theta)}")

print(f"mean_e_obs = {np.sum(e_obs)}")
print(f"mean_d_obs = {np.sum(d_obs)}")



axes[0].plot(x_hours, hourly["energy_forecast"], label=r"$P_{PV}^{q50}$", marker='.')
axes[0].plot(x_hours, hourly["demand_forecast"], label=r"$P_{Demand}^{q50}$", marker='.')
axes[0].set_title("a. Original Forecasts",fontsize=11, fontweight='bold')

axes[1].plot(x_hours, hourly["energy_modified"], label=r"$P_{PV}^{\theta}$", marker='.')
axes[1].plot(x_hours, hourly["demand_modified"], label=r"$P_{Demand}^{\theta}$", marker='.')
axes[1].set_title("b. Modified Forecasts",fontsize=11, fontweight='bold')

axes[2].plot(x_hours, hourly["energy_observed"], label=r"$P_{PV}^{realized}$", marker='.')
axes[2].plot(x_hours, hourly["demand_observed"], label=r"$P_{Demand}^{realized}$", marker='.')
axes[2].set_title("c. Observed Values",fontsize=11, fontweight='bold')

axes[3].plot(x_hours, theta_e_plot, label=r"$\theta_{PV}$", marker='.')
axes[3].plot(x_hours, theta_d_plot, label=r"$\theta_{Demand}$", marker='.')
axes[3].set_title("d. Forecast Modifications (Thetas)",fontsize=11, fontweight='bold')
axes[3].set_xlabel("Time [h]")

for ax in axes:
    if ax in axes[:-1]:
        ax.set_ylabel("Power [KW]")
        ax.legend(loc='upper left',fontsize=9, framealpha=0.85, borderpad=0.35, labelspacing=0.25, handlelength=1.6)
    else:
        ax.legend(loc='lower left',fontsize=9, framealpha=0.85, borderpad=0.35, labelspacing=0.25, handlelength=1.6)
    ax.grid(True)
# constrained_layout handles spacing for article figures
save_article_figure(fig, f"{opt_type}_Hourly_Profiles")

# --- 4. FLUSSI BATTERIA ---
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
for key, val in hourly["optimized_fluxes"].items():
    # Assegna al subplot 1 se è un flusso di storage, altrimenti al subplot 0
    ax_idx = 1 if "Storage" in key else 0
    axes[ax_idx].plot(x_hours, val, label=key)


axes[0].set_title("Direct Fluxes")
axes[1].set_title("Battery Fluxes")


for ax in axes:
    ax.legend(loc='upper right')
    ax.grid(True)
# constrained_layout handles spacing for article figures
save_article_figure(fig, f"{opt_type}_Battery_Management")
### validation plot
try:
    print("Generating Validation Plot...")
    
    # Estrazione dati (gestione errori se le chiavi mancano)
    g_plan = hourly.get("grid_exchange", hourly.get("grid_exchange_planned")) # Fallback se non trovi la chiave esatta
    g_real = hourly.get("grid_realized", hourly.get("Grid_Exchange_realized"))
    s_plan = hourly.get("soc", hourly.get("soc"))
    s_real = hourly.get("soc_realized", hourly.get("Realized_SOC")) # Se non hai il realizzato, usa il piano per non rompere il codice
    imb_vol = hourly.get("imbalance_volumes", hourly.get("imbalance_volumes"))
    
    # Chiamata alla funzione
    val_fig = plot_validation_results(
        grid_forecast=g_plan,
        grid_realized=g_real,
        soc_forecast=s_plan,
        soc_realized=s_real,
        imbalance_volumes = imb_vol
    )
    
    #val_fig.suptitle(f"Validation: {start_str} to {end_str} ({opt_type})", fontsize=14)
    
    img_name = f"{opt_type}_Validation_Extended.png"
    save_article_figure(val_fig, f"{opt_type}_Validation_Extended")
    print(f"✅ Validation plot saved: {img_name}")

except Exception as e:
    print(f"⚠️ Errore durante la creazione del validation plot: {e}")


plt.show()
print(f"📊 All plots generated successfully from JSON data in: {fig_dir}")
