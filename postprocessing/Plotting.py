#import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from validationplot import plot_validation_results, comulative_results
import json
from datetime import datetime

start_date_plot = datetime(2024,7,1) # Choice how many days you want to load as Data and to train
end_date_plot = datetime(2024,7,31)
Opt_Type = "Deterministic"  #Type of Optimization ['Deterministic', 'Parametric']
#"gamma_1_cheby"
models = ["gamma_zscore0_punto_3","gamma0punto4_optuna","gamma0punto5_optuna","gamma0punto6_optuna","gamma_zscore0_punto_7",]
Model_Name = models[2]

# --- CONFIGURAZIONE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(current_dir, 'results')
start_str = start_date_plot.strftime('%d-%m-%Y')
end_str   = end_date_plot.strftime('%d-%m-%Y')


if Opt_Type == "Parametric":
    JSON_FILE = f"Results_{start_str}_to_{end_str}_{Opt_Type}_{Model_Name}.json"
else:
    JSON_FILE = f"Results_{start_str}_to_{end_str}_{Opt_Type}.json"

json_path = os.path.join(json_dir, JSON_FILE)

main_dir = os.path.dirname(current_dir)
save_fig_dir = os.path.join(main_dir, 'figure', '_Code_repository')

os.makedirs(save_fig_dir, exist_ok=True)

def save_article_figure(fig, filename):
    """Save cropped figures for LaTeX/Overleaf in vector and raster formats."""
    fig.savefig(os.path.join(save_fig_dir, f"{filename}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(save_fig_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0.02, dpi=300)

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
# plt.figure(figsize=(10, 6))
# cum_costs = np.cumsum(daily["costs_realized_total"])
# plt.plot(range(1, len(cum_costs)+1), cum_costs, color='green', marker='o', linewidth=2.5)
# plt.axhline(0, color='black', alpha=0.3)
# plt.title(f"Andamento Economico Cumulato ({opt_type})")
# plt.xlabel("Giorno")
# plt.ylabel("Euro Cumulati (€)")
# plt.grid(True, alpha=0.3)
fig = comulative_results(opt_type,models,start_str,end_str,json_dir)
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
print(f"📊 All plots generated successfully from JSON data in: {save_fig_dir}")
# # %% ###########################################################     PLOTTING RESULTS     ###################################################################################################### 
#     number_days=(end_date_plot-start_date_plot).days + 1
#     start_str = start_date_plot.strftime('%d-%m-%Y')
#     end_str   = end_date_plot.strftime('%d-%m-%Y')
#     if number_days > len(DATA):
#         number_days = len(DATA)
#         print(f"⚠️  Not enough Data. only {number_days} days plotted ⚠️ ")

#     #validation

#     validation_fig =plot_validation_results(data, Grid_Exchange_day, grid_realized, soc_day, Realized_SOC_day, imbalance)
#     plt.tight_layout(pad=2.0)
#     img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_validation_fig.png"
#     plt.savefig(os.path.join(save_fig_dir,img_name))
#     ############################################################
# ########################  PROFITTO CUMULATO NEL TEMPO  ########################
#     plt.figure(figsize=(10, 6))
    
#     # Calcoliamo la somma cumulata dei costi realizzati
#     # Ricorda: essendo costi, se il valore scende (diventa più negativo) stai accumulando profitto
#     cumulative_costs = np.cumsum(Costs_Realized_Total)
    
#     days_range = np.arange(1, len(cumulative_costs) + 1)
    
#     plt.plot(days_range, cumulative_costs, label=f"Modello: {Opt_Type}", 
#              color='green', linewidth=2.5, marker='o', markersize=4)

#     # Aggiungiamo una linea orizzontale sullo zero per distinguere area profitto/perdita
#     plt.axhline(0, color='black', linestyle='-', alpha=0.3)

#     plt.title(f"Andamento Economico Cumulato ({start_str} - {end_str})", fontsize=14)
#     plt.xlabel("Giorno di Simulazione", fontsize=12)
#     plt.ylabel("Euro Cumulati (€)", fontsize=12)
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
#     # Inseriamo un testo con il valore finale per renderlo immediato
#     plt.text(len(cumulative_costs), cumulative_costs[-1], f" Totale: {cumulative_costs[-1]:.2f}€", 
#              verticalalignment='center', fontweight='bold')

#     # constrained_layout handles spacing for article figures
#     plt.savefig(os.path.join(save_fig_dir, "Single_Model_Cumulative_Performance.png"))
#  ########################  EVOLUZIONE GIORNALIERA DEL RISCHIO  ########################
#     fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    
#     # Creiamo l'asse X (i giorni)
#     days_range = np.arange(1, len(Costs_) + 1)
    
#     # Trasformiamo le liste in array per operazioni matematiche veloci
#     planned = np.array(Costs_)
#     realized = np.array(Costs_Realized_Total)
#     cvar_risk = np.array(Imbalance_Costs_CVaR)
    
#     # Area di confidenza: dal piano DA fino al rischio massimo stimato (CVaR)
#     plt.fill_between(days_range, planned, planned + cvar_risk, 
#                      color='orange', alpha=0.3, label='Fascia di Rischio Stimata (DA + CVaR)')

#     # Linea del Piano Day-Ahead
#     plt.plot(days_range, planned, 'b--', marker='o', label='Piano DA (Target)')

#     # Linea del Realizzato
#     plt.plot(days_range, realized, 'r-', marker='x', linewidth=2, label='Costo Reale Realizzato')

#     plt.title(f"Performance Giornaliera: Piano vs Realizzato ({start_str} - {end_str})")
#     plt.xlabel("Giorno di Simulazione")
#     plt.ylabel("Euro (€)")
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.6)
#     img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_Daily_Risk_Timeline.png"
#     plt.savefig(os.path.join(save_fig_dir, img_name))

# ####################################################################

#     # #scenario difference distribution
#     # n_subplot = 2
#     # fig, axes = plt.subplots(n_subplot, 1, figsize=(8, 10))
#     # #VAR_imb =
#     # Da_plan = sum(Costs_)
#     # I_cost_obs = sum(Imbalance_Cost_Observed)
#     # I_Cvar,I_var,_ = RightCVaR(Imbalance_costs_scenarios,alpha=0.95)
#     # sns.histplot(Imbalance_costs_scenarios, kde=True, stat="frequency", ax=axes[0], color='blue', edgecolor='black')
#     # axes[0].axvline(I_cost_obs, color='red', linestyle='--', linewidth=2, label=f'Imbalance Costs Observed: {I_cost_obs:.2f}€')
#     # axes[0].axvline(I_var, color='green', linestyle='--', linewidth=2, label=f'Imbalance Costs VAR: {I_var:.2f}€')
#     # axes[0].axvline(I_Cvar, color='blue', linestyle='--', linewidth=2, label=f'Imbalance Costs CVAR: {I_Cvar:.2f}€')
 
#     # #axes[1].axvline(Da_plan, color='red', linestyle='--', linewidth=2, label=f'DA_Plan Total: {Da_plan:.2f}€')
#     # axes[0].set_title("Imbalance Costs")
#     # axes[0].set_xlabel("euro")
#     # axes[0].set_ylabel("Frequency")  # if you meant "Densità", use density=True
#     # axes[0].grid(True,linestyle=':', alpha=0.7)
#     # axes[0].legend() # Necessario per mostrare la label della linea

#     # sns.histplot(Costs_scenarios, kde=True, stat="frequency", ax=axes[1], color='green', edgecolor='black')
#     # # Aggiunta linea verticale
#     # axes[1].axvline(Da_plan, color='red', linestyle='--', linewidth=2, label=f'DA_Plan Total: {Da_plan:.2f}€')
#     # axes[1].set_title("Scenario Costs Distribution")
#     # axes[1].set_xlabel("euro")
#     # axes[1].set_ylabel("Frequency")  # if you meant "Densità", use density=True
#     # axes[1].grid(True,linestyle=':', alpha=0.7)
#     # axes[1].legend()

#     # plt.tight_layout(pad=2.0)
#     # img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_Distribution.png"
#     # plt.savefig(os.path.join(save_fig_dir,img_name))

#     ### Plotting of the Energy and Demand DATA ###
#     n_subplot = 4
#     fig, axes = plt.subplots(n_subplot, 1, figsize=(6, 10))


#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Energy_forecast, label="PV Energy forecast",ax=axes[0],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Demand_forecast, label= "Demand Energy forecast",  ax=axes[0],marker = 'o')

#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Energy_forecast_modified, label= "PV Energy modified",  ax=axes[1],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Demand_forecast_modified, label= "Demand modified",  ax=axes[1],marker = 'o')

#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Energy_observed, label="PV Energy observed",ax=axes[2],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Demand_observed, label="Demand observed",ax=axes[2],marker = 'o')

#     sns.lineplot(x=range(1,number_days*timesteps+1), y=theta_e, label="theta_e",ax=axes[3],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=theta_d, label="theta_d",ax=axes[3],marker = 'o')
#     # if Opt_Type == "Parametric":
#     #     sns.lineplot(x=range(1,number_days*timesteps+1), y=out_e, label="out_e",ax=axes[3],marker = 'o')
#     #     sns.lineplot(x=range(1,number_days*timesteps+1), y=out_d, label="out_d",ax=axes[3],marker = 'o')
#     titles = ["Original Forecasts","Modified Forecasts","Observed Values","thetas"]
#     units  = ["kW","kW","kW",""]

#     for ax in range(len(axes)):
#         axes[ax].set_title(titles[ax])
#         axes[ax].set_ylabel(units[ax])
#         axes[ax].grid(True)
#         axes[ax].set_xlabel("time")
        
#     #### put the same scale
#     ymin = min(
#         min(Energy_forecast), min(Demand_forecast),
#         min(Energy_forecast_modified), min(Demand_forecast_modified),
#         min(Energy_observed), min(Demand_observed)
#     )
#     ymax = max(
#         max(Energy_forecast), max(Demand_forecast),
#         max(Energy_forecast_modified), max(Demand_forecast_modified),
#         max(Energy_observed), max(Demand_observed)
#     )

#     for ax in axes[:3]:
#         ax.set_ylim(ymin, ymax+10)


#     plt.tight_layout(pad=2.0)
#     img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_forecasts.png"
#     plt.savefig(os.path.join(save_fig_dir,img_name))
   

#     ### Plotting of the Optimized Energy Fluxes ###

#     fig, axes = plt.subplots(3, 1, figsize=(6, 10))

#     for key in Optimized_fluxes.keys():
#         if "Storage" in key:
#             sns.lineplot(x=range(1,number_days*timesteps+1), y=Optimized_fluxes[key], label=key,ax=axes[1], marker = "o")
#         else:
#             sns.lineplot(x=range(1,number_days*timesteps+1), y=Optimized_fluxes[key], label=key,ax=axes[0], marker = "o")

#     sns.lineplot(x=range(1,number_days*timesteps+1), y=soc, label="SoC",ax=axes[2], marker = 'o')

#     titles = ["Direct Fluxes","Battery Fluxes","SOC"]
#     units  = ["kW","kW",""]

#     for ax in range(len(axes)):
#         axes[ax].set_title(titles[ax])
#         axes[ax].set_xlabel("time")
#         axes[ax].set_ylabel(units[ax])
#         axes[ax].grid(True)

#     plt.tight_layout(pad=2.0)
#     img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_BatteryManagement.png"
#     plt.savefig(os.path.join(save_fig_dir,img_name))
#      ### Grid exchange ####
#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Grid_Exchange, label="Grid Exchange",ax=ax, marker = 'o')
#     ax.set_title("Grid Exchange")
#     ax.set_xlabel("time")
#     ax.set_ylabel("kWh")
#     ax.grid(True)
#     img_name = f"from_{start_str}_to_{end_str}_{Opt_Type}_GridExchange.png"
#     plt.savefig(os.path.join(save_fig_dir,img_name))
#      ### Prices ###
#     fig, axes = plt.subplots(3, 1, figsize=(6, 10))
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=Market_Price_sell, label="Market Price",ax=axes[0],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=positive_price_imbalance, label="imbalance Price +",ax=axes[1],marker = 'o')
#     sns.lineplot(x=range(1,number_days*timesteps+1), y=negative_price_imbalance, label="imbalance Price -",ax=axes[2],marker = 'o')

#     titles = ["Market Price","imbalance Price +","imbalance Price -"]
#     units  = ["€/kWh","€/kWh","€/kWh"]
#     for ax in range(len(axes)):
#         axes[ax].set_title(titles[ax])
#         axes[ax].set_ylabel(units[ax])
#         axes[ax].grid(True)
#         axes[ax].set_xlabel("time")
#     plt.tight_layout(pad=2.0)
#     img_name = f"ffrom_{start_str}_to_{end_str}_Prices.png"
#     plt.savefig(os.path.join(save_fig_dir,img_name))
   
#     plt.show()

