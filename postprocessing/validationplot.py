import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_validation_results( grid_forecast, grid_realized,soc_forecast, soc_realized,imbalance_volumes,max_soc = 90, min_soc=30):
    
    # Setup asse temporale
    timesteps = len(grid_forecast)
    time = np.arange(timesteps)
    
    # Creazione della figura con 3 subplot (Grid, SoC, Imbalance)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True, constrained_layout=True)

    
    step_ore = 6 
    ticks_x = np.arange(0, timesteps + 1, step_ore)
    
    grid_Y_LIMITS = (-200, 500)  # stesso asse per tutti i run
    
    imb_Y_LIMITS = (-600, 0)  # stesso asse per tutti i run
    
    axes = [ax1, ax2, ax3]
    for i, ax in enumerate(axes):
        # --- COMANDI PER I NUMERI SU TUTTI GLI ASSI ---
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(ticks_x, fontsize=8)
        
        # Rendiamo i tick visibili anche se i grafici sono sovrapposti
        ax.tick_params(axis='x', which='both', labelbottom=True)
        
        # Label solo sull'ultimo grafico per pulizia
        if i == len(axes) - 1:
            ax.set_xlabel("Time [h]")
        
        # --- GRIGLIA E LINEE VERTICALI ---
        ax.grid(True, alpha=0.3)
        for h in ticks_x:
            ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8, zorder=0)
    
    step_day = 24 
    ticks_x_day = np.arange(0, timesteps + 1, step_day)
    for ax in axes:
        for h in ticks_x_day:
            ax.axvline(x=h, color='black', linestyle='--', alpha=0.8, linewidth=1, zorder=0)
    # --- SUBPLOT 1: GRID EXCHANGE (Il Target vs La Realtà) ---
    ax1.set_title("A. Grid Exchange Commitment vs Realized", fontsize=11, fontweight='bold')
    ax1.plot(time, grid_forecast, color='black', linestyle='--', linewidth=2, label='DA Plan')
    ax1.plot(time, grid_realized, color='blue', linewidth=1.5, alpha=0.8, label='Realized')
    ax1.set_ylim(*grid_Y_LIMITS)
    
    # Evidenzia le aree dove il target non è stato raggiunto
    ax1.fill_between(time, grid_forecast, grid_realized, color='red', alpha=0.3, label='Tracking Error')
    
    ax1.set_ylabel("Power [kW]")
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.85, borderpad=0.35, labelspacing=0.25, handlelength=1.6)
    ax1.grid(True, alpha=0.3)

    # --- SUBPLOT 2: STATE OF CHARGE (Il Drift) ---
    ax2.set_title("B. Battery SoC: Planned vs Realized Trajectory", fontsize=11, fontweight='bold')
    ax2.plot(time, 100*np.array(soc_forecast), color='grey', linestyle='--', linewidth=2, label='Planned SoC')
    ax2.plot(time, 100*np.array(soc_realized), color='green', linewidth=2, label='Realized SoC')
    
    # Aggiungi linee orizzontali per i limiti della batteria
    
    # Se il tuo SOC è in percentuale (0-1), usa questi:
    ax2.axhline(y=max_soc, color='r', linestyle=':', alpha=0.5, label='SoC Limits')
    ax2.axhline(y=min_soc, color='r', linestyle=':', alpha=0.5)
    
    ax2.set_ylabel("SoC [%]")
    ax2.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.85, borderpad=0.35, labelspacing=0.25, columnspacing=0.8, handlelength=1.6)
    ax2.grid(True, alpha=0.3)

    # --- SUBPLOT 3: IMBALANCE VOLUME (Lo Slack) ---
    ax3.set_title("C. Imbalance Volumes", fontsize=11, fontweight='bold')
    
    # Bar chart per gli sbilanciamenti
    imbalance_volumes = np.array(imbalance_volumes)
    print(f"imbalnce = {sum(imbalance_volumes)}")
    positive_mask = imbalance_volumes >= 0
    negative_mask = imbalance_volumes < 0
    #ax3.bar(time[positive_mask], imbalance_volumes[positive_mask], color='orange', alpha=0.7, label='Positive imbalance')
    ax3.bar(time[negative_mask], imbalance_volumes[negative_mask], color='red', alpha=0.7, label='Negative imbalance')
    ax3.set_ylim(*imb_Y_LIMITS)
    
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.set_ylabel("Imbalance [kWh]")
    ax3.set_xlabel("Time [h]")
    ax3.legend(loc='lower left', fontsize=8, framealpha=0.85, borderpad=0.35, labelspacing=0.25, handlelength=1.6)
    ax3.grid(True, alpha=0.3)

    # # All'interno della funzione plot_validation_results, dopo aver definito 'timesteps'
    # for h in range(0, timesteps, 24):
    #     # Aggiunge la linea su ax1 (Grid)
    #     ax1.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    #     # Aggiunge la linea su ax2 (SoC)
    #     ax2.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

    #plt.suptitle(f"Active Validation Results (Scenario: {data.Simulation_Data.get('scenario_name', 'Generic')})", fontsize=14)
    return fig

def comulative_results(opt_type,models,start_str,end_str,json_dir):
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    plot_count= 0
    baseline_x = baseline_cum = None
    perf_x = perf_cum = None


    labels = {#models[0]:"min_max average lambda",
              models[0]:r"$\lambda = 0.3$",
              models[1]:r"$\lambda = 0.4$",
              models[2]:r"$\lambda = 0.5$",
              models[3]:r"$\lambda = 0.6$",
              models[4]:r"$\lambda = 0.7$"}
    colors = {
            models[0]: "#1f77b4",  # blue
            models[1]: "#2ca02c",  # green
            models[2]: "#d62728",  # red
            models[3]: "#9467bd",
            models[4]: "#E30BEB"
        }
    ########## PArametriccc ######################################


    for model_name in models:
       
        filename = f"Results_{start_str}_to_{end_str}_Parametric_{model_name}.json"
  
        path = os.path.join(json_dir, filename)

        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Estrazione e calcolo cumulata
            daily_costs = np.array(data["daily_data"]["costs_realized_total"])
            cum_costs = np.cumsum(daily_costs)
            
            # Plot della linea per il modello corrente
            plt.plot(range(1, len(cum_costs) + 1), cum_costs, label=labels[model_name],color=colors[model_name] ,marker='x', linewidth=2)
            plot_count += 1
        else:
            print(f"⚠️ Salto modello {model_name}: file non trovato in {path}")


    ##################################BAseline########################
    filename = f"Results_{start_str}_to_{end_str}_Deterministic.json"
    path = os.path.join(json_dir, filename)

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    # Estrazione e calcolo cumulata
        daily_costs = np.array(data["daily_data"]["costs_realized_total"])
        cum_costs = np.cumsum(daily_costs)
        
        # Plot della linea per il modello corrente
        baseline_x = np.arange(1, len(cum_costs) + 1)
        baseline_cum = cum_costs
        plt.plot(baseline_x, baseline_cum, label="Baseline q50", color="#000000", linestyle="--", linewidth=2)
        plot_count += 1
    
    filename = f"Results_{start_str}_to_{end_str}_Deterministic_perf.json"
    path = os.path.join(json_dir, filename)

    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        daily_costs = np.array(data["daily_data"]["costs_realized_total"])
        cum_costs = np.cumsum(daily_costs)
        perf_x = np.arange(1, len(cum_costs) + 1)
        perf_cum = cum_costs
        plt.plot(perf_x, perf_cum, label="Perfect Forecast", color="#8B0000", linestyle="--", linewidth=2)
        plot_count += 1
    else:
        print(f"Warning: Deterministic_perf file not found in {path}")


    # fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    # cum_costs = np.cumsum(daily_costs)
    #plt.plot(range(1, len(cum_costs)+1), cum_costs, color='green', marker='o', linewidth=2.5)
    if baseline_cum is not None and perf_cum is not None:
        common_len = min(len(baseline_cum), len(perf_cum))
        plt.fill_between(
            baseline_x[:common_len],
            baseline_cum[:common_len],
            perf_cum[:common_len],
            color="orange",
            alpha=0.3,
            zorder=0,
        )

    if plot_count > 0:
        # 'best' sceglie l'angolo più vuoto del grafico
        # frameon=True aggiunge un riquadro per renderla più leggibile
        plt.legend(loc='best', fontsize='small', frameon=True, shadow=True)

    plt.axhline(0, color='black', alpha=0.3)

    plt.xlabel("day")
    plt.ylabel("Costs (€)")
    plt.grid(True, alpha=0.3)
    return fig
