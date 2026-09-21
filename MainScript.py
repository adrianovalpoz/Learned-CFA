import os
import sys
import json
import itertools
import numpy as np
import torch
import optuna
import time
import random
from optuna.samplers import TPESampler
from datetime import datetime, timedelta
from learned_cfa_src.evolution_strategy.Policy import PolicyNetwork
from learned_cfa_src.milp.Optimize_Energy_Flux import optimize_energy_flux
from learned_cfa_src.Optimize_Risk  import Optimize_Risk , Real_Loss_Assessment                                                                                                                                                                                                                                                                                                                                                                             
from learned_cfa_src.evolution_strategy.es_Policy import Evolution_Strategy_Training_HPC,Theta_Bounds,Theta_Choice
from learned_cfa_src.evolution_strategy.NN_sizing import NN_opt_objective
from learned_cfa_src.data_pipeline.Problem_Data   import Problem_Data
# Directory path creation
from learned_cfa_src.directory_names import base_dir, save_NN_dir, results_dir, json_repository_dir, NN_size_dir, config_dir

########## inizialize the seeds ###################
sid = 101 #42, 101, 1337, 2025]:
torch.manual_seed(sid)
np.random.seed(sid)
random.seed(sid)
fixed_sampler = TPESampler(seed=sid)

start_time = time.time()
# %% ########################################################   SIMULATION CONTROL INPUTS  ########################################################################################################################################################

start_date = datetime(2024,7,1) # Choice how many days you want to load as Data and to train
end_date = datetime(2024,7,31)
timesteps  = 24       # Dont Change this

##### Choice of the Optimization type #####

Opt_Type   = 'Learned_CFA'     # Type of Optimization # ['Deterministic', 'Static_CFA', 'Learned_CFA']
Data_Type = "forecast"        # Type of Determinsitic Data ['Observed', 'forecast']

##### Learned-CFA inputs ######

Model_Name = "gamma0punto5_optuna"
input_size  = 10        # Number of Input Features for the NN Model [mean E, variance E, mean D, variance D, sin(T), cos(T)]
output_size = 2        # Number of Output Features for the NN Model [theta E, theta D] 

##### TRAINIG CONTROL #####

train_months = [1,3,4,5,6,8,9,11,12] #1=jenuary 12 = december
NN_Training = 'n'     # put y If want to train the NN with the load data
                       # otherwise put n if you want to use a trained NN for the optimization
NN_Optimization = 'n' # put y If want to optimize the NN size and hyperparameters
                       # otherwise put n if you want to use a already sized NN

# %% ################################################## IMPORT DATA FROM JSON AND DATA OBJECT CREATION  ##################################################################################
if __name__ == '__main__':
    ## section to load the Json files for the data of the days considered ##
    # data folder directories
    data_json_dir  =  os.path.join(base_dir, 'Data','json_repository_for_simulation','SystemData')
    scenario_json_dir  =  os.path.join(base_dir, 'Data','json_repository_for_simulation','ScenarioData')

    # Get list of all .json files in the directory
    data_json_files = [file for file in os.listdir(data_json_dir) if file.endswith('.json')]
    scenario_json_files = [file for file in os.listdir(scenario_json_dir) if file.endswith('.json')]
    # load .Json Statistics of DATAFRAME
    Data_Statistics_path = os.path.join(base_dir, 'Data','json_repository_for_simulation','normalization_stats.json')
    with open(Data_Statistics_path, "r") as file:
        Data_Statistics = json.load(file)

    # load .Json training hyperparameters (edit config/training_hyperparameters.json to change them)
    with open(os.path.join(config_dir, 'training_hyperparameters.json'), "r") as file:
        hyperparameters = json.load(file)

    Evolution_Params = {key: hyperparameters[key] for key in ("population_size", "sigma", "alpha", "n_iterations", "weight_decay")}
    NN_Structure = {"N_layers": hyperparameters["n_layers"], "Hidden_size": hyperparameters["hidden_dim"]}
    # Dictionary to store contents

    DATA = {}

    # Load each JSON file and create the problem_Data from them
    date = start_date

    while date <= end_date:
        # 2. WHITELIST: Se stiamo addestrando o ottimizzando la NN, salta tutto ciò che non è in train_months
        if Opt_Type == 'Parametric' and (NN_Training == 'y' or NN_Optimization == 'y'):
            if date.month not in train_months:
                date += timedelta(days=1)
                continue # Salta al giorno successivo senza leggere il JSON
        y = date.strftime('%Y')
        m = date.strftime('%m')
        d = date.strftime('%d')
        day= f'{y}-{m}-{d}'
        data_found = False
        Scenario_found = False
        for filename in data_json_files:
            if filename.endswith(f"_{day}.json"):
                file_path = os.path.join(data_json_dir, filename)
                with open(file_path, 'r') as file:
                    Data = json.load(file)
                data_found = True

        for filename in scenario_json_files:
            if filename.endswith(f"_{day}.json"):
                file_path = os.path.join(scenario_json_dir, filename)
                with open(file_path, 'r') as file:
                    scenario_Milps = json.load(file)
                Scenario_found = True

        if data_found == True & Scenario_found == True:
            print(f"[OK] {day} Loaded.")#✅
            # Creation of a Problem_Data OBJ for each day
            DATA[day] = Problem_Data(Data["Battery"],Data["Simulation"],Data["Fluxes"],Data["Prices"],Data["Energy"],Data["Demand"],scenario_Milps["Costs"],scenario_Milps["Grid_Exchange"],scenario_Milps["Grid_Exchange_Baseline"], scenario_Milps["C_baseline"],scenario_Milps["Best_DA"],scenario_Milps["Worst_DA"],scenario_Milps["Best_ICVaR"],scenario_Milps["Worst_ICVaR"])
        else:
            print(f"[WARNING]  {day} Data not Found.") #⚠️
        date+=timedelta(days=1) # needed to progress


    if Opt_Type == 'Learned_CFA':
        dummy_vec = torch.zeros(4)
        policy = PolicyNetwork(input_size, output_size, dummy_vec, dummy_vec, NN_Structure["Hidden_size"], NN_Structure["N_layers"])
        policy.load_state_dict(torch.load(os.path.join(save_NN_dir, f"{Model_Name}.pth")))
        policy.eval()
    elif Opt_Type == 'Static_CFA':
        with open(os.path.join(save_NN_dir, 'static_cfa.json'), "r") as file:
            policy = json.load(file)
    else:
        policy = None # Deterministic: nothing to load

    
    # %% ##########################################################    OPTIMIZATION SECTION     ###########################################################################################################
    # %% ################################################################### TRAINING SECTION ##############################################################################################################################################################################

    if Opt_Type == 'Learned_CFA' and NN_Optimization == 'y':
        print("NN Optimization Start...")

        study = optuna.create_study(direction="maximize",sampler=fixed_sampler,pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
        study.optimize(lambda trial: NN_opt_objective(trial, DATA, Data_Statistics["mu"], Data_Statistics["sigma"], Data_Type, Model_Name), n_trials=50)
        print("Migliori parametri:", study.best_params)

        save_best_hp_dir = os.path.join(base_dir,'evolution_strategy','NN_Size','best_hyperparameters.json')

        with open(save_best_hp_dir, "w") as f:
            json.dump(study.best_params, f, indent=4)
        print("Parametri salvati in 'best_hyperparameters.json'")
        sys.exit("NN_Optimization completed — stopping script.")

    if Opt_Type == 'Learned_CFA' and NN_Training == 'y':
        mu_for_norm = torch.tensor(Data_Statistics["mu"], dtype=torch.float32)
        sigma_for_norm = torch.tensor(Data_Statistics["sigma"], dtype=torch.float32)
        model = PolicyNetwork(input_size, output_size,mu_for_norm,sigma_for_norm,NN_Structure["Hidden_size"],NN_Structure["N_layers"])             # NN Initializationl

 
        os.makedirs(save_NN_dir, exist_ok=True)
        start_time = time.perf_counter()
        Evolution_Strategy_Training_HPC(Optimize_Risk,model,DATA,Data_Type,Evolution_Params,Model_Name,save_NN_dir,training_check_dir)    # Training function Call
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"tempo totale: {elapsed_time:.2f} secondi ({elapsed_time/60:.2f} minuti)")
        sys.exit("Training completed — stopping script.")


  
    # %% ################################################## Run of the Energy Flux Optimization with the values thetas ############################################################################################################

    # results storage inizialization:
    Energy_forecast          = []
    Energy_forecast_modified = []
    Energy_observed          = []
    Demand_forecast          = []
    Demand_forecast_modified = []
    Demand_observed          = []
    theta_e                  = []
    out_e                    = []
    theta_d                  = []
    out_d                    = []
    soc                      = []
    Realized_SOC             = []
    Market_Price_sell        = []
    Optimized_fluxes         = {}
    Costs_                   = []
    Costs_CVaR               = []
    Costs_Observed           = []
    Costs_Realized_Total     = []
    Imbalance_Costs_CVaR     = []
    Imbalance_Cost_Observed  = []
    imbalance_volumes        = []
    negative_price_imbalance = []
    positive_price_imbalance = []
    Grid_Exchange            = []
    Grid_Exchange_realized   = []
    Costs_diff_scenarios = None
    Imbalance_costs_scenarios = None
    Costs_scenarios = None
    inference_times_daily = []
    infernece_time_daily_real= []
    optimization_times_daily = []
    ## Loop over the days --> the optimization is done for each of it
    date = start_date
    
    while date <= end_date:
        y = date.strftime('%Y')
        m = date.strftime('%m')
        d = date.strftime('%d')
        day = f'{y}-{m}-{d}'
        if day not in DATA:
            print(f"⚠️ ⚠️ ⚠️  {day} Data not Found. Loop Broken ⚠️ ⚠️ ⚠️")
            break 
        ## Bounds theta computation
        theta_max_e,theta_min_e,theta_max_d,theta_min_d = Theta_Bounds(DATA[day])
        # theta_e_manual = theta_min_e
        # theta_d_manual = theta_max_d
        ## theta computations 
        t_start_ann = time.perf_counter()
        theta_e_day,theta_d_day,out_e_day,out_d_day,timer = Theta_Choice(DATA[day],Opt_Type,policy)#ThetaChoice,theta_e_manual,theta_d_manual,input_size,output_size,Model_Name,NN_Structure,save_NN_dir)
        t_end_ann = time.perf_counter()
        infernece_time_daily_real.append(timer)
        inference_times_daily.append(t_end_ann - t_start_ann)

        if date != start_date:
            DATA[day].Battery_Data["SoC_in"] = SoC_end # condition needed to assure that the next day the status of the storage is the same as the end of prevous day
    
        ## Energy Fluxes Optimization

        SoC_in = DATA[day].Battery_Data["SoC_in"]
        print(f"Daily Storage Initial SoC is {SoC_in}")
        t_start_opt = time.perf_counter()
        Costs_day,Optimized_Fluxes_day,soc_day,_,_,data,timesteps,Energy_forecast_day,Energy_forecast_modified_day,Demand_forecast_day,Demand_forecast_modified_day,Market_Price_sell_day,Grid_Exchange_day,_ = optimize_energy_flux(DATA[day],theta_e_day,theta_d_day,Data_Type)
        t_end_opt = time.perf_counter()
        optimization_times_daily.append(t_end_opt - t_start_opt)
        
    
        ## Risk Assessment

        RiskCosts_day,_,Imbalance_scenarios,Costs_diff_scenarios_day,Costs,Costs_CVaR_day,Imbalance_Costs_CVaR_day,Imbalance_costs_scenarios_day =  Optimize_Risk(DATA[day],theta_e_day,theta_d_day,Data_Type)
        if Costs != Costs_day:
            print("Costs different")
            print(f"cost from energy flux = {Costs_day} | Costs from Obj Risk = {Costs}")


        ## Economic Loss assessment
        Imbalance_Cost_day, Costs_observed_day , _ ,Energy_observed_day,Demand_observed_day,Realized_SOC_day,grid_realized_day,imbalance_day,Costs_Realized_Total_day = Real_Loss_Assessment(DATA[day],theta_e_day,theta_d_day,Data_Type)
        
        #SoC_end = soc_day[-1] # condition needed to assure that the next day the status of the storage is the same as the end of prevous day
        SoC_end = Realized_SOC_day[-1]
        ## appending results for each day
        if Costs_diff_scenarios is None:
            Costs_diff_scenarios = Costs_diff_scenarios_day.copy()
        else:
            Costs_diff_scenarios += Costs_diff_scenarios_day

        if Imbalance_costs_scenarios is None:
         Imbalance_costs_scenarios = Imbalance_costs_scenarios_day.copy()
        else:
            Imbalance_costs_scenarios += Imbalance_costs_scenarios_day

        Costs_scenarios_Day= np.array(DATA[day].Costs_Milps)
        if Costs_scenarios is None:
            Costs_scenarios = Costs_scenarios_Day.copy()
        else:
            Costs_scenarios += Costs_scenarios_Day

        Energy_forecast.append(Energy_forecast_day) 
        Energy_forecast_modified.append(Energy_forecast_modified_day) 
        Energy_observed.append(Energy_observed_day)
        Demand_forecast.append(Demand_forecast_day) 
        Demand_forecast_modified.append(Demand_forecast_modified_day) 
        Demand_observed.append(Demand_observed_day)
        theta_e.append(theta_e_day) 
        out_e.append(out_e_day)
        theta_d.append(theta_d_day)
        out_d.append(out_d_day)
        soc.append(soc_day)
        Realized_SOC.append(Realized_SOC_day)
        Market_Price_sell.append(Market_Price_sell_day)
        Costs_.append(Costs_day)
        Costs_CVaR.append(Costs_CVaR_day)
        Costs_Observed.append(Costs_observed_day)
        Costs_Realized_Total.append(Costs_Realized_Total_day)
        Imbalance_Costs_CVaR.append(Imbalance_Costs_CVaR_day)
        Imbalance_Cost_Observed.append(Imbalance_Cost_day)
        imbalance_volumes.append(imbalance_day)
        positive_price_imbalance.append(DATA[day].Prices["Imbalance_positive"])
        negative_price_imbalance.append(DATA[day].Prices["Imbalance_negative"])
        Grid_Exchange.append(Grid_Exchange_day)
        Grid_Exchange_realized.append(grid_realized_day)
        for key in Optimized_Fluxes_day.keys():
            if key not in Optimized_fluxes:
                Optimized_fluxes[key] = []
            Optimized_fluxes[key].append(Optimized_Fluxes_day[key])

        ## Optimized reward Log for each day
        print(f"__________________________DAY {day}__________________________________________________________________________")
        print("_______________________COST ANALYSIS:_________________________________________________________________________")
        print(f"Optimized Costs: {Costs_day:.2f} | Costs CVaR: {Costs_CVaR_day:.2f} | Imbalance_Costs_CVaR: {Imbalance_Costs_CVaR_day:.2f} | Observed Costs: {Costs_observed_day:.2f} | Imbalance Costs Observed: {Imbalance_Cost_day:.2f} ")
        print("_______________________ENERGY ANALYSIS:_______________________________________________________________________")
        #print(f"DAY {day} --> Optimized Costs: {Costs_day:.2f} | Costs CVaR: {Costs_CVaR_day:.2f} | Imbalance_Costs_CVaR: {Imbalance_Costs_CVaR_day:.2f}  ")
        print(f" Positive Imbalance: {sum([x for x in imbalance_day if x > 0]):.2f} | Negative Imbalance: {sum([x for x in imbalance_day if x < 0]):.2f}  ")
        print("____________________________________________________________________________________________________________")  
        date += timedelta(days=1)
    ## Optimized reward Log for the whole timespan
    print(f"\033[32m__________________________TOTAL RESULTS: from {start_date} to {end_date}________________________________________________________\033[0m")
    print("\033[32m_____________________________________COST ANAYSIS_____________________________________________________________\033[0m")
    print(f"\033[32mOptimized DA Costs : {sum(Costs_):.2f} | Costs CVaR: {sum(Costs_CVaR):.2f} | Imbalance_Costs_CVaR: {sum(Imbalance_Costs_CVaR):.2f}\033[0m")
    print(f"\033[32mPerfect Forecast Costs: {sum(Costs_Observed):.2f} | Imbalance Costs Observed: {sum(Imbalance_Cost_Observed):.2f} | Cost_Realized_Total: {sum(Costs_Realized_Total) :.2f}\033[0m")


    ## concatenation of the stored results for the plotting
    Energy_forecast = list(itertools.chain(*Energy_forecast))
    Energy_forecast_modified = list(itertools.chain(*Energy_forecast_modified))
    Energy_observed = list(itertools.chain(*Energy_observed))
    Demand_forecast = list(itertools.chain(*Demand_forecast))
    Demand_forecast_modified = list(itertools.chain(*Demand_forecast_modified))
    Demand_observed = list(itertools.chain(*Demand_observed))
    theta_e = list(itertools.chain(*theta_e))
    out_e= list(itertools.chain(*out_e))
    theta_d = list(itertools.chain(*theta_d))
    out_d= list(itertools.chain(*out_d))
    soc = list(itertools.chain(*soc))
    Realized_SOC = list(itertools.chain(*Realized_SOC))
    Market_Price_sell = list(itertools.chain(*Market_Price_sell))
    negative_price_imbalance=list(itertools.chain(*negative_price_imbalance))
    positive_price_imbalance=list(itertools.chain(*positive_price_imbalance))
    Grid_Exchange = list(itertools.chain(*Grid_Exchange))
    Grid_Exchange_realized = list(itertools.chain(*Grid_Exchange_realized))
    imbalance_volumes = list(itertools.chain(*imbalance_volumes))
    for key in Optimized_fluxes.keys():
        Optimized_fluxes[key] = list(itertools.chain(*Optimized_fluxes[key]))

    SOC_max= [x for x in Realized_SOC if x>=0.9]
    SOC_min= [x for x in Realized_SOC if x<=0.3]
    hour_SOC_max = len(SOC_max)
    hour_SOC_min = len(SOC_min)

    tolleranza = 0.1
    # Conta quante volte il valore assoluto dello sbilanciamento è inferiore alla tolleranza
    hour_commitment_met = sum(abs(x) < tolleranza for x in imbalance_volumes)
    # hour_overdelivey    = len([x for x in imbalance_volumes if x > 0])
    # hour_underdelivery  = len([x for x in imbalance_volumes if x < 0])
    hour_overdelivery   = sum(x > tolleranza for x in imbalance_volumes)
    hour_underdelivery  = sum(x < -tolleranza for x in imbalance_volumes)
    #

    print("\033[32m______________________________________ENERGY ANALYSIS___________________________________________________________\033[0m") 
    print(f"\033[32mPositive Imbalance: {sum([x for x in imbalance_volumes if x > 0]):.2f} | Negative Imbalance: {sum([x for x in imbalance_volumes if x < 0]):.2f}\033[0m") 
    print(f"\033[32mHour at Max SOC (%): {100*hour_SOC_max/len(Realized_SOC):.2f} | Hour at min SOC (%): {100*hour_SOC_min/len(Realized_SOC):.2f}\033[0m") 
    print(f"\033[32mHours with commitment met (%): {100*hour_commitment_met/len(imbalance_volumes):.2f}| Hours with Overdelivery (%): {100*hour_overdelivery/len(imbalance_volumes):.2f}| Hours with Underdelivery (%): {100*hour_underdelivery/len(imbalance_volumes):.2f}| \033[0m") 
    print("\033[32m____________________________________________________________________________________________________________\033[0m")

    ## Calculate the total time taken and Print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print("_______________________________________________________________________")
    print(f"inference_times_total_real ={sum(infernece_time_daily_real)}")
    print("_______________________________________________________________________")


    print("_______________________________________________________________________")
    print(f"inference_times_total_not_true ={sum(inference_times_daily)}")
    print("_______________________________________________________________________")

    
    print("_______________________________________________________________________")
    print(f"Optimization_times_total ={sum(optimization_times_daily)}")
    print("_______________________________________________________________________")

        
    print("_______________________________________________________________________")
    print(f"Optimization_times_total ={sum(optimization_times_daily)}")
    print("_______________________________________________________________________")
    print("\033[32m____________________________________________________________________________________________________________\033[0m")
    print(f"\033[32mend-to-end day-ahead computataional time ={sum(infernece_time_daily_real)+sum(optimization_times_daily)}\033[0m")
    print("\033[32m____________________________________________________________________________________________________________\033[0m")  

# %% ########################  SAVE DATA FOR POST-PROCESSING  ########################
    number_days=(end_date-start_date).days + 1
    start_str = start_date.strftime('%d-%m-%Y')
    end_str   = end_date.strftime('%d-%m-%Y')
    if number_days > len(DATA):
        number_days = len(DATA)
        print(f"⚠️  Not enough Data. only {number_days} days plotted ⚠️ ")

    # Prepariamo il dizionario con tutti i dati estratti dalle liste del loop
    results_data = {
        "metadata": {
            "start_date": start_str,
            "end_date": end_str,
            "opt_type": Opt_Type,
            "number_days": int(number_days),
            "timesteps_per_day": int(timesteps)
        },
        "hourly_data": {
            "energy_forecast": [float(x) for x in Energy_forecast],
            "energy_modified": [float(x) for x in Energy_forecast_modified],
            "energy_observed": [float(x) for x in Energy_observed],
            "demand_forecast": [float(x) for x in Demand_forecast],
            "demand_modified": [float(x) for x in Demand_forecast_modified],
            "demand_observed": [float(x) for x in Demand_observed],
            "theta_e": [float(x) for x in theta_e],
            "theta_d": [float(x) for x in theta_d],
            "soc": [float(x) for x in soc],
            "Realized_SOC": [float(x) for x in Realized_SOC],
            "grid_exchange_planned": [float(x) for x in Grid_Exchange],
            "Grid_Exchange_realized": [float(x) for x in Grid_Exchange_realized],
            "imbalance_volumes": [float(x) for x in imbalance_volumes],
            "market_price": [float(x) for x in Market_Price_sell],
            "imb_price_pos": [float(x) for x in positive_price_imbalance],
            "imb_price_neg": [float(x) for x in negative_price_imbalance],
            "optimized_fluxes": {k: [float(x) for x in v] for k, v in Optimized_fluxes.items()}
        },
        "daily_data": {
            "costs_planned": [float(x) for x in Costs_],
            "costs_realized_total": [float(x) for x in Costs_Realized_Total],
            "costs_observed_optimal": [float(x) for x in Costs_Observed],
            "imbalance_costs_observed": [float(x) for x in Imbalance_Cost_Observed],
            "imbalance_costs_cvar": [float(x) for x in Imbalance_Costs_CVaR],
            "costs_cvar": [float(x) for x in Costs_CVaR],
            "inference_times": inference_times_daily,
            "optimization_times": optimization_times_daily
        },
        "scenarios_data": {
            "imbalance_costs_scenarios": Imbalance_costs_scenarios.tolist() if Imbalance_costs_scenarios is not None else [],
            "costs_scenarios": Costs_scenarios.tolist() if Costs_scenarios is not None else []
        }
    }

    # Salvataggio in formato JSON
    
    if Opt_Type == "Parametric":
        json_name = f"Results_{start_str}_to_{end_str}_{Opt_Type}_{Model_Name}.json"
    else:
        json_name = f"Results_{start_str}_to_{end_str}_{Opt_Type}.json"
    json_path = os.path.join(results_dir, json_name)
    
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"✅ Data saved for plotting in: {json_name}")
