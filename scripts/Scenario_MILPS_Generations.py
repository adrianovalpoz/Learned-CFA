import json
import os
import seaborn as sns
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from Problem_Data import Problem_Data
from optimize_energy_flux import optimize_energy_flux


from datetime import datetime, timedelta
import time
import numpy as np
sid = 101 #42, 101, 1337, 2025]:

np.random.seed(sid)
random.seed(sid)
start_time = time.time()

def theta_bounds(DATA):
    ################################### THETA BOUNDS#####################################################
        # Energy
    e_forecast = DATA.Energy_Data["forecast"]
    e_bound_max = DATA.Energy_Data["bands"]["max"]
    e_bound_min = DATA.Energy_Data["bands"]["min"]
   # Demand
    d_forecast = DATA.Demand_Data["forecast"]
    d_bound_max = DATA.Demand_Data["bands"]["max"]
    d_bound_min = DATA.Demand_Data["bands"]["min"]

     # List of arrays to check and convert
    arrays = [e_forecast, e_bound_max,e_bound_min, d_forecast, d_bound_max,d_bound_min]

    # Loop through each array and convert to NumPy array if not already
    for i in range(len(arrays)):
        if not isinstance(arrays[i], np.ndarray):
             arrays[i] = np.array(arrays[i])   

    theta_max_e = arrays[1]/arrays[0]
    theta_min_e = arrays[2]/arrays[0]
    theta_max_d = arrays[4]/arrays[3]
    theta_min_d = arrays[5]/arrays[3]
    # cleaning of the element when forecast = 0
    for t in range(len(arrays[0])):
        if arrays[0][t] <= 1:
            theta_max_e[t] = 0
            theta_min_e[t] = 0
        if arrays[3][t] <= 1:
            theta_max_d[t] = 0
            theta_min_d[t] =  0   
    
    r = 1

    theta_max_e = np.minimum(1 + r, theta_max_e)
    theta_min_e = np.maximum(1 - r, theta_min_e)

    theta_max_d = np.minimum(1 + r, theta_max_d)
    theta_min_d = np.maximum(1 - r, theta_min_d)


    e_forecast = np.array(e_forecast)
    d_forecast = np.array(d_forecast)
    #max secure profit,max risk
    d_min = d_forecast*theta_min_d
    d_min = d_min.tolist()
    e_max  = e_forecast*theta_max_e
    e_max = e_max.tolist()
        
    d_max =d_forecast*theta_max_d
    d_max = d_max.tolist()
    e_min =e_forecast*theta_min_e
    e_min =e_min.tolist()

    return e_max, e_min, d_max, d_min

def RightCVaR(Data,alpha):

    Data = np.asarray(Data)
    if Data.size == 0:
        return 0,0,0  # O un altro valore predefinito
    
    VaR = np.percentile(Data, (alpha)*100)
    tail_data = Data[Data >= VaR]

    
    CVaR = tail_data.mean()
    sigma_tail = tail_data.std() + 1e-6
    #print(f"CVaR = {CVaR:.2f}| VaR = {VaR:.2f} | lenght = {len(Data[Data >= VaR]):.2f}/500")
  
    return CVaR,VaR,sigma_tail

def ImbalanceCostsCVaR(data,Grid_Exchange_Milps,Grid_Exchange):
    N_scenarios = len(Grid_Exchange_Milps)
    timesteps = data.Simulation_Data["time_horizon"]*data.Simulation_Data["time_resolution"]
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]

    Imbalance_costs_tot = []
    for s in range(N_scenarios):
        Imbalance_costs = []
       
        for t in range(timesteps):
            imbalance= Grid_Exchange_Milps[s][t] - Grid_Exchange[t]
         
            if imbalance >= 0:
                if positive_price_imbalance[t]<0:
                    Imbalance_costs.append(-positive_price_imbalance[t]*imbalance)
                else:
                    Imbalance_costs.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
            elif imbalance <0:

                if negative_price_imbalance[t]>0:
                    Imbalance_costs.append(-negative_price_imbalance[t]*imbalance)
                else:
                    Imbalance_costs.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        

        Imbalance_costs_tot.append(sum(Imbalance_costs))

    ImbalanceCostsCVaR,Imbalance_Costs_VaR,sigma_tail = RightCVaR(Imbalance_costs_tot,alpha=0.95)    

        
    return ImbalanceCostsCVaR

def Scenario(input_file_name,output_file_name,date,start_date,SoC_End):
    ########################################## JSON IMPORT #######################################################
    # Get the directory where this script is located

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    # Specify the folder that is outside the current script folder
    folder_path_read = os.path.join(script_dir, '..', 'json_repository_for_simulation','SystemData')  # Going one level up
    data_path = os.path.join(folder_path_read, input_file_name)
    
    folder_path_save= os.path.join(script_dir, '..', 'json_repository_for_simulation','ScenarioData') # Going one level up
    full_path = os.path.join(folder_path_save, output_file_name)
    # Open and load the JSON file

    with open(data_path, "r") as file:
        Data = json.load(file)  # Convert JSON to a Python dictionary

    #########################################################    DATA OBJECT CREATION    ##################################################################################

    # Creation of Proble_Data object to store all the Data needed for the simulation, and pass them to the optimization functions.

    DATA = Problem_Data(Data["Battery"],Data["Simulation"],Data["Fluxes"],Data["Prices"],Data["Energy"],Data["Demand"],"","","","","","","")
    timesteps = DATA.Simulation_Data["time_horizon"]*DATA.Simulation_Data["time_resolution"]

    ################# loop for scenario MILP ###########################################
    Number_Scenarios_e = DATA.Energy_scenarios.shape[1]
    Number_Scenarios_d = DATA.Demand_scenarios.shape[1]

    Grid_Exchange_Scenarios = []
    Costs_Scenarios = []
    #TO DO: suddividere scenari e e d con altro ciclo for
    n_samples = 1000
    # Generiamo indici casuali indipendenti per e e d
    # replace=True permette di pescare più volte lo stesso scenario se necessario
    random_indices_e = np.random.choice(range(Number_Scenarios_e), size=n_samples, replace=True)
    random_indices_d = np.random.choice(range(Number_Scenarios_d), size=n_samples, replace=True)
    
    for i in tqdm(range(n_samples), desc="Evaluating Scenarios"):
        e = int(random_indices_e[i])
        d = int(random_indices_d[i])
        if date != start_date:
            DATA.Battery_Data["SoC_in"] = SoC_End[i] # condition needed to assure that the next day the status of the storage is the same as the end of prevous day
  
        #for d in range(Number_Scenarios_d):
        Costs,_,soc_day,_,_,_,_,_,_,_,_,_,Grid_Exchange = optimize_energy_flux(DATA,e,d,"scenario")
        SoC_end = soc_day[-1]

    
        #Grid_Exchange = sum(Grid_Exchange)
        SoC_End.append(SoC_end)
        

        Costs_Scenarios.append(Costs)
        Grid_Exchange_Scenarios.append(Grid_Exchange)
    # Dopo il loop tqdm...
    Costs_Scenarios = np.array(Costs_Scenarios)

    # 1. Identifica gli indici del Best e Worst DA osservati negli scenari
    idx_best_DA = np.argmin(Costs_Scenarios)  # Minimo costo = Massimo profitto
    idx_worst_DA = np.argmax(Costs_Scenarios) # Massimo costo = Minimo profitto

    # 2. Estrai i valori per la normalizzazione DA
    Best_DA_empirico = Costs_Scenarios[idx_best_DA]
    Worst_DA_empirico = Costs_Scenarios[idx_worst_DA]

    # 3. Estrai i profili Grid_Exchange corrispondenti (le nuove "Ancore")
    Grid_Exchange_best_DA = Grid_Exchange_Scenarios[idx_best_DA]
    Grid_Exchange_worst_DA = Grid_Exchange_Scenarios[idx_worst_DA]

    # 4. Calcola il CVaR di imbalance su questi profili specifici
    # Nota: qui misuriamo il rischio generato dalle scelte "ottime" e "pessime" del MILP
    ICVaR_at_Best_DA = ImbalanceCostsCVaR(DATA, Grid_Exchange_Scenarios, Grid_Exchange_best_DA)
    ICVaR_at_Worst_DA = ImbalanceCostsCVaR(DATA, Grid_Exchange_Scenarios, Grid_Exchange_worst_DA)

    #baseline grid exchange
    C_baseline,_,_,_,_,_,_,_,_,_,_,_,Grid_Exchange_Baseline = optimize_energy_flux(DATA,"","","forecast") # flux optimization function
    ICVaR_baseline = ImbalanceCostsCVaR(DATA,Grid_Exchange_Scenarios,Grid_Exchange_Baseline)
   ############## min max forumulation #####################
    e_max,e_min,d_max,d_min= theta_bounds(DATA)
 #max secure profit,max risk
    C_best_cheby,_,_,_,_,_,_,_,_,_,_,_,Grid_Exchange_maxrisk_cheby = optimize_energy_flux(DATA,e_max,d_min,"bounds") 
    
    ICVaR_worst_Cheby = ImbalanceCostsCVaR(DATA,Grid_Exchange_Scenarios,Grid_Exchange_maxrisk_cheby)

#min secure profit,min risk
    C_worst_cheby,_,_,_,_,_,_,_,_,_,_,_,Grid_Exchange_minrisk_cheby = optimize_energy_flux(DATA,e_min,d_max,"bounds") 
    ICVaR_best_cheby =  ImbalanceCostsCVaR(DATA,Grid_Exchange_Scenarios,Grid_Exchange_minrisk_cheby)

    Scenario_MILP_Results ={"Costs":Costs_Scenarios.tolist(),
                            "Grid_Exchange": Grid_Exchange_Scenarios,
                            "C_baseline":C_baseline,
                            "Grid_Exchange_Baseline": Grid_Exchange_Baseline,
                            "Best_DA":float(Best_DA_empirico),
                            "Worst_DA":float(Worst_DA_empirico),
                            "Best_ICVaR":float(ICVaR_at_Worst_DA),
                            "Worst_ICVaR":float(ICVaR_at_Best_DA),
                            "C_best_cheby" : C_best_cheby,
                            "C_worst_cheby": C_worst_cheby,
                            "ICVaR_worst_Cheby": float(ICVaR_worst_Cheby),
                            "ICVaR_best_cheby":float(ICVaR_best_cheby)}
    #### JSON PRINT 
    
    #file_path='C:/Users/adria/Desktop/INESCTEC/00.Code/Architecture_Neural_Network/Data/json_repository_for_simulation'



    with open(full_path, "w") as file:
        json.dump(Scenario_MILP_Results, file, indent=4)
        
    print(f"JSON saved at: {folder_path_save}")
    return SoC_End

def Distribution(input_file_name):
    ########################################## JSON IMPORT #######################################################
 
    # Get the directory where this script is located

    script_dir  = os.path.dirname(__file__)
    # Specify the folder that is outside the current script folder
    folder_path = os.path.join(script_dir, '..', 'json_repository_for_simulation')  # Going one level up
    data_path = os.path.join(folder_path, input_file_name)

    # Open and load the JSON file

    with open(data_path, "r") as file:
        Data = json.load(file)  # Convert JSON to a Python dictionary

    ####### UNPACK ##################
    Costs = Data["Costs"]
    Grid_Exchange = Data["Grid_Exchange"]

    n_subplot = 2
    fig, axes = plt.subplots(n_subplot, 1, figsize=(8, 10))
    sns.histplot(Costs, bins=30, kde=True, stat="density", ax=axes[0], color='red', edgecolor='black')

    axes[0].set_title("Costs")
    axes[0].set_xlabel("Euro")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    sns.histplot(Grid_Exchange, kde=True, stat="density", ax=axes[1], color='blue', edgecolor='black')
    axes[1].set_title("Grid Exchange")
    axes[1].set_xlabel("kWh")
    axes[1].set_ylabel("Frequency")  # if you meant "Densità", use density=True
    axes[1].grid(True)

    plt.show()
    return

start_date = datetime(2024,1,1)
end_date   = datetime(2024,12,31)

date = start_date
SoC_End = []
while date <= end_date:
    y = date.strftime('%Y')
    m = date.strftime('%m')
    d = date.strftime('%d')
    day= f'{y}-{m}-{d}'
    output_file_name = f"Scenarios_Results_{day}.json"
    input_flie_name = f"data_{day}.json"
    SoC_End_s = Scenario(input_flie_name,output_file_name,date,start_date,SoC_End)
    SoC_End = SoC_End_s
    print(f"✅ {day} Scenarios Evaluated and Saved.")
    date += timedelta(days=1)
#input_file_name = f"Scenarios_Results.json"
#Distribution(input_file_name)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")


# n_subplot = 2
# fig, axes = plt.subplots(n_subplot, 1, figsize=(6, 10))


# sns.lineplot(x=range(1,Number_Scenarios+1), y=Costs_Scenarios, label="Costs",ax=axes[0],marker = 'o')
# sns.lineplot(x=range(1,Number_Scenarios+1), y=Grid_Exchange_Scenarios, label= "GRID EXCHANGE",  ax=axes[1],marker = 'o')

# plt.show()