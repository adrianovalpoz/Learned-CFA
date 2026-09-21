from .milp.Optimize_Energy_Flux import optimize_energy_flux, optimize_tracking_fixed_grid

import numpy as np
def normalize_value(c, worst, best):
    # Calcoliamo lo swing totale
    delta = worst - best
    # Applichiamo la formula
    v = (worst - c) / delta
    # Clipping di sicurezza (fondamentale per la stabilità della NN)
    return v #np.clip(v, 0, 1)

def gamma_computation(Best_DA,Worst_DA,Best_ICVaR,Worst_ICVaR):
    P = [Best_DA,Worst_ICVaR]#maxrisk extreme scenario
    Q = [Worst_DA ,Best_ICVaR]#minrisk extreme scenario
    M = [None,Worst_ICVaR]
    M[0] = Q[0] + Q[1] - M[1]
    DA_valuefunction_M= normalize_value(M[0], Worst_DA, Best_DA)
    gamma= DA_valuefunction_M
    return gamma

def imbalance_costs(realized,planned,imb_price_pos,imb_price_neg):
    """Hourly imbalance costs of a realized grid exchange against the planned one.
    Dual pricing: deviations that help the system are not rewarded (cost 0)."""
    costs = []
    Imbalance = []
    for t in range(len(planned)):
        imbalance = realized[t] - planned[t]
        Imbalance.append(imbalance)
        if imbalance >= 0:
            costs.append(-imb_price_pos[t] * imbalance if imb_price_pos[t] < 0 else 0)
        elif imbalance < 0:
            costs.append(-imb_price_neg[t] * imbalance if imb_price_neg[t] > 0 else 0)
    return costs,Imbalance
   

def Optimize_Risk(data,theta_e,theta_d,Data_Type):
    timesteps = data.Simulation_Data["time_horizon"]*data.Simulation_Data["time_resolution"]
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]
    Imbalance_costs_scenarios =[]
    Imbalance_Scenarios = []
    Costs_diff_scenarios = []
    Total_Costs = []
    # Milp Scenario Data
    Costs_Milps         = data.Costs_Milps
    Grid_Exchange_Milps = data.Grid_Exchange_Milps #list of list [primo indice numero MILps secodno indice timestep]
    if len(Costs_Milps)>= len(Grid_Exchange_Milps):
        N_scenarios = len(Grid_Exchange_Milps)
    else:
        N_scenarios = len(Costs_Milps)
############################## Modified Forecast Optimization ################################################################

    Planned_Costs,_,SoC,_,_,_,_,_,_,_,_,_,Planned_Grid_Exchange,_ = optimize_energy_flux(data,theta_e,theta_d,Data_Type) # flux optimization function
    SoC_end = SoC[-1]


    ##baseline computation
    Baseline_Grid_Exchange = data.Grid_Exchange_Baseline
    Cost_Baseline = data.Cost_Baseline
    Imbalance_costs_baseline_tot = []
    for s in range(N_scenarios):
        #Imbalance_costs_baseline = []
       
        # for t in range(timesteps):
        #     imbalance_baseline= Grid_Exchange_Milps[s][t] - Baseline_Grid_Exchange[t]
         
        #     if imbalance_baseline >= 0:
        #         if positive_price_imbalance[t]<0:
        #             Imbalance_costs_baseline.append(-positive_price_imbalance[t]*imbalance_baseline)
        #         else:
        #             Imbalance_costs_baseline.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        #     elif imbalance_baseline <0:

        #         if negative_price_imbalance[t]>0:
        #             Imbalance_costs_baseline.append(-negative_price_imbalance[t]*imbalance_baseline)
        #         else:
        #             Imbalance_costs_baseline.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        Imbalance_costs_baseline,_ = imbalance_costs(Grid_Exchange_Milps[s],Baseline_Grid_Exchange,positive_price_imbalance,negative_price_imbalance)
        Imbalance_costs_baseline_tot.append(sum(Imbalance_costs_baseline))

    Imbalance_Costs_CVaR_baseline,Imbalance_Costs_VaR_baseline,sigma_tail_baseline = RightCVaR(Imbalance_costs_baseline_tot,alpha=0.95) 
        
################################ Imbalance Risk ##############################################################################

    for s in range(N_scenarios):
       # Imbalance_costs = []
        #imbalance_s = []
        
        # for t in range(timesteps):
        #     imbalance = Grid_Exchange_Milps[s][t] - Planned_Grid_Exchange[t]
        #     imbalance_s.append(imbalance)
        #     if imbalance >= 0:
        #         if positive_price_imbalance[t]<0:
        #             Imbalance_costs.append(-positive_price_imbalance[t]*imbalance)
        #         else:
        #             Imbalance_costs.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        #     elif imbalance <0:

        #         if negative_price_imbalance[t]>0:
        #             Imbalance_costs.append(-negative_price_imbalance[t]*imbalance)
        #         else:
        #             Imbalance_costs.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        Imbalance_costs,imbalance_s = imbalance_costs(Grid_Exchange_Milps[s],Planned_Grid_Exchange,positive_price_imbalance,negative_price_imbalance)

        Total_Costs_scenarios = Planned_Costs+sum(Imbalance_costs)
        Imbalance_Scenarios.append(sum(imbalance_s))
        Imbalance_costs_scenarios.append(sum(Imbalance_costs))
        Total_Costs.append(Total_Costs_scenarios)
        #Costs_diff_scenarios.append(Costs_Milps[s]-Planned_Costs) #-Costs to reintroduce
    
    Costs_diff_scenarios = np.array(Costs_diff_scenarios)
    Imbalance_costs_scenarios = np.array(Imbalance_costs_scenarios)
    Worst_Scenarios=Costs_diff_scenarios[Costs_diff_scenarios >= 0]

    Costs_CVaR = LeftCVaR(Costs_diff_scenarios,alpha=0.95)
  
    Imbalance_Costs_CVaR,Imbalance_Costs_VaR,sigma_tail = RightCVaR(Imbalance_costs_scenarios,alpha=0.95)

################################# REWARD FUNCTION ##########################################################################
######## Normalization min-max ############
    # Best_DA = data.Best_DA
    # Worst_DA = data.Worst_DA
    # DA_valuefunction= normalize_value(Planned_Costs, Worst_DA, Best_DA)
    # Best_ICVaR=data.Best_ICVaR
    # Worst_ICVaR=data.Worst_ICVaR
    # ICVaR_valuefunction= normalize_value(Imbalance_Costs_CVaR, Worst_ICVaR, Best_ICVaR)

##### Normalization##### Normalization z-score
    Da_Scenarios_mean = np.mean(np.array(Costs_Milps))
    Da_Scenarios_variance = np.std(np.array(Costs_Milps))
    Planned_Costs_norm = (Planned_Costs - Da_Scenarios_mean )/Da_Scenarios_variance
    
    Imbalance_Costs_CVaR_norm = (Imbalance_Costs_CVaR-Imbalance_Costs_VaR_baseline)/sigma_tail_baseline 
#### reward ####à##  
    #gamma = gamma_computation(Best_DA,Worst_DA,Best_ICVaR,Worst_ICVaR)
    gamma = 0.5
    #obj_risk = DA_valuefunction +gamma*ICVaR_valuefunction
    obj_risk = Planned_Costs_norm +gamma*Imbalance_Costs_CVaR_norm
   
   
    return obj_risk,SoC_end,Imbalance_Scenarios, Costs_diff_scenarios,Planned_Costs,Costs_CVaR,Imbalance_Costs_CVaR,Imbalance_costs_scenarios

def Optimize_Risk_Generation(data,theta_e,theta_d):
    timesteps = data.Simulation_Data["time_horizon"]*data.Simulation_Data["time_resolution"]
    Energy_forecast   = data.Energy_Data["forecast"]
    Energy_scenarios   = data.Energy_scenarios    # rivedere forma database dovrebbe
    Market_Price   = data.Prices['Market']
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]
    Modified_Energy_forecast = []
    profits = []
    for t in range(timesteps):
        Modified_Energy_forecast.append(theta_e[t]*Energy_forecast[t])
        profits.append(Market_Price[t]*theta_e[t]*Energy_forecast[t])

###### IMBALANCE #######
    Imbalance_Costs = []
    Imbalance_Scenarios = []
    N_scenarios = 0
    for s in range(N_scenarios):
        imbalance_s = []
        Imbalance_costs = []
        for t in range(timesteps):
            
            imbalance = SCENARIO - Modified_Energy_forecast[t]
            imbalance_s.append(imbalance)
            if imbalance >= 0:
                Imbalance_costs.append(-positive_price_imbalance[t]*imbalance)
            elif imbalance <0:
                Imbalance_costs.append(negative_price_imbalance[t]*imbalance)

        Imbalance_Scenarios.append(sum(imbalance_s))
        Imbalance_Costs.append(sum(Imbalance_costs))
    return

def LeftCVaR(Data,alpha):
    Data = np.asarray(Data)
    if Data.size == 0:
        return 0  # O un altro valore predefinito
    VaR = np.percentile(Data, (1-alpha)*100)
    CVaR = Data[Data <= VaR].mean()
    return CVaR

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

 

def Real_Loss_Assessment(data, theta_e, theta_d, Data_Type):
    timesteps = data.Simulation_Data["time_horizon"] * data.Simulation_Data["time_resolution"]
    
    # -----------------------------------------------------------------------------------------
    # 1. DAY AHEAD PLAN (Forecast)
    # -----------------------------------------------------------------------------------------
    # (Grid_Exchange_forecast)
    Costs_forecast,_,_,_,_,_,_,_,_,_,_,_, Grid_Exchange_forecast, Costs_hourly_forecasted = optimize_energy_flux(data, theta_e, theta_d, Data_Type)

    # -----------------------------------------------------------------------------------------
    # 2. REALIZED SCENARIO (VALIDATION - CONCEPT B) -> USA LA NUOVA FUNZIONE QUI
    # -----------------------------------------------------------------------------------------
    # Qui usiamo la logica "Tracking": la batteria cerca di rispettare Grid_Exchange_forecast.
    # Se non ci riesce, le slack variables ci daranno la differenza.
    # Grid_Exchange_REALIZED è quello che fisicamente scambiamo alla fine.
    
    # Nota: Assicurati di aver definito 'optimize_tracking_fixed_grid' come ti ho scritto sopra
    Grid_Exchange_REALIZED, Slack_Volumes, Realized_SOC = optimize_tracking_fixed_grid(data, "Observed", Grid_Exchange_forecast)

    # -----------------------------------------------------------------------------------------
    # 3. OBSERVED OPTIMUM (BENCHMARK - CONCEPT A) -> TI SERVE ANCORA!
    # -----------------------------------------------------------------------------------------
    # Ti serve questo run per sapere quale sarebbe stato il costo MINIMO assoluto (colonna A della tabella)
    # Non lo usiamo per l'imbalance, ma solo per il confronto finale.
    Costs_observed_optimal,_,_,_,_,_,_,Energy_observed,_,Demand_observed,_,_,_,_ = optimize_energy_flux(data, [1]*timesteps, [1]*timesteps, "Observed")

    # -----------------------------------------------------------------------------------------
    # 4. IMBALANCE COMPUTATION
    # -----------------------------------------------------------------------------------------
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]
    
    # Imbalance_costs = []
    # Imbalance = []
    # for t in range(timesteps):
    #     # QUI LA MODIFICA CHIAVE:
    #     # L'imbalance non è (Ottimo - Forecast), ma (Reale_da_Tracking - Forecast)
    #     imbalance = Grid_Exchange_REALIZED[t] - Grid_Exchange_forecast[t]
    #     Imbalance.append(imbalance)
    #     # --- Logica Prezzi (La tua logica originale) ---
    #     if imbalance >= 0:
    #         if positive_price_imbalance[t] < 0:
    #             Imbalance_costs.append(-(positive_price_imbalance[t]) * imbalance)
    #         else:
    #             Imbalance_costs.append(0) 
    #     elif imbalance < 0:
    #         if negative_price_imbalance[t] > 0:
    #             Imbalance_costs.append(-(negative_price_imbalance[t]) * imbalance)
    #         else:
    #             Imbalance_costs.append(0)
    Imbalance_costs,Imbalance = imbalance_costs(Grid_Exchange_REALIZED,Grid_Exchange_forecast,positive_price_imbalance,negative_price_imbalance)

    Total_Imbalance_Cost = sum(Imbalance_costs)

    # -----------------------------------------------------------------------------------------
    # 5. RESULT AGGREGATION
    # -----------------------------------------------------------------------------------------
    
    # Costo Reale (B) = Costo previsto dal piano DA + Costo degli sbilanciamenti reali
    Costs_Realized_Total = Costs_forecast + Total_Imbalance_Cost


    return  Total_Imbalance_Cost, Costs_observed_optimal , Costs_forecast,Energy_observed,Demand_observed, Realized_SOC,Grid_Exchange_REALIZED,Imbalance,Costs_Realized_Total