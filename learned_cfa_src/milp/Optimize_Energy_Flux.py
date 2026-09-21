from pulp import *
from .Battery import Battery
from .Energy_Flux import Energy_Flux
import numpy as np
from scipy.stats import norm
import os
#from Data.scripts.Problem_Data import Problem_Data

def optimize_energy_flux(data,Theta_e,Theta_d,Data_Type):
    
    ################################################################## UNPACK DATA FROM DATA ###############################################################################################################
    
    ### Simulation Data ###
    time_resolution  = data.Simulation_Data["time_resolution"] 
    time_horizon     = data.Simulation_Data["time_horizon"]
    timesteps        = time_resolution*time_horizon      

    ### Battery Data ###
    SoC_in           = data.Battery_Data["SoC_in"]
    Storage_Capacity = data.Battery_Data["Storage_Capacity"] 
  

    eff_charge       = data.Battery_Data["eff_charge"]
    eff_discharge    = data.Battery_Data["eff_discharge"]
    Max_Charge       = data.Battery_Data["Max_Charge"]
    Max_Discharge    = data.Battery_Data["Max_Discharge"]

    ### Demand Data ###
    Demand_forecast = data.Demand_Data[Data_Type]
    #Demand_forecast =  data.Demand_scenarios[1]

    ### Energy Data ###    
    Energy_forecast   = data.Energy_Data[Data_Type]
    #Energy_forecast =  data.Energy_scenarios[1]

    ### Prices Data ###
    Market_Price_sell   = data.Prices['Market']
   # Market_Price_Aquire = Market_Price_sell
    Market_Price_Aquire = []
    Access_tarif_day = 0.018
    Access_tarif_night = 0.014
    tax = 1.2
    for t in range(timesteps):
        if t<=7 or t >=21:
            Market_Price_Aquire.append((Market_Price_sell[t]+Access_tarif_night)*(tax))
        else:
            Market_Price_Aquire.append((Market_Price_sell[t]+Access_tarif_day)*(tax))

        # Market_Price_Aquire = [p + 0.01 for p in Market_Price_sell]

    ### Names Fluxes ###
    name_variables_bat = data.Fluxes_Names["Battery Fluxes"]
    type               = data.Fluxes_Names["Battery Fluxes Type"]
    other_fluxes       = data.Fluxes_Names["Other Fluxes"]
    name_fluxes        = other_fluxes + name_variables_bat

    ################################################################## OPTIMIZATION PROBLEM #####################################################################################################################
    ### problem creation ###
    prob = LpProblem("Test", LpMinimize)
    #####################################################################  VARIABLES  #######################################################################################################################
    ### energy fluxes variables ###
    x_Energy_Market      = LpVariable.dicts(other_fluxes[0], range(timesteps), lowBound=0, cat='Continuous')    # From Energy to Market
    x_Energy_Demand      = LpVariable.dicts(other_fluxes[1], range(timesteps), lowBound=0, cat='Continuous')    # From Energy to Demand
    x_Market_Demand      = LpVariable.dicts(other_fluxes[2], range(timesteps), lowBound=0, cat='Continuous')    # From Market to Demand
    # comment this if you want to activate the fact that you can both sell and buy from the grid
    z_battery_mode       = LpVariable.dicts("z_battery_mode",range(timesteps), cat='Binary') #if = 0 cahrging allowed if = 1 discharged allowed
    Flux_battery = {}
    for var in name_variables_bat:   
        Flux_battery[var] = Energy_Flux(type[var],var,timesteps)    # Battery fluxes ["x_Market_Storage","x_Energy_Storage","x_Storage_Market","x_Storage_Demand]

    ### Forecast modification variables ###


     # Theta_m_s = np.ones(timesteps)
      #  Theta_m_a = np.ones(timesteps)   #problem with theta_m_s a s variable non l,inear obj funct

       
    ##################################################################  GENERIC CONSTRAINTS  #######################################################################################################################
    ### Energy and Demands flow balances ###
    for t in range(timesteps):
        prob += x_Energy_Demand[t] + x_Market_Demand[t] + Flux_battery["x_Storage_Demand"].variable[t] == Demand_forecast[t] * Theta_d[t], f"Demand_t{t:02d}"           
        prob += x_Energy_Demand[t] + x_Energy_Market[t] + Flux_battery["x_Energy_Storage"].variable[t] == Energy_forecast[t] * Theta_e[t], f"Energy_Generation_t{t:02d}"

    ### Battery Constraints ###
    SOC = LpVariable.dicts("SOC", range(timesteps), lowBound=0.3, upBound= 0.9, cat='Continuous')                           # State of Charge Variable definition
    bat=Battery(SoC_in,name_variables_bat,Flux_battery,Storage_Capacity,eff_charge,eff_discharge,Max_Charge,Max_Discharge)  # Batter Object creation with the Battery Data
    bat.mode_binaries = z_battery_mode
    for t in range(timesteps):   
        for name, constraints in bat.Optimization_Constraints(SOC,t):
            prob += constraints, name                                                                                       # battery Constraint Definition through the Object Battery
    
    # Esempio: Il SOC finale deve essere almeno uguale al SOC iniziale
    prob += SOC[timesteps-1] >= SoC_in, "Final_SOC_Constraint" #da valutare
    ##################################################################  OBJECTIVE FUNCTION  ####################################################################################################################################
    ### Costs related to the energy fluxes with a penalty of the low useage of the battery system ###
    #prob += (Market_Price_Aquire[-1]*Storage_Capacity*(SoC_in - SOC[timesteps-1]) if SOC[timesteps-1]<= SoC_in else 0) + \
    # prob +=    sum(Market_Price_Aquire[t]*(
    #            x_Market_Demand[t] +
    #            Flux_battery["x_Market_Storage"].variable[t] -
    #            Flux_battery["x_Storage_Demand"].variable[t] -
    #            x_Energy_Demand[t]) - 
    #            Market_Price_sell[t]*(
    #            x_Energy_Market[t] +
    #            Flux_battery["x_Storage_Market"].variable[t]) for t in range(timesteps))
    # below only grid exchnage
    prob +=     sum(Market_Price_Aquire[t]*(
                x_Market_Demand[t] +
                Flux_battery["x_Market_Storage"].variable[t]) - 
                Market_Price_sell[t]*(
                x_Energy_Market[t] +
                Flux_battery["x_Storage_Market"].variable[t]) for t in range(timesteps))


########################################################      SOLUTION        #################################################################################################################################### 
    unique_id = f"{os.getpid()}_{id(prob)}"
    prob.name = f"Problem_{unique_id}"
    solution = prob.solve(pulp.PULP_CBC_CMD(msg=False, keepFiles=False))
   # solution = prob.solve(pulp.PULP_CBC_CMD(msg=False))     # Solve the Optimization problem
    #print("Solver Status:", LpStatus[solution])             # Print status (Optimal, Infeasible, etc.)

    if LpStatus[solution] == "Optimal":
       # Costs = value(prob.objective)                       # Storing of the optimized cost value
        #print("Optimized Results")
        # below it consider only money exchange with grid
        Costs = sum(Market_Price_Aquire[t]*(
               x_Market_Demand[t].varValue +
               Flux_battery["x_Market_Storage"].variable[t].varValue)
               - Market_Price_sell[t]*(
               x_Energy_Market[t].varValue +
               Flux_battery["x_Storage_Market"].variable[t].varValue) for t in range(timesteps))
        Costs_hourly=[]
        for t in range(timesteps):
            Costs_hourly.append(Market_Price_Aquire[t]*(
               x_Market_Demand[t].varValue +
               Flux_battery["x_Market_Storage"].variable[t].varValue)
               - Market_Price_sell[t]*(
               x_Energy_Market[t].varValue +
               Flux_battery["x_Storage_Market"].variable[t].varValue))
        #prob.writeLP("energy_flux_opt.lp")                  # Creation of a LP file with the whole formulation
    else:
        Costs = 1e6                                         # giving an infeasible value of cost if the solution is infeasible
        #print("\033[31mNo optimal solution found!\033[0m")
        #prob.writeLP("energy_flux_Error.lp")                # Creation of a LP file with the whole formulation


    ##################################################################    RESULTS STORE      #################################################################################################################################### 
    ### DICTIONARY FOR STORING THE OPTIMIZED RESULTS OF THE ENERGY FLUXES 
    other_fluxes_var ={'x_Energy_Market':x_Energy_Market,
                       'x_Energy_Demand':x_Energy_Demand, 
                       'x_Market_Demand':x_Market_Demand}

        
    Optimized_Fluxes={}

    for key in name_fluxes:
        flux = []
        if key in other_fluxes: 
            for t in range(timesteps):
                flux.append(other_fluxes_var[key][t].varValue)
        else:
            for t in range(timesteps):
                flux.append(Flux_battery[key].variable[t].varValue)     
        Optimized_Fluxes[key] = flux

    ### LIST STORING SOC, GRID EXCHANGE,   THETAS AND THEIR RELATIVE FORECAST MODIFICATION
    soc = []
    Grid_Exchange = []
    theta_e = []
    theta_d = []
    Energy_forecast_modified = []
    Demand_forecast_modified = []
    for t in range(timesteps):
        soc.append(round(SOC[t].varValue,2))
        Grid_Exchange.append( x_Energy_Market[t].varValue + Flux_battery["x_Storage_Market"].variable[t].varValue - Flux_battery["x_Market_Storage"].variable[t].varValue - x_Market_Demand[t].varValue)
        
        theta_e.append(round(Theta_e[t],2))
        theta_d.append(round(Theta_d[t],2))
        Energy_forecast_modified.append(theta_e[t]*Energy_forecast[t])
        Demand_forecast_modified.append(theta_d[t]*Demand_forecast[t])

    return Costs,Optimized_Fluxes, soc,theta_e,theta_d, data, timesteps, Energy_forecast,Energy_forecast_modified,Demand_forecast,Demand_forecast_modified,Market_Price_sell,Grid_Exchange,Costs_hourly
def optimize_tracking_fixed_grid(data, Realized_Data_Type, Target_Grid_Exchange):
    """
    Esegue il MILP sui dati REALI cercando di inseguire il Target_Grid_Exchange (Day-Ahead Plan).
    Usa variabili di slack per evitare crash (Infeasibility) se la batteria non ce la fa.
    """
    
    # 1. SETUP DATA 
    time_resolution  = data.Simulation_Data["time_resolution"] 
    time_horizon     = data.Simulation_Data["time_horizon"]
    timesteps        = time_resolution * time_horizon      

    SoC_in           = data.Battery_Data["SoC_in"]
    Storage_Capacity = data.Battery_Data["Storage_Capacity"]
    eff_charge       = data.Battery_Data["eff_charge"]
    eff_discharge    = data.Battery_Data["eff_discharge"]
    Max_Charge       = data.Battery_Data["Max_Charge"]
    Max_Discharge    = data.Battery_Data["Max_Discharge"]

    #Theta = 1 
    Demand_real = data.Demand_Data[Realized_Data_Type]
    Energy_real = data.Energy_Data[Realized_Data_Type]

    name_variables_bat = data.Fluxes_Names["Battery Fluxes"]
    type_bat           = data.Fluxes_Names["Battery Fluxes Type"]
    other_fluxes       = data.Fluxes_Names["Other Fluxes"]
    
    # 2. PROBLEM CREATION
    prob = LpProblem("Tracking_Validation", LpMinimize)

    # 3. VARIABLES
  
    x_Energy_Market  = LpVariable.dicts(other_fluxes[0], range(timesteps), lowBound=0, cat='Continuous')
    x_Energy_Demand  = LpVariable.dicts(other_fluxes[1], range(timesteps), lowBound=0, cat='Continuous')
    x_Market_Demand  = LpVariable.dicts(other_fluxes[2], range(timesteps), lowBound=0, cat='Continuous')
    z_battery_mode   = LpVariable.dicts("z_battery_mode", range(timesteps), cat='Binary')
    
    Flux_battery = {}
    for var in name_variables_bat:   
        Flux_battery[var] = Energy_Flux(type_bat[var], var, timesteps)

    # --- NUOVE VARIABILI PER IL TRACKING (SLACK) ---
    # Queste rappresentano la deviazione INEVITABILE dal piano

    slack_pos = LpVariable.dicts("Slack_Pos", range(timesteps), lowBound=0, cat='Continuous')
    slack_neg = LpVariable.dicts("Slack_Neg", range(timesteps), lowBound=0, cat='Continuous')

    # 4. CONSTRAINTS
    # Bilanci Energetici (Standard)
    for t in range(timesteps):
        prob += x_Energy_Demand[t] + x_Market_Demand[t] + Flux_battery["x_Storage_Demand"].variable[t] == Demand_real[t], f"Demand_Bal_t{t}"           
        prob += x_Energy_Demand[t] + x_Energy_Market[t] + Flux_battery["x_Energy_Storage"].variable[t] == Energy_real[t], f"Energy_Bal_t{t}"

    # Battery Constraints (Standard)
    SOC = LpVariable.dicts("SOC", range(timesteps), lowBound=0.3, upBound=0.9, cat='Continuous') # OCCHIO AI LIMITI SOC NEL TUO CODICE
    bat = Battery(SoC_in, name_variables_bat, Flux_battery, Storage_Capacity, eff_charge, eff_discharge, Max_Charge, Max_Discharge)
    bat.mode_binaries = z_battery_mode
    for t in range(timesteps):   
        for name, constraints in bat.Optimization_Constraints(SOC, t):
            prob += constraints, name
   # prob += SOC[timesteps-1] >= SoC_in, "Final_SOC"

    # --- IL VINCOLO DI TRACKING  ---
    for t in range(timesteps):
        # Calcolo lo scambio netto ATTUALE del modello (Export - Import)
        # Export = Energy->Market + Battery->Market
        current_export = x_Energy_Market[t] + Flux_battery["x_Storage_Market"].variable[t]
        # Import = Market->Demand + Market->Battery
        current_import = x_Market_Demand[t] + Flux_battery["x_Market_Storage"].variable[t]
        
        current_net_exchange = current_export - current_import
        
        # Recupero il target (Day Ahead Plan)
        target = Target_Grid_Exchange[t]
        
        # EQUAZIONE DI TRACKING:
        # Reale = Target + (DeviazionePositiva - DeviazioneNegativa)
        # Se slack_pos e slack_neg sono 0, allora Reale == Target (Perfetto)
        prob += current_net_exchange == target + slack_pos[t] - slack_neg[t], f"Tracking_Constraint_t{t}"

    # 5. OBJECTIVE FUNCTION
    # 
    
    PENALTY_WEIGHT = 1e6 # Un milione di euro per MWh di errore. Priorità assoluta.
    

    
    prob += sum(
        # Penalità Enorme per ogni deviazione dal piano Pgrid
        PENALTY_WEIGHT * (slack_pos[t] + slack_neg[t])
        
        for t in range(timesteps)
    )

    solution = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 7. EXTRACT RESULTS
    Realized_Grid_Exchange = []
    Imbalance_Volumes = [] # Questo è l'errore che la batteria NON è riuscita a coprire
    Realized_SOC = []
    if LpStatus[solution] == "Optimal":
        for t in range(timesteps):
            # Ricostruisco lo scambio reale
            exp = x_Energy_Market[t].varValue + Flux_battery["x_Storage_Market"].variable[t].varValue
            imp = x_Market_Demand[t].varValue + Flux_battery["x_Market_Storage"].variable[t].varValue
            net_real = exp - imp
            
            Realized_Grid_Exchange.append(net_real)
            
            # L'imbalance qui è ESATTAMENTE il valore delle variabili slack
            # Se slack > 0, significa che non siamo riusciti a seguire il piano
            imb_val = slack_pos[t].varValue - slack_neg[t].varValue
            Imbalance_Volumes.append(imb_val)
            # SOC Extraction (NEW)
            Realized_SOC.append(round(SOC[t].varValue, 2))
            
        return Realized_Grid_Exchange, Imbalance_Volumes,Realized_SOC
    else:
        print("Errore: Il problema è ancora Infeasible (molto raro con le slack variables)")
        return [], [] 