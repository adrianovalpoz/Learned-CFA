from pulp import *
import numpy as np
from scipy.stats import norm

######### import models from parent folder #######################
import sys
import os
# Add the parent directory to the system path
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parent_folder')))

# Add models folder to sys.path
sys.path.append(models_path)


from Battery import Battery
from Energy_Flux import Energy_Flux

def optimize_energy_flux(data,scenario_e,scenario_d,type):
    
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
    if type == "scenario":
        Demand_forecast =  data.Demand_scenarios[scenario_d]

        ### Energy Data ###    
        
        Energy_forecast =  data.Energy_scenarios[scenario_e]
    elif type == "forecast":
        Demand_forecast =  data.Demand_Data["forecast"]

        ### Energy Data ###    
        
        Energy_forecast =  data.Energy_Data["forecast"]

    elif type == "bounds":
        Demand_forecast =  scenario_d

        ### Energy Data ###    
        
        Energy_forecast =  scenario_e
    

    ### Prices Data ###
    Market_Price_sell   = data.Prices['Market']
    Market_Price_Aquire = []
    Access_tarif_day = 0.018
    Access_tarif_night = 0.014
    tax = 1.2
    for t in range(timesteps):
        if t<=7 or t >=21:
            Market_Price_Aquire.append((Market_Price_sell[t]+Access_tarif_night)*(tax))
        else:
            Market_Price_Aquire.append((Market_Price_sell[t]+Access_tarif_day)*(tax))

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
    z_battery_mode       = LpVariable.dicts("z_battery_mode",range(timesteps), cat='Binary') #if = 0 cahrging allowed if = 1 discharged allowed

    Flux_battery = {}
    for var in name_variables_bat:   
        Flux_battery[var] = Energy_Flux(type[var],var,timesteps)    # Battery fluxes ["x_Market_Storage","x_Energy_Storage","x_Storage_Market","x_Storage_Demand]

    ### Forecast modification variables ###
    Theta_e   = np.ones(timesteps)
    Theta_d   = np.ones(timesteps)
    Theta_m_s = np.ones(timesteps)
    Theta_m_a = np.ones(timesteps)   #problem with theta_m_s a s variable non l,inear obj funct

       
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
    # prob += (Theta_m_a[-1]*Market_Price_Aquire[-1]*Storage_Capacity*(SoC_in - SOC[timesteps-1]) if SOC[timesteps-1]<= SoC_in else 0) + \
    prob += sum(Theta_m_a[t]*Market_Price_Aquire[t]*(x_Market_Demand[t]+Flux_battery["x_Market_Storage"].variable[t]) - Theta_m_s[t]*Market_Price_sell[t]*(x_Energy_Market[t]+Flux_battery["x_Storage_Market"].variable[t]) for t in range(timesteps))
    
    ##################################################################      SOLUTION        #################################################################################################################################### 
    solution = prob.solve(pulp.PULP_CBC_CMD(msg=False))     # Solve the Optimization problem
    #print("Solver Status:", LpStatus[solution])             # Print status (Optimal, Infeasible, etc.)

    if LpStatus[solution] == "Optimal":
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
        #print("Optimized Results")
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


    return Costs,Optimized_Fluxes, soc,theta_e,theta_d, data, timesteps, Energy_forecast,Energy_forecast_modified,Demand_forecast,Demand_forecast_modified,Market_Price_sell,Grid_Exchange
   