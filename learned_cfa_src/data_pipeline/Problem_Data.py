import numpy as np
import pandas as pd
from io import StringIO
class Problem_Data:
    def __init__(self,Battery_Data,Simulation_Data,Fluxes_Name,Prices,Energy_Data,Demand_Data,Costs_Milps,Grid_Exchange_Milps,Grid_Exchange_Baseline,Cost_Baseline,Best_DA,Worst_DA,Best_ICVaR,Worst_ICVaR):
      
        self.Battery_Data = Battery_Data
        self.Simulation_Data = Simulation_Data
        self.Fluxes_Names = Fluxes_Name
        self.Energy_Data = Energy_Data
        self.Demand_Data = Demand_Data
        self.Prices = Prices
        self.Energy_scenarios = pd.read_json(StringIO(self.Energy_Data["scenario"]))
        self.Demand_scenarios = pd.read_json(StringIO(self.Demand_Data["scenario"]))
        self.Costs_Milps = Costs_Milps
        self.Grid_Exchange_Milps =Grid_Exchange_Milps
        self.Grid_Exchange_Baseline = Grid_Exchange_Baseline
        self.Cost_Baseline = Cost_Baseline
        self.Best_DA = Best_DA
        self.Worst_DA = Worst_DA
        self.Best_ICVaR=Best_ICVaR
        self.Worst_ICVaR=Worst_ICVaR

    def quantile(self,alpha_e,alpha_d):

        quantile_energy = []
        quantile_demand = []
        timesteps = len(self.Energy_scenarios)
  
        for t in range(timesteps):
            hour_energy_scenarios = self.Energy_scenarios.iloc[t]
            hour_energy_quantile =np.quantile(hour_energy_scenarios,alpha_e)
            quantile_energy.append(hour_energy_quantile)

            hour_demand_scenarios = self.Demand_scenarios.iloc[t]
            hour_demand_quantile =np.quantile(hour_demand_scenarios,1-alpha_d)
            quantile_demand.append(hour_demand_quantile)

        return quantile_energy, quantile_demand
    
    def risk_computation(self,alpha_e,alpha_d):
      
        timesteps = len(self.Energy_scenarios)
        quantile_energy, quantile_demand = self.quantile(alpha_e,alpha_d)
        energy_risk = []
        demand_risk = []
        for t in range(timesteps):
            hour_energy_scenarios = self.Energy_scenarios.iloc[t]
            hour_demand_scenarios = self.Demand_scenarios.iloc[t]

            lower_values_energy = hour_energy_scenarios[hour_energy_scenarios <= quantile_energy[t]]
            lower_values_demand = hour_demand_scenarios[hour_demand_scenarios >= quantile_demand[t]]

            errors_energy = quantile_energy[t]-lower_values_energy
            errors_demand = lower_values_demand - quantile_demand[t]

            mean_error_energy = np.mean(errors_energy)
            mean_error_demand = np.mean(errors_demand)

            risk_energy = alpha_e*self.Market_Prices[t]*mean_error_energy
            risk_demand = alpha_d*self.Market_Prices[t]*mean_error_demand
            energy_risk.append(risk_energy)
            demand_risk.append(risk_demand)

        return energy_risk, demand_risk