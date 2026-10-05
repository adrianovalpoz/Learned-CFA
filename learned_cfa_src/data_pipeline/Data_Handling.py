import pickle
import numpy as np
import json
import pandas as pd
import os
from .simult_intervals import chebyshev_interval
from datetime import datetime, timedelta

# Class created to store and manipulate the data and save them for the simulation

class Data_Handling:
    def __init__(self,Battery_Data,Simulation_Data,Fluxes_Name,quantile_probability,File_Path_Energy_Scenario='', File_Path_Energy_Forecast='',File_Path_Prices='',File_Path_Prices_Imbalance ='',column_names='',File_Path_Observed_Values='',scaling_PV='1'):
        # parameter optionals
        self.scaling_PV=scaling_PV
        self.File_Path_Energy_Scenario=File_Path_Energy_Scenario
        self.File_Path_Energy_Forecast = File_Path_Energy_Forecast
        self.File_Path_Observed_Values = File_Path_Observed_Values
        self.File_Path_Prices=File_Path_Prices
        self.File_Path_Prices_Imbalance = File_Path_Prices_Imbalance

        self.Battery_Data = Battery_Data
        self.Simulation_Data = Simulation_Data
        self.Fluxes_Names = Fluxes_Name
        self.quantile_probability=quantile_probability
      
        self.Energy_Data,self.Demand_Data,self.Energy_Scenario_df,self.Demand_Scenario_df = self.energy_scenarios()
        self.Energy_Forecast_df,self.Demand_Forecast_df = self.energy_forecast()
        self.Energy_Observed, self.Demand_Observed = self.obeserved_values()
        self.Market_Prices = self.market_prices()
        self.Imbalance_negative, self.Imbalance_positive =self.imbalance_prices()


    def xlsx_reading(self,path):
        df = pd.read_excel(path, engine='openpyxl')
        return df
    
    def csv_reading(self,path):
        df = pd.read_csv(path)
        return df
    
    def pickle_reading(self,path):
        with open(path,"rb") as file:
            Data = pickle.load(file)

        Energy ={}
        Demand ={}

        for key in Data.keys():
            day,type=key.split("_")
        
            if type =="load":
                Demand[day]=Data[key]
            elif type =="pv":
                Energy[day]=Data[key]
        return Energy,Demand

    def energy_scenarios(self):
        path = self.File_Path_Energy_Scenario
        Energy_Scenario,Demand_Scenario=self.pickle_reading(path)

        days = list(Energy_Scenario.keys())
        n_scenaros = len(Energy_Scenario[days[0]])

        Energy_Scenario_df_dict = {}
        Demand_Scenario_df_dict = {}
        for key in days:
            Energy_Scenario_arr = np.empty((24, n_scenaros))
            Demand_Scenario_arr = np.empty((24, n_scenaros))
            for s in range(n_scenaros):
                Energy_Scenario_arr[:, s] = Energy_Scenario[key][s]
                Demand_Scenario_arr[:, s] = Demand_Scenario[key][s]

            Energy_Scenario_df =  self.scaling_PV*pd.DataFrame(Energy_Scenario_arr, columns=range(n_scenaros), index=range(24))
            Demand_Scenario_df =  pd.DataFrame(Demand_Scenario_arr, columns=range(n_scenaros), index=range(24))
            Energy_Scenario_df_dict[key] = Energy_Scenario_df
            Demand_Scenario_df_dict[key] = Demand_Scenario_df

        return Energy_Scenario,Demand_Scenario,Energy_Scenario_df_dict,Demand_Scenario_df_dict
    
    def energy_forecast(self):
        path = self.File_Path_Energy_Forecast
        Energy_Forecasts,Demand_Forecasts=self.pickle_reading(path)
        
        days = list(Energy_Forecasts.keys())

        Energy_Forecast_df_dict = {}
        Demand_Forecast_df_dict = {}
        for key in days:
            dict = json.loads(Energy_Forecasts[key])
            p =list(dict['q50'].items())
            Energy_Forecast_df_dict[key] = self.scaling_PV*pd.DataFrame(list(dict['q50'].values()), columns=['value'])
            #Energy_Forecast_df_dict[key]['timestamp'] = pd.to_datetime(Energy_Forecast_df_dict[key]['timestamp'].astype(int), unit='ms')

            dict = json.loads(Demand_Forecasts[key])
            Demand_Forecast_df_dict[key] = pd.DataFrame(list(dict['q50'].values()), columns=['value'])
            #Demand_Forecast_df_dict[key]['timestamp'] = pd.to_datetime(Demand_Forecast_df_dict[key]['timestamp'].astype(int), unit='ms')

        return Energy_Forecast_df_dict,Demand_Forecast_df_dict
    
    def obeserved_values(self):
        path = self.File_Path_Observed_Values
        df = self.csv_reading(path)
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        Energy_observed = {}
        Demand_Observed = {}
        start = df.index.min().date()
        end   = df.index.max().date()
        date = start

        while date <= end:
            day = date.strftime('%Y-%m-%d')
            energy_observed = self.scaling_PV*df.loc[day]['res']
            load_observed   = df.loc[day]['load']
            energy_observed = pd.to_numeric(energy_observed,errors = 'coerce').to_list()
            load_observed   = pd.to_numeric(load_observed,errors = 'coerce').to_list()
            Energy_observed[day] = energy_observed
            Demand_Observed[day] = load_observed
            date+= timedelta(days=1)

        return Energy_observed, Demand_Observed
 
    def market_prices(self):
        df = pd.read_csv(self.File_Path_Prices)#, sep=";", header=None, skiprows=1)
        prices = df.iloc[:,1]
        # np.where(condition, value_if_true, value_if_false)
        processed_prices = np.where(prices > 0, prices / 1000, 0.001)
        Market_Price = processed_prices.tolist()
        return Market_Price
    
    def imbalance_prices(self):   ##################################
       # df = self.xlsx_reading(self.File_Path_Prices_Imbalance)
        df = pd.read_csv(self.File_Path_Prices_Imbalance)
        # convertire , in .
        # df['Prezzo Medio di Vendita (€/MWh)'] = df['Prezzo Medio di Vendita (€/MWh)'].astype(str).str.replace(',', '.', regex=False)
        # df['Prezzo Medio di Acquisto (€/MWh)'] = df['Prezzo Medio di Acquisto (€/MWh)'].astype(str).str.replace(',', '.', regex=False)
        # #to n umeric
        # df['Prezzo Medio di Vendita (€/MWh)'] = pd.to_numeric(df['Prezzo Medio di Vendita (€/MWh)'], errors='coerce')
        # df['Prezzo Medio di Acquisto (€/MWh)'] = pd.to_numeric(df['Prezzo Medio di Acquisto (€/MWh)'], errors='coerce')
        # 4. RIPARARE I BUCHI (NaN)
        # ffll() copia il prezzo dell'ora precedente se manca un dato
        df.ffill(inplace=True) 
        df.bfill(inplace=True) # Per sicurezza se manca la prima ora
        Imbalance_negative = df["Short"]
        Imbalance_positive = df["Long"]
        Imbalance_negative = Imbalance_negative/1000
        Imbalance_positive = Imbalance_positive/1000
        Imbalance_negative = Imbalance_negative.to_list()
        Imbalance_positive = Imbalance_positive.to_list()
        # Imbalance_positive.rename({'Prezzo Medio di Acquisto (€/MWh)': 'Imbalance_positive_buy'}, inplace=True)
        # Imbalance_negative.rename(columns={'Prezzo Medio di Vendita (€/MWh)': 'Imbalance_negative_sell'}, inplace=True)
        return Imbalance_negative,Imbalance_positive
    
    def chebyshev_interval(self,day):
        coverage = 0.9
        energy_bands = chebyshev_interval(coverage, self.Energy_Scenario_df[day])
        demand_bands = chebyshev_interval(coverage, self.Demand_Scenario_df[day])
        
        energy_limits ={'min': energy_bands["min"].tolist(),
                        'max': energy_bands["max"].tolist()}
        
        demand_limits ={'min': demand_bands["min"].tolist(),
                        'max': demand_bands["max"].tolist()}

        if self.Energy_Forecast_df[day].values.flatten().tolist():
            # 1) assure that minimum flexibility is less or equal to next day load
            energy_limits['min'] = [float(min(x, y)) for x, y in zip(self.Energy_Forecast_df[day].values.flatten().tolist(), energy_limits['min'])]
            # 2) assure that maximum flexibility is more or equal to next day load
            energy_limits['max'] = [float(max(x, y)) for x, y in zip(self.Energy_Forecast_df[day].values.flatten().tolist(), energy_limits['max'])]

        if self.Demand_Forecast_df[day].values.flatten().tolist():
            # 1) assure that minimum flexibility is less or equal to next day load
            demand_limits['min'] = [float(min(x, y)) for x, y in zip(self.Demand_Forecast_df[day].values.flatten().tolist(), demand_limits['min'])]
            # 2) assure that maximum flexibility is more or equal to next day load
            demand_limits['max'] = [float(max(x, y)) for x, y in zip(self.Demand_Forecast_df[day].values.flatten().tolist(), demand_limits['max'])]

        return energy_limits, demand_limits
    
    def PDF(self, day):
        #daily
        Energy = self.Energy_Data[day]
        Demand = self.Demand_Data[day]
        n_scenarios = len(Energy)
        timesteps = len(Energy[0])
        energy_mean = []
        energy_deviation = []
        demand_mean = []
        demand_deviation = []
        for t in range(timesteps):
            energy_values = [Energy[s][t] for s in range(n_scenarios)]
            demand_values = [Demand[s][t] for s in range(n_scenarios)]
            energy_values = self.scaling_PV*np.array(energy_values)
            energy_mean.append(np.mean(energy_values))
            energy_deviation.append(np.std(energy_values, ddof=1))
            demand_mean.append(np.mean(demand_values))
            demand_deviation.append(np.std(demand_values, ddof=1))

        return energy_mean, energy_deviation, demand_mean, demand_deviation

    def quantile(self,day):
        day_energy_scenarios_df = self.Energy_Scenario_df[day]
        day_demand_scenarios_df = self.Demand_Scenario_df[day]
        quantile_energy = []
        quantile_demand = []
        timesteps = len(day_energy_scenarios_df)
  
        for t in range(timesteps):
            hour_energy_scenarios = day_energy_scenarios_df.iloc[t]
            hour_energy_quantile =np.quantile(hour_energy_scenarios,self.quantile_probability)
            quantile_energy.append(hour_energy_quantile)

            hour_demand_scenarios = day_demand_scenarios_df.iloc[t]
            hour_demand_quantile =np.quantile(hour_demand_scenarios,1-self.quantile_probability)
            quantile_demand.append(hour_demand_quantile)

        return quantile_energy, quantile_demand
    
    def risk_computation(self, day):
        #PV#
        timesteps = len(self.Energy_Scenario_df[day])
        quantile_energy, quantile_demand = self.quantile(day)
        energy_risk = []
        demand_risk = []
        for t in range(timesteps):
            hour_energy_scenarios = self.Energy_Scenario_df[day].iloc[t]
            hour_demand_scenarios = self.Demand_Scenario_df[day].iloc[t]

            lower_values_energy = hour_energy_scenarios[hour_energy_scenarios <= quantile_energy[t]]
            lower_values_demand = hour_demand_scenarios[hour_demand_scenarios >= quantile_demand[t]]

            errors_energy = quantile_energy[t]-lower_values_energy
            errors_demand = lower_values_demand - quantile_demand[t]

            mean_error_energy = np.mean(errors_energy)
            mean_error_demand = np.mean(errors_demand)

            risk_energy = self.quantile_probability*self.Market_Prices[t]*mean_error_energy
            risk_demand = self.quantile_probability*self.Market_Prices[t]*mean_error_demand
            energy_risk.append(risk_energy)
            demand_risk.append(risk_demand)

        return energy_risk, demand_risk
    
    def to_dict(self,day):

        energy_mean, energy_deviation, demand_mean, demand_deviation = self.PDF(day)
        energy_limits, demand_limits = self.chebyshev_interval(day)
        quantile_energy, quantile_demand = self.quantile(day)
        energy_forecast = [max(1e-6, val) for val in self.Energy_Forecast_df[day].values.flatten().tolist()]
        demand_forecast = [max(1e-6, val) for val in self.Demand_Forecast_df[day].values.flatten().tolist()]

        Energy = {'mean':energy_mean,
                  'deviation': energy_deviation,
                  'quantile': quantile_energy,
                  'scenario': self.Energy_Scenario_df[day].to_json(),
                  'forecast': energy_forecast,
                  'bands':  energy_limits,
                  'Observed': self.Energy_Observed[day]}
        
        Demand = {'mean':demand_mean,
                  'deviation': demand_deviation,
                  'quantile': quantile_demand,
                  'scenario': self.Demand_Scenario_df[day].to_json(),
                  'forecast': demand_forecast ,
                  'bands': demand_limits,
                  'Observed': self.Demand_Observed[day]}
        
        Prices = {'Market': self.Market_Prices,
                  'Imbalance_positive': self.Imbalance_positive,
                  'Imbalance_negative': self.Imbalance_negative}

        data_dict = {"Battery":self.Battery_Data,
                    "Simulation": self.Simulation_Data,
                    "Fluxes": self.Fluxes_Names,
                    "Energy": Energy,
                    "Demand": Demand,
                    "Prices": Prices}
        return data_dict 
    
    def json_print(self,data_dict,folder_path,name_file):

        file_name = name_file + ".json"
        full_path = os.path.join(folder_path, file_name)
        
        with open(full_path, "w") as file:
            json.dump(data_dict, file, indent=4)
    
        print(f"JSON saved at: {full_path}")
        return