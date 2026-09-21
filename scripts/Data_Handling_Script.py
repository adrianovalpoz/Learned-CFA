### Script per mopdificare i dati della simulazione crea grazie a problem data clas un json con dentro un dizionario con i dati
from Data_Handling import Data_Handling
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


################################################## INPUTS AND DATA #############################################################
####### Simulation #######
Simulation= {"time_resolution": 1, # sec,min,h
              "time_horizon"  : 24, # hours
            }

timesteps = Simulation["time_horizon"]

timesteps = np.arange(Simulation["time_horizon"])
sin_enc = np.sin(2 * np.pi * timesteps / Simulation["time_horizon"])
cos_enc = np.cos(2 * np.pi * timesteps / Simulation["time_horizon"])

Simulation["sin_enc"]= sin_enc.tolist()
Simulation["cos_enc"]= cos_enc.tolist()

####### Battery ##########
capacity = 755/0.6
Battery_Data = {"SoC_in"          : 0.5,  # Initial State of Charge
                "Storage_Capacity": capacity,  # [kWh]
                "eff_charge"      : 0.9,  #
                "eff_discharge"   : 0.9,  #
                "Max_Charge"      : capacity*0.7, #
                "Max_Discharge"   : capacity*0.7  #
                 }
######## Variable Name #######
Fluxes_Names= {"Battery Fluxes"      : ["x_Market_Storage","x_Energy_Storage","x_Storage_Market","x_Storage_Demand"],
                "Battery Fluxes Type": {"x_Market_Storage":1,"x_Energy_Storage":1,"x_Storage_Market":0,"x_Storage_Demand":0},
                "Other Fluxes"       : ["x_Energy_Market" , "x_Energy_Demand", "x_Market_Demand"]
                }

Fluxes_Names["Total Fluxes"]= Fluxes_Names["Other Fluxes"] + Fluxes_Names["Battery Fluxes"]
##### plot inputs###################
plot = "no"

quantile_probability= 0.3
####### energy and price Data ########

File_path_Energy           = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/scenarios_full.pkl"
File_path_Forecast         = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/forecasts_full.pkl"
File_Path_Observed_Values  = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/observed_values.csv"

column_names = ["Year", "Month", "Day", "Index", "Price[euro/MWh]", "Value2","0"]

### day iteration ###
start_date = datetime(2024, 1, 1)#2024, 5, 1)
end_date = datetime(2024, 12, 31) #2024, 10, 22
date = start_date
scaling_PV =1.5
while date <= end_date:
  #df = pd.read_excel(File_Path_Prices_Imbalance)
  y = date.strftime('%Y')
  m = date.strftime('%m')
  d = date.strftime('%d')
  ymd = date.strftime('%Y%m%d')
########################################################################################################################
  File_Path_Prices_Imbalance = f"C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/imbalance/Imbalance_prices_{ymd}.csv"
  File_Path_Prices           = f"C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/prices/DA_prices_{ymd}.csv"
  #excel = pd.read_excel(File_Path_Prices_Imbalance, engine='openpyxl')
  DATA = Data_Handling(Battery_Data, Simulation, Fluxes_Names,quantile_probability,File_path_Energy,File_path_Forecast, File_Path_Prices,File_Path_Prices_Imbalance,column_names,File_Path_Observed_Values,scaling_PV)
  #print(DATA.Energy_Scenario_df[day][1])

  #day = '2024-05-02' ############### walways check ---->number0 0 ----<'2024-05-01'
  day= f'{y}-{m}-{d}'
  
  coverage = 0.9
  data_dict = DATA.to_dict(day)
  name_file = f"data_{day}"
  file_path='C:/Users/adria/Desktop/INESCTEC/00.Code/Architecture_Neural_Network/Data/json_repository_for_simulation/SystemData'

  DATA.json_print(data_dict,file_path,name_file)
 
  date+= timedelta(days=1)



# %% Plottig
# DATA ENERGY



if plot == "YES":
  fig, axes = plt.subplots(2, 1, figsize=(6, 10))

  #sns.lineplot(x=range(1,timesteps+1), y=data_dict["Energy"]["forecast"], label="PV Energy",ax=axes[0],marker = 'o')
  # sns.lineplot(x=range(1,timesteps+1), y=data_dict["Energy"]["bands"]["min"], label= "Chebyshev lower bound",  ax=axes[0],marker = 'o')
  # sns.lineplot(x=range(1,timesteps+1), y=data_dict["Energy"]["bands"]["max"], label= "Chebyshev upper bound",  ax=axes[0],marker = 'o')
  # #sns.lineplot(x=range(1,timesteps+1), y=data_dict["Demand"]["forecast"], label="Demand",ax=axes[1],marker = 'o')
  # sns.lineplot(x=range(1,timesteps+1), y=data_dict["Demand"]["bands"]["min"], label= "Chebyshev lower bound",  ax=axes[1],marker = 'o')
  # sns.lineplot(x=range(1,timesteps+1), y=data_dict["Demand"]["bands"]["max"], label= "Chebyshev upper bound",  ax=axes[1],marker = 'o')
  quantiles = [0,0.2,0.4,0.6,0.8,1]
  fig = plt.figure(figsize=(6, 10))
  for q in quantiles:
    DATA = Data_Handling(Battery_Data, Simulation, Fluxes_Names,q,File_path_Energy,File_path_Forecast, File_Path_Prices,File_Path_Prices_Imbalance, column_names)
    coverage = 0.9
    data_dict = DATA.to_dict(day)
    energy_risk, demand_risk = DATA.risk_computation(day)
    print(f" alpha = {q}, energy_risk = {sum(energy_risk)} and demand_risk = {sum(demand_risk)}")
    sns.lineplot(x=range(1,timesteps+1), y=data_dict["Energy"]["quantile"], label= f"quantile {q} upper bound",marker = 'o')
    #sns.lineplot(x=range(1,timesteps+1), y=data_dict["Demand"]["quantile"], label= f"quantile {q} lower bound",  ax=axes[1],marker = 'o')


  titles = ["PV","Demand"]
  units  = ["kW","kW"]
  plt.title("PV")
  plt.xlabel("time")
  plt.ylabel("kW")
  plt.grid(True)
  # for ax in range(len(axes)):
  #     axes[ax].set_title(titles[ax])
  #     axes[ax].set_xlabel("time")
  #     axes[ax].set_ylabel(units[ax])
  #     axes[ax].grid(True)

  fig = plt.figure(figsize=(6, 10))
  for q in quantiles:
    DATA = Data_Handling(Battery_Data, Simulation, Fluxes_Names,q,File_path_Energy,File_path_Forecast, File_Path_Prices,File_Path_Prices_Imbalance,column_names)
    coverage = 0.9
    data_dict = DATA.to_dict(day)
    energy_risk, demand_risk = DATA.risk_computation(day)
    print(f" alpha = {q}, energy_risk = {sum(energy_risk)} and demand_risk = {sum(demand_risk)}")
    #sns.lineplot(x=range(1,timesteps+1), y=data_dict["Energy"]["quantile"], label= f"quantile {q} upper bound",  ax=axes[0],marker = 'o')
    sns.lineplot(x=range(1,timesteps+1), y=data_dict["Demand"]["quantile"], label= f"quantile {q} lower bound",marker = 'o')
  
  plt.title("Demand")
  plt.xlabel("time")
  plt.ylabel("kW")
  plt.grid(True)
 
  plt.show()

