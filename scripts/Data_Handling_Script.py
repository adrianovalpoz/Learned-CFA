### Script per mopdificare i dati della simulazione crea grazie a problem data clas un json con dentro un dizionario con i dati
import os
import json
import numpy as np
from datetime import datetime, timedelta

import _path
from learned_cfa_src.data_pipeline.Data_Handling import Data_Handling
from learned_cfa_src.directory_names import data_json_dir, prices_dir, imbalance_dir, raw_data_dir, config_dir
################################################## INPUTS AND DATA #############################################################
# Parametri fisici del caso studio (modificarli in config/system_parameters.json)
with open(os.path.join(config_dir, 'system_parameters.json'), "r") as f:
    system_parameters = json.load(f)

####### Simulation #######
Simulation = dict(system_parameters["simulation"])

timesteps = Simulation["time_horizon"]

timesteps = np.arange(Simulation["time_horizon"])
sin_enc = np.sin(2 * np.pi * timesteps / Simulation["time_horizon"])
cos_enc = np.cos(2 * np.pi * timesteps / Simulation["time_horizon"])

Simulation["sin_enc"]= sin_enc.tolist()
Simulation["cos_enc"]= cos_enc.tolist()

####### Battery ##########
Battery_Data = system_parameters["battery"]
######## Variable Name #######
Fluxes_Names= {"Battery Fluxes"      : ["x_Market_Storage","x_Energy_Storage","x_Storage_Market","x_Storage_Demand"],
                "Battery Fluxes Type": {"x_Market_Storage":1,"x_Energy_Storage":1,"x_Storage_Market":0,"x_Storage_Demand":0},
                "Other Fluxes"       : ["x_Energy_Market" , "x_Energy_Demand", "x_Market_Demand"]
                }

Fluxes_Names["Total Fluxes"]= Fluxes_Names["Other Fluxes"] + Fluxes_Names["Battery Fluxes"]


quantile_probability = 0.3        # usato solo da questo script
####### energy and price Data ########

# File_path_Energy           = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/scenarios_full.pkl"
# File_path_Forecast         = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/forecasts_full.pkl"
# File_Path_Observed_Values  = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/observed_values.csv"
File_path_Energy          = os.path.join(raw_data_dir, 'scenarios_full.pkl')
File_path_Forecast        = os.path.join(raw_data_dir, 'forecasts_full.pkl')
File_Path_Observed_Values = os.path.join(raw_data_dir, 'observed_values.csv')

column_names = ["Year", "Month", "Day", "Index", "Price[euro/MWh]", "Value2","0"]

if __name__ == "__main__":
    ### day iteration ###
    start_date = datetime(2024, 1, 1)#2024, 5, 1)
    end_date = datetime(2024, 12, 31) #2024, 10, 22
    date = start_date
    scaling_PV = 1.5                  # usato solo da questo script
    while date <= end_date:

      y = date.strftime('%Y')
      m = date.strftime('%m')
      d = date.strftime('%d')
      ymd = date.strftime('%Y%m%d')
    ########################################################################################################################

      File_Path_Prices_Imbalance = os.path.join(imbalance_dir, f"Imbalance_prices_{ymd}.csv")
      File_Path_Prices           = os.path.join(prices_dir, f"DA_prices_{ymd}.csv")
      #excel = pd.read_excel(File_Path_Prices_Imbalance, engine='openpyxl')
      DATA = Data_Handling(Battery_Data, Simulation, Fluxes_Names,quantile_probability,File_path_Energy,File_path_Forecast, File_Path_Prices,File_Path_Prices_Imbalance,column_names,File_Path_Observed_Values,scaling_PV)
  
      #day = '2024-05-02' ############### walways check ---->number0 0 ----<'2024-05-01'
      day= f'{y}-{m}-{d}'
  
      coverage = 0.9
      data_dict = DATA.to_dict(day)
      name_file = f"data_{day}"
  
      os.makedirs(data_json_dir, exist_ok=True)
      DATA.json_print(data_dict,data_json_dir,name_file)
 
      date+= timedelta(days=1)


