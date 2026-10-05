from datetime import datetime, timedelta
import os
import json
import numpy as np
import _path
from learned_cfa_src.directory_names import data_json_dir, results_dir

# base_dir  = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.dirname(base_dir)
# data_json_dir  =  os.path.join(data_dir,'json_repository_for_simulation','SystemData')
# #scenario_json_dir  =  os.path.join(base_dir, 'Data','json_repository_for_simulation','ScenarioData')

if __name__ == "__main__":
    # Get list of all .json files in the directory
    data_json_files = [file for file in os.listdir(data_json_dir) if file.endswith('.json')]
    # scenario_json_files = [file for file in os.listdir(scenario_json_dir) if file.endswith('.json')]
    # load .Json Statistics of DATAFRAME
    # Data_Statistics_path = os.path.join(base_dir, 'Data','json_repository_for_simulation','normalization_stats.json')
    # with open(Data_Statistics_path, "r") as file:
    #     Data_Statistics = json.load(file)
    # # Dictionary to store contents

    # Load each JSON file and create the problem_Data from them

    start_date_data = datetime(2024,1,1) # Choice how many days you want to load as Data and to train
    end_date_data = datetime(2024,12,31)
    date = start_date_data

    corrupted_demands_days= []
    ANOMALIE_LOG_TOT = {}
    ANOMALIE_LOG_TEST = {}
    test_months = [2, 7, 10]

    while date <= end_date_data:
        # 2. WHITELIST: Se stiamo addestrando o ottimizzando la NN, salta tutto ciò che non è in train_months

        y = date.strftime('%Y')
        m = date.strftime('%m')
        d = date.strftime('%d')
        day= f'{y}-{m}-{d}'
        data_found = False
    
        for filename in data_json_files:
            if filename.endswith(f"_{day}.json"):
                file_path = os.path.join(data_json_dir, filename)
                with open(file_path, 'r') as file:
                    Data = json.load(file)
                data_found = True

        # for filename in scenario_json_files:
        #     if filename.endswith(f"_{day}.json"):
        #         file_path = os.path.join(scenario_json_dir, filename)
        #         with open(file_path, 'r') as file:
        #             scenario_Milps = json.load(file)
        #         Scenario_found = True

        if data_found == True:
            print(f"[OK] {day} Loaded.")#
        
            observed_load = np.array(Data["Demand"]["Observed"])
            current_month = int(date.strftime('%m'))
            # --- DEFINIZIONE CRITERI ---
            is_bad = False
            reasons = []
        
            # 1. Check Zeri (Sensore spento)
            if np.count_nonzero(observed_load == 0) > (len(observed_load) * 0.4):
                is_bad = True
                reasons.append("Troppi Zeri (Data loss)")
            
            # 2. Check Valori Negativi
            if np.any(observed_load < 0):
                is_bad = True
                reasons.append("Carico Negativo (Sensor Error)")
            
            # 3. Check Picchi Irrealistici (es. > 1000 kW se la media è 50)
            # if np.max(observed_load) > 500: # Inserisci un valore limite fisico per il tuo impianto
            #     is_bad = True
            #     reasons.append("Picco fuori scala")

            if is_bad:
                corrupted_demands_days.append(day)
                ANOMALIE_LOG_TOT[day] = {
                    "reasons": reasons,
                    "demand_data": observed_load.tolist() # Convertiamo in lista per JSON
                }
                print("_____________________________________________________________________________")
                print(f"[REJECTED] {day} | Motivi: {reasons} ")
                print(f"Observed Values corrupted | observed_load ")

                if current_month in test_months:
                    ANOMALIE_LOG_TEST[day] = {
                    "reasons": reasons,
                    "demand_data": observed_load.tolist() # Convertiamo in lista per JSON
                }
           # else:
                #print("_____________________________________________________________________________")
                #print(f"[OK] {day} Loaded. Mean Load: {np.mean(observed_load):.2f}")
        
            # Creation of a Problem_Data OBJ for each day
            #DATA[day] = Problem_Data(Data["Battery"],Data["Simulation"],Data["Fluxes"],Data["Prices"],Data["Energy"],Data["Demand"],scenario_Milps["Costs"],scenario_Milps["Grid_Exchange"],scenario_Milps["Grid_Exchange_Baseline"])
        else:
            print(f"[WARNING]  {day} Data not Found.") #⚠️

        date+=timedelta(days=1) # needed to progress

    # --- SALVATAGGIO DEL LOG ---
    os.makedirs(results_dir, exist_ok=True)
    path_log_tot  = os.path.join(results_dir, 'anomalie_dataset.json')
    with open(path_log_tot, 'w') as f:
        json.dump(ANOMALIE_LOG_TOT, f, indent=4)

    path_log_test  = os.path.join(results_dir, 'anomalie_data_Test.json')
    with open(path_log_test, 'w') as f:
        json.dump(ANOMALIE_LOG_TEST, f, indent=4)

    print(f"\nAnalisi completata. Trovati {len(ANOMALIE_LOG_TOT)} giorni corrotti.")
    print(f"Log salvato in: {results_dir}")