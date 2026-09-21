import os
import json
import numpy as np
from datetime import datetime, timedelta

# SCRIPT PER LA NORMALIZZAZIONE

# CONFIGURAZIONE
JSON_FOLDER = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/json_repository_for_simulation/SystemData"  # La cartella dove hai i file data_2024-05-01.json
OUTPUT_FILE = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/json_repository_for_simulation/normalization_stats.json"
test_months = [2, 7, 10] #1=jenuary 12 = december
train_months = [1,3,4,5,6,8,9,11,12] #1=jenuary 12 = december
def generate_global_stats():
    print(f"--- Inizio scansione cartella: {JSON_FOLDER} ---")
    
    # Accumulatori
    all_energy_mean = []
    all_energy_dev  = []
    all_demand_mean = []
    all_demand_dev  = []

    files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
    
    if not files:
        print("ERRORE: Nessun file JSON trovato!")
        return

    for filename in files:
        _,day_json = filename.split("_")
        day_string,_ = day_json.split(".")
        data_obj = datetime.fromisoformat(day_string)
        if data_obj.month in train_months:
            filepath = os.path.join(JSON_FOLDER, filename)

        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                # Estraiamo le liste orarie (24 valori per file)
                all_energy_mean.extend(data['Energy']['mean'])
                all_energy_dev.extend(data['Energy']['deviation'])
                all_demand_mean.extend(data['Demand']['mean'])
                all_demand_dev.extend(data['Demand']['deviation'])
            except KeyError:
                print(f"Skipping {filename}: Formato non valido")

    print(f"Totale campioni processati: {len(all_energy_mean)} ore")

    # CALCOLO STATISTICHE GLOBALI
    # Struttura Vettore: [En_Mean, En_Dev, Dem_Mean, Dem_Dev, Sin, Cos]
    
    stats = {
        "mu": [
            float(np.mean(all_energy_mean)),
            float(np.mean(all_energy_dev)),
            float(np.mean(all_demand_mean)),
            float(np.mean(all_demand_dev)),
        ],
        "sigma": [
            float(np.std(all_energy_mean)),
            float(np.std(all_energy_dev)),
            float(np.std(all_demand_mean)),
            float(np.std(all_demand_dev)),
        ],
        "info": "Calculated on entire dataset. Order: [EnMean, EnDev, DemMean, DemDev]"
    }

    # SALVATAGGIO
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"✅ File salvato con successo: {OUTPUT_FILE}")
    print(f"Mean Vector: {stats['mu']}")
    print(f"Std Vector:  {stats['sigma']}")


generate_global_stats()
# if __name__ == "__main__":
#     generate_global_stats()
