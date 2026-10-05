import os
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import _path
from learned_cfa_src.directory_names import prices_dir, imbalance_dir

# 1. INCOLLA QUI IL TUO TOKEN
API_TOKEN = ""
if not API_TOKEN:
    raise SystemExit(
        "No ENTSO-E token set. Paste your token into API_TOKEN at the top of this file. "
        "The token is free: register at https://transparency.entsoe.eu and generate it "
        "under Account Settings -> Web Api Security Token."
    )
client = EntsoePandasClient(api_key=API_TOKEN)

# 2. CONFIGURAZIONE
country_code = 'PT'  # Bidding zone del Portogallo


save_dir_imb = imbalance_dir
save_dir_DA = prices_dir
os.makedirs(save_dir_imb, exist_ok=True)
os.makedirs(save_dir_DA, exist_ok=True)

### day iteration ###
start_date = datetime(2024, 1, 1)#2024, 5, 1)
end_date = datetime(2024, 12, 31) #2024, 10, 22
date = start_date

while date <= end_date:
    #df = pd.read_excel(File_Path_Prices_Imbalance)
    y = date.strftime('%Y')
    m = date.strftime('%m')
    d = date.strftime('%d')

    ymd = date.strftime('%Y%m%d')

    start = pd.Timestamp(date, tz='Europe/Lisbon')
    end = start + pd.Timedelta(days=1)-pd.Timedelta(minutes=1)

    filename_imb = os.path.join(save_dir_imb, f"Imbalance_prices_{ymd}.csv")
    filename_DA = os.path.join(save_dir_DA, f"DA_prices_{ymd}.csv")

    # Days already on disk are skipped, so that a second run only fills the gaps left by the
    # days that failed, instead of querying the public API again for the whole year.
    if os.path.isfile(filename_DA) and os.path.isfile(filename_imb):
        print(f"{ymd} gia' presente, salto.")
        date += timedelta(days=1)
        continue

    # 3. DOWNLOAD E SALVATAGGIO
    print(f"Contattando il server ENTSO-E per il giorno {ymd}...")
    try:
        # La libreria gestisce automaticamente impaginazione e rate-limiting
        imbalance_prices = client.query_imbalance_prices(country_code, start=start, end=end)
        da_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        da_prices.to_csv(filename_DA)
        print(f"[OK] Prezzi Day-Ahead salvati in: {filename_DA}")
        # Salva il DataFrame direttamente in CSV
        imbalance_prices.to_csv(filename_imb)
        print(f"[OK] Download completato con successo! Dati salvati in: {filename_imb}")
        
    except Exception as e:
        print(f"[X] Errore durante l'interrogazione dell'API: {e}")

    date+= timedelta(days=1)
