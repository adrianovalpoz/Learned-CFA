import os
import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta

# 1. INCOLLA QUI IL TUO TOKEN
API_TOKEN = os.environ["ENTSOE_API_TOKEN"]
client = EntsoePandasClient(api_key=API_TOKEN)

# 2. CONFIGURAZIONE
country_code = 'PT'  # Bidding zone del Portogallo


save_dir_imb = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/imbalance"
save_dir_DA = "C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/Data/prices"
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
    # 3. DOWNLOAD E SALVATAGGIO
    print(f"Contattando il server ENTSO-E per il giorno {ymd}...")
    try:
        # La libreria gestisce automaticamente impaginazione e rate-limiting
        imbalance_prices = client.query_imbalance_prices(country_code, start=start, end=end)
        da_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        da_prices.to_csv(filename_DA)
        print(f"✅ Prezzi Day-Ahead salvati in: {filename_DA}")
        # Salva il DataFrame direttamente in CSV
        imbalance_prices.to_csv(filename_imb)
        print(f"✅ Download completato con successo! Dati salvati in: {filename_imb}")
        
    except Exception as e:
        print(f"❌ Errore durante l'interrogazione dell'API: {e}")

    date+= timedelta(days=1)
