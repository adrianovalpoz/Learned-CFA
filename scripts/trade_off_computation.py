from datetime import datetime, timedelta
import os
import json
import numpy as np
import _path
from learned_cfa_src.directory_names import data_json_dir, scenario_json_dir
from learned_cfa_src.Optimize_Risk import imbalance_costs, RightCVaR

def z_scoreNormalization(value,mean,sigma):
    norm = (value-mean)/sigma
    return norm
def trade_off_computation(Cx_n,CDA_w_n,CVar_w_n,CVaR_b_n):
    lamba = (CDA_w_n-Cx_n)/(CVar_w_n-CVaR_b_n)
    return lamba
def indifferent_condition(Q,M,P,q50):
    M[0]=q50 
    beta =  (Q[0] - M[0])/ (Q[0] - P[0])
    return M[0],beta


def imbalance(Baseline_Grid_Exchange, Grid_Exchange_Milps, positive_price_imbalance, negative_price_imbalance):
    """VaR and tail sigma of the scenario imbalance costs, against the baseline exchange."""
    totali = [sum(imbalance_costs(g, Baseline_Grid_Exchange,
                                  positive_price_imbalance, negative_price_imbalance)[0])
              for g in Grid_Exchange_Milps]
    _, VaR, sigma_tail = RightCVaR(totali, alpha=0.95)
    return VaR, sigma_tail

if __name__ == "__main__":
    # Get list of all .json files in the directory
    scenario_json_files = [file for file in os.listdir(scenario_json_dir) if file.endswith('.json')]
    system_data_json_files = [file for file in os.listdir(data_json_dir) if file.endswith('.json')]


    DATA = {}

    # Load each JSON file and create the problem_Data from them

    start_date_data = datetime(2024,1,1) # Choice how many days you want to load as Data and to train
    end_date_data = datetime(2024,12,31)
    date = start_date_data
    train_months = [1,3,4,5,6,8,9,11,12]

    LAMBDA = []
    BETA = []
    error_ouliners= 0
    while date <= end_date_data:
        # 2. WHITELIST: Se stiamo addestrando o ottimizzando la NN, salta tutto ciò che non è in train_months

        if date.month not in train_months:
            date+=timedelta(days=1)
            continue

        y = date.strftime('%Y')
        m = date.strftime('%m')
        d = date.strftime('%d')
        day= f'{y}-{m}-{d}'
        data_found = False
        system_data_found = False
    
        for filename in scenario_json_files:
            if filename.endswith(f"_{day}.json"):
                file_path = os.path.join(scenario_json_dir, filename)
                with open(file_path, 'r') as file:
                    Data = json.load(file)
                data_found = True

        for filename in system_data_json_files:
            if filename.endswith(f"_{day}.json"):
                file_path = os.path.join(data_json_dir, filename)
                with open(file_path, 'r') as file:
                    System_Data = json.load(file)
                system_data_found = True

        if data_found == True and system_data_found == True:
            print(f"[OK] {day} Loaded.")#
            positive_price_imbalance = System_Data["Prices"]["Imbalance_positive"]
            negative_price_imbalance = System_Data["Prices"]["Imbalance_negative"]
            ## extremes ######

            P = [Data["C_best_cheby"],Data["ICVaR_worst_Cheby"]]
            Q = [Data["C_worst_cheby"],Data["ICVaR_best_cheby"]]
            M = [None,Data["C_worst_cheby"]]
            q50 = Data["C_baseline"]
            ##### Normalization z-score
            Da_Scenarios_mean = np.mean(np.array(Data["Costs"]))
            Da_Scenarios_variance = np.std(np.array(Data["Costs"]))
            Imbalance_Costs_VaR_baseline,sigma_tail_baseline= imbalance(Data["Grid_Exchange_Baseline"],Data["Grid_Exchange"],positive_price_imbalance,negative_price_imbalance)
       
            CVaR_w_n = z_scoreNormalization(P[1],Imbalance_Costs_VaR_baseline,sigma_tail_baseline)
            CVaR_b_n = z_scoreNormalization(Q[1],Imbalance_Costs_VaR_baseline,sigma_tail_baseline)
            CDA_w_n = z_scoreNormalization(Q[0],q50,Da_Scenarios_variance)
            Cx,beta = indifferent_condition(Q,M,P,q50)
            Cx_n = z_scoreNormalization(Cx,q50,Da_Scenarios_variance)
            gamma = trade_off_computation(Cx_n,CDA_w_n,CVaR_w_n,CVaR_b_n)
            if gamma <0:
                # a negative lambda has no meaning here: the day is discarded instead of being
                # clamped to zero, so that it does not drag the average down
                print(f"[DISCARDED] {day}: lambda = {gamma}")
                error_ouliners += 1
            else:
                LAMBDA.append(gamma)

            BETA.append(beta)
        else:
            print(f"[WARNING]  {day} Data not Found.") #⚠️

        date+=timedelta(days=1) # needed to progress

    print(f"\naverage lambda = {np.mean(LAMBDA)} over {len(LAMBDA)} days | average beta = {np.mean(BETA)} over {len(BETA)} days")

    print(f"\nnumber of discarded days = {error_ouliners}")
