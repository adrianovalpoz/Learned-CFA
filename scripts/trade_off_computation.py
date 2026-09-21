from datetime import datetime, timedelta
import os
import json
import numpy as np

def z_scoreNormalization(value,mean,sigma):
    norm = (value-mean)/sigma
    return norm
def trade_off_computation(Cx_n,CDA_w_n,CVar_w_n,CVaR_b_n):
    lamba = (CDA_w_n-Cx_n)/(CVar_w_n-CVaR_b_n)
    return lamba
def indifferent_condition(Q,M,P,q50):
#     deltaCVaR = P[1]-Q[1]
#     deltaCDA = Q[0] - P[0]
#     M[0] = Q[0] + Q[1] - M[1]
#     beta=0.6568755020962364
#     #rangetorange
#     # beta = deltaCVaR/(deltaCVaR+deltaCDA)
#    # beta = 1
#     M[0] = Q[0] - beta * (Q[0] - P[0])
    #M[0] = P[0]+P[1]
    M[0]=q50 
    beta =  (Q[0] - M[0])/ (Q[0] - P[0])
    return M[0],beta

def RightCVaR(Data,alpha):

    Data = np.asarray(Data)
    if Data.size == 0:
        return 0,0,0  # O un altro valore predefinito
    
    VaR = np.percentile(Data, (alpha)*100)
    tail_data = Data[Data >= VaR]

    
    CVaR = tail_data.mean()
    sigma_tail = tail_data.std() + 1e-6
    #print(f"CVaR = {CVaR:.2f}| VaR = {VaR:.2f} | lenght = {len(Data[Data >= VaR]):.2f}/500")
  
    return CVaR,VaR,sigma_tail

def imbalance(Baseline_Grid_Exchange,Grid_Exchange_Milps,positive_price_imbalance,negative_price_imbalance):

    
    N_scenarios = len(Grid_Exchange_Milps)


    timesteps = 24
    Imbalance_costs_baseline_tot = []
    for s in range(N_scenarios):
        Imbalance_costs_baseline = []
       
        for t in range(timesteps):
            imbalance_baseline= Grid_Exchange_Milps[s][t] - Baseline_Grid_Exchange[t]
         
            if imbalance_baseline >= 0:
                if positive_price_imbalance[t]<0:
                    Imbalance_costs_baseline.append(-positive_price_imbalance[t]*imbalance_baseline)
                else:
                    Imbalance_costs_baseline.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
            elif imbalance_baseline <0:

                if negative_price_imbalance[t]>0:
                    Imbalance_costs_baseline.append(-negative_price_imbalance[t]*imbalance_baseline)
                else:
                    Imbalance_costs_baseline.append(0) #### nullify the effects on imbalance when we are helping the system (we dont consider it as a profit)
        

        Imbalance_costs_baseline_tot.append(sum(Imbalance_costs_baseline))

    Imbalance_Costs_CVaR_baseline,Imbalance_Costs_VaR_baseline,sigma_tail_baseline = RightCVaR(Imbalance_costs_baseline_tot,alpha=0.95) 
    return Imbalance_Costs_VaR_baseline,sigma_tail_baseline


base_dir  = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(base_dir)
data_json_dir  =  os.path.join(data_dir,'json_repository_for_simulation','ScenarioData')
system_data_json_dir = os.path.join(data_dir,'json_repository_for_simulation','SystemData')


# Get list of all .json files in the directory
data_json_files = [file for file in os.listdir(data_json_dir) if file.endswith('.json')]
system_data_json_files = [file for file in os.listdir(system_data_json_dir) if file.endswith('.json')]


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
    
    for filename in data_json_files:
        if filename.endswith(f"_{day}.json"):
            file_path = os.path.join(data_json_dir, filename)
            with open(file_path, 'r') as file:
                Data = json.load(file)
            data_found = True

    for filename in system_data_json_files:
        if filename.endswith(f"_{day}.json"):
            file_path = os.path.join(system_data_json_dir, filename)
            with open(file_path, 'r') as file:
                System_Data = json.load(file)
            system_data_found = True

    # for filename in scenario_json_files:
    #     if filename.endswith(f"_{day}.json"):
    #         file_path = os.path.join(scenario_json_dir, filename)
    #         with open(file_path, 'r') as file:
    #             scenario_Milps = json.load(file)
    #         Scenario_found = True

    if data_found == True and system_data_found == True:
        print(f"[OK] {day} Loaded.")#
        positive_price_imbalance = System_Data["Prices"]["Imbalance_positive"]
        negative_price_imbalance = System_Data["Prices"]["Imbalance_negative"]
        ## extremes ######

        # P = [Data["Best_DA"],Data["Worst_ICVaR"]]
        # Q = [Data["Worst_DA"],Data["Best_ICVaR"]]
        # M = [None,Data["Worst_ICVaR"]]

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
            print(f"{day} e {gamma}")
            gamma=0
            error_ouliners += 1
            
        LAMBDA.append(gamma)
        BETA.append(beta)
    else:
        print(f"[WARNING]  {day} Data not Found.") #⚠️

    date+=timedelta(days=1) # needed to progress

print(f"\navarage lambda = {np.mean(LAMBDA)}| varage beta = {np.mean(BETA)}")

print(f"\nnumber of outliners ={error_ouliners}")
