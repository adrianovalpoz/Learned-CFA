import numpy as np
import pandas as pd
import torch
import optuna
import copy
import os
import time
import concurrent.futures
import psutil
try:
    from .Policy import PolicyNetwork
except ImportError:
    from Policy import PolicyNetwork

# #from .Policy import PolicyNetworkLTSM
# import matplotlib
# # Disabilitiamo l'interfaccia grafica a finestre prima di caricare pyplot
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns


# def global_reward_func(sum_reward):
#         # alpha = 0
#     # beta  = 1
#  # Total Costs considering the risks
#     # gamma = 1
#     # DA_best = -33115.6
#     # DA_worst = 55854.81
#     # Imbalance_Costs_CVaR_worst = 71567.92
#     # Imbalance_Costs_CVaR_best =  9130.47
#     # DA_baseline = -2431.72
#     # Imbalance_Costs_CVaR_baseline = 27558.1
#     # Reward_aggressive = DA_best + gamma*Imbalance_Costs_CVaR_worst    
#     # Reward_baseline =DA_baseline  + gamma*Imbalance_Costs_CVaR_baseline
#     # Reward_conservative = DA_worst + gamma*Imbalance_Costs_CVaR_best

#     # reward_norm = (sum_reward-Reward_conservative)/(Reward_aggressive-Reward_conservative)
#     #Planned_Costs_norm = Planned_Costs/2432.79#-DA_min)/(DA_max-DA_min)
#     # Planned_Costs_norm = (Planned_Costs-DA_best)/(DA_worst-DA_best)
#     # #Imbalance_Costs_CVaR_norm = Imbalance_Costs_CVaR/27558.1
#     # Imbalance_Costs_CVaR_worst = 71567.92
#     # Imbalance_Costs_CVaR_best =  9130.47
#     # Imbalance_Costs_CVaR_norm = (Imbalance_Costs_CVaR-Imbalance_Costs_CVaR_best)/(Imbalance_Costs_CVaR_worst-Imbalance_Costs_CVaR_best)
#     # obj_risk = Planned_Costs_norm +gamma*Imbalance_Costs_CVaR_norm  # Total Costs considering the risks
#     # reward_baseline =
#     # reward_norm = sum_reward/reward_baseline
#     reward_norm = sum_reward
#     return reward_norm

def solve_single_day_worker(args):
    # wrapper per il multiprocessing
    func,data_day,theta_e,theta_d,gamma,Data_Type = args

    #start_worker = time.perf_counter()
    result = func(data_day, theta_e, theta_d,gamma, Data_Type)
    # end_worker = time.perf_counter()
    
    # # Stampiamo quanto ci mette IL SINGOLO OPERAIO a risolvere un giorno
    # #print(f"Tempo calcolo puro MILP: {end_worker - start_worker:.4f} sec")
    return  result

# def evaluate(func,model,data,Data_Type):
#     reward         = []
#     Costs          = []
#     Costs_CVaR     = []
#     Imbalance_CVaR = []

#     start_day = list(data.keys())[0]

#     for day in data.keys():

#         # Extract the statistic from data
#         energy_mean       = data[day].Energy_Data["mean"]
#         energy_deviation  = data[day].Energy_Data["deviation"]
#         demand_mean       = data[day].Demand_Data["mean"]
#         demand_deviation  = data[day].Demand_Data["deviation"]
#         sin_enc           = data[day].Simulation_Data["sin_enc"]
#         cos_enc           = data[day].Simulation_Data["cos_enc"]
#         price             = data[day].Prices["Market"]
#         # convert them to torch tensors and concatenate
#         energy_mean       = torch.tensor(energy_mean,dtype=torch.float32).unsqueeze(-1)
#         energy_deviation  = torch.tensor(energy_deviation,dtype=torch.float32).unsqueeze(-1)
#         demand_mean       = torch.tensor(demand_mean,dtype=torch.float32).unsqueeze(-1)
#         demand_deviation  = torch.tensor(demand_deviation,dtype=torch.float32).unsqueeze(-1)
#         sin_enc           = torch.tensor(sin_enc,dtype=torch.float32).unsqueeze(-1)
#         cos_enc           = torch.tensor(cos_enc,dtype=torch.float32).unsqueeze(-1)
#         #price             = torch.tensor(price,dtype=torch.float32).unsqueeze(-1)
#             #statistics = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation),dim=0)
#         # theta boundaries 
#         theta_max_E,theta_min_E,theta_max_D,theta_min_D = Theta_Bounds(data[day])
#         theta_max_e       = torch.tensor(theta_max_E,dtype=torch.float32).unsqueeze(-1)
#         theta_min_e       = torch.tensor(theta_min_E,dtype=torch.float32).unsqueeze(-1)
#         theta_max_d       = torch.tensor(theta_max_D,dtype=torch.float32).unsqueeze(-1)
#         theta_min_d       = torch.tensor(theta_min_D,dtype=torch.float32).unsqueeze(-1)
#         statistics = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation,sin_enc,cos_enc,theta_max_e,theta_min_e,theta_max_d,theta_min_d),dim=-1)
        
#         if statistics.dtype != next(model.parameters()).dtype:
#              statistics = statistics.to(next(model.parameters()).dtype)
#         # # bounds rec
#         #theta_max_e,theta_min_e,theta_max_d,theta_min_d = Theta_Bounds(data[day])

#         #Theta Prediction based on the statistic Vectors
#         theta,delta_theta = model(statistics)
#         theta_e = theta[:,0].tolist()
#         theta_d = theta[:,1].tolist()
#         Theta = theta_e+theta_d
#         last_delta_theta = delta_theta  # Salviamo l'ultimo per analizzarlo
#         #compute the Obj which is the reward for the NN training

#         if day != start_day:
#             data[day].Battery_Data["SoC_in"] = soc_end

#         obj            = func(data[day],theta_e,theta_d,Data_Type)
#         Rwrd           = obj[0]
#         soc_end        = obj[1]
#         costs          = obj[4]
#         costs_CVaR     = obj[5]
#         imbalance_CVaR = obj[6]
#         # comulative reward
#         reward.append(Rwrd)
#         Costs.append(costs)
#         Costs_CVaR.append(costs_CVaR)
#         Imbalance_CVaR.append(imbalance_CVaR)
    
#     #P#####Penalty if theta goes out of the bounds ( NOT NEEDED NOW)
#     theta_min = np.concatenate((theta_min_E , theta_min_D))
#     theta_max = np.concatenate((theta_max_E , theta_max_D))
#     theta_np = np.array(Theta)
#     below = np.maximum(0.0, theta_min - theta_np)
#     above = np.maximum(0.0, theta_np - theta_max)
#     penalty = below**2 + above**2  # soft penalty
#     lambda_pen = 0
   
#     ############################ IMbalance Penalty ############################################################  
#     #Reward with Penalty ALWAYS CHECK THE SIGN
#     Reward = - sum(np.array(reward)) - sum(lambda_pen*penalty)
#     return Reward,sum(Costs),sum(Costs_CVaR),sum(Imbalance_CVaR),sum(lambda_pen*penalty),theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d ,last_delta_theta

# def evaluate_batch(func,model,data,Data_Type,executor):
#     # --- 1. SETUP ---
#     # Salviamo lo stato training/eval per ripristinarlo dopo
#     was_training = model.training
#     model.eval() # Fondamentale: blocca Dropout e BatchNorm per risultati stabili
#     # Assicuriamo un ordine temporale coerente
#     days_sequence = sorted(list(data.keys()))
#     # Liste temporanee per costruire il "Batch"
#     input_tensors_list = []
#     # --- 2. PREPARAZIONE DATI (Costruzione del Batch) ---
#     # Qui NON chiamiamo la rete. Prepariamo solo gli input.
#     for day in days_sequence:
#         d = data[day]

#         # Extract the statistic from data
#         energy_mean       = d.Energy_Data["mean"]
#         energy_deviation  = d.Energy_Data["deviation"]
#         demand_mean       = d.Demand_Data["mean"]
#         demand_deviation  = d.Demand_Data["deviation"]
#         sin_enc           = d.Simulation_Data["sin_enc"]
#         cos_enc           = d.Simulation_Data["cos_enc"]
#         # convert them to torch tensors and concatenate
#         energy_mean       = torch.tensor(energy_mean,dtype=torch.float32).unsqueeze(-1)
#         energy_deviation  = torch.tensor(energy_deviation,dtype=torch.float32).unsqueeze(-1)
#         demand_mean       = torch.tensor(demand_mean,dtype=torch.float32).unsqueeze(-1)
#         demand_deviation  = torch.tensor(demand_deviation,dtype=torch.float32).unsqueeze(-1)
#         sin_enc           = torch.tensor(sin_enc,dtype=torch.float32).unsqueeze(-1)
#         cos_enc           = torch.tensor(cos_enc,dtype=torch.float32).unsqueeze(-1)    
#          # theta boundaries 
#         theta_max_E,theta_min_E,theta_max_D,theta_min_D = Theta_Bounds(d)
#         theta_max_e       = torch.tensor(theta_max_E,dtype=torch.float32).unsqueeze(-1)
#         theta_min_e       = torch.tensor(theta_min_E,dtype=torch.float32).unsqueeze(-1)
#         theta_max_d       = torch.tensor(theta_max_D,dtype=torch.float32).unsqueeze(-1)
#         theta_min_d       = torch.tensor(theta_min_D,dtype=torch.float32).unsqueeze(-1)

#         Day_Stats = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation,sin_enc,cos_enc,theta_max_e,theta_min_e,theta_max_d,theta_min_d),dim=-1)

#         # Aggiungiamo alla lista
#         input_tensors_list.append(Day_Stats)

#     # --- 3. BATCH (Il cuore della modifica) ---
#     # Stackiamo tutto in un unico Tensorone [N_Giorni, Features]
#     batch_input = torch.stack(input_tensors_list)
#     # --- CONVERSIONE MASSIVA (Fuori dal for) ---
#     target_dtype = next(model.parameters()).dtype
#     if batch_input.dtype != target_dtype:
#         batch_input = batch_input.to(target_dtype)
#     # # Spostiamo il batch sullo stesso device del modello (gestisce CPU/GPU automaticamente) dubbii
#     # device = next(model.parameters()).device
#     # batch_input = batch_input.to(device)

#     with torch.no_grad():
#         # LA RETE RISPONDE UNA VOLTA SOLA PER TUTTI I GIORNI
#         # Output shape: [N_Giorni, 2]
#         batch_theta, batch_delta = model(batch_input)
#         #theta,delta_theta = model(statistics)

# # #   Riportiamo i dati su CPU/Numpy per i calcoli fisici
# #     batch_theta_np = batch_theta.cpu().numpy()
#     # --- 3. MILP REWARD EVALUATION  WITH MULTIPROCESSING---
 
#     # --- 3.1 Daily task geenrations for the CPU cores ---
#     tasks = []
#     for i, day in enumerate(days_sequence):
#         theta_e = batch_theta[i,:,0].tolist()
#         theta_d = batch_theta[i,:,1].tolist()

#         # tuple for every daydata[day]
#         tasks.append((func,data[day],theta_e,theta_d,Data_Type))
#     # 3.2 execute in paralle within the avibale processors
#     results = []

#     #with concurrent.futures.ProcessPoolExecutor() as executor:
#         # map subdivide the tasks on the free core
#         # all the results are in order for the days
#     results = list(executor.map(solve_single_day_worker,tasks))
#     # --- 3. MILP REWARD EVALUATION ---
    
#     reward         = 0
#     Costs          = 0
#     Costs_CVaR     = 0
#     Imbalance_CVaR = 0

#     for obj in results:
#         reward += obj[0]
#         Costs  += obj[4]
#         Costs_CVaR+= obj[5]
#         Imbalance_CVaR+= obj[6]

  
#     penalty= 0
#     # ⏱️ FINE CRONOMETRO: Subito dopo il for
#     # ==========================================
#     # end_time = time.perf_counter()

#     # --- CALCOLO E STAMPA DEI RISULTATI ---
#     # num_days = len(days_sequence)
#     # total_pulp_time = end_time - start_time
#     # avg_time = total_pulp_time / num_days if num_days > 0 else 0

#     # # print(f"\n--- ANALISI BOTTLENECK MILP ---")
#     # # print(f"Giorni processati: {num_days}")
#     # # print(f"Tempo totale ciclo: {total_pulp_time:.4f} sec")
#     # # print(f"Tempo medio/giorno: {avg_time:.4f} sec")
#     # # print(f"-------------------------------\n")
#     # --- 5. CLEANUP ---
#     if was_training:
#         model.train()
    

#     ############################ IMbalance Penalty ############################################################  
#     #Reward with Penalty ALWAYS CHECK THE SIGN
#     Reward = - reward
#     # Poiché batch_delta è un tensore [N, 2], per il log prendiamo la media o l'ultimo
#     last_delta_theta = batch_delta[-1]

#     return Reward,Costs,Costs_CVaR,Imbalance_CVaR,penalty,theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d ,last_delta_theta

 

def breakconditions(reward_history, patience, tolerance, trial):

    # Keep only the last `patience` rewards
    if len(reward_history) >= patience:
        #reward_history.pop(0)
        recent_rewards = reward_history[-patience:]
        reward_std = np.std(recent_rewards)
        if trial is None:
            print(f"std reward = {reward_std}")
        # Check convergence: std of recent rewards
        if (reward_std) < tolerance:
            #if trial is None:
            print("\033[92mConverged: reward variation below tolerance.\033[0m")
            return True,reward_std
    else:
        reward_std =  float('nan')

    return False,reward_std

# Define the function for adjusting hyperparameters
def adjust_hyperparameters(iteration, sigma, alpha, Evolution_Params,trial):
    """
    Dynamically adjusts hyperparameters (e.g., mutation rate, learning rate) based on performance.
    """
    alpha_o = Evolution_Params["alpha"]
    sigma_o = Evolution_Params["sigma"]
    n_iterations = Evolution_Params["n_iterations"]
   # decay_step =max(1, n_iterations//50)
    decay_step = 5
    if iteration != 0:
        if iteration % decay_step == 0:  # For example, adjust every 5 iteration
                alpha = max(alpha * 0.98, alpha_o*0.01)  # Reduce learning rate but ensure it doesn't go below a small threshold
                sigma = max(sigma * 0.98, sigma_o*0.01)  # Reduce learning rate but ensure it doesn't go below a small threshold
                if trial is None:
                    print(f"Iteration {iteration}: Adjusting alpha to {alpha:.4f} | Adjusting sigma to {sigma:.4f}.")
    
    return sigma, alpha

def Theta_Bounds(data): 
    # Energy
    e_forecast = data.Energy_Data["forecast"]
    e_bound_max = data.Energy_Data["bands"]["max"]
    e_bound_min = data.Energy_Data["bands"]["min"]
    # Demand
    d_forecast = data.Demand_Data["forecast"]
    d_bound_max = data.Demand_Data["bands"]["max"]
    d_bound_min = data.Demand_Data["bands"]["min"]

     # List of arrays to check and convert
    arrays = [e_forecast, e_bound_max,e_bound_min, d_forecast, d_bound_max,d_bound_min]

    # Loop through each array and convert to NumPy array if not already
    for i in range(len(arrays)):
        if not isinstance(arrays[i], np.ndarray):
             arrays[i] = np.array(arrays[i])   

    theta_max_e = arrays[1]/arrays[0]
    theta_min_e = arrays[2]/arrays[0]
    theta_max_d = arrays[4]/arrays[3]
    theta_min_d = arrays[5]/arrays[3]
    # cleaning of the element when forecast = 0
    for t in range(len(arrays[0])):
        if arrays[0][t] <= 1:
            theta_max_e[t] = 0
            theta_min_e[t] = 0
        if arrays[3][t] <= 1:
            theta_max_d[t] = 0
            theta_min_d[t] = 0      


    return theta_max_e,theta_min_e,theta_max_d,theta_min_d

def Theta_Choice(data,Opt_Type,policy=None):#Theta_Choice,theta_e_manual,theta_d_manual,input_size,output_size,Model_Name,NN_Structure,save_dir):
    if Opt_Type == 'Learned_CFA':
        # Extract the statistic from data
        energy_mean       = data.Energy_Data["mean"]
        energy_deviation  = data.Energy_Data["deviation"]
        demand_mean       = data.Demand_Data["mean"]
        demand_deviation  = data.Demand_Data["deviation"]
        sin_enc           = data.Simulation_Data["sin_enc"]
        cos_enc           = data.Simulation_Data["cos_enc"]
        #price             = data.Prices["Market"]
            # convert them to torch tensors and concatenate IS THIS POINTELESS IN THE MAIN CODE? CHECK todo
        energy_mean       = torch.tensor(energy_mean,dtype=torch.float32).unsqueeze(-1)
        energy_deviation  = torch.tensor(energy_deviation,dtype=torch.float32).unsqueeze(-1)
        demand_mean       = torch.tensor(demand_mean,dtype=torch.float32).unsqueeze(-1)
        demand_deviation  = torch.tensor(demand_deviation,dtype=torch.float32).unsqueeze(-1)
        sin_enc           = torch.tensor(sin_enc,dtype=torch.float32).unsqueeze(-1)
        cos_enc           = torch.tensor(cos_enc,dtype=torch.float32).unsqueeze(-1)
       # price             = torch.tensor(price,dtype=torch.float32).unsqueeze(-1)
               # theta boundaries 
        theta_max_e,theta_min_e,theta_max_d,theta_min_d = Theta_Bounds(data)
        theta_max_e       = torch.tensor(theta_max_e,dtype=torch.float32).unsqueeze(-1)
        theta_min_e       = torch.tensor(theta_min_e,dtype=torch.float32).unsqueeze(-1)
        theta_max_d       = torch.tensor(theta_max_d,dtype=torch.float32).unsqueeze(-1)
        theta_min_d       = torch.tensor(theta_min_d,dtype=torch.float32).unsqueeze(-1)
        #statistics = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation),dim=0)
        statistics = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation,sin_enc,cos_enc,theta_max_e,theta_min_e,theta_max_d,theta_min_d),dim=-1)
        # inizializzation NN
        # dummy_vec = torch.zeros(4)
        # hidden_dim = NN_Structure["Hidden_size"]
        # n_layers = NN_Structure["N_layers"]
        # model = PolicyNetwork(input_size, output_size,dummy_vec,dummy_vec,hidden_dim,n_layers)  


        # model_load_path = os.path.join(save_dir, f"{Model_Name}.pth")
        # model.load_state_dict(torch.load(model_load_path))
        # model.eval()

        inference_start_real = time.perf_counter()
        theta,out = policy(statistics)
        inference_end_real   = time.perf_counter()
        inference_time_daily_real = inference_end_real-inference_start_real
        timer= inference_time_daily_real

        out_e = out[:,0].tolist()
        out_d = out[:,1].tolist()
        theta_e = theta[:,0].tolist()
        theta_d = theta[:,1].tolist()

    elif Opt_Type == 'Static_CFA':
        timer=0
        out_e = []
        out_d = []
        theta_e = policy["theta_e"]
        theta_d = policy["theta_d"]

    elif Opt_Type == 'Deterministic':
        
        timesteps = data.Simulation_Data["time_horizon"]
    ### Deterministic Forcasts q 50
        timer=0     
        out_e = []
        out_d = []
        theta_e = [1]*timesteps
        theta_d = [1]*timesteps
    else:
        raise ValueError(f"Unknown Opt_Type: {Opt_Type}")


    return theta_e,theta_d,out_e, out_d,timer

# def Evolution_Strategy_Training(func,model,data,Data_Type,Evolution_Params,Model_Name,save_dir,trial = None):
    
#     population_size = Evolution_Params["population_size"]       # Number of samples per iteration
#     sigma           = Evolution_Params["sigma"]               # Noise standard deviation
#     alpha           = Evolution_Params["alpha"]            # Learning rate
#     n_iterations    = Evolution_Params["n_iterations"]        # Total iterations
#     weight_decay    = Evolution_Params["weight_decay"]        # to save from saturation

#     # 3. --- Initialization ---

#     NN_params_dim =  sum(p.numel() for p in model.parameters())  # Total number of parameters in the model
#     reward_history =[] #needed to track the rewards and use the breack condition

#     logic_processors_usage = True #False if i want to use only the real physical cores
#     cpu_available   = psutil.cpu_count(logical=logic_processors_usage)
#     optimal_workers = max(1, cpu_available-1) # cpu_available-1 we keep one core free to run the main file
#     print("\n Global Pool inizialization ...")
#     print(f"Core Types requested: {'logials' if logic_processors_usage else 'physical'}")
#     print(f"Workers assigned to CPU: {optimal_workers}")

#     with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as global_executor:
#         # --- Starting (BASELINE) ---
#         if trial is None:
#             print("\n--- BASELINE CHECK (Before Training) ---")
#             base_reward, base_costs,base_costcvar, base_imbcvar, _, _, _, _, _, _, _,_ = evaluate_batch(func, model, data, Data_Type,global_executor)
#             print(f"BASELINE | Reward: {base_reward:.4f} | Costs: {base_costs:.2f} |  Cost CVaR: {base_costcvar:.2f} | imb CVaR: {base_imbcvar:.2f}")
#             print("----------------------------------------\n")

#         # 4. --- Main ES Loop ---
#         log_df = pd.DataFrame(columns=["iteration", "reward", "std_reward","penalty","Costs", "Costs_CVaR","Imbalance_CVaR","Last Layer saturation [%]", "Last Layer mean Output"]) # dataframe for logging
#         best_run_reward = -float('inf')
        
#         for iteration in range(n_iterations):
#             start_iteration = time.perf_counter()
#             noise = np.random.randn(population_size, NN_params_dim)
#             rewards = np.zeros(population_size)

#             # Evaluate all perturbations

#             for i in range(population_size):
#                 perturbed_model = copy.deepcopy(model)
#                 # Perturb each parameter (weight) of the model
#                 idx = 0
#                 for param in perturbed_model.parameters():
#                     numel = param.numel()
#                     perturbation = torch.tensor(noise[i, idx:idx + numel], dtype=torch.float32).view(param.size())
#                     param.data += sigma * perturbation  # Apply the noise perturbation
#                     idx += numel

#                 # Calculate the reward for this perturbed model
#                 rewards[i],_,_,_,_,_,_,_,_,_,_,_= evaluate_batch(func,perturbed_model,data,Data_Type,global_executor)

#             # Normalize rewards
#             rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

#             # Estimate gradient
#            # gradient = np.dot(noise.T, rewards) / (population_size * sigma) #è la formamatematica giusta ma al diminuire di sigma il gradienten esplode per questo rimuovere
#             gradient = np.dot(noise.T, rewards) / (population_size)
#             # Update the model parameters using the gradient estimate
#             with torch.no_grad():
#                 idx = 0
#                 for param in model.parameters():
#                     numel = param.numel()
#                     param.data = param.data*(1-weight_decay)+ alpha * gradient[idx:idx + numel].reshape(param.size())  # Update weights
#                     idx += numel

#                     # for param in mask_model.parameters():
#                     #   param.data = nn.parameter.Parameter(torch.ones_like(param))
#             # valutation and tracking #####
#             Reward,Costs,Costs_CVaR,Imbalance_CVaR,penalty,theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d,last_delta_theta = evaluate_batch(func,model,data,Data_Type,global_executor)
#             reward_history.append(Reward)
#             ######### Saturation info ##############
#             with torch.no_grad():
#                 mean_act = torch.abs(last_delta_theta).mean().item()
#                 pct_saturated = (torch.abs(last_delta_theta) > 0.95).float().mean().item() * 100

#             if Reward > best_run_reward:
#                 best_run_reward=Reward

#             #### optuna check (PRUNING)
#             if trial is not None:
#                 trial.report(Reward,iteration)
#                 if trial.should_prune():
#                     raise optuna.exceptions.TrialPruned()
#                 if pct_saturated > 80.0:
#                     print(f"[Trial {trial.number}] KILLED due to SATURATION ({pct_saturated:.1f}%)")
#                     raise optuna.exceptions.TrialPruned()
            
#             ########## convergence condition#################
#             converged,std_reward = breakconditions(reward_history,patience=10,tolerance=10e-4,trial=trial)

#             ###################### hyperparameters updtate
#             sigma, alpha = adjust_hyperparameters(iteration, sigma, alpha,Evolution_Params,trial=trial)    

#            ################### LOGGING AND PRINT ##############################################
#             if trial is None:
#                         ######### saturation check##################################

#             # delta_theta è l'output della tanh che ritorna la tua rete
            
#                 # Stampa diagnostica intelligente
#                 if pct_saturated > 50.0:
#                     # ROSSO: Allarme
#                     print(f"\033[91m[ALLARME SATURAZIONE] {pct_saturated:.1f}% neuroni bloccati! (Mean: {mean_act:.3f})\033[0m")
#                 else:
#                     # VERDE: Tutto ok
#                     print(f"\033[92m[OK] Saturazione: {pct_saturated:.1f}% (Mean: {mean_act:.3f})\033[0m")
#                ######################## Plotting thetas
#                 # Plot only every 25 iterations
#                 if iteration % 25 == 0 or iteration == 1:
                    
#                     # Funzione helper per convertire tutto in numpy array puliti
#                     def to_numpy(x):
#                         if isinstance(x, torch.Tensor):
#                             return x.detach().cpu().numpy().flatten()
#                         return np.array(x).flatten()

#                     # Conversione dati
#                     t_e = to_numpy(theta_e)
#                     t_d = to_numpy(theta_d)
#                     max_e = to_numpy(theta_max_e)
#                     min_e = to_numpy(theta_min_e)
#                     max_d = to_numpy(theta_max_d)
#                     min_d = to_numpy(theta_min_d)
                    
#                     # Asse X
#                     x_axis = range(1, 24 + 1)

#                     # Creazione Figura
#                     fig, axes = plt.subplots(2, 1, figsize=(8, 10))
                    
#                     # --- PLOT ENERGIA (Subplot 0) ---
#                     # 1. Disegna l'area dei vincoli (Min-Max) come banda grigia
#                     axes[0].fill_between(x_axis, min_e, max_e, color='gray', alpha=0.3, label='Feasible Region')
#                     # 2. Disegna la linea del theta attuale
#                     sns.lineplot(x=x_axis, y=t_e, ax=axes[0], marker='o', color='blue', label='Theta E (Actual)')
                    
#                     axes[0].set_title(f"Energy Theta Evolution (Iter {iteration})")
#                     axes[0].set_ylabel("Theta Multiplier")
#                     axes[0].grid(True, linestyle='--', alpha=0.6)
#                     axes[0].legend()

#                     # --- PLOT DOMANDA (Subplot 1) ---
#                     # 1. Disegna l'area dei vincoli
#                     axes[1].fill_between(x_axis, min_d, max_d, color='gray', alpha=0.3, label='Feasible Region')
#                     # 2. Disegna la linea del theta attuale
#                     sns.lineplot(x=x_axis, y=t_d, ax=axes[1], marker='o', color='orange', label='Theta D (Actual)')
                    
#                     axes[1].set_title(f"Demand Theta Evolution (Iter {iteration})")
#                     axes[1].set_xlabel("Hour")
#                     axes[1].set_ylabel("Theta Multiplier")
#                     axes[1].grid(True, linestyle='--', alpha=0.6)
#                     axes[1].legend()

#                     # --- SALVATAGGIO ---
#                     plt.tight_layout()
#                     save_path = f"C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/figure/TrainingCheck/Iteration_{iteration}.png"
#                     plt.savefig(save_path)
#                     plt.close(fig)

#                 # print of the log
#                 end_iteration = time.perf_counter()
#                 iteration_time=end_iteration - start_iteration
#                 # # Stampiamo quanto ci mette IL SINGOLO OPERAIO a risolvere un giorno
#                 print("\033[96m____________________________________________________________________________________________________________\033[0m")
#                 print(f"\033[96mIteration Time: {iteration_time:.4f} sec | {iteration_time/60:.4f} min \033[0m")
#                 print(f"\033[96mIteration {iteration:3d} | Reward: {Reward:.4f} | Penalty: {penalty:.4f} | Costs: {Costs:.2f} | Costs_CVaR: {Costs_CVaR:.2f} | Imbalance_CVaR: {Imbalance_CVaR:.2f}\033[0m")
#                 print("\033[96m____________________________________________________________________________________________________________\033[0m")


#             # sigma, alpha = adjust_hyperparameters(iteration, reward_history, sigma, alpha, patience=10, tolerance=1e-3)
            
            
#                 # Logging into dataframe
#                 log_df.loc[len(log_df)] ={"iteration"     : iteration,
#                                         "reward"        : Reward,
#                                         "std_reward"    : std_reward,
#                                         "penalty"       : penalty,
#                                         "Costs"         : Costs,
#                                         "Costs_CVaR"    : Costs_CVaR,
#                                         "Imbalance_CVaR": Imbalance_CVaR,
#                                         "Last Layer saturation [%]": pct_saturated,
#                                         "Last Layer mean Output": mean_act} 

            
#                 ################break conditions and standard deviation logging
            



#             if converged: #break the training loop
#                 break


# ############### RETURN SECTION ###########################
#     if trial is not None:
#     # Se siamo in Optuna, restituiamo SOLO il numero (miglior reward ottenuto)
#         return best_run_reward 
#                 # Save the training log as an excel
#     log_df.to_csv("C:/Users/adria/Desktop/INESCTEC/00.code/Architecture_Neural_Network/debugger/training_log.csv", index=False)
#     # Save the model's state_dict after training
#     model_save_path = os.path.join(save_dir, f"{Model_Name}.pth")
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved at {model_save_path}")

#     return model,model_save_path

#%% #################### funzioni per l'High Performance Computing
def Input_batch(data,model):
    """
    Estrae le statistiche dai dizionari storici e costruisce un singolo tensore PyTorch
    allineato al tipo e al device del modello per il forward pass massivo.
    """
    days_sequence = sorted(list(data.keys()))
    # Liste temporanee per costruire il "Batch"
    input_tensors_list = []
    # --- 2. PREPARAZIONE DATI (Costruzione del Batch) ---
    # Qui NON chiamiamo la rete. Prepariamo solo gli input.
    for day in days_sequence:
        d = data[day]

        # Extract the statistic from data
        energy_mean       = d.Energy_Data["mean"]
        energy_deviation  = d.Energy_Data["deviation"]
        demand_mean       = d.Demand_Data["mean"]
        demand_deviation  = d.Demand_Data["deviation"]
        sin_enc           = d.Simulation_Data["sin_enc"]
        cos_enc           = d.Simulation_Data["cos_enc"]
        # convert them to torch tensors and concatenate
        energy_mean       = torch.tensor(energy_mean,dtype=torch.float32).unsqueeze(-1)
        energy_deviation  = torch.tensor(energy_deviation,dtype=torch.float32).unsqueeze(-1)
        demand_mean       = torch.tensor(demand_mean,dtype=torch.float32).unsqueeze(-1)
        demand_deviation  = torch.tensor(demand_deviation,dtype=torch.float32).unsqueeze(-1)
        sin_enc           = torch.tensor(sin_enc,dtype=torch.float32).unsqueeze(-1)
        cos_enc           = torch.tensor(cos_enc,dtype=torch.float32).unsqueeze(-1)    
         # theta boundaries 
        theta_max_E,theta_min_E,theta_max_D,theta_min_D = Theta_Bounds(d)
        theta_max_e       = torch.tensor(theta_max_E,dtype=torch.float32).unsqueeze(-1)
        theta_min_e       = torch.tensor(theta_min_E,dtype=torch.float32).unsqueeze(-1)
        theta_max_d       = torch.tensor(theta_max_D,dtype=torch.float32).unsqueeze(-1)
        theta_min_d       = torch.tensor(theta_min_D,dtype=torch.float32).unsqueeze(-1)

        Day_Stats = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation,sin_enc,cos_enc,theta_max_e,theta_min_e,theta_max_d,theta_min_d),dim=-1)

        # if Day_Stats.dtype != next(model.parameters()).dtype:
        #     Day_Stats = Day_Stats.to(next(model.parameters()).dtype)
        # Aggiungiamo alla lista
        input_tensors_list.append(Day_Stats)

    # --- 3. BATCH (Il cuore della modifica) ---
    # Stackiamo tutto in un unico Tensorone [N_Giorni, Features]
    batch_input = torch.stack(input_tensors_list)
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        batch_input = batch_input.to(target_dtype)
    # # Spostiamo il batch sullo stesso device del modello (gestisce CPU/GPU automaticamente) dubbii
    # device = next(model.parameters()).device
    # batch_input = batch_input.to(device)
    return batch_input,days_sequence

def build_clone_tasks(model,gamma,batch_input,days_sequence,data, Data_Type, func):
    """
    Esegue il forward pass per un singolo clone e genera la lista dei task MILP.
    """
    was_training = model.training
    model.eval()
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        batch_input = batch_input.to(target_dtype)
    with torch.no_grad():
        # LA RETE RISPONDE UNA VOLTA SOLA PER TUTTI I GIORNI del clone
        # Output shape: [N_Giorni, 2]
        batch_theta, batch_delta = model(batch_input)

        # --- 3.1 Daily task geenrations for the CPU cores ---
    tasks = []
    for i, day in enumerate(days_sequence):
        theta_e = batch_theta[i,:,0].tolist()
        theta_d = batch_theta[i,:,1].tolist()
        # tuple for every daydata[day]
        tasks.append((func,data[day],theta_e,theta_d,gamma,Data_Type))
    if was_training:
        model.train()
    return tasks

def calculate_clone_reward(risultati_clone):
    """
    Somma i risultati MILP dei 365 giorni per restituire il reward del clone.
    """
    reward = 0
    for obj in risultati_clone:
        reward += obj[0] # obj[0] corrisponde a Rwrd nel tuo worker
    # Nel tuo codice originale il Reward finale aveva il segno invertito
    #Global_Reward = global_reward_func(reward)
    Reward = -reward
    penalty = 0
    return Reward#,Costs,Costs_CVaR,Imbalance_CVaR,penalty#,theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d ,last_delta_theta

def evaluate_model_HPC(func, model,gamma, batch_input, days_sequence, data, Data_Type, executor):
    """
    Valuta in modo completo il modello principale, calcolando tutte le metriche (Costi, CVaR, ecc.).
    """
    was_training = model.training
    model.eval()
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        # Molto importante: sovrascriviamo la variabile locale con la versione convertita
        batch_input = batch_input.to(target_dtype)
    # ------------------------------
    with torch.no_grad():
        batch_theta, batch_delta = model(batch_input)
    # --- EMERGENCY DTYPE CHECK ---

    tasks = []
    for i, day in enumerate(days_sequence):
        theta_e = batch_theta[i,:,0].tolist()
        theta_d = batch_theta[i,:,1].tolist()
        tasks.append((func, data[day], theta_e, theta_d, gamma, Data_Type))

    # --- Esecuzione Multi-Core HPC ---
    # Usiamo il chunksize per annullare l'overhead di Python  chunksize=50
    results = list(executor.map(solve_single_day_worker, tasks))
    
    # --- Aggregazione Risultati ---
    sum_reward = 0
    Costs = 0
    Costs_CVaR = 0
    Imbalance_CVaR = 0

    for obj in results:
        sum_reward +=obj[0]
        Costs  += obj[4]
        Costs_CVaR += obj[5]
        Imbalance_CVaR += obj[6]

    penalty = 0 
    #global_reward = global_reward_func(sum_reward)
    Reward = -sum_reward
    
    # Per il plotting e il tracking, estraiamo i theta dell'ultimo giorno o le medie
    last_delta_theta = batch_delta[-1]
    
    # Estraiamo i boundaries dell'ultimo giorno per i plot
    theta_max_e, theta_min_e, theta_max_d, theta_min_d = Theta_Bounds(data[days_sequence[-1]])

    if was_training:
        model.train()

    return Reward, Costs, Costs_CVaR, Imbalance_CVaR, penalty, theta_e, theta_d, theta_max_e, theta_min_e, theta_max_d, theta_min_d, last_delta_theta

def Evolution_Strategy_Training_HPC(func,model,data,gamma,Data_Type,optimal_workers,Evolution_Params,Model_Name,save_dir,training_check_dir,trial = None):
    
    population_size = Evolution_Params["population_size"]       # Number of samples per iteration
    sigma           = Evolution_Params["sigma"]               # Noise standard deviation
    alpha           = Evolution_Params["alpha"]            # Learning rate
    n_iterations    = Evolution_Params["n_iterations"]        # Total iterations
    weight_decay    = Evolution_Params["weight_decay"]        # to save from saturation
    days_per_clone  = len(data)
    print("\n Pre-calcolo tensori storici (Ottimizzazione HPC)...")
    batch_input,days_sequence = Input_batch(data,model)
    # 3. --- Initialization ---

    NN_params_dim =  sum(p.numel() for p in model.parameters())  # Total number of parameters in the model
    reward_history =[] #needed to track the rewards and use the breack condition

    # core analysis and cpu usage
    logic_processors_usage = False #False if i want to use only the real physical cores
    #cpu_available   = psutil.cpu_count(logical=logic_processors_usage)
   # optimal_workers = max(1, cpu_available-1) # cpu_available-1 we keep one core free to run the main file

    print("\n Global Pool inizialization HPC...")
    print(f"Core Types requested: {'logicals' if logic_processors_usage else 'physical'}")
    print(f"Workers assigned to CPU: {optimal_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as global_executor:
        # --- Starting (BASELINE) ---
        if trial is None:
            print("\n--- BASELINE CHECK (Before Training) ---")
            base_reward, base_costs,base_costcvar, base_imbcvar, _, _, _, _, _, _, _,_ = evaluate_model_HPC(func, model,gamma,batch_input,days_sequence ,data, Data_Type,global_executor)
            print(f"BASELINE | Reward: {base_reward:.4f} | Costs: {base_costs:.2f} |  Cost CVaR: {base_costcvar:.2f} | imb CVaR: {base_imbcvar:.2f}")
            print("----------------------------------------\n")

        # 4. --- Main ES Loop ---
        log_df = pd.DataFrame(columns=["iteration", "reward", "std_reward","penalty","Costs", "Costs_CVaR","Imbalance_CVaR","Last Layer saturation [%]", "Last Layer mean Output"]) # dataframe for logging
        best_run_reward = -float('inf')
        
        for iteration in range(n_iterations):
            start_iteration = time.perf_counter()
            noise = np.random.randn(population_size, NN_params_dim)
            rewards = np.zeros(population_size)
            # =========================================================================
            # INIZIO ARCHITETTURA HPC (FLAT MULTIPROCESSING)
            # =========================================================================
           
            mega_lista_task = []
            # FASE 1: Preparazione Massiva (Veloce, Seriale, Solo Inferenza PyTorch)
            for i in range(population_size):
                perturbed_model = copy.deepcopy(model)
                # Perturb each parameter (weight) of the model
                idx = 0
                for param in perturbed_model.parameters():
                    numel = param.numel()
                    perturbation = torch.tensor(noise[i, idx:idx + numel], dtype=torch.float32).view(param.size())
                    param.data += sigma * perturbation  # Apply the noise perturbation
                    idx += numel
                # LA MAGIA: Non calcoliamo i MILP qui. Chiediamo solo i dati pronti da impacchettare.
                clone_task = build_clone_tasks(perturbed_model,gamma,batch_input, days_sequence, data, Data_Type, func)

                mega_lista_task.extend(clone_task)

            # FASE 2: L'Esplosione Parallela (1 singolo `.map` per 14.600 giorni)
            # chunksize=50 azzera l'overhead di comunicazione di Python
            mega_risultati = list(global_executor.map(solve_single_day_worker, mega_lista_task))

                # FASE 3: Ricomposizione e Contabilità
            for i in range(population_size):
                # Tagliamo la mega lista a fette esatte da 365 giorni
                inizio = i * days_per_clone
                fine = inizio + days_per_clone
                risultati_singolo_clone = mega_risultati[inizio:fine]
                
                # Il contabile calcola il reward per questa fetta
                rewards[i] = calculate_clone_reward(risultati_singolo_clone)
             
            # Normalize rewards
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

            # Estimate gradient
           # gradient = np.dot(noise.T, rewards) / (population_size * sigma) #è la formamatematica giusta ma al diminuire di sigma il gradienten esplode per questo rimuovere
            gradient = np.dot(noise.T, rewards) / (population_size)
            # Update the model parameters using the gradient estimate
            with torch.no_grad():
                idx = 0
                for param in model.parameters():
                    numel = param.numel()
                    param.data = param.data*(1-weight_decay)+ alpha * gradient[idx:idx + numel].reshape(param.size())  # Update weights
                    idx += numel

                    # for param in mask_model.parameters():
                    #   param.data = nn.parameter.Parameter(torch.ones_like(param))
            # valutation and tracking #####
            Reward,Costs,Costs_CVaR,Imbalance_CVaR,penalty,theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d,last_delta_theta = evaluate_model_HPC(func,model,gamma,batch_input,days_sequence,data,Data_Type,global_executor)
            reward_history.append(Reward)
            ######### Saturation info ##############
            with torch.no_grad():
                mean_act = torch.abs(last_delta_theta).mean().item()
                pct_saturated = (torch.abs(last_delta_theta) > 0.95).float().mean().item() * 100

            if Reward > best_run_reward:
                best_run_reward=Reward

            #### optuna check (PRUNING)
            if trial is not None:
                trial.report(Reward,iteration)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if pct_saturated > 80.0:
                    print(f"[Trial {trial.number}] KILLED due to SATURATION ({pct_saturated:.1f}%)")
                    raise optuna.exceptions.TrialPruned()
            
            ########## convergence condition#################
            converged,std_reward = breakconditions(reward_history,patience=20,tolerance=10e-3,trial=trial)

            ###################### hyperparameters updtate
            sigma, alpha = adjust_hyperparameters(iteration, sigma, alpha,Evolution_Params,trial=trial)    

           ################### LOGGING AND PRINT ##############################################
            if trial is None:
                        ######### saturation check##################################

            # delta_theta è l'output della tanh che ritorna la tua rete
            
                # Stampa diagnostica intelligente
                if pct_saturated > 50.0:
                    # ROSSO: Allarme
                    print(f"\033[91m[ALLARME SATURAZIONE] {pct_saturated:.1f}% neuroni bloccati! (Mean: {mean_act:.3f})\033[0m")
                else:
                    # VERDE: Tutto ok
                    print(f"\033[92m[OK] Saturazione: {pct_saturated:.1f}% (Mean: {mean_act:.3f})\033[0m")
               ######################## Plotting thetas
                # # Plot only every 25 iterations
                # if iteration % 10 == 0 or iteration == 1:
                    
                #     # Funzione helper per convertire tutto in numpy array puliti
                #     def to_numpy(x):
                #         if isinstance(x, torch.Tensor):
                #             return x.detach().cpu().numpy().flatten()
                #         return np.array(x).flatten()

                #     # Conversione dati
                #     t_e = to_numpy(theta_e)
                #     t_d = to_numpy(theta_d)
                #     max_e = to_numpy(theta_max_e)
                #     min_e = to_numpy(theta_min_e)
                #     max_d = to_numpy(theta_max_d)
                #     min_d = to_numpy(theta_min_d)
                    
                #     # Asse X
                #     x_axis = range(1, 24 + 1)

                #     # Creazione Figura
                #     fig, axes = plt.subplots(2, 1, figsize=(8, 10))
                    
                #     # --- PLOT ENERGIA (Subplot 0) ---
                #     # 1. Disegna l'area dei vincoli (Min-Max) come banda grigia
                #     axes[0].fill_between(x_axis, min_e, max_e, color='gray', alpha=0.3, label='Feasible Region')
                #     # 2. Disegna la linea del theta attuale
                #     sns.lineplot(x=x_axis, y=t_e, ax=axes[0], marker='o', color='blue', label='Theta E (Actual)')
                    
                #     axes[0].set_title(f"Energy Theta Evolution (Iter {iteration})")
                #     axes[0].set_ylabel("Theta Multiplier")
                #     axes[0].grid(True, linestyle='--', alpha=0.6)
                #     axes[0].legend()

                #     # --- PLOT DOMANDA (Subplot 1) ---
                #     # 1. Disegna l'area dei vincoli
                #     axes[1].fill_between(x_axis, min_d, max_d, color='gray', alpha=0.3, label='Feasible Region')
                #     # 2. Disegna la linea del theta attuale
                #     sns.lineplot(x=x_axis, y=t_d, ax=axes[1], marker='o', color='orange', label='Theta D (Actual)')
                    
                #     axes[1].set_title(f"Demand Theta Evolution (Iter {iteration})")
                #     axes[1].set_xlabel("Hour")
                #     axes[1].set_ylabel("Theta Multiplier")
                #     axes[1].grid(True, linestyle='--', alpha=0.6)
                #     axes[1].legend()

                #     # --- SALVATAGGIO ---
                #     plt.tight_layout()
                #     fig_name = f"Iteration_{iteration}.png"
                #     iter_fig_path = os.path.join(training_check_dir,fig_name) 
                #     plt.savefig(iter_fig_path)
                #     plt.close(fig)

                # print of the log
                end_iteration = time.perf_counter()
                iteration_time=end_iteration - start_iteration
                # # Stampiamo quanto ci mette IL SINGOLO OPERAIO a risolvere un giorno
                print("\033[96m____________________________________________________________________________________________________________\033[0m")
                print(f"\033[96mIteration Time: {iteration_time:.4f} sec | {iteration_time/60:.4f} min \033[0m")
                print(f"\033[96mIteration {iteration:3d} | Reward: {Reward:.4f} | Penalty: {penalty:.4f} | Costs: {Costs:.2f} | Costs_CVaR: {Costs_CVaR:.2f} | Imbalance_CVaR: {Imbalance_CVaR:.2f}\033[0m")
                print("\033[96m____________________________________________________________________________________________________________\033[0m")


            # sigma, alpha = adjust_hyperparameters(iteration, reward_history, sigma, alpha, patience=10, tolerance=1e-3)
            
            
                # Logging into dataframe
                log_df.loc[len(log_df)] ={"iteration"     : iteration,
                                        "reward"        : Reward,
                                        "std_reward"    : std_reward,
                                        "penalty"       : penalty,
                                        "Costs"         : Costs,
                                        "Costs_CVaR"    : Costs_CVaR,
                                        "Imbalance_CVaR": Imbalance_CVaR,
                                        "Last Layer saturation [%]": pct_saturated,
                                        "Last Layer mean Output": mean_act} 

            
                ################break conditions and standard deviation logging
            



            if converged: #break the training loop
                break


############### RETURN SECTION ###########################
    if trial is not None:
    # Se siamo in Optuna, restituiamo SOLO il numero (miglior reward ottenuto)
        return best_run_reward 
                # Save the training log as an excel
    Csv_training = os.path.join(training_check_dir,f'training_log_{Model_Name}.csv')
    log_df.to_csv(Csv_training, index=False)
    # Save the model's state_dict after training
    model_save_path = os.path.join(save_dir, f"{Model_Name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    return model,model_save_path
