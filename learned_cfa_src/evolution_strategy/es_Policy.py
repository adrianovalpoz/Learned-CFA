import numpy as np
import pandas as pd
import torch
import optuna
import copy
import os
import time
import concurrent.futures



def solve_single_day_worker(args):
    """One day solved in a worker process: unpacks the task and calls the reward function.

    Parameters
    ----------
    args : tuple (func, data_day, theta_e, theta_d, gamma, Data_Type), packed by the caller
           because executor.map passes a single argument

    Returns
    -------
    result : whatever func returns for that day, the six values of Optimize_Risk
    """
 
    func,data_day,theta_e,theta_d,gamma,Data_Type = args
    result = func(data_day, theta_e, theta_d,gamma, Data_Type)

    return  result


def breakconditions(reward_history, patience, tolerance, trial):
    """Stop the training when the reward stops moving.

    Looks at the last `patience` rewards and measures their standard deviation: below
    `tolerance` the search is considered converged.
    Returns (converged, reward_std); reward_std is NaN until `patience` iterations have been
    accumulated, and is logged as such.
    
    Parameters
    ----------
    reward_history : rewards of every iteration so far
    patience       : how many recent iterations the variation is measured on
    tolerance      : threshold below which the search is considered converged
    trial          : an Optuna trial, which silences the printing, None otherwise

    Returns
    -------
    converged  : True when the training should stop
    reward_std : standard deviation of the last `patience` rewards, NaN until that many
                 iterations have been accumulated, and logged as such
    """

    if len(reward_history) >= patience:
        
        recent_rewards = reward_history[-patience:]
        reward_std = np.std(recent_rewards)
        if trial is None:
            print(f"std reward = {reward_std}")
        # Check convergence: std of recent rewards
        if (reward_std) < tolerance:
            print("\033[92mConverged: reward variation below tolerance.\033[0m")
            return True,reward_std
    else:
        reward_std =  float('nan')

    return False,reward_std


def adjust_hyperparameters(iteration, sigma, alpha, Evolution_Params,trial):
    """
    Dynamically adjusts hyperparameters (e.g., mutation rate, learning rate) based on performance.
    """
    alpha_o = Evolution_Params["alpha"]
    sigma_o = Evolution_Params["sigma"]
   
    decay_step = 5                                       # adjust every 5 iteration
    if iteration != 0:
        if iteration % decay_step == 0:  
                alpha = max(alpha * 0.98, alpha_o*0.01)  # Reduce learning rate but ensure it doesn't go below a small threshold
                sigma = max(sigma * 0.98, sigma_o*0.01)  # Reduce learning rate but ensure it doesn't go below a small threshold
                if trial is None:
                    print(f"Iteration {iteration}: Adjusting alpha to {alpha:.4f} | Adjusting sigma to {sigma:.4f}.")
    
    return sigma, alpha

def Theta_Bounds(data): 
    """Per-hour bounds of the multipliers, derived from the forecast uncertainty of the day.

    The Chebyshev band stored in the dataset is expressed in kW; dividing it by the q50
    forecast turns it into the multiplicative range the policy is allowed to move in, which
    is what the policy actually outputs.

    Parameters
    ----------
    data : Problem_Data of the day, for its forecasts and uncertainty bands

    Returns
    -------
    theta_max_e : list[T] upper bound of the PV multiplier
    theta_min_e : list[T] lower bound of the PV multiplier
    theta_max_d : list[T] upper bound of the demand multiplier
    theta_min_d : list[T] lower bound of the demand multiplier
    """

    # Energy
    e_forecast = data.Energy_Data["forecast"]
    e_bound_max = data.Energy_Data["bands"]["max"]
    e_bound_min = data.Energy_Data["bands"]["min"]
    # Demand
    d_forecast = data.Demand_Data["forecast"]
    d_bound_max = data.Demand_Data["bands"]["max"]
    d_bound_min = data.Demand_Data["bands"]["min"]


    arrays = [e_forecast, e_bound_max,e_bound_min, d_forecast, d_bound_max,d_bound_min]

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

def Theta_Choice(data,Opt_Type,policy=None):
    """The multipliers of one day, from whichever policy Opt_Type selects.

    Parameters
    ----------
    data     : Problem_Data of the day
    Opt_Type : "Learned_CFA", "Static_CFA" or "Deterministic"
    policy   : the LearnedCFA module, or the dict read from the Static-CFA JSON, or None

    Returns
    -------
    theta_e : list[T] multipliers applied to the PV forecast
    theta_d : list[T] multipliers applied to the demand forecast
    out_e   : the raw tanh activations for PV, empty for the policies that have none
    out_d   : the raw tanh activations for demand, empty for the policies that have none
    timer   : inference time in seconds, 0 for the policies that do not infer
    
    """
    if Opt_Type == 'Learned_CFA':
        # Extract the statistic from data
        energy_mean       = data.Energy_Data["mean"]
        energy_deviation  = data.Energy_Data["deviation"]
        demand_mean       = data.Demand_Data["mean"]
        demand_deviation  = data.Demand_Data["deviation"]
        sin_enc           = data.Simulation_Data["sin_enc"]
        cos_enc           = data.Simulation_Data["cos_enc"]
       
        # convert them to torch tensors and concatenate
        energy_mean       = torch.tensor(energy_mean,dtype=torch.float32).unsqueeze(-1)
        energy_deviation  = torch.tensor(energy_deviation,dtype=torch.float32).unsqueeze(-1)
        demand_mean       = torch.tensor(demand_mean,dtype=torch.float32).unsqueeze(-1)
        demand_deviation  = torch.tensor(demand_deviation,dtype=torch.float32).unsqueeze(-1)
        sin_enc           = torch.tensor(sin_enc,dtype=torch.float32).unsqueeze(-1)
        cos_enc           = torch.tensor(cos_enc,dtype=torch.float32).unsqueeze(-1)
       
        # theta boundaries 
        theta_max_e,theta_min_e,theta_max_d,theta_min_d = Theta_Bounds(data)
        theta_max_e       = torch.tensor(theta_max_e,dtype=torch.float32).unsqueeze(-1)
        theta_min_e       = torch.tensor(theta_min_e,dtype=torch.float32).unsqueeze(-1)
        theta_max_d       = torch.tensor(theta_max_d,dtype=torch.float32).unsqueeze(-1)
        theta_min_d       = torch.tensor(theta_min_d,dtype=torch.float32).unsqueeze(-1)
       
        statistics = torch.cat((energy_mean,energy_deviation,demand_mean,demand_deviation,sin_enc,cos_enc,theta_max_e,theta_min_e,theta_max_d,theta_min_d),dim=-1)

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
        timer=0     
        out_e = []
        out_d = []
        theta_e = [1]*timesteps
        theta_d = [1]*timesteps
    else:
        raise ValueError(f"Unknown Opt_Type: {Opt_Type}")

    return theta_e,theta_d,out_e, out_d,timer

#%% ####################  high-performance computing functions ######################################################
def Input_batch(data,model):
  
    """Pack the daily contexts of the whole training set into one tensor, once per run.

    The features of every day never change during training, so they are built here and reused
    at every iteration: this is what lets a clone be evaluated with a single forward pass
    instead of one per day.

    Parameters
    ----------
    data  : {day: Problem_Data} of the training set
    model : used only to match its dtype and device

    Returns
    -------
    batch_input   : tensor [n_days, 24, 10] with the features of every day
    days_sequence : the days in the order of the first dimension, so that results can be
                    mapped back to the day they belong to
    """

    days_sequence = sorted(list(data.keys()))
    
    input_tensors_list = []
    # ---  Batch Data Construction ---
    
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

        input_tensors_list.append(Day_Stats)

    # ---  BATCH  ---
    
    batch_input = torch.stack(input_tensors_list)
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        batch_input = batch_input.to(target_dtype)
  
    return batch_input,days_sequence

def build_clone_tasks(model,gamma,batch_input,days_sequence,data, Data_Type, func):

    """One forward pass of a perturbed clone, turned into the list of MILP tasks to dispatch.
    Returns
    -------
    tasks : list of (func, data_day, theta_e, theta_d, gamma, Data_Type), one per day, in the
            order of days_sequence
    """
    was_training = model.training
    model.eval()
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        batch_input = batch_input.to(target_dtype)

    with torch.no_grad():
        batch_theta, _ = model(batch_input)

    tasks = []
    for i, day in enumerate(days_sequence):
        theta_e = batch_theta[i,:,0].tolist()
        theta_d = batch_theta[i,:,1].tolist()
        tasks.append((func,data[day],theta_e,theta_d,gamma,Data_Type))
    if was_training:
        model.train()
    return tasks

def calculate_clone_reward(clone_results):

    """Sum the per-day results of one clone into the reward the evolutionary strategy ranks it by.

    Parameters
    ----------
    risultati_clone : the results returned by the pool for the days of one clone

    Returns
    -------
    Reward : the summed reward with the sign flipped, because the objective is a cost and the
             strategy maximises
    """
    reward = 0
    for obj in clone_results:
        reward += obj[0] # obj[0] corrisponding to workerd reward   
    Reward = -reward

    return Reward

def evaluate_model_HPC(func, model,gamma, batch_input, days_sequence, data, Data_Type, executor):
  
    """Evaluate the current, policy and collect everything the training log needs.

    Returns
    -------
    Reward         : summed reward with the sign flipped
    Costs          : day-ahead cost summed over the training days [EUR]
    Imbalance_CVaR : imbalance CVaR summed over the training days [EUR]
    theta_e, theta_d           : multipliers of the last day, as diagnostics
    theta_max_e, theta_min_e   : their bounds for that day
    theta_max_d, theta_min_d   : the same for demand
    last_delta_theta           : raw tanh activations of the last day, from which the
                                 saturation percentage in the log is computed
    """
    was_training = model.training
    model.eval()
    target_dtype = next(model.parameters()).dtype
    if batch_input.dtype != target_dtype:
        batch_input = batch_input.to(target_dtype)
    
    with torch.no_grad():
        batch_theta, batch_delta = model(batch_input)

    # --- EMERGENCY DTYPE CHECK ---
    tasks = []
    for i, day in enumerate(days_sequence):
        theta_e = batch_theta[i,:,0].tolist()
        theta_d = batch_theta[i,:,1].tolist()
        tasks.append((func, data[day], theta_e, theta_d, gamma, Data_Type))

    # ---  Multi-Core HPC execution---
    
    results = list(executor.map(solve_single_day_worker, tasks))
    
    # --- Results Aggregation---
    sum_reward = 0
    Costs = 0
    Imbalance_CVaR = 0

    for obj in results:
        sum_reward +=obj[0]
        Costs  += obj[3]
        Imbalance_CVaR += obj[4]

    
    Reward = -sum_reward
    last_delta_theta = batch_delta[-1]
    theta_max_e, theta_min_e, theta_max_d, theta_min_d = Theta_Bounds(data[days_sequence[-1]])

    if was_training:
        model.train()

    return Reward, Costs, Imbalance_CVaR, theta_e, theta_d, theta_max_e, theta_min_e, theta_max_d, theta_min_d, last_delta_theta

def Evolution_Strategy_Training_HPC(func,model,data,gamma,Data_Type,optimal_workers,Evolution_Params,Model_Name,save_dir,training_check_dir,trial = None):
    """Estimate the parameters of a policy with an evolutionary strategy.

    Each iteration perturbs the parameters along `population_size` random directions, solves
    the MILP of every training day for every clone in a process pool, and moves the parameters
    along the reward-weighted average of those directions. There are no gradients: the policy
    is evaluated, which is what allows the MILP to sit inside the objective.

    LearnedCFA and StaticCFA are both trained by this same function.

    Parameters
    ----------
    func               : the reward, Optimize_Risk
    model              : the policy to train, modified in place
    data               : {day: Problem_Data} of the training months
    gamma              : weight of the risk term, passed through to func
    Data_Type          : profiles the MILP is fed, normally "forecast"
    optimal_workers    : processes in the pool, best set to the physical cores
    Evolution_Params   : population_size, sigma, alpha, n_iterations, weight_decay
    Model_Name         : names the artefacts written to disk
    save_dir           : where <Model_Name>.pth is written
    training_check_dir : where training_log_<Model_Name>.csv is written
    trial              : an Optuna trial during the hyperparameter search, None otherwise

    Returns
    -------
    model           : the trained policy
    model_save_path : path of the .pth just written

    When called from Optuna, i.e. with trial not None, it returns the best reward alone and
    writes nothing to disk, because the search needs a single number per trial.
    """
    
    population_size = Evolution_Params["population_size"]     # Number of samples per iteration
    sigma           = Evolution_Params["sigma"]               # Noise standard deviation
    alpha           = Evolution_Params["alpha"]               # Learning rate
    n_iterations    = Evolution_Params["n_iterations"]        # Total iterations
    weight_decay    = Evolution_Params["weight_decay"]        # to save from saturation
    days_per_clone  = len(data)


    print("\n Data tensors pre-computation (HPC optimization)...")

    batch_input,days_sequence = Input_batch(data,model)
    #  --- Initialization ---

    NN_params_dim =  sum(p.numel() for p in model.parameters())  # Total number of parameters in the model
    reward_history =[]                                           # needed to track the rewards and use the breack condition

    # cpu usage
    logic_processors_usage = False #False if i want to use only the real physical cores

    print("\n Global Pool inizialization HPC...")
    print(f"Core Types requested: {'logicals' if logic_processors_usage else 'physical'}")
    print(f"Workers assigned to CPU: {optimal_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as global_executor:
        # --- Starting (BASELINE) ---
        if trial is None:
            print("\n--- BASELINE CHECK (Before Training) ---")
            base_reward, base_costs, base_imbcvar, _, _, _, _, _, _,_ = evaluate_model_HPC(func, model,gamma,batch_input,days_sequence ,data, Data_Type,global_executor)
            print(f"BASELINE | Reward: {base_reward:.4f} | Costs: {base_costs:.2f} | imb CVaR: {base_imbcvar:.2f}")
            print("----------------------------------------\n")

        #  --- Main ES Loop ---
        log_df = pd.DataFrame(columns=["iteration", "reward", "std_reward","Costs","Imbalance_CVaR","Last Layer saturation [%]", "Last Layer mean Output"]) # dataframe for logging
        best_run_reward = -float('inf')
        
        for iteration in range(n_iterations):
            start_iteration = time.perf_counter()
            noise = np.random.randn(population_size, NN_params_dim)
            rewards = np.zeros(population_size)
            # =========================================================================
            # HPC (FLAT MULTIPROCESSING)
            # =========================================================================
           
            mega_lista_task = []
            
            for i in range(population_size):
                perturbed_model = copy.deepcopy(model)
                # Perturb each parameter (weight) of the model
                idx = 0
                for param in perturbed_model.parameters():
                    numel = param.numel()
                    perturbation = torch.tensor(noise[i, idx:idx + numel], dtype=torch.float32).view(param.size())
                    param.data += sigma * perturbation  # Apply the noise perturbation
                    idx += numel
                
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
            Reward,Costs,Imbalance_CVaR,theta_e,theta_d,theta_max_e,theta_min_e,theta_max_d,theta_min_d,last_delta_theta = evaluate_model_HPC(func,model,gamma,batch_input,days_sequence,data,Data_Type,global_executor)
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
                print(f"\033[96mIteration {iteration:3d} | Reward: {Reward:.4f} | Costs: {Costs:.2f} | Imbalance_CVaR: {Imbalance_CVaR:.2f}\033[0m")
                print("\033[96m____________________________________________________________________________________________________________\033[0m")


            # sigma, alpha = adjust_hyperparameters(iteration, reward_history, sigma, alpha, patience=10, tolerance=1e-3)
            
            
                # Logging into dataframe
                log_df.loc[len(log_df)] ={"iteration"     : iteration,
                                        "reward"        : Reward,
                                        "std_reward"    : std_reward,
                                        "Costs"         : Costs,
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
