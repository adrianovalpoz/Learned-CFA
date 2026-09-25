import sys
import os
import optuna
import torch
from .es_Policy import Evolution_Strategy_Training_HPC
from .Policy import PolicyNetwork

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
from ..Optimize_Risk  import Optimize_Risk


def NN_opt_objective(trial,DATA,mu_data,sigma_data,Data_Type,gamma,optimal_workers,Model_Name):
    # Nota: DATA, mu_data, ecc. devono essere disponibili (globali o caricati qui)
    # --- 1. Optuna suggerisce i parametri ---
    
    # archtecture
    hidden_size = trial.suggest_categorical("hidden_dim", [32, 48, 64])
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # ES parameters
    sigma = trial.suggest_float("sigma", 0.05, 0.3)
    alpha = trial.suggest_float("alpha", 0.01, 0.1, log=True) # Log scale è meglio per Learning Rate
   # weight_decay = trial.suggest_float("weight_decay", 0.0, 0.01,log=True)
    pop_size = trial.suggest_categorical("pop_size", [25, 50])

    # --- AGGIUNGI QUESTO PRINT ---
    print(f"\n [Trial {trial.number}] GUESS: Layers={n_layers} | Hidden={hidden_size} | Pop={pop_size} | Alpha={alpha:.5f} | Sigma={sigma:.3f}")
    # Iterazioni ridotte per il tuning (Proxy Run)
    ITERATIONS_PROXY = 100

    Evolution_Params = {"theta_dim"    :48,   # Dimension of the parameter vector
                    "population_size"  :pop_size,   # Number of samples per iteration
                    "sigma"            :sigma,  # Noise standard deviation
                    "alpha"            :alpha,  # Learning rate
                    "n_iterations"     :ITERATIONS_PROXY,
                    "weight_decay"     :0}   # Total iterations

    
    # --- 2. Setup  e ES ---
    # Creiamo la rete con l'hidden dim suggerito
    mu_for_norm = torch.tensor(mu_data, dtype=torch.float32)
    sigma_for_norm = torch.tensor(sigma_data, dtype=torch.float32)

    model = PolicyNetwork(input_size=10, output_size=2, 
                          hidden_size=hidden_size, n_layers= n_layers,
                          mu_for_norm=mu_for_norm, sigma_for_norm=sigma_for_norm)
    

    
    # --- 3. Il Loop di Training (Inner Loop) ---

    reward = Evolution_Strategy_Training_HPC(Optimize_Risk,model,DATA,gamma,Data_Type,optimal_workers,Evolution_Params,Model_Name,save_dir ='',training_check_dir = '',trial=trial)    # Training function Call
    
    
    print(f"[Trial {trial.number}] END   | Final Reward: {reward:.4f}")
    print("---------------------------------------------------------------")

    return reward

