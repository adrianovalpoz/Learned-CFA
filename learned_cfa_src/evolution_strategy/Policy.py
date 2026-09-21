import torch
import torch.nn as nn
import numpy as np

# Define the Policy Network class
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size,mu_for_norm,sigma_for_norm,hidden_size=48,n_layers=1):
        super(PolicyNetwork, self).__init__()
        
#################### Define the network architecture (Dynamic)##############
        layers = []

        # input layer #
        layers.append(nn.Linear(input_size,hidden_size))  
        layers.append(nn.ReLU())
        # hidden layers #
        for _ in range(n_layers -1):
                layers.append(nn.Linear(hidden_size,hidden_size))
                layers.append(nn.ReLU())
        # toa ppòly the features to the layers
        self.feature_extractor = nn.Sequential(*layers)
        # output layers
        self.output_layer = nn.Linear(hidden_size, output_size)  # Output layer (theta values)

        # --- FORZATURA ZERO ---
        # Inizializzando a 0, l'output grezzo iniziale sarà [0, 0]
        nn.init.constant_(self.output_layer.weight, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

        ############ Normalization
        self.epsilon = 1e-6
        # --- SALVATAGGIO STATISTICHE NEL MODELLO ---

        self.register_buffer('mu', mu_for_norm)
        self.register_buffer('sigma', sigma_for_norm)


    def forward(self, x):
######### Normalization ######################################
        x_norm = x.clone()
        x_norm[..., :4] = (x_norm[..., :4]-self.mu) / (self.sigma + self.epsilon)
######### Maximum bound ######################################
        theta_max_e = x[..., -4]  # Assuming last four features are bounds for energy and demand
        theta_min_e = x[..., -3]
        theta_max_d = x[..., -2]
        theta_min_d = x[..., -1]

# modifiing the bounds 
        r=1
        upper_limit = 1.0 + r
        lower_limit = 1.0 - r
        theta_max_e=torch.clamp(theta_max_e,max=upper_limit)
        theta_max_d=torch.clamp(theta_max_d,max=upper_limit)
        theta_min_e=torch.clamp(theta_min_e,min=lower_limit)
        theta_min_d=torch.clamp(theta_min_d,min=lower_limit)
# Soluzione: Il minimo non può mai essere superiore al massimo
        theta_min_e = torch.minimum(theta_min_e, theta_max_e)
        theta_min_d = torch.minimum(theta_min_d, theta_max_d)       
# min cant be higehr than max
        theta_max = torch.stack((theta_max_e,theta_max_d),dim=-1)
        theta_min = torch.stack((theta_min_e,theta_min_d),dim=-1)
        
################### Forward pass through the network ##########################
        features = self.feature_extractor(x_norm) # Ora usa ReLU interna
        out_raw = self.output_layer(features)
        delta_theta = torch.tanh(out_raw)
# --- LOGICA ASIMMETRICA ---
        # Calcoliamo quanto spazio abbiamo sopra e sotto l'1
        up_room = theta_max - 1.0     
        down_room = 1.0 - theta_min   
        
        # Se delta_theta > 0, usiamo lo spazio superiore (up_room)
        # Se delta_theta < 0, usiamo lo spazio inferiore (down_room)
        # torch.where(condition, if_true, if_false)
        scale_factor = torch.where(delta_theta > 0, up_room, down_room)
        
        # Calcolo finale
        # Se delta_theta è 0 -> theta = 1.0 + 0 * scale = 1.0 (ESATTO)
        # Se delta_theta è +1 -> theta = 1.0 + 1 * up_room = theta_max
        # Se delta_theta è -1 -> theta = 1.0 - 1 * down_room = theta_min
        theta = 1.0 + delta_theta * scale_factor

        # Sicurezza finale: nel caso in cui 1.0 fosse FUORI dai bounds (caso raro/errore dati)
        #theta = torch.clamp(theta, theta_min, theta_max)

        return theta,delta_theta  # Return the vector of theta values
    

