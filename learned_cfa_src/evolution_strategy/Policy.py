import torch
import torch.nn as nn

# Define the Policy Network class
class LearnedCFA(nn.Module):
    """Context-aware CFA policy: reads the uncertainty context of a day and reshapes its forecasts.
    One multiplier per hour and per variable, produced by a feed-forward network from
    the daily context. 

    Input, 10 features per hour, in this order:
        0-3   PV mean, PV deviation, demand mean, demand deviation normalised with mu, sigma
        4-5   sine and cosine encoding of the hour
        6-9   theta_max_e, theta_min_e, theta_max_d, theta_min_d     the day's uncertainty envelope

    The output layer is initialised to zero on purpose, so that an untrained policy returns
    theta = 1 exactly and reproduces the deterministic q50 plan. Training therefore starts
    from the baseline and departs from it only when the reward improves.
    """
    def __init__(self, input_size, output_size,mu_for_norm,sigma_for_norm,hidden_size=48,n_layers=1):
        super(LearnedCFA, self).__init__()
        
#################### Define the network architecture (Dynamic)##############
        layers = []

        # input layer #
        layers.append(nn.Linear(input_size,hidden_size))  
        layers.append(nn.ReLU())
        # hidden layers #
        for _ in range(n_layers -1):
                layers.append(nn.Linear(hidden_size,hidden_size))
                layers.append(nn.ReLU())

       
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)  

        # --- INIZIALIZATION OUTPUT LAYER ---
        nn.init.constant_(self.output_layer.weight, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # --- NORMALIZATION ---
        self.epsilon = 1e-6
        self.register_buffer('mu', mu_for_norm)
        self.register_buffer('sigma', sigma_for_norm)

    def forward(self, x):
        # --- NORMALIZATION --- 
        x_norm = x.clone()
        x_norm[..., :4] = (x_norm[..., :4]-self.mu) / (self.sigma + self.epsilon)
        # --- Maximum bound  ---

        theta_max_e = x[..., -4] 
        theta_min_e = x[..., -3]
        theta_max_d = x[..., -2]
        theta_min_d = x[..., -1]

        r=1
        upper_limit = 1.0 + r
        lower_limit = 1.0 - r
        theta_max_e=torch.clamp(theta_max_e,max=upper_limit)
        theta_max_d=torch.clamp(theta_max_d,max=upper_limit)
        theta_min_e=torch.clamp(theta_min_e,min=lower_limit)
        theta_min_d=torch.clamp(theta_min_d,min=lower_limit)

        theta_min_e = torch.minimum(theta_min_e, theta_max_e)
        theta_min_d = torch.minimum(theta_min_d, theta_max_d)       

        theta_max = torch.stack((theta_max_e,theta_max_d),dim=-1)
        theta_min = torch.stack((theta_min_e,theta_min_d),dim=-1)
        
        # --- FORWARD PASS --- 
        features = self.feature_extractor(x_norm)
        out_raw = self.output_layer(features)
        delta_theta = torch.tanh(out_raw)

        up_room = theta_max - 1.0     
        down_room = 1.0 - theta_min   
        
        scale_factor = torch.where(delta_theta > 0, up_room, down_room)
        
        theta = 1.0 + delta_theta * scale_factor

        return theta,delta_theta



class StaticCFA(nn.Module):
    """Context-independent CFA: a single fixed theta vector shared by all days.
        48 free parameters (24 hours x 2 variables) and no daily contextual input. 
        Zero init reproduces the deterministic q50 baseline, mirroring the zero-initialized
        output layer of LearnedCFA, and tanh enforces the same global +/-100% range set by
        the regularizer r = 1.0.
    """

    def __init__(self, timesteps=24, n_vars=2):
        super(StaticCFA, self).__init__()
        
        self.w = nn.Parameter(torch.zeros(timesteps, n_vars))

    def forward(self, x):
        """x is ignored except for its batch size: the same theta is broadcast to every day."""
        delta_theta = torch.tanh(self.w)          # [24, 2] in [-1, 1]
        theta = 1.0 + delta_theta                 # [24, 2] in [0, 2]
        if x.dim() == 3:                          # [N_days, 24, features]
            n_days = x.shape[0]
            theta = theta.unsqueeze(0).expand(n_days, -1, -1)
            delta_theta = delta_theta.unsqueeze(0).expand(n_days, -1, -1)
        return theta, delta_theta


