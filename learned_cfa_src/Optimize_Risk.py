from .milp.Optimize_Energy_Flux import optimize_energy_flux, optimize_tracking_fixed_grid
import numpy as np


def imbalance_costs(realized,planned,imb_price_pos,imb_price_neg):
    """Hourly imbalance costs of a realized grid exchange against the planned one.
    Dual pricing: deviations that help the system are not rewarded (cost 0).
      Returns
    -------
    costs     : list[T] hourly imbalance cost [EUR]
    Imbalance : list[T] hourly signed deviation, realized - planned [kWh]
    """

    costs = []
    Imbalance = []
    for t in range(len(planned)):
        imbalance = realized[t] - planned[t]
        Imbalance.append(imbalance)
        if imbalance >= 0:
            costs.append(-imb_price_pos[t] * imbalance if imb_price_pos[t] < 0 else 0)
        elif imbalance < 0:
            costs.append(-imb_price_neg[t] * imbalance if imb_price_neg[t] > 0 else 0)
    return costs,Imbalance
   

def Optimize_Risk(data,theta_e,theta_d,gamma,Data_Type):
    """Reward of one day: day-ahead cost plus the risk of the imbalance it exposes to.
    
    Plans the day with the MILP fed by the forecasts rescaled by theta_e and theta_d, then
    measures the imbalance cost that plan would incur against each of the pre-computed
    scenarios. Both terms are normalised before being combined, so that days of different
    magnitude contribute comparably.

        Parameters
    ----------
    data      : Problem_Data of the day, carrying forecasts, prices and the scenario MILPs
    theta_e   : list[T] multipliers applied to the PV forecast
    theta_d   : list[T] multipliers applied to the demand forecast
    gamma     : weight of the risk term
    Data_Type : "forecast" or "Observed", which profiles the MILP is fed

        Returns
    -------
    obj_risk                  : the reward [-]
    SoC_end                   : state of charge at the end of the planned day
    Imbalance_Scenarios       : list[N] total signed deviation of each scenario [kWh]
    Planned_Costs             : day-ahead cost of the plan [EUR]
    Imbalance_Costs_CVaR      : CVaR at 95% of the scenario imbalance costs [EUR]
    Imbalance_costs_scenarios : list[N] imbalance cost of each scenario [EUR]
    """

     
    timesteps = data.Simulation_Data["time_horizon"]*data.Simulation_Data["time_resolution"]
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]
    Imbalance_costs_scenarios =[]
    Imbalance_Scenarios = []
    Total_Costs = []

    # Milp Scenario Data
    Costs_Milps         = data.Costs_Milps
    Grid_Exchange_Milps = data.Grid_Exchange_Milps

    if len(Costs_Milps)>= len(Grid_Exchange_Milps):
        N_scenarios = len(Grid_Exchange_Milps)
    else:
        N_scenarios = len(Costs_Milps)
############################## Modified Forecast Optimization ################################################################
    ##modified computation
    Planned_Costs,_,SoC,_,_,_,_,_,_,_,_,_,Planned_Grid_Exchange,_ = optimize_energy_flux(data,theta_e,theta_d,Data_Type) 
    SoC_end = SoC[-1]

    ##baseline computation
    Baseline_Grid_Exchange = data.Grid_Exchange_Baseline
    Cost_Baseline = data.Cost_Baseline
    Imbalance_costs_baseline_tot = []
    
    for s in range(N_scenarios):
        Imbalance_costs_baseline,_ = imbalance_costs(Grid_Exchange_Milps[s],Baseline_Grid_Exchange,positive_price_imbalance,negative_price_imbalance)
        Imbalance_costs_baseline_tot.append(sum(Imbalance_costs_baseline))

    _,Imbalance_Costs_VaR_baseline,sigma_tail_baseline = RightCVaR(Imbalance_costs_baseline_tot,alpha=0.95) 
        
################################ Imbalance Risk ##############################################################################

    for s in range(N_scenarios):
       
        Imbalance_costs,imbalance_s = imbalance_costs(Grid_Exchange_Milps[s],Planned_Grid_Exchange,positive_price_imbalance,negative_price_imbalance)

        Total_Costs_scenarios = Planned_Costs+sum(Imbalance_costs)
        Imbalance_Scenarios.append(sum(imbalance_s))
        Imbalance_costs_scenarios.append(sum(Imbalance_costs))
        Total_Costs.append(Total_Costs_scenarios)
    
    Imbalance_costs_scenarios = np.array(Imbalance_costs_scenarios)

  
    Imbalance_Costs_CVaR,_,_ = RightCVaR(Imbalance_costs_scenarios,alpha=0.95)

################################# REWARD FUNCTION ##########################################################################


##### Normalization##### Normalization z-score
    Da_Scenarios_mean = np.mean(np.array(Costs_Milps))
    Da_Scenarios_variance = np.std(np.array(Costs_Milps))
    Planned_Costs_norm = (Planned_Costs - Da_Scenarios_mean )/Da_Scenarios_variance
    
    Imbalance_Costs_CVaR_norm = (Imbalance_Costs_CVaR-Imbalance_Costs_VaR_baseline)/sigma_tail_baseline 
#### reward ######  
 
    obj_risk = Planned_Costs_norm +gamma*Imbalance_Costs_CVaR_norm
   
    return obj_risk,SoC_end,Imbalance_Scenarios,Planned_Costs,Imbalance_Costs_CVaR,Imbalance_costs_scenarios

def RightCVaR(Data,alpha):
    """Mean of the worst alpha-tail on the right, i.e. of the highest values.

    Applied to costs, where the risk lies in the upper tail: the alpha fraction of scenarios
    that cost the most.

    Returns
    -------
    CVaR       : mean of the values at or above the alpha quantile
    VaR        : the alpha quantile itself
    sigma_tail : standard deviation of that tail, plus 1e-6 so that it stays a safe divisor
                 when the tail is degenerate and its spread collapses to zero
    """
    Data = np.asarray(Data)
    if Data.size == 0:
        return 0,0,0  
    
    VaR = np.percentile(Data, (alpha)*100)
    tail_data = Data[Data >= VaR]

    
    CVaR = tail_data.mean()
    sigma_tail = tail_data.std() + 1e-6
  
    return CVaR,VaR,sigma_tail

 

def Real_Loss_Assessment(data, theta_e, theta_d, Data_Type):
    """Ex-post evaluation of a day: what the committed plan actually cost once reality unfolded.

    Three MILPs are solved. The day-ahead plan on the rescaled forecasts, which fixes the
    commitment. A tracking problem on the observed profiles, which follows that commitment as
    closely as the battery allows: whatever it cannot deliver is the physical imbalance, and
    is settled at the imbalance prices. And the optimum on the observed profiles, the
    unreachable perfect-foresight benchmark used only for comparison.

    Parameters
    ----------
    data      : Problem_Data of the day
    theta_e   : list[T] multipliers applied to the PV forecast
    theta_d   : list[T] multipliers applied to the demand forecast
    Data_Type : profiles the day-ahead plan is built on, normally "forecast"

    Returns
    -------
    Total_Imbalance_Cost   : settlement cost of the deviations [EUR]
    Costs_observed_optimal : cost of the perfect-foresight plan [EUR]
    Costs_forecast         : day-ahead cost of the committed plan [EUR]
    Energy_observed        : list[T] observed PV [kW]
    Demand_observed        : list[T] observed demand [kW]
    Realized_SOC           : list[T] state of charge actually followed
    Grid_Exchange_REALIZED : list[T] grid exchange actually delivered [kWh]
    Imbalance              : list[T] hourly deviation from the commitment [kWh]
    Costs_Realized_Total   : Costs_forecast + Total_Imbalance_Cost [EUR]
    """
    timesteps = data.Simulation_Data["time_horizon"] * data.Simulation_Data["time_resolution"]
    
    # -----------------------------------------------------------------------------------------
    # 1. DAY AHEAD PLAN (Forecast)
    # -----------------------------------------------------------------------------------------

    Costs_forecast,_,_,_,_,_,_,_,_,_,_,_, Grid_Exchange_forecast, Costs_hourly_forecasted = optimize_energy_flux(data, theta_e, theta_d, Data_Type)

    # -----------------------------------------------------------------------------------------
    # 2. REALIZED SCENARIO 
    # -----------------------------------------------------------------------------------------
    
    Grid_Exchange_REALIZED, Slack_Volumes, Realized_SOC = optimize_tracking_fixed_grid(data, "Observed", Grid_Exchange_forecast)

    # -----------------------------------------------------------------------------------------
    # 3. PERFECT FORECAST
    # -----------------------------------------------------------------------------------------
 
    Costs_observed_optimal,_,_,_,_,_,_,Energy_observed,_,Demand_observed,_,_,_,_ = optimize_energy_flux(data, [1]*timesteps, [1]*timesteps, "Observed")

    # -----------------------------------------------------------------------------------------
    # 4. IMBALANCE COMPUTATION
    # -----------------------------------------------------------------------------------------
    positive_price_imbalance = data.Prices["Imbalance_positive"]
    negative_price_imbalance = data.Prices["Imbalance_negative"]
    

    Imbalance_costs,Imbalance = imbalance_costs(Grid_Exchange_REALIZED,Grid_Exchange_forecast,positive_price_imbalance,negative_price_imbalance)

    Total_Imbalance_Cost = sum(Imbalance_costs)

    # -----------------------------------------------------------------------------------------
    # 5. RESULT AGGREGATION
    # -----------------------------------------------------------------------------------------
  
    Costs_Realized_Total = Costs_forecast + Total_Imbalance_Cost

    return  Total_Imbalance_Cost, Costs_observed_optimal , Costs_forecast,Energy_observed,Demand_observed, Realized_SOC,Grid_Exchange_REALIZED,Imbalance,Costs_Realized_Total