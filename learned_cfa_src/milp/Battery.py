import math


class Battery:
    def __init__(self,SoC_in:float=0.7,flux_names:list=[],Energy_Fluxes:dict={},Capacity=60,Eff_Charge=0.9,Eff_Discharge=0.9,Max_Charge=5,Max_Discharge=5,timesteps = 24,mode_binaries= None):
        self.Capacity       = Capacity
        self.Eff_Charge     = Eff_Charge
        self.Eff_Discharge  = Eff_Discharge
        self.Max_Charge     = Max_Charge
        self.Max_Discharge  = Max_Discharge
        self.mode_binaries  = mode_binaries # if = 0 cahrging allowed if = 1 discharged allowed
        self.SoC_in         = SoC_in
        self.flux_names     = flux_names
        self.Fluxes_in,self.Fluxes_Out = self.Fluxes_Type(Energy_Fluxes)
      
    def __str__(self):
        return f" {self.SoC_in,self.N_Flux_in, self.N_flux_out ,self.Capacity,self.Eff_Charge,self.Eff_Discharge,self.Max_Charge,self.Max_Discharge}"

    def Optimization_Constraints(self,SOC,t)->list:
        constraints = [self.Max_Charge_Constraint(t), 
                       self.Max_Discharge_Constraint(t),  
                       self.Overcharge_Limit(SOC,t),
                       self.Actual_Capacity_Limit(SOC,t),
                       self.SOC_Evolution(SOC,t)]
           #            self.Actual_Capacity_Limit(SOC,t),
        name =[f"Max_Charge_t{t:02d}",
               f"Max_Discharge_t{t:02d}",
               f"Max_Overcharge_t{t:02d}",
               f"Actual_Capacity_Limit_t{t:02d}",
               f"SOC_evolution_t{t:02d}"]
        return list(zip(name,constraints))
    
    def SOC_Evolution(self,SOC,t):
        if t == 0:
            SoC_Evolution = self.Capacity*SOC[t] == self.Capacity*self.SoC_in + self.Eff_Charge*sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in))) - sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out)))/self.Eff_Discharge
             
             #SoC_Evolution = SOC[t] == self.SoC_in    
        else:
            SoC_Evolution = self.Capacity*SOC[t] == self.Capacity*SOC[t-1] + self.Eff_Charge*sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in))) - sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out)))/self.Eff_Discharge
        return SoC_Evolution
    
    def Fluxes_Type(self,Energy_Fluxes:dict):
        Fluxes_In = []
        Fluxes_Out = []
        for name in self.flux_names:
            if Energy_Fluxes[name].type == 1:
                Fluxes_In.append(Energy_Fluxes[name].variable)
            else:
                Fluxes_Out.append(Energy_Fluxes[name].variable)
        return Fluxes_In, Fluxes_Out
    
    def Max_Charge_Constraint(self,t): 
        if self.mode_binaries:
            return sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in)))  <= self.Max_Charge*(1-self.mode_binaries[t])
        else:
            return sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in)))  <= self.Max_Charge
    
    def Max_Discharge_Constraint(self,t):
        if self.mode_binaries:
            return sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out))) <= self.Max_Discharge*self.mode_binaries[t]
        else:
            return sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out))) <= self.Max_Discharge

    def Actual_Capacity_Limit(self,SOC,t):
        if t == 0:
            return sum(self.Fluxes_Out[i][t] for i in range(len(self.Fluxes_Out)))/ self.Eff_Discharge <= self.Capacity * self.SoC_in
        return sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out)))/ self.Eff_Discharge <= self.Capacity*SOC[t-1]
        # if t == 0:
        #     return sum(self.Fluxes_Out[i][t] for i in range(len(self.Fluxes_Out)))<= self.Capacity * self.SoC_in
        # return sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out))) <= self.Capacity*SOC[t-1]
    
    def Overcharge_Limit(self,SOC,t):
        if t == 0:
            return self.Eff_Charge*sum(self.Fluxes_in[i][t] for i in range(len(self.Fluxes_in))) <= self.Capacity * (0.9 - self.SoC_in)
        return  self.Eff_Charge*sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in))) <= self.Capacity*(0.9 - SOC[t-1])
         #     if t == 0:
    #         return sum(self.Fluxes_in[i][t] for i in range(len(self.Fluxes_in))) <= self.Capacity * (0.9 - self.SoC_in)
    #     return  sum(self.Fluxes_in[i][t]for i in range(len(self.Fluxes_in))) <= self.Capacity*(0.9 - SOC[t-1])
    
    def Cycle_Counter(self,timesteps):
        discharge_energy= []
        for t in range(timesteps):
            discharge_energy.append(sum(self.Fluxes_Out[i][t]for i in range(len(self.Fluxes_Out))))
        cycles =sum(discharge_energy)/self.Capacity
        return cycles,discharge_energy
    
    def Derating(self,x_out,timesteps):
        [cycles,discharge_energy] = self.Cycle_Counter(timesteps)
        Cap_loss = 30.330*(math.exp(-(31.5/8.314)*278)*(cycles*self.Capacity)**0.5)
        New_Capacity =self.Capacity*(1-Cap_loss)
        return Cap_loss, New_Capacity