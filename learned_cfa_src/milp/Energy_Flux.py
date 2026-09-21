from pulp import LpVariable
# class to store the energy fluxes considering their direction in-out of the storage
class Energy_Flux:
    def __init__(self,type,name,timesteps):
        self.type = type #1 input, 0 output
        self.name = name
        self.variable = LpVariable.dicts(self.name, range(timesteps), lowBound=0, cat='Continuous')
