from mealpy.swarm_based import GWO
import scipy.stats as stats
import numpy as np
from copy import deepcopy

class CG_GWO(GWO.OriginalGWO):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)
        
    def evolve(self, epoch):
        super().evolve(epoch)
        for idx in range(self.pop_size):
            agent = self.pop[idx].copy()
            
            r = np.random.rand()
            if r < 0.5:
                mutation = stats.cauchy.rvs(loc=0, scale=1, size=self.problem.n_dims)
                step = agent.solution * mutation
            else:
                mutation = stats.norm.rvs(loc=0, scale=1, size=self.problem.n_dims)
                step = agent.solution * mutation
                
            a = 2 - epoch * (2 / self.epoch)
            agent.solution = agent.solution + a * step
            agent.solution = np.clip(agent.solution, self.problem.lb, self.problem.ub)
            agent.target = self.get_target(agent.solution)
            
            if self.compare_target(agent.target, self.pop[idx].target):
                self.pop[idx] = agent
        _, self.g_best = self.update_global_best_agent(self.pop)
