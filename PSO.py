import numpy as np
import math
import time
import matplotlib.pyplot as plt
from typing import Callable, List


### objective function: test1(Rastrigin function)
def Rastrigin_fun(val:List[int], dim:int):
    obj = 0
    for dim_ite in range(dim): 
        obj += (val[:, dim_ite] ** 2 - 10 * np.cos(2 * np.pi * val[:, dim_ite]))
    obj += dim * 10
    return obj

### objective function: test2(STYBLINSKI-TANG function)
def Styblinski_Tang_fun(val:List[int], dim:int):
    obj = 0
    for dim_ite in range(dim):
        obj += val[:, dim_ite] ** 4 - 16 * val[:, dim_ite] ** 2 + 5 * val[:, dim_ite]
    obj /= 2
    return obj

class PSO:
    '''
    Formula: v = w ∙ v + c1 ∙ rand() ∙ ( Pbest - x ) + c2 ∙ Rand() ∙ ( Gbest - x )
    --------------------------------------------------------------------------
    Paramter:
    obj: objective funcion
    obj_sta: 'max' or 'min'
    size_popu: total number of partitial
    ite: total number of iteration
    w: inertia weight
    c1: cognition parameter
    c2: social parameter
    LB: the lower bound for different dimension
    UB: the upper bound for different dimension
    v_max: acceptable maximum speed(including different directions)
    '''
    def __init__(self, obj:Callable[[List[int]], int], obj_sta:str, size_popu:int, ite:int, dim:int, w:List[float], c1:int, c2:int, LB:List[int], UB:List[int], v_max:List[float]):
        ### setting the parameter
        self.obj = obj
        self.obj_sta = obj_sta
        self.size_popu = size_popu
        self.ite = ite
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.LB = LB
        self.UB = UB
        self.v_max = v_max
        
        ### setting the object properties(initial status)
        self.loc = np.add(LB, np.subtract(UB, LB) * np.random.uniform(size=(size_popu,dim))) # personal location
        self.fit = self.obj(self.loc, self.dim) 
        self.vel = v_max * np.random.uniform(-1, 1, size=(size_popu, dim)) # personal moving velocity
        self.pbest = self.loc # best personal location in the past
        self.gbest = self.pbest[np.argmin(self.obj(self.pbest, self.dim))] if self.obj_sta == 'min' else self.pbest[np.argmax(self.obj(self.pbest))] # best group location in the past

    ### other limitation for different combination of dimensions
    def other_limit(self) -> None:
        x, y = self.loc[:,0], self.loc[:,1]
        num = len(self.loc[x + y > 1])
        # random produce the loction untill it satisfy the limitation
        while num != 0:
            self.loc[x + y > 1] = np.add(LB, np.subtract(UB, LB) * np.random.uniform(size=(num,dim)))
            num = len(self.loc[x + y > 1])

    ### update the location and velocity of each particle
    def update(self, counts:int) -> None:
        # update the fitness value
        self.fit = self.obj(self.loc, self.dim) 
        con = self.fit < self.obj(self.pbest, self.dim) if self.obj_sta == 'min' else self.fit > self.obj(self.pbest, self.dim)
        # update the pbest
        self.pbest[con] = self.loc[con] 
        # update the gbest
        self.gbest = self.pbest[np.argmin(self.obj(self.pbest, self.dim))] if self.obj_sta == 'min' else self.pbest[np.argmax(self.obj(self.pbest, self.dim))]
        # Inertial weight
        w_inertia = self.w[1] - (counts/self.ite) * (self.w[1] - self.w[0])
        # update velocity
        self.vel = w_inertia * self.vel + np.random.uniform() * self.c1 * (self.pbest - self.loc) + np.random.uniform() * self.c2 * (self.gbest - self.loc) 
        
        # restrict the velocity into the boundary
        for dim in range(self.dim):
            if np.sum(abs(self.vel[:,dim]) > v_max[dim]):
                self.vel[:,dim][self.vel[:,dim] > v_max[dim]] = v_max[dim]
                self.vel[:,dim][abs(self.vel[:,dim]) > v_max[dim]] = -1 * v_max[dim]

        # restrict the location in the boundary
        self.loc = self.loc + self.vel
        for dim in range(self.dim):
            if np.sum(self.loc[:,dim] < self.LB[dim]):
                self.loc[:,dim][self.loc[:,dim] < self.LB[dim]] = self.LB[dim]
            elif np.sum(self.loc[:,dim] > self.UB[dim]):
                self.loc[:,dim][self.loc[:,dim] > self.UB[dim]] = self.UB[dim]

        # restrict the location in other boundary
        self.other_limit()

    ### connect the all process in one function
    def process(self) -> None:
        record_gbest_val = [] # record gbest value
        record_best_pbest_val = [] # record pbest value
        for counts in range(self.ite):
            self.update(counts)
            record_gbest_val.append(self.obj(np.expand_dims(self.gbest, axis=0), self.dim)[0])
            record_best_pbest_val.append(np.min(self.obj(self.pbest, self.dim))) if self.obj_sta == 'min' else record_best_pbest_val.append(np.max(self.obj(self.pbest, self.dim)))
        print(self.gbest)
        print(record_gbest_val[-1])
        end_time = time.time()
        print(f'It totally use {end_time - star_time:.2f} seconds')
        
        ### Show the graph
        plt.plot(record_gbest_val)
        plt.plot(record_gbest_val, label='best record so far')
        plt.plot(record_best_pbest_val, label='best in population')
        plt.legend()
        plt.title('The convergence histories of PSO')
        plt.xlabel('The iteration of moving')
        plt.ylabel('The best fitness value in these particles')
        plt.show()
        
### parameter setting
obj_sta = 'min' # 'max' or 'min'
size_popu, dim = 200, 2 # total number of partitial and dimension
ite = 100 # total number of iteration
w = [0.2, 1] # inertia weight
c1, c2 = 1, 1.5 # cognition parameter and social parameter
LB, UB = [-5] * dim, [5] * dim # the lower bound for different dimension, the upper bound for different dimension
v_max = [0.5] * dim # acceptable maximum speed(including different directions)

# main program
star_time = time.time()
P1 = PSO(Styblinski_Tang_fun, obj_sta, size_popu, ite, dim, w, c1, c2, LB, UB, v_max)
P1.process()
