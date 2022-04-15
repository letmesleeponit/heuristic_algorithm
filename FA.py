import numpy as np
import matplotlib.pyplot as plt
import time

### parameter setting ###
D=50  #dimension of problem
Lb=[-5] * D #lower bound variables
Ub=[5] * D #upper bound
N=200  #population size
alpha=1 #randomness strength 0-1 (步長因子)
beta=1 #attractiveness constant (螢火蟲最大吸引度，通常當成吸引力常數)
gamma=1 #absorption coefficient (光吸收係數)
theta=1 #randomness reduction factor (矢向量)
iter_max=100 #max iterations
obj_sta = 'min'

### objective function ###
def Rastrigin_fun(val, dim):
    obj = 0
    for dim_ite in range(dim): 
        obj += (val[:, dim_ite] ** 2 - 10 * np.cos(2 * np.pi * val[:, dim_ite]))
    obj += dim * 10
    return obj

def Styblinski_Tang_fun(val, dim):
    obj = 0
    for dim_ite in range(dim):
        obj += val[:, dim_ite] ** 4 - 16 * val[:, dim_ite] ** 2 + 5 * val[:, dim_ite]
    obj /= 2
    return obj

### main program ###
sta_time = time.time()
pop = Lb + np.random.uniform(size=(N,D)) * np.subtract(Ub, Lb)
fit = Styblinski_Tang_fun(pop, D)
record = [] # record for plot
record_pos = []

for iter in range(iter_max): # main iteration loop
    for i in range(N):       # for all the fireflies
        for j in range(N):   # for all the fireflies
            # if j more attractive, then we should update
            if obj_sta == 'max':
                con = fit[i] < fit[j]
            else:
                con = fit[i] > fit[j]
            if con: #change > or <
                steps = alpha * (np.random.uniform(size=D) - 0.5) * np.abs(np.subtract(Ub, Lb))
                r = np.linalg.norm(pop[i] - pop[j], ord=2, axis=0)
                beta_attr = beta * np.exp(-gamma * r)
                #Find Xnew using Xi and Xj
                X_new = pop[i] + beta_attr * (pop[j] - pop[i]) + steps
                #check Xnew is within bounds
                for dim in range(D):
                    if X_new[dim] > Ub[dim]:
                        X_new[dim] = Ub[dim]
                    elif X_new[dim] < Lb[dim]:
                        X_new[dim] = Ub[dim]
                #get the new function value for Xnew
                fnew = Styblinski_Tang_fun(np.expand_dims(X_new, axis=0), D)
                
                #if fnew is better than fx[i]
                if (obj_sta == 'max') and (fnew > fit[i]):
                    fit[i] = fnew
                    pop[i] = X_new
                    
                elif (obj_sta == 'min') and (fnew < fit[i]):
                    fit[i] = fnew
                    pop[i] = X_new
    record.append(np.min(fit)) if obj_sta == 'min' else record.append(np.max(fit))
    record_pos.append(pop[np.argmin(fit)]) if obj_sta == 'min' else record_pos.append(pop[np.argmax(fit)])

# visualize the result
end_time = time.time()
print(record_pos[-1])
print(Styblinski_Tang_fun(np.expand_dims(record_pos[-1], axis=0), D))
print(f'Total consumse time is {end_time - sta_time: .2f}')

plt.plot(record)
plt.title('The convergence histories of FA')
plt.xlabel('The iteration of moving')
plt.ylabel('The best fitness value in these particles')
plt.show()