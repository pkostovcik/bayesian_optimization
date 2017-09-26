import numpy as np
from bayesian_optimization import *

def f_1dim(x):
    
    if x<4:
        return 4*np.sin(x+np.pi/3) - x + 4
    elif x<7:
        return 4*np.sin(x+np.pi/3) - x + 7
    else:
        return 10 - np.exp(x-7.5) + 3


def test1():
    print("RUNNING BO for 1 dimensional funcion with log info:")
    opti = BO(f_1dim,[(0,10)],100)
    
    print("========== RESULTS ==========")
    print("True max: %s, finded max: %s" %(f_1dim(7),opti.optimum))
    print("True argmax: %s, finded argmax %s"  %(7,opti.optimal_x))






j = np.random.randint(0,29,2)

def f_30dim(x):
    return np.exp(-(x[j[0]]+0.5)**2-(x[j[1]]-0.3)**2 + 0.4*np.cos(10*x[j[0]]+5))



rem_dim2 = []
rem_dim3 = []

for i in range(3):
    rem_dim2.append(REMBO(f_30dim, 30, 2, 120,log_info=False))
    rem_dim3.append(REMBO(f_30dim, 30, 3, 120,log_info=False))