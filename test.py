import numpy as np
import pandas as pd
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
    print("True max: %s, found max: %s" %(f_1dim(7),opti.optimum))
    print("True argmax: %s, found argmax %s"  %(7,opti.optimal_x))






j = np.random.randint(0,29,2)

def f_30dim(x):
    return np.exp(-(x[j[0]]+0.5)**2-(x[j[1]]-0.3)**2 + 0.4*np.cos(10*x[j[0]]+5))

def test2():
    rem_dim2 = []
    rem_dim3 = []

    for i in range(3):
        rem_dim2.append(REMBO(f_30dim, 30, 2, 120,log_info=False))
        rem_dim3.append(REMBO(f_30dim, 30, 3, 120,log_info=False))
    
    dic = {"dim2":[],"dim3":[]}
    for i in range(3):
        dic["dim2"].append(rem_dim2[i][0].optimum)
        dic["dim3"].append(rem_dim3[i][0].optimum)
    
    table = pd.DataFrame(dic)
    table.columns = ["d=2, n=120", "d=3, n=120"]
    epsilon = np.array([np.log(2)/np.sqrt(2), np.log(3)/np.sqrt(3)])
    table = table.append({"d=2, n=120": np.max(dic["dim2"]),"d=3, n=120": np.max(dic["dim3"])}, ignore_index=True)
    table = table.append({"d=2, n=120": epsilon[0],"d=3, n=120": epsilon[1]}, ignore_index=True)
    table = table.append({"d=2, n=120": epsilon[0]**3,"d=3, n=120": epsilon[1]**3}, ignore_index=True)
    table.index = [1,2,3,"max","e", "e^3"]
    print("=====================================================")
    print("RESULTS after 3 independent runs (true max is 1.4918):")
    print()
    print(table.round(4))
    print("e is probability when real maximum is not included in reduced space")
    print("e^3 is joint probability of the same event for 3 independet runs")
    print("(thats why we need more runs)")
    
if __name__ == "__main__":
    print("Testing BO for 1 dimensional function.")
    test1()
    print()
    print("Testing REMBO for 30 dimensional function with 2 effective dimensions...")
    print("Objective function and effective dimensions are 'unknown'.")
    print("This test can last few minutes (because of 3 independent runs for 2 and 3 dimensions)")
    choice = input("Type 'y' if you want to continue: ")
    if choice.lower() == "y":
        test2()
    else:
        print("You can run REMBO example by typing test2().")
    