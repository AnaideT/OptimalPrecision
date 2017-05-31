import numpy as np

def ControlInitFunction(n, gain=17., rho=0.86):
    seg_1 = np.ones(int(rho*n)+1)
    seg_2 = np.ones(int(n-1-rho*n))
    seg_3 = np.zeros(int(n+1))
    return np.concatenate((gain/n*seg_1, -2*gain/n*seg_2, seg_3))