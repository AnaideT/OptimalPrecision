import numpy as np

def ControlInitFunction(n):
	return np.concatenate((40/n*np.ones(int(0.86*n)+1),-2*40/n*np.ones(n-1-int(0.86*n)),np.zeros(n+1)))