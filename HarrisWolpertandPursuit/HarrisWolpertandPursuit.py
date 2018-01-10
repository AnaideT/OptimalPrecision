import numpy as np
from decimal import Decimal

# Harris and Wolpert 1998 article's review


### Set all the needed variables to run the functions and classes bellow ??? Or indicate them in the corresponding notebooks (better if we need to change to test the value of the variables)?
#dt = 0.001
#tau = .013
#A = np.array([[1., dt], [0., 1-dt/tau]])
#B = np.array([0., dt])
#t_T = .05
#t_R = .05
#T = int(t_T/dt)    
#R = int(t_R/dt) 




# Creation of low-level functions used for mathematical purpose to obtain the formulas of the curves to study (position, speed, control signal,...)

# factorial :
def fact(n): 
    """The factorial function
    Returns n!, that is n x (n-1) x (n-2) x ... x 1
    Recursive implementation
    """
    if n == 0:
        return 1
    else:
        return(n*fact(n-1))

# power :
def power(A, n): 
    """
    renvoie A puissance n où A est une matrice carrée
    """
    if n == 0:
        return(np.eye(int(np.sqrt(np.size(A)))))
    elif n == 1:
        return A
    else:
        if n % 2 == 0:
            A_half = power(A, n//2)
            return(A_half.dot(A_half)) # dot product of the A_half with itself
        else:
            A_half = power(A, (n-1)//2)
            return(A.dot(A_half.dot(A_half))) # dot product of A by A_half by A_half

        
### à quoi servent les matrices ci0 et ci1 ?
# 
def A_pow(A, B, T, R):
    """
    compute the array of A^i of shape (T+R+1, 2, 2)
    """
    # create 101 matrices of 2x2 dimensions
    A_pow_array = np.zeros((T+R+1, 2, 2))
    # create 2 matrices of 1x101 dimensions
    ci0_array = np.zeros(T+R+1)
    ci1_array = np.zeros(T+R+1)

    for i in np.arange(T+R+1): # from 0 to 100 with T=50 and R=50
        #A_pow_array[i, :, :] = power(A, i)
        A_pow_array[i] = power(A, i) #
        #ci0_array[i] = (A_pow_array[i, :, :].dot(B))[0]
        ci0_array[i] = (A_pow_array[i].dot(B))[0] # dot product of the 2 matrices A and B

        #ci1_array[i] = (A_pow_array[i, :, :].dot(B))[1]
        ci1_array[i] = (A_pow_array[i].dot(B))[1]

        ci_array = np.array([ci0_array,ci1_array])
    
    return A_pow_array#, ci0_array, ci1_array, ci_array

    ###A_pow_array = A_pow(A, B, T, R) 
    ###ci_array = np.array([ci0_array,ci1_array])

    
###enlever cette function (on peut appeler pow_fast(n) en faisant 'A_pow(A, B, T, R)[n]')
#def pow_fast(n):
#    #return A_pow_array[n,:,:]
#    return A_pow_array[n] 
#    #return A_pow(A, B, T, R)[n]


def expectation(u, t):
    """
    compute the expectation at time t given the control signal u
    array of shape (2, 1)
    """
    if t == 0:
        return x0
    else:
        #return pow_fast(t).dot(x0)+(ci_array[:,0:t]*np.flipud(u[0:t])).sum(axis = 1)
        return A_pow(A, B, T, R)[t].dot(x0)+(ci_array[:,0:t]*np.flipud(u[0:t])).sum(axis = 1)


class BangBang:
    def __init__(self,
                 ###control_init = None, 
                 tau = 0.013, # a plant parameter
                 k = 0.0001,
                 dt = 0.001, # timestep of the algorithm (s)
                 t_T = .05, # movement period (s)
                 t_R = .05, # post-movement period (s)
                 x0 = np.array([0, 0]),
                 xT = 10, #np.array([10, 0]), #values at time T of both position and velocity
                 #v = 0.,
                 v0 = 20, # initial velocity of the target
                 n_iter = 2000,
                 eta = 5000,
                 record_each = 100):
                        
        ###self.control_init = control_init
        self.tau = tau
        self.k = k
        self.dt = dt
        self.t_T = t_T
        self.t_R = t_R
        self.x0 = x0
        self.xT = xT
        #self.v = v
        self.v0 = v0
        self.n_iter = n_iter
        self.eta = eta
        self.record_each = record_each
    
    "primary functions"
    # define the number of bends during the movement period (t_T) regarding the step of time (dt)
    def T(self):
        T = int(self.t_T/self.dt)
        return T

    # define the number of bends during the post-movement period (t_R) regarding the step of time (dt)
    def R(self):
        R = int(self.t_R/self.dt)
        return R
    
    # define the time scale (in sec) during the recording : from 0 to 100ms (t_T+t_R) with 101 samples to generate
    def time(self):
        time = np.linspace(0, self.t_T+self.t_R, self.R()+self.T()+1)
        return time
    
    # define the time scale in ms
    def time_ms(self):
        time_ms = self.time()*1000
        return time_ms


    def rho(self):
        rho = self.tau/self.t_T*np.log((1+np.exp(self.t_T/self.tau))/2)
        return rho
    
    
    def rhoT(self):
        rhoT = int(np.round(self.T()*self.rho()))
        return rhoT
    
    
    def Umax(self):
        Umax = (1/self.tau) * self.xT / ((2*self.rho()-1)*self.t_T-self.tau*(2-np.exp(-self.rho()*self.t_T/self.tau)-np.exp((1-self.rho())*self.t_T/self.tau)))
        return Umax
    
    # measure the control signal during symmetrical bang-bang
    def command_u(self):
        u = np.concatenate((self.Umax()*np.ones(self.rhoT()),
                            -self.Umax()*np.ones(self.T()-self.rhoT()),
                            np.zeros(self.R()+1)))
        return u


    def theory_x(self):
        x = np.concatenate((self.Umax()*self.tau*(self.time()[0:self.rhoT()]-self.tau*(1-np.exp(-1/self.tau*self.time()[0:self.rhoT()]))),
                            self.xT+self.Umax()*self.tau*(self.t_T-self.time()[self.rhoT():self.T()]+self.tau*(1-np.exp(1/self.tau*(self.t_T-self.time()[self.rhoT():self.T()])))),
                            self.xT*np.ones(self.R()+1)))
        return x
    
    
    def theory_v(self):
        v = np.concatenate((self.Umax()*self.tau*(1-np.exp(-1/self.tau*self.time()[0:self.rhoT()])),
                            -self.Umax()*self.tau*(1-np.exp(1/self.tau*(self.t_T-self.time()[self.rhoT():self.T()]))),
                            np.zeros(self.R()+1)))
        return v


class MinimumVarianceControl:
    pass
    