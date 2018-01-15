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




# Creation of low-level functions used for mathematical purpose to obtain the formulas of the curves to study (position, velocity, control signal,...)

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
            return(A_half.dot(A_half)) # dot product of A_half by itself
        else:
            A_half = power(A, (n-1)//2)
            return(A.dot(A_half.dot(A_half))) # dot product of A by A_half by A_half

        
### à quoi servent les matrices ci0 et ci1 ?
# 
def A_pow(A, T, R):
    """
    compute the array of A^i of shape (T+R+1, 2, 2)
    """
    # create 101 matrices of 2x2 dimensions
    A_pow_array = np.zeros((T+R+1, 2, 2))
    for i in np.arange(T+R+1): # from 0 to 100 with T=50 and R=50
        A_pow_array[i] = power(A, i)
        #A_pow_array = power(A, i) #   
    return A_pow_array#, ci0_array, ci1_array, ci_array

    ###A_pow_array = A_pow(A, T, R) 

    
###enlever cette function (on peut appeler pow_fast(n) en faisant 'A_pow(A, B, T, R)[n]')







class BangBang:
    """"""
    def __init__(self,
                 ###control_init = None, 
                 tau = 0.013, # a plant parameter
                 k = 0.0001, ### .00001 # timestep of the algorithm (s)
                 dt = 0.001, # timestep of the algorithm (s)
                 t_T = .05, # movement period (s)
                 t_R = .05, # post-movement period (s)
                 x0 = 0, #np.array([10, 0]), #values at time t=0 of both position
                 xT = 10, #np.array([10, 0]), #values at time T of both position
                 #v = 0., # velocity of the target at time t
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


    # theoretical position (x) of the reach movement of the effector (eye) regarding the bang-bang model
    def theory_x(self):
        x = np.concatenate((self.Umax()*self.tau*(self.time()[0:self.rhoT()]-self.tau*(1-np.exp(-1/self.tau*self.time()[0:self.rhoT()]))),
                            self.xT+self.Umax()*self.tau*(self.t_T-self.time()[self.rhoT():self.T()]+self.tau*(1-np.exp(1/self.tau*(self.t_T-self.time()[self.rhoT():self.T()])))),
                            self.xT*np.ones(self.R()+1)))
        return x
    
    
    # theoretical velocity (v) of the reach movement of the effector (eye) regarding the bang-bang model
    def theory_v(self):
        v = np.concatenate((self.Umax()*self.tau*(1-np.exp(-1/self.tau*self.time()[0:self.rhoT()])),
                            -self.Umax()*self.tau*(1-np.exp(1/self.tau*(self.t_T-self.time()[self.rhoT():self.T()]))),
                            np.zeros(self.R()+1)))
        return v


    # prediction of the position (x) during    
#    def linear_pred(u):
#        x_pred = self.expectation(u, t) 






class MinimumVarianceControl:
    """"""
    def __init__(self,
                 ###control_init = None,
                 n = 2, # number of coordinates for the arrays A and B
                 n_rho = 100, # number of rho's values
                 tau = 0.013, # a plant parameter
                 k = 0.00001, # coefficient of intensity of the multiplicative noise
                             # kind of accuracy parameter: the smaller k is, the more accurate is the pointing
                 dt = 0.001, # timestep of the algorithm (s)
                 t_T = .05, # movement period (s)
                 t_R = .05, # post-movement period (s)
                 x0 = np.array([5, 20]), #values at time t=0 of both position and velocity ###
                 xT = 10, #np.array([10, 0]), #values at time T of the position
                 #v = 0., # velocity of the target at time t
                 v0 = 20, # initial velocity of the target
                 n_iter = 2000,
                 eta = 5000,
                 record_each = 100):
        
        ###self.control_init = control_init
        self.n = n
        self.n_rho = n_rho
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


    ###  step of rhos
    def rho(self):
        rho = np.linspace(0.5, 0.999, self.n_rho) # rho's tested values
        return rho


    #paramètres de position de l'oeil au temps t0
    def Uminus(self):
        #Uminus = 1/tau*((xT+v0*(t_T+tau))*(1-np.exp(-rho*t_T/tau))-v0*rho*t_T*np.exp((1-rho)*t_T/tau)) \
        #    / (t_T-(1-rho)*t_T*np.exp(-rho*t_T/tau)-rho*t_T*np.exp((1-rho)*t_T/tau))
        Uminus = 1/self.tau*((self.xT-self.x0[0]+self.v0*(self.t_T+self.tau)-0*self.x0[1]*(self.rho()*self.t_T+self.tau))*(1-np.exp(-self.rho()*self.t_T/self.tau))-self.rho()*self.t_T*(self.v0*np.exp((1-self.rho())*self.t_T/self.tau)-0*self.x0[1]))/(self.t_T-(1-self.rho())*self.t_T*np.exp(-self.rho()*self.t_T/self.tau)-self.rho()*self.t_T*np.exp((1-self.rho())*self.t_T/self.tau))
        return Uminus


    def Uplus(self):
        Uplus = (1-np.exp((1-self.rho())*self.t_T/self.tau))/(1-np.exp(-self.rho()*self.t_T/self.tau))*self.Uminus()+1/self.tau*(self.v0*np.exp((1-self.rho())*self.t_T/self.tau)-0*self.x0[1])/(1-np.exp(-self.rho()*self.t_T/self.tau)) ### -0*self.x0[1]
        return Uplus


    def U(self):
        u = np.zeros((self.n_rho-1, self.T()+self.R()+1))
        for i in np.arange(self.n_rho-1):
            rhoT = int(self.T()*self.rho()[i])
            u[i, :] = np.concatenate((self.Uplus()[i]*np.ones(rhoT),
                                     self.Uminus()[i]*np.ones(self.T()-rhoT),
                                     1/self.tau*self.v0*np.ones(self.R()+1)))
        return u




    def array_A(self):
        ###A = np.array([[1., self.dt], [0., 1-self.dt/self.tau]])
        A = np.zeros((self.n, self.n))
        for i in np.arange(self.n): ###np.arange
            for j in np.arange(i, self.n): ###np.arange
                A[i,j] = self.dt**(j-i)/fact(j-i) # calling the fact(n) function from the beginning of this script
        A[self.n-1, self.n-1] = 1-self.dt/self.tau
        return A


    def array_B(self):
        ###B = np.array([0., self.dt])
        B = np.zeros(self.n)
        B[self.n-1] = self.dt
        return B

### à enlever ??
    """
    if (t_T is None):
    t_Tv = (0.02468 + 0.001739*np.abs(xT[0]-x0[0]))/(1-0.001739*np.abs(v))
    t_T =  float(round(Decimal(t_Tv),3)) #.05 # saccade duration
    t_R =  float(round(Decimal(0.15-t_T),3)) # .05 # fixing / pursuit duration
    """
    

    def pow_fast(self, n):
        A_pow_array = A_pow(self.array_A(), self.T(), self.R())
        #return A_pow_array[n,:,:]
        return A_pow_array[n] 
        #return A_pow(A, B, T, R)[n]


    def ci0_array(self):
        ci0 = np.zeros(self.T()+self.R()+1)
        #A_pow_array = A_pow(self.array_A, self.T(), self.R())
        for i in np.arange(self.T()+self.R()+1):
            #ci0[i] = (A_pow[i, :, :].dot(B))[0]
            ci0[i] = (self.pow_fast(i).dot(self.array_B()))[0] # dot product of the 2 matrices A and B
        return ci0


    def ci1_array(self):
        ci1 = np.zeros(self.T()+self.R()+1)
        for i in np.arange(self.T()+self.R()+1):
            #ci1_array[i] = (A_pow[i, :, :].dot(B))[1]
            ci1[i] = (self.pow_fast(i).dot(self.array_B()))[1]
        return ci1


    def ci_array(self):
        ci_array = np.array([self.ci0_array(), self.ci1_array()])
        return ci_array


    def expectation(self, u, t):
        """"""
        #Computation of the expectation of the state vector at time t, given u
        """"""
        if t == 0:
            return self.x0
        else:
            #return pow_fast(t).dot(x0)+(ci_array[:,0:t]*np.flipud(u[0:t])).sum(axis = 1)
            return (self.pow_fast(t).dot(self.x0)+(self.ci_array()[:,0:t]*np.flipud(u[0:t])).sum(axis = 1))  


    def variance(self, u, t):
        """Computation of the variance of the state vector at time t, given u
        """
        return self.k*(np.flipud(self.ci0_array()[0:t]**2)*u[0:t]**2).sum()


    def position(self):
        position = np.zeros((self.n_rho-1, self.T()+self.R()+1))
        for i in np.arange(self.n_rho-1):
            for j in np.arange(self.T()+self.R()+1):
                position[i, j] = self.expectation(self.U()[i,:], j)[0]
        return position
    
    
    def velocity(self):
        velocity = np.zeros((self.n_rho-1, self.T()+self.R()+1))
        for i in np.arange(self.n_rho-1):
            for j in np.arange(self.T()+self.R()+1): 
                velocity[i, j] = self.expectation(self.U()[i,:], j)[1]
        return velocity


    def variancev(self):
        variancev = np.zeros((self.n_rho-1, self.T()+self.R()+1))
        for i in range(self.n_rho-1):
            for j in range(self.T()+self.R()+1):
                variancev[i,j] = self.variance(self.U()[i,:], j)
        return variancev




    def control_learning(self):
        """"""
        pass

