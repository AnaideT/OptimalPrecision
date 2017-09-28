# -*- coding: utf-8 -*

import numpy as np
from decimal import Decimal


# Harris and Wolpert 1998 article's review
class MinimumVarianceControl:
	"""Minimum variance control
	In presence of signal-dependent noise, the shape of the trajectory is selected
	to minimize the variance of the final eye or arm position
	Parameters
	----------
	control_init : array of shape ((t_T+t_R)/dt+1),
		initial value of the control
	tau : float,
		a plant parameter (s)
	k : float,
		coefficient of intensity of the multiplicative noise
	dt : float,
		timestep of the algorithm (s)
	t_T : float,
		movement period (s)
	t_R  : float,
		post-movement period (s)
	x0 : array of shape (2,1)
		initial values of both position and velocity
	xT : array of shape (2,1)
		values at time T of both position and velocity
	v : float,
		velocity of the target (deg/s)
	n_iter : int,
		total number of iterations to perform
	eta : float,
		step of the gradient descent
	record_each :
		if set to 0, it does nothing. Else it records every record_each step the
		statistics during the learning phase (variance and kurtosis of coefficients).

	Attributes
	----------
	control : array, [(t_T+t_R)/dt+1]
		control extracted from the data
	Notes
	-----
	**References:**
	Harris & Wolpert (1998).
	Signal-dependent noise determines motor planning.
	Nature, 394, 20 August. (https://homes.cs.washington.edu/~todorov/courses/amath533/HarrisWolpert98.pdf)

	"""

	def __init__(self, control_init = None, tau = 0.013, k=0.0001,
				 dt=0.001, t_T=None, t_R=None, x0=np.array([0,0]), xT=np.array([10,0]), v=0.,
				 n_iter=2000, eta=5000,
				 record_each=100):
		self.control_init = control_init
		self.tau = tau
		self.k = k
		self.dt = dt
		self.t_T = t_T
		self.t_R = t_R
		self.x0 = x0
		self.xT = xT
		self.v = v
		self.n_iter = n_iter
		self.eta = eta
		self.record_each = record_each

	def fit(self):
		"""Fit the model from self.
		Parameters
		----------

		Returns
		-------
		self : object
	        Returns the instance itself.
		"""

		return_fn = control_learning(self.control_init, self.tau, self.k,
	    							 self.dt, self.t_T, self.t_R, self.x0, self.xT, self.v,
	    							 self.n_iter, self.eta,
									 self.record_each)

		if self.record_each==0:
			self.control = return_fn
		else:
			self.control, self.record = return_fn



def control_learning(control_init=None, tau = 0.013, k=0.0001,
					 dt=0.001, t_T=None, t_R=None, x0=np.array([0,0]), xT=np.array([10,0]), v=0.,
					 n_iter=2000, eta=5000,
					 record_each=100):
	"""
	Solves the optimization problem::
		u^* = argmin_{(u_0,u_1,...,u_{T+R})} C(u)
					where C is a cost function (C = C_1 + C_2
					where C_1 is the bias term and C_2 the variance term)
	where U is the control signal during the period [0, T+R]. This is
	accomplished by  iterating a gradient descent algorithm.
	u_new = u_old - eta*grad(C(u))
	Parameters
	----------
	control_init : array of shape ((t_T+t_R)/dt+1),
		initial value of the control
	tau : float
		a plant parameter (s)
	k : float,
		coefficient of intensity of the multiplicative noise
	dt : float,
		timestep of the algorithm (s)
	t_T : float,
		movement period (s)
	t_R  : float,
		post-movement period (s)
	x0 : array of shape (2, 1)
		initial values of both position and velocity
	xT : array of shape (2, 1)
		values at time T of both position and velocity
	v : float,
	velocity of the target (deg/s)
	n_iter : int,
		total number of iterations to perform
	eta : float,
		step of the gradient descent
	record_each :
		if set to 0, it does nothing. Else it records every record_each step the
		statistics during the learning phase (variance and kurtosis of coefficients).

	Returns
	-------
	control : array of shape ((t_T+t_R)/dt+1),
		the solutions to the control learning problem
	"""


	A = np.array([[1., dt],[0., 1-dt/tau]])
	B = np.array([0., dt])

	if (t_T is None):
		t_Tv = (0.02468 + 0.001739*np.abs(xT[0]-x0[0]))/(1-0.001739*np.abs(v))
		t_T =  float(round(Decimal(t_Tv),3)) #.05 # saccade duration
		t_R =  float(round(Decimal(0.15-t_T),3)) # .05 # fixing / pursuit duration


	T = int(t_T/dt)
	R = int(t_R/dt)
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000
	mult = 0.01




	def SymmetricalBangbang(tau, x0, xT, dt, t_T, t_R, v0):
		"""
		Returns the symmetrical (U+ = - U-) bangbang solution
		"""
		T = int(t_T/dt)
		R = int(t_R/dt)
		time = np.linspace(0, t_T+t_R, R+T+1)

		if v0==0.:
			vrho = np.linspace(0.5,1,1000001)
			y = (xT-x0[0]-x0[1]*vrho*t_T)*(2-np.exp(-vrho*t_T/tau)-np.exp((1-vrho)*t_T/tau))+x0[1]*((2*vrho-1)*t_T-tau*(2-np.exp(-vrho*t_T/tau)-np.exp((1-vrho)*t_T/tau)))

			rho = vrho[np.argmin(np.abs(y))]
			rhoT = int(np.round(T*rho))

			Umax = 1/tau*(xT-x0[0]-x0[1]*rho*t_T)/((2*rho-1)*t_T-tau*(2-np.exp(-rho*t_T/tau)-np.exp((1-rho)*t_T/tau)))

			xx = np.concatenate((Umax*tau*(time[0:rhoT]-tau*(1-np.exp(-time[0:rhoT]/tau)))+x0[1]*time[0:rhoT]+x0[0],
								xT+Umax*tau*(t_T-time[rhoT:T]+tau*(1-np.exp((t_T-time[rhoT:T])/tau))),
								xT*np.ones(R+1)))

			vv = np.concatenate((Umax*tau*(1-np.exp(-time[0:rhoT]/tau))+x0[1],
								-Umax*tau*(1-np.exp((t_T-time[rhoT:T])/tau)),
								np.zeros(R+1)))

			uu = np.concatenate((Umax*np.ones(rhoT),
								-Umax*np.ones(T-rhoT),
								np.zeros(R+1)))

			return uu, xx, vv

		else:
			vrho = np.linspace(0.5,1,1000001)
			y = (xT+v0*t_T-x0[0]-x0[1]*vrho*t_T+v0*tau*(1-np.exp((1-vrho)*t_T/tau)))*(2-np.exp(-vrho*t_T/tau)-np.exp((1-vrho)*t_T/tau))-(v0*np.exp((1-vrho)*t_T/tau)-x0[1])*((2*vrho-1)*t_T-tau*(2-np.exp(-vrho*t_T/tau)-np.exp((1-vrho)*t_T/tau)))

			rho_pursuit = vrho[np.argmin(np.abs(y))]
			rhoT_pursuit = int(np.round(T*rho_pursuit))

			Umax_pursuit = 1/tau*(v0*np.exp((1-rho_pursuit)*t_T/tau)-x0[1])/(2-np.exp((1-rho_pursuit)*t_T/tau)-np.exp(-rho_pursuit*t_T/tau))

			x_pursuit = np.concatenate((Umax_pursuit*tau*(time[0:rhoT_pursuit]-tau*(1-np.exp(-time[0:rhoT_pursuit]/tau)))+x0[1]*time[0:rhoT_pursuit]+x0[0],
										xT+v0*t_T+Umax_pursuit*tau*(t_T-time[rhoT_pursuit:T]+tau*(1-np.exp((t_T-time[rhoT_pursuit:T])/tau)))+tau*v0*(1-np.exp((t_T-time[rhoT_pursuit:T])/tau)),
										xT+v0*t_T+v0*(time[T:(T+R+1)]-t_T)))

			v_pursuit = np.concatenate((Umax_pursuit*tau*(1-np.exp(-time[0:rhoT_pursuit]/tau))+x0[1],
										-Umax_pursuit*tau*(1-np.exp((t_T-time[rhoT_pursuit:T])/tau))+v0*np.exp((t_T-time[rhoT_pursuit:T])/tau),
										v0*np.ones(R+1)))

			u_pursuit = np.concatenate((Umax_pursuit*np.ones(rhoT_pursuit),
										-Umax_pursuit*np.ones(T-rhoT_pursuit),
										1/tau*v0*np.ones(R+1)))

			return u_pursuit, x_pursuit, v_pursuit



	def AsymmetricalBangbang(tau, x0, xT, dt2, t_T, t_R, v0):
		"""
		Returns the asymmetrical (U+ =/= - U-) bangbang solution
		"""
		T2 = int(t_T/dt2)
		R2 = int(t_R/dt2)
		time = np.linspace(0, t_T+t_R, R2+T2+1)

		A2 = np.array([[1, dt2], [0, 1-dt2/tau]])
		B2 = np.array([0., dt2])

		def A2_pow(A):
			"""
			compute the array of A^i of shape (T+R+1, 2, 2)
			"""
			A2_pow_array = np.zeros((T2+R2+1, 2, 2))
			for i in np.arange(T2+R2+1):
				A2_pow_array[i, :, :] = power(A, i)
			return A2_pow_array

		A2_pow_array = A2_pow(A2)

		def pow_fast2(n):
			return A2_pow_array[n,:,:]

		ci0_array2 = np.zeros(T2+R2+1)
		ci1_array2 = np.zeros(T2+R2+1)

		for i in np.arange(T2+R2+1):
			ci0_array2[i] = (A2_pow_array[i, :, :].dot(B2))[0]
			ci1_array2[i] = (A2_pow_array[i, :, :].dot(B2))[1]

		ci_array2 = np.array([ci0_array2, ci1_array2])


		def expectation2(u, t):
			"""
			compute the expectation at time t given the control signal u

			array of shape (2, 1)
			"""
			if t == 0:
				return x0
			else:
				return pow_fast2(t).dot(x0)+(ci_array2[:,0:t]*np.flipud(u[0:t])).sum(axis = 1)

		def variance2(u, t):
			"""
			compute the variance at time t given the control signal u
			"""
			return k*(np.flipud(ci0_array2[0:t]**2)*u[0:t]**2).sum()


		n = 100 # number of rho's values
		rho = np.linspace(0.5,0.999,n) # rho's tested values


		Umoins = 1/tau*((xT-x0[0]+v0*(t_T+tau)-x0[1]*(rho*t_T+tau))*(1-np.exp(-rho*t_T/tau))-rho*t_T*(v0*np.exp((1-rho)*t_T/tau)-x0[1]))/(t_T-(1-rho)*t_T*np.exp(-rho*t_T/tau)-rho*t_T*np.exp((1-rho)*t_T/tau))

		Uplus = (1-np.exp((1-rho)*t_T/tau))/(1-np.exp(-rho*t_T/tau))*Umoins+1/tau*(v0*np.exp((1-rho)*t_T/tau)-x0[1])/(1-np.exp(-rho*t_T/tau))

		u = np.zeros((n-2, T2+R2+1))

		for i in np.arange(n-2):
			rhoT = np.round(T2*rho[i])
			u[i,:] = np.concatenate((Uplus[i]*np.ones(rhoT), Umoins[i]*np.ones(T2-rhoT), 1/tau*v0*np.ones(R2+1)))


		position = np.zeros((n-2, T2+R2+1))
		velocity = np.zeros((n-2, T2+R2+1))

		for i in np.arange(n-2):
			for j in np.arange(T2+R2+1):
				mean = expectation2(u[i,:], j)
				position[i,j] = mean[0]
				velocity[i,j] = mean[1]


		variancev = np.zeros((n-2, T2+R2+1))

		for i in np.arange(n-2):
			for j in np.arange(T2+R2+1):
				variancev[i,j] = variance2(u[i,:], j)


		somme = np.zeros(n-2)
		for i in np.arange(n-2):
			for j in T2+np.arange(R2+1):
				somme[i] += variancev[i,j]

		ind_best = np.argmin(somme)
		rho = rho[ind_best]

		Umoins = 1/tau*((xT-x0[0]+v0*(t_T+tau)-x0[1]*(rho*t_T+tau))*(1-np.exp(-rho*t_T/tau))-rho*t_T*(v0*np.exp((1-rho)*t_T/tau)-x0[1]))/(t_T-(1-rho)*t_T*np.exp(-rho*t_T/tau)-rho*t_T*np.exp((1-rho)*t_T/tau))

		Uplus = (1-np.exp((1-rho)*t_T/tau))/(1-np.exp(-rho*t_T/tau))*Umoins+1/tau*(v0*np.exp((1-rho)*t_T/tau)-x0[1])/(1-np.exp(-rho*t_T/tau))

		rhoT = np.round(T2*rho)

		position[ind_best,0:rhoT] = tau*Uplus*(time[0:rhoT]-tau*(1-np.exp(-time[0:rhoT]/tau)))+x0[1]*time[0:rhoT]+x0[0]
		position[ind_best,rhoT:T2] = xT+v0*t_T+v0*tau*(1-np.exp((t_T-time[rhoT:T2])/tau))-tau*Umoins*(t_T-time[rhoT:T2]+tau*(1-np.exp((t_T-time[rhoT:T2])/tau)))
		position[ind_best,T2:(T2+R2+1)] = xT+v0*t_T+v0*(time[T2:(T2+R2+1)]-t_T)


		velocity[ind_best,0:rhoT] = tau*Uplus*(1-np.exp(-time[0:rhoT]/tau))+x0[1]
		velocity[ind_best,rhoT:T2] = v0*np.exp((t_T-time[rhoT:T2])/tau)+tau*Umoins*(1-np.exp((t_T-time[rhoT:T2])/tau))
		velocity[ind_best,T2:(T2+R2+1)] = v0*np.ones(R2+1)


		return u[ind_best, :], position[ind_best, :], velocity[ind_best, :], variancev[ind_best, :]*dt/dt2


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
				return(A_half.dot(A_half))
			else:
				A_half = power(A, (n-1)//2)
				return(A.dot(A_half.dot(A_half)))


	def A_pow(A):
		"""
		compute the array of A^i of shape (T+R+1, 2, 2)
		"""
		A_pow_array = np.zeros((T+R+1, 2, 2))
		for i in np.arange(T+R+1):
			A_pow_array[i, :, :] = power(A, i)
		return A_pow_array

	A_pow_array = A_pow(A)

	ci0_array = np.zeros(T+R+1)
	ci1_array = np.zeros(T+R+1)

	for i in np.arange(T+R+1):
		ci0_array[i] = (A_pow_array[i, :, :].dot(B))[0]
		ci1_array[i] = (A_pow_array[i, :, :].dot(B))[1]

	ci_array = np.array([ci0_array,ci1_array])


	def pow_fast(n):
		return A_pow_array[n,:,:]

	def expectation(u, t):
		"""
		compute the expectation at time t given the control signal u

		array of shape (2, 1)
		"""
		if t == 0:
			return x0
		else:
			return pow_fast(t).dot(x0)+(ci_array[:,0:t]*np.flipud(u[0:t])).sum(axis = 1)


	def vexpectation(u):
		"""
		vectorized version of expectation
		"""
		exp = np.zeros((T+R+1, 2))
		for i in np.arange(T+R+1):
			exp[i, :] = expectation(u,i)
		return exp


	def variance(u, t):
		"""
		compute the variance at time t given the control signal u
		"""
		return k*(np.flipud(ci0_array[0:t]**2)*u[0:t]**2).sum()


	def vvariance(u):
		"""
		vectorized version of variance
		"""
		var = np.zeros(T+R+1)
		for i in np.arange(T+R+1):
			var[i] = variance(u,i)
		return var


	def bias(u, t):
		"""
		compute the bias at time t given the control signal u
		"""
		return (((expectation(u, t)-(xT+np.array([v*t*dt,v])))**2)*np.array([1,mult])).sum()


	def cost(u):
		"""
		compute the post-movement cost given the control signal u
		"""
		def var1d(t):
			return(variance(u,t))
		var_vec = np.vectorize(var1d)
		def bias1d(t):
			return((bias(u,t)**2).sum())
		bias_vec = np.vectorize(bias1d)

		return var_vec(T+1+np.arange(R)).sum() + bias_vec(T+np.arange(R+1)).sum()


	def cost_deriv(u, i):
		"""
		Derivative of the cost function with respect to u_i
		"""
		if i < T:
			return (2*np.transpose(ci_array[:,(T-i-1):(T+R-i)])*np.array([((expectation(u,t)-xT-np.array([v*t*dt,v]))*np.array([1,mult])).tolist() for t in (T+np.arange(R+1))])).sum() + 2*k*u[i]*(ci0_array[(T+1-i-1):(T+R-i)]**2).sum()
		else:
			return (2*np.transpose(ci_array[:,0:(T+R-i)])*np.array([((expectation(u,t)-xT-np.array([v*t*dt,v]))*np.array([1,mult])).tolist() for t in (i+1+np.arange(R+T-i))])).sum() + 2*k*u[i]*(ci0_array[0:(T+R-i)]**2).sum()


	def vcost_deriv(u):
		"""
		vectorized version of cost_deriv
		"""
		deriv_cost = np.zeros(T+R+1)
		for i in np.arange(T+R+1-1):
			deriv_cost[i] = cost_deriv(u,i)
		return deriv_cost



	# First : test if the file already exists which mean that the function has already been used for this parameters
	import pickle
	import os
	from os.path import isfile

	# if the file exists we use it
	if os.path.isfile('../2017_OptimalPrecision/DataRecording/'+'dt_'+str(dt)+'/'+'HW_tau='+str(tau)+'_dt='+str(dt)+'_tT='+str(t_T)+'_tR='+str(t_R)+'_k='+str(k)+'_niter='+str(n_iter)+'_xT='+str(xT[0])+'_v='+str(v)+'.pkl'):
		import pandas as pd
		record = pd.read_pickle('../2017_OptimalPrecision/DataRecording/'+'dt_'+str(dt)+'/'+'HW_tau='+str(tau)+'_dt='+str(dt)+'_tT='+str(t_T)+'_tR='+str(t_R)+'_k='+str(k)+'_niter='+str(n_iter)+'_xT='+str(xT[0])+'_v='+str(v)+'.pkl')
		control = record.signal[n_iter]

		# We compute the sym bangbang to have nice plots (that's why the time step is reduced)
		control_bang1, pos_bang1, vel_bang1 = SymmetricalBangbang(tau, x0, xT[0], 0.0000001, t_T, t_R, v)

		# useful to compute the variance
		control_bang1bis, pos_bang1bis, vel_bang1bis = SymmetricalBangbang(tau, x0, xT[0], dt, t_T, t_R, v)

		var_bang1 = vvariance(control_bang1bis)


		bang_data = pd.DataFrame([{'signal':control_bang1,
								   'position':pos_bang1,
								   'velocity':vel_bang1,
								   'variance':var_bang1}],
								   index=[0])

		# We compute the asym bangbang to have nice plots (that's why the time step is reduced)
		control_bang2, pos_bang2, vel_bang2, var_bang2 = AsymmetricalBangbang(tau, x0, xT[0], 0.00001, t_T, t_R, v)

		bang_data2 = pd.DataFrame([{'signal':control_bang2,
								   'position':pos_bang2,
								   'velocity':vel_bang2,
								   'variance':var_bang2}],
								   index=[1])

		bang_data = pd.concat([bang_data, bang_data2])

		return control, record, bang_data, t_T, t_R

	# if not, we start from zero
	else:

		if record_each>0:
			import pandas as pd
			record = pd.DataFrame()

		# if the initial control is given by the inputs, we use it
		if not (control_init is None):
			control = control_init.copy()

		# if not, we start from a constant signal (faster than starting from a bangbang)
		else:
			control = np.ones(T+R+1)*v/tau

		# We compute the sym bangbang to have nice plots (that's why the time step is reduced)
		control_bang1, pos_bang1, vel_bang1 = SymmetricalBangbang(tau, x0, xT[0], 0.0000001, t_T, t_R, v)

		# useful to compute the variance
		control_bang1bis, pos_bang1bis, vel_bang1bis = SymmetricalBangbang(tau, x0, xT[0], dt, t_T, t_R, v)

		var_bang1 = vvariance(control_bang1bis)


		bang_data = pd.DataFrame([{'signal':control_bang1,
								   'position':pos_bang1,
								   'velocity':vel_bang1,
								   'variance':var_bang1}],
								   index=[0])

		# We compute the asym bangbang to have nice plots (that's why the time step is reduced)
		control_bang2, pos_bang2, vel_bang2, var_bang2 = AsymmetricalBangbang(tau, x0, xT[0], 0.00001, t_T, t_R, v)

		bang_data2 = pd.DataFrame([{'signal':control_bang2,
								   'position':pos_bang2,
								   'velocity':vel_bang2,
								   'variance':var_bang2}],
								   index=[1])

		bang_data = pd.concat([bang_data, bang_data2])



		control[T+R] = 1/tau*v


		cost_iter = np.zeros(0)
		posT_iter = np.zeros(0)

		for i_iter in np.arange(n_iter):

			# Gradient descent
			control_old = control.copy()
			control[0:T+R] = control_old[0:T+R] - eta*np.array([cost_deriv(control_old, i) for i in np.arange(T+R)])
			cost_iter = np.concatenate((cost_iter, np.array([cost(control_old)])))
			posT_iter = np.concatenate((posT_iter, np.array([expectation(control_old, T)[0]])))

			if record_each>0:
				if i_iter % int(record_each) == 0:
					control_rec = control_old.copy()
					pos_rec = vexpectation(control_old)[:, 0]
					vel_rec = vexpectation(control_old)[:, 1]
					var_rec = vvariance(control_old)
					cost_rec = cost_iter.copy()
					cost_iter = np.zeros(0)
					posT_rec = posT_iter.copy()
					posT_iter = np.zeros(0)


					record_one = pd.DataFrame([{'signal':control_rec,
												'position':pos_rec,
												'velocity':vel_rec,
												'variance':var_rec,
												'cost':cost_rec,
												'positionT':posT_rec}],
												index=[i_iter])
					record = pd.concat([record, record_one])

		record_last = pd.DataFrame([{'signal':control,
									 'position':vexpectation(control)[:, 0],
									 'velocity':vexpectation(control)[:, 1],
									 'variance':vvariance(control),
									 'cost':cost_iter,
									 'positionT':posT_iter}],
									 index=[n_iter])

		record = pd.concat([record, record_last])

# '/home/baptiste/Documents/2017_OptimalPrecision' = '.'
# import os
# fname = os.path.join('DataRecording', 'machin', 'truc') 
		record.to_pickle('../2017_OptimalPrecision/DataRecording/'+'dt_'+str(dt)+'/'+'HW_tau='+str(tau)+'_dt='+str(dt)+'_tT='+str(t_T)+'_tR='+str(t_R)+'_k='+str(k)+'_niter='+str(n_iter)+'_xT='+str(xT[0])+'_v='+str(v)+'.pkl')

		if record_each==0:
			return control, bang_data, t_T, t_R
		else:
			return control, record, bang_data, t_T, t_R