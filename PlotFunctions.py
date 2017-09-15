import matplotlib.pyplot as plt
import numpy as np


def plot_signal(record_signal, record_each, n_iter, t_T, t_R, dt):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(np.round(t_T/dt))
	R = int(np.round(t_R/dt))
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	col = np.zeros((N_show,3))
	for i in np.arange(N_show):
		col[i] = (N_show-i)/(2*N_show),0,i/(2*N_show)

	for i in np.arange(N_show):
		if i == 0:
			p = len(record_signal[0])
			ax.step(np.linspace(0, (t_T+t_R)*1000, p), record_signal[0], '-', where = 'post', color = col[i])
		else:
			ax.plot(time_ms, record_signal[i*record_each], '-', color = col[i])
	ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
	ax.set_title(r'Control signal $u$', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel(r'$u$', fontsize=12)


def plot_position(record_position, record_each, n_iter, t_T, t_R, dt):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(np.round(t_T/dt))
	R = int(np.round(t_R/dt))
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000


	col = np.zeros((N_show,3))
	for i in np.arange(N_show):
		col[i] = (N_show-i)/(2*N_show),0,i/(2*N_show)


	for i in np.arange(N_show):
		if i == 0:
			p = len(record_position[0])
			ax.plot(np.linspace(0, (t_T+t_R)*1000, p), record_position[0], '-', color = col[i])
		else:
			ax.plot(time_ms, record_position[i*record_each], '-', color = col[i])

	ax.set_title(r'Position $x_t$', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel('Angle (deg)', fontsize=12)


def plot_velocity(record_velocity, record_each, n_iter, t_T, t_R, dt):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(np.round(t_T/dt))
	R = int(np.round(t_R/dt))
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000


	col = np.zeros((N_show,3))
	for i in np.arange(N_show):
		col[i] = (N_show-i)/(2*N_show),0,i/(2*N_show)


	for i in np.arange(N_show):
		if i == 0:
			p = len(record_velocity[0])
			ax.plot(np.linspace(0, (t_T+t_R)*1000, p), record_velocity[0], '-', color = col[i])
		else:
			ax.plot(time_ms, record_velocity[i*record_each], '-', color = col[i])

	ax.set_title('Velocity', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel(r'Velocity $(deg.s^{-1})$ ', fontsize=12)


def plot_variance(record_variance, record_each, n_iter, t_T, t_R, dt):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(np.round(t_T/dt))
	R = int(np.round(t_R/dt))
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000


	col = np.zeros((N_show,3))
	for i in np.arange(N_show):
		col[i] = (N_show-i)/(2*N_show),0,i/(2*N_show)


	for i in np.arange(N_show):
		ax.plot(time_ms, record_variance[i*record_each], '-', color = col[i])

	ax.set_title('Positional variance', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel('Positional variance', fontsize=12)


def plot_cost(record_cost, record_each, n_iter):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	record_cost_conc = np.zeros(0)
	for i in np.arange(N_show)*record_each:
		record_cost_conc = np.concatenate((record_cost_conc, record_cost[i]))

	ax.plot(record_cost_conc, lw=2)
	ax.set_title('Cost', fontsize = 15)
	ax.set_xlabel('Number of iterations', fontsize=12)
	ax.set_ylabel('Cost', fontsize=12)


def plot_posT(record_posT, record_each, n_iter):
	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	record_posT_conc = np.zeros(0)
	for i in np.arange(N_show)*record_each:
		record_posT_conc = np.concatenate((record_posT_conc, record_posT[i]))

	ax.plot(record_posT_conc, lw=2)
	ax.set_title('Position at time T', fontsize = 15)
	ax.set_xlabel('Number of iterations', fontsize=12)
	ax.set_ylabel('Angle (deg)', fontsize=12)


def all_plots(record, bang_data, record_each, n_iter, t_T, t_R, dt, xT, v):
	record_signal = record.signal
	record_position = record.position
	record_velocity = record.velocity
	record_variance = record.variance
	record_cost = record.cost
	record_posT = record.positionT


	plot_signal(record_signal, record_each, n_iter, t_T, t_R, dt)

	T = int(np.round(t_T/dt))
	R = int(np.round(t_R/dt))
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	ax.plot(time_ms, record_signal[n_iter], lw=2, label="Minimum-variance")
	p = len(bang_data.signal[0])
	ax.step(np.linspace(0, (t_T+t_R)*1000, p), bang_data.signal[0], lw=1, where = 'post', label="Sym. bangbang")
	p = len(bang_data.signal[1])
	ax.step(np.linspace(0, (t_T+t_R)*1000, p), bang_data.signal[1], lw=1, where = 'post', label="Asym. bangbang")
	ax.plot([0,1000*(t_T+t_R)], [0,0],'r--')
	ax.set_title(r'Control signal $u$', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel(r'$u$', fontsize=12)
	ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
	ax.legend(prop={'size':10})


	plot_position(record_position, record_each, n_iter, t_T, t_R, dt)

	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	ax.plot(time_ms, record_position[n_iter], lw=2, label="Minimum-variance")
	p = len(bang_data.position[0])
	ax.plot(np.linspace(0, (t_T+t_R)*1000, p), bang_data.position[0], lw=1, label="Sym. bangbang")
	p = len(bang_data.position[1])
	ax.plot(np.linspace(0, (t_T+t_R)*1000, p), bang_data.position[1], lw=1, label="Asym. bangbang")
	ax.plot(time_ms, xT[0]+v*np.linspace(0,t_T+t_R,T+R+1), 'r--')
	ax.set_title(r'Position $x_t$', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel('Angle (deg)', fontsize=12)
	ax.legend(loc='lower right',prop={'size':10})


	plot_velocity(record_velocity, record_each, n_iter, t_T, t_R, dt)

	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	ax.plot(time_ms, record_velocity[n_iter], lw=2, label="Minimum-variance")
	p = len(bang_data.velocity[0])
	ax.plot(np.linspace(0, (t_T+t_R)*1000, p), bang_data.velocity[0], lw=1, label="Sym. bangbang")
	p = len(bang_data.velocity[1])
	ax.plot(np.linspace(0, (t_T+t_R)*1000, p), bang_data.velocity[1], lw=1, label="Asym. bangbang")
	ax.plot(time_ms, v*np.ones(T+R+1),'r--')
	ax.set_title('Velocity', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel(r'Velocity $(deg.s^{-1})$ ', fontsize=12)
	ax.legend(prop={'size':10})


	plot_variance(record_variance, record_each, n_iter, t_T, t_R, dt)

	fig_width = 5
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	ax.plot(time_ms, record_variance[n_iter], lw=1, label="Minimum-variance")
	ax.plot(time_ms, bang_data.variance[0], lw=1, label="Sym. bangbang")
	p = len(bang_data.variance[1])
	ax.plot(np.linspace(0, (t_T+t_R)*1000, p), bang_data.variance[1], lw=1, label="Asym. bangbang")
	ax.set_title('Positional variance', fontsize = 15)
	ax.set_xlabel('Time (ms)', fontsize=12)
	ax.set_ylabel('Positional variance', fontsize=12)
	ax.legend(loc='lower right',prop={'size':10})


	plot_cost(record_cost, record_each, n_iter)


	plot_posT(record_posT, record_each, n_iter)
