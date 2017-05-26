import matplotlib.pyplot as plt
import numpy as np


def plot_signal(record_signal, record_each, n_iter, t_T, t_R, dt):
	fig_width = 15
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(t_T/dt)
	R = int(t_R/dt)
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	for i in np.arange(N_show):
		ax.plot(time_ms, record_signal[i*record_each], '-')
	ax.set_title(r'Control signal $u$')
	ax.set_xlabel('Time (ms)', fontsize=14)
	ax.set_ylabel(r'$u$', fontsize=14)


def plot_position(record_position, record_each, n_iter, t_T, t_R, dt):
	fig_width = 15
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(t_T/dt)
	R = int(t_R/dt)
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	for i in np.arange(N_show):
		ax.plot(time_ms, record_position[i*record_each], '-')
	ax.set_title(r'Position $x_t$')
	ax.set_xlabel('Time (ms)', fontsize=14)
	ax.set_ylabel('Angle (deg)', fontsize=14)


def plot_velocity(record_velocity, record_each, n_iter, t_T, t_R, dt):
	fig_width = 15
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(t_T/dt)
	R = int(t_R/dt)
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	for i in np.arange(N_show):
		ax.plot(time_ms, record_velocity[i*record_each], '-')
	ax.set_title('Velocity')
	ax.set_xlabel('Time (ms)', fontsize=14)
	ax.set_ylabel(r'Velocity $(deg.s^{-1})$ ', fontsize=14)


def plot_variance(record_variance, record_each, n_iter, t_T, t_R, dt):
	fig_width = 15
	fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.618))
	N_show = n_iter//record_each+1

	T = int(t_T/dt)
	R = int(t_R/dt)
	time = np.linspace(0, t_T+t_R, R+T+1)
	time_ms = time*1000

	for i in np.arange(N_show):
		ax.plot(time_ms, record_variance[i*record_each], '-')
	ax.set_title('Positional variance')
	ax.set_xlabel('Time (ms)', fontsize=14)
	ax.set_ylabel('Positional variance', fontsize=14)

