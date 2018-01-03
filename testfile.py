    
### retrieve the files (for the cells are long to run in GeneralCoordinates. pynb) ### 
    
def record_pickle():
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