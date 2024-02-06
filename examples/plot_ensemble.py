import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from lpi_control_demo import ensemble_fit_controller
from lpi_control_demo import create_interp_func
from lpi_control_demo import LPIController


param_file = 'lpi_ensemble_fit.pkl'

print('loading fit parameters')
with open(param_file, 'rb') as f:
    param = pickle.load(f)
     

# File containing data to fit
file_list = [ 
        pathlib.Path('data_files', 'data_3s.pkl'), 
        pathlib.Path('data_files', 'data_10s.pkl'), 
        pathlib.Path('data_files', 'data_30s.pkl'), 
        pathlib.Path('data_files', 'data_40s.pkl'), 
        pathlib.Path('data_files', 'data_60s.pkl'), 
        pathlib.Path('data_files', 'data_80s.pkl'), 
        pathlib.Path('data_files', 'data_90s.pkl'), 
        pathlib.Path('data_files', 'data_100s.pkl'), 
        pathlib.Path('data_files', 'data_180s.pkl'), 
        ]

datasets = []

print('loading data files')
for data_file in file_list:
    print(f'  {data_file}')
    # Load in data and unpack the values
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    t = data['t']
    omega = data['omega']
    setpt = data['setpt']
    disable = data['disable']
    num_pts = t.shape[0]

    # fix setpt error
    mask = setpt < 0
    setpt[mask] = 0

    datasets.append({
        't'         : t, 
        'omega'     : omega, 
        'setpt'     : setpt, 
        'disable'   : disable,
        'data_file' : data_file
        })

print()


print('plotting results')
for ds in datasets:

    print(f'  {ds['data_file']}')

    t = ds['t']
    omega = ds['omega']
    setpt = ds['setpt']
    disable = ds['disable']

    _param = dict(param)
    _param['setpt'] = create_interp_func(t, setpt)
    _param['disable'] = create_interp_func(t, disable)

    # Evaluate controller, fit above, and get angular velocity for model fit
    ctlr = LPIController(_param)
    y = ctlr.solve(t)
    omega_fit = y[0]
    
    # Plot results
    fig, ax = plt.subplots(2,1,sharex=True)
    omega_line, = ax[0].plot(t, omega, 'b')
    omega_fit_line, = ax[0].plot(t, omega_fit, 'g')
    setpt_line, = ax[0].plot(t, setpt, 'r')
    ax[0].set_ylabel('angular velocity')
    ax[0].set_title(f'{ds["data_file"]}')
    ax[0].legend(
            (omega_line, omega_fit_line, setpt_line), 
            (r'$\omega$ true', r'$\omega$ fit', r'setpt'), 
            loc='lower right'
            )
    ax[0].grid(True)
    ax[1].plot(t, disable, 'k')
    ax[1].set_ylabel('disable')
    ax[1].set_xlabel('t (sec)')
    ax[1].grid(True)
plt.show()


