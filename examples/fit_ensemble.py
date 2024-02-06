import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from lpi_control_demo import ensemble_fit_controller
from lpi_control_demo import create_interp_func
from lpi_control_demo import LPIController

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

# Search bounds for parameters for model fit
bounds = {
        'dcoef'   : (1.0e-8, 4.0),
        'pgain'   : (1.0e-8, 5.0),
        'igain'   : (1.0e-8, 4.0),
        'ileak'   : (1.0e-8, 4.0),
        }

# Type of controller to fit
controller = 'lpi'

# Flag for displaying cost at every evaluation of cost function
disp_cost = False 

datasets = []

for data_file in file_list:
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


    # Plot the data we are going to fit
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t, omega, 'b')
    ax[0].plot(t, setpt, 'r')
    ax[0].set_ylabel('angular velocity')
    ax[0].set_title('Data to be fit - close figure to run')
    ax[0].grid(True)
    ax[1].plot(t, disable, 'k')
    ax[1].set_ylabel('disable')
    ax[1].set_xlabel('t (sec)')
    ax[1].grid(True)
    ax[0].set_title(str(data_file))

plt.show()

# Fit controller 
param = ensemble_fit_controller(
        datasets,            # list of datasets
        bounds,              # search bounds for fitting the model parameters
        controller,          # type of controller to fit
        disp_cost=disp_cost  # whether to display the cost every evaluation
        )


print(param)

print('saving fit parameters')
with open(f'{controller}_ensemble_fit.pkl', 'wb') as f:
    pickle.dump(param, f)

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


