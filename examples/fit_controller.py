import pickle
import numpy as np
import matplotlib.pyplot as plt
from lpi_control_demo import fit_controller
from lpi_control_demo import LPIController

# File containing data to fit
data_file = 'data.pkl'

# Search bounds for parameters for model fit
bounds = {
        'dcoef'   : (1.0e-8, 4.0),
        'pgain'   : (1.0e-8, 4.0),
        'igain'   : (1.0e-8, 4.0),
        'ileak'   : (1.0e-8, 4.0),
        }

# Type of controller to fit
controller = 'lpi'

# Flag for displaying cost at every evaluation of cost function
disp_cost = False 

# Load in data and unpack the values
with open(data_file, 'rb') as f:
    data = pickle.load(f)
t = data['t']
omega = data['omega']
setpt = data['setpt']
disable = data['disable']
num_pts = t.shape[0]

# Add some noise to the omega values we are fitting. I'm doing this a
# the data I have comes from simulation and is unrealisticly clean. 
noise_std = 2.0
omega_noisy = omega + noise_std*np.random.randn(num_pts)

# Plot the data we are going to fit
fig, ax = plt.subplots(1,1)
omega_noisy_line, = ax.plot(t, omega_noisy, 'gray')
omega_line, = ax.plot(t, omega, 'b')
ax.set_xlabel('t (sec)')
ax.set_ylabel('angular velocity')
ax.set_title('Data to be fit - close figure to run')
ax.legend(
        (omega_noisy_line, omega_line), 
        (r'$\omega$ with noise added', r'$\omega$ true'), 
        loc='lower right'
        )
ax.grid(True)
plt.show()

# Fit controller 
param = fit_controller(
        t,                   # array of time points
        omega_noisy,         # array of measured angular velocities
        setpt,               # array of set point values 
        disable,             # array of flags indicating when controller disabled
        bounds,              # search bounds for fitting the model parameters
        controller,          # type of controller to fit
        disp_cost=disp_cost  # whether to display the cost every evaluation
        )

# Evaluate controller, fit above, and get angular velocity for model fit
ctlr = LPIController(param)
y = ctlr.solve(data['t'])
omega_fit = y[0]

# Plot results
fig, ax = plt.subplots(1,1)
omega_noisy_line, = ax.plot(t, omega_noisy, 'gray')
omega_line, = ax.plot(t, omega, 'b')
omega_fit_line, = ax.plot(t, omega_fit, 'g')
ax.set_xlabel('t (sec)')
ax.set_ylabel('angular velocity')
ax.set_title('Fit Results')
ax.legend(
        (omega_noisy_line, omega_line, omega_fit_line), 
        (r'$\omega$ with noise added', r'$\omega$ true', r'$\omega$ fit'), 
        loc='lower right'
        )
ax.grid(True)
plt.show()


