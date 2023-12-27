import functools
import numpy as np
import matplotlib.pyplot as plt
from lpi_control_demo import LPIController
from lpi_control_demo.utility_funcs import square_wave
from lpi_control_demo.utility_funcs import timed_disable

data_file = 'data.pkl'
save_data = False

# Square wave parameters
amplitude = 10.0
period = 25.0
t_wave_start = 2*period
num_cycle = 2.5 
t_wave_stop = t_wave_start + num_cycle*period

# Disable signal parameters
t_disable = t_wave_stop

# Time points 
num_pts = 1000
t_end = t_wave_stop + 4*period 
t_vals = np.linspace(0,t_end,num_pts)

# Get set point function - a square wave with given parameters 
setpt_func = functools.partial(
        square_wave, 
        amplitude=amplitude,
        period=period,
        t0=t_wave_start, 
        num_cycle=num_cycle,
        )

# Disable signal. Timed disable, enabled until t0 then  disabled 
disable_func = functools.partial(timed_disable, t0=t_disable) 

# Set up and solve ODE model of LPI controller
param = {
        'dcoef'   : 0.1, 
        'pgain'   : 1.5, 
        'igain'   : 0.0,
        'ileak'   : 0.0,
        'setpt'   : setpt_func,
        'disable' : disable_func, 
        }
ctlr = LPIController(param)
y = ctlr.solve(t_vals, method='RK23')

# Get set point values for plot
setpt_vals = setpt_func(t_vals)
disable_vals = disable_func(t_vals)

# Save data
if save_data: 
    data = {
            't'       : t_vals,
            'setpt'   : setpt_vals,
            'disable' : disable_vals, 
            'omega'   : y[0],
            }
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)

# Get ylims for plotting
angvel_max = max(np.absolute(y[0]).max(), amplitude)
ylim_scale = 1.15
ylim_val = ylim_scale*angvel_max
ylim = (-ylim_val, ylim_val)

fig, ax = plt.subplots(1,1)
setpt_line, = ax.plot(t_vals, setpt_vals)
omega_line, = ax.plot(t_vals, y[0])

rect_pos = (t_disable, -ylim_val)
rect_width = t_end - t_disable
rect_height = 2*ylim_val
rect = plt.Rectangle(rect_pos,rect_width, rect_height, facecolor='gray', alpha=0.2)
ax.add_patch(rect)

text_pos = (0.5*t_disable, 0.90*ylim_val)
ax.text(*text_pos, 'enabled')

text_pos = (t_disable + 0.5*rect_width, 0.90*ylim_val)
ax.text(*text_pos, 'disabled')

ax.set_label('t (s)')
ax.set_ylabel('angular velocity')
ax.grid(True)
ax.legend((setpt_line, omega_line), (r'$\omega$ set point', r'$\omega$'), loc='lower right')
ax.set_ylim(*ylim)

ax.set_title('Proportional Controller')
plt.show()
