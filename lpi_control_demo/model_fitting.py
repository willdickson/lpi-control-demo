import numpy as np
import scipy as sp
from .lpi_controller import LPIController


def fit_controller(t, omega, setpt, disable, bounds, controller, tol=0.01, 
        popsize=15, workers=-1, disp_cost=False):
    """
    Fit a controller model to the provided data

    Parameters:
    ----------

    t          : ndarray
                 1D array (float) of time values

    omega      : ndarray
                 1D array (float) of angular velocities (data). These are 
                 the values we are fitting to. 

    setpt      : ndarray
                 1D array (float) of set point values.

    disable    : ndarray
                 1D array (bool) of disable values.  True => controller disabled.

    bounds     : dict 
                 Dictionary of bounds for variables, dcoef, pgain, igain, ileak.

    controller : string
                 A string indicating the controller type, lpi, pi or p
                 for leaky-protortional-integral, proportional-integral,
                 and proportional respectively. 

    tol        : float, optional
                 Relative tolerance for convergence. 
                 See scipy.optimize.differential_evolution 

    popsize    : int, optional
                 A multiplier for setting the total population size. 
                 See scipy.optimize.differential_evolution

    workers    : int or map-like callable, optional
                 If `workers` is an int the population is subdivided into `workers`
                 sections and evaluated in parallel.
                 See scipy.optimize.differential_evolution.

    disp_cost  : bool, optional
                 Whether or not to display the cost at every cost function evaluation.


    Returns:
    --------

    param      : dict
                 Dictionary containing the LPI controller parameters
                 param = {
                         'dcoef'    # drag coefficient 
                         'pgain'    # proportional gain 
                         'igain'    # integrator gain
                         'ileak'    # integrator leakiness coefficient
                         'setpt'    # set point function 
                         'disable'  # controller disable function 
                         }

    """
    match controller:
        case 'lpi':
            _bounds = [bounds['dcoef'], bounds['pgain'], bounds['igain'], bounds['ileak']]
        case 'pi':
            _bounds = [bounds['dcoef'], bounds['pgain'], bounds['igain']]
        case 'p':
            _bounds = [bounds['dcoef'], bounds['pgain']]
        case _:
            raise ValueError(f'unknown controller type {controller}')

    # Create interpolated functions for set point and disable 
    setpt_func = sp.interpolate.interp1d(t, setpt, fill_value='extrapolate')
    disable_func = sp.interpolate.interp1d(t, disable, fill_value='extrapolate')

    results = sp.optimize.differential_evolution(
            cost_func, 
            _bounds, 
            args=(t, omega, setpt_func, disable_func, disp_cost, controller),
            disp=True,
            polish=False,
            tol=tol,
            updating='deferred',
            popsize=popsize, 
            workers=workers,
            )
    print(results)

    match controller:
        case 'lpi':
            param = {
                    'dcoef'   : results.x[0], 
                    'pgain'   : results.x[1], 
                    'igain'   : results.x[2],
                    'ileak'   : results.x[3],
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case 'pi':
            param = {
                    'dcoef'   : results.x[0], 
                    'pgain'   : results.x[1], 
                    'igain'   : results.x[2],
                    'ileak'   : 0.0,
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case 'p':
            param = {
                    'dcoef'   : results.x[0], 
                    'pgain'   : results.x[1], 
                    'igain'   : 0.0,
                    'ileak'   : 0.0,
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case _:
            raise ValueError(f'unknown controller type {controller}')

    return param



def cost_func(x, t, omega, setpt_func, disable_func, disp_cost, controller):
    """
    Cost function for lpi, pi and p controllers. 

    Parameters:
    ----------

    x            :  ndarray
                    1D array of floats, shape (4,), (3,), (2,) for  lpi, pi
                    and p controllers respectively

    t            :  float
                    The current time

    omega        : ndarray
                   1D array (floats) of desired angular velocities.  This
                   is the data we fitting to. 

    setpt_func   : callable 
                   Set point function. Takes single argument, the time t 
                   in seconds, and returns the current set point value.

    disable_func : callable 
                   Disable function. Takes single argument, the time t
                   in seconds, and returns bool indicating whether or
                   not the controller should be disabled.

    disp_cost    : bool
                   Whether or not to display the cost at every evaluation.

    controller   : string
                   A string indicating the controller type, lpi, pi or p
                   for leaky-protortional-integral, proportional-integral,
                   and proportional respectively. 

    Returns:
    --------
    
    dy           : ndarray
                   1D array (floats) of shape (4,), (3,) or (2,) for the
                   case of a lpi, pi or p controller
    """
    match controller:
        case 'lpi':
            param = {
                    'dcoef'   : x[0], 
                    'pgain'   : x[1], 
                    'igain'   : x[2],
                    'ileak'   : x[3],
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case 'pi':
            param = {
                    'dcoef'   : x[0], 
                    'pgain'   : x[1], 
                    'igain'   : x[2],
                    'ileak'   : 0.0,
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case 'p':
            param = {
                    'dcoef'   : x[0], 
                    'pgain'   : x[1], 
                    'igain'   : 0.0,
                    'ileak'   : 0.0,
                    'setpt'   : setpt_func,
                    'disable' : disable_func, 
                    }
        case _:
            raise ValueError(f'unknown controller type {controller}')

    ctlr = LPIController(param)
    y = ctlr.solve(t, method='RK23')
    omega_fit = y[0]
    cost = np.sum((omega_fit - omega)**2)/omega.shape[0]
    if disp_cost:
        print(f'  cost: {cost}')
    return cost



