import numpy as np
import scipy as sp
from .lpi_controller import LPIController


def ensemble_fit_controller(datasets, bounds, controller, tol=0.01, popsize=15, 
        workers=-1, disp_cost=False):

    """
    Fit a controller model to an ensemble of datasets
    Fit a controller model to the provided data

    Parameters:
    ----------

    datasets   : list 
                 list of datasets to fit control model to  

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

    _datasets = [dict(ds) for ds in datasets]

    for ds in _datasets:
        ds['setpt_func'] = create_interp_func(ds['t'], ds['setpt'])
        ds['disable_func'] = create_interp_func(ds['t'], ds['disable'])

    results = sp.optimize.differential_evolution(
            ensemble_cost_func, 
            _bounds, 
            args=(_datasets, disp_cost, controller),
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
                    }
        case 'pi':
            param = {
                    'dcoef'   : results.x[0], 
                    'pgain'   : results.x[1], 
                    'igain'   : results.x[2],
                    'ileak'   : 0.0,
                    }
        case 'p':
            param = {
                    'dcoef'   : results.x[0], 
                    'pgain'   : results.x[1], 
                    'igain'   : 0.0,
                    'ileak'   : 0.0,
                    }
        case _:
            raise ValueError(f'unknown controller type {controller}')

    return param


def ensemble_cost_func(x, datasets, disp_cost, controller):
    """
    Ensemble cost function for lpi, pi and p controllers. 

    Cost function for lpi, pi and p controllers. 

    Parameters:
    ----------

    x            :  ndarray
                    1D array of floats, shape (4,), (3,), (2,) for  lpi, pi
                    and p controllers respectively

    datasets     : list 
                   list of datasets to fit control model to  

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
                    }
        case 'pi':
            param = {
                    'dcoef'   : x[0], 
                    'pgain'   : x[1], 
                    'igain'   : x[2],
                    'ileak'   : 0.0,
                    }
        case 'p':
            param = {
                    'dcoef'   : x[0], 
                    'pgain'   : x[1], 
                    'igain'   : 0.0,
                    'ileak'   : 0.0,
                    }
        case _:
            raise ValueError(f'unknown controller type {controller}')

    total_cost = 0.0
    for ds in datasets:
        param['setpt'] = ds['setpt_func']
        param['disable'] = ds['disable_func']
        ctlr = LPIController(param)
        y = ctlr.solve(ds['t'], method='RK23')
        omega_fit = y[0]
        cost = np.sum((omega_fit - ds['omega'])**2)/ds['omega'].shape[0]
        total_cost += cost
    total_cost = total_cost/len(datasets)
    if disp_cost:
        print(f'  cost: {totalcost}')
    return total_cost



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
    setpt_func = create_interp_func(t, setpt)
    disable_func = create_interp_func(t, disable)

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



def create_interp_func(t, data): 
    interp_func = sp.interpolate.interp1d(t, data, fill_value='extrapolate')
    return interp_func
