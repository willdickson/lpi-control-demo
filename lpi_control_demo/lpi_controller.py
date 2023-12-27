import numpy as np
import scipy.integrate as integrate
from .utility_funcs import false_func

class LPIController:
    
    """
    Implements a simple Leaky proportional-integral controller, i.e., a
    proportional controller with a leaky integrator. 

    Note, moment of inertia I is assumed to be = 1.0. 
    """

    def __init__(self, param):
        self.dcoef = param['dcoef'] # damping coefficient
        self.pgain = param['pgain'] # controller proportional gain
        self.igain = param['igain'] # controller integral gain
        self.ileak = param['ileak'] # integrator leakiness coefficient
        self.setpt = param['setpt'] # controller set point
        try:
            self.disable = param['disable']
        except KeyError:
            self.disable = false_func 

    def state_func(self, t, y):
        """
        Dynamical system state function

        Parameters: 
        -----------
        t : float 
            time in seconds

        y : ndarray 
            1D array with shape=(2,) and float type containing the angular
            velocity y[0] and angular acceleration y[1]. 

        Returns:
        --------
        dy : ndarry
             1D array with shape=(2,) and float type containing the derivatives
             of the angular velocity dy[0] and angular acceleration dy[1]. 

        """
        dy = np.zeros(y.shape)
        if self.disable(t):
            dy[0] = -self.dcoef*y[0] + y[1] 
            dy[1] = -self.ileak*y[1] 
        else:
            err = self.setpt(t) - y[0]
            dy[0] = -self.dcoef*y[0] + y[1] + self.pgain*err
            dy[1] = -self.ileak*y[1] + self.igain*err
        return dy


    def solve(self, t_vals, y_init=None, method='RK45'):
        """
        Solves the LPI controller differential equations for the specified time values. 

        Parameters:
        -----------
        t_vals : ndarray
                 1D array with float type containing the time values at which return
                 the numerical solution of the differential equation. 

        y_init : ndarray, optional
                 1D array with initial condition.  Default all zeros

        method : string, optional
                 a string specifiying ODE solver method, e.g. 'RK45', 'LSODA', etc

        Returns:
        --------
        y_vals : ndarray
                2D array, shape = (2,N), LPI controller solution at specified time points. 

        """
        y_init = np.array([0.0,0.0]) if y_init is None else y_init
        t_min, t_max = t_vals[0], t_vals[-1]
        max_step = t_vals[1] - t_vals[0]
        result = integrate.solve_ivp(
                self.state_func, 
                (t_min, t_max), 
                y_init, 
                t_eval=t_vals, 
                method=method, 
                max_step=max_step,
                )
        y_vals = result.y
        return y_vals 

    def get_determinant(self):
        """
        Get determinant of LPI linear system
        """
        return self.ileak*(self.pgain + self.dcoef) + self.igain

    def get_trace(self):
        """
        Get trace of LPI linear system
        """
        return -(self.pgain + self.dcoef + self.ileak)

    def get_discriminant(self):
        """
        Get discriminant of LPI linear system
        """
        return self.get_trace()**2 - 4*self.get_determinant()

    def get_eigenvalues(self):
        """
        Get eigenvalues of LPI linear system
        """
        A = np.array([[-(self.pgain + self.dcoef), 1], [-self.igain, -self.ileak]])
        return np.linalg.eig(A)







