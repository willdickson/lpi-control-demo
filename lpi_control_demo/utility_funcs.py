import numpy as np

@np.vectorize
def pulse_func(t, t0, t1, step):

    """
    A simple pulse function specified by the start time, t0, the stop time
    t1 and the step value of the pulse.

    Parameters:
    ----------
    t  : float
         time in (s)

    t0 : float
         step start time

    t1 : float
         step end time

    step : float
           step value for pulse

    Returns:
    --------
    val : float
          the value of the pulse function at time t

    """

    if t < t0:
        val = 0.0
    elif t > t0 and t < t1:
        val = step
    else:
        val = 0.0
    return val


@np.vectorize
def timed_disable(t, t0):
    """
    A simple disable function which returns False (no disable) until t>t0 and 
    the returns True (disable).

    Parameters:
    t  : float
         time in (s)

    t0 : float
         time after which disable should be true.

    Returns:
    vals : bool
           the disable value, True = disable
         
    """

    if t <= t0:
        val = False
    else:
        val = True
    return val


@np.vectorize
def square_wave(t, amplitude=1.0, period=1.0, t0=0.0, num_cycle=None):
    """
    Square wave function specified by amplitude, period and
    time at which squarewave begins.

    Parameters:
    -----------
    t         : float
                time in (s)

    amplitude : float, optional
                the amplitude of the square wave

    period    : float, optional
                the period of the square wave

    t0        : float, optional
                the time at which the square wave begins

    num_cycle : float, optional
                the number of cycles to perform. None => infinite

    Returns:
    val       : float
                the value of the square wave at time t
     
    """
    if t < t0:
        val = 0.0
    elif (t-t0)%period < 0.5*period:
        val = amplitude
    else:
        val = -amplitude
    if num_cycle is not None:
        if t > t0 + period*num_cycle:
            val = 0.0
    return val


@np.vectorize
def periodic_disable(t, period=1.0, t0=0.0):
    """
    Parameters:
    -----------
    t         : float
                time in (s)

    period    : float, optional
                the period of the square wave

    t0        : float, optional
                the time at which the square wave begins

    Returns:
    val       : float
                the value of the square wave at time t

    """
    if t < t0:
        val = 0.0
    elif (t-t0)%period < 0.5*period:
        val = False 
    else:
        val = True 
    return val


@np.vectorize
def false_func(t):
    """
    Function which always returns false
    
    Parameters:
    -----------
    t : float
        time (s)

    Returns:
    --------
    val : bool
          always false
    """
    return False 


