import numpy as np

from sim.DREAM.DREAMSimulation import DREAMSimulation
from sim.DREAM.transport import TransportSimulation

# objective function parameters
CRITICAL_RE_CURRENT             = 150e3
CRITICAL_OHMIC_CURRENT          = 300e3

HEAT_LOSS_WEIGHT                = 100
CRITICAL_TRANSPORTED_FRACTION   = 1e-2

CQ_WEIGHT                       = 100
CQ_TIME_MIN                     = 50e-3
CQ_TIME_MAX                     = 150e-3

SLOPE_LEFT                      = 3e2
SLOPE_RIGHT                     = 3e2


def sigmoid(x, x0=0, k=1):
    """
    Hyperbolic tangent sigmoid function.

    :param x:   np.ndarray of input values.
    :param x0:  Offset.
    :param k:   Sets how steep the transitions is from 0 to 1.
    """
    return 1/2 + 1/2*np.tanh(k*(x-x0))

def baseObjective(output, **kwargs):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM.

    :param output:      DREAMSimulation output object.

    Optional keyword arguments:
    :param critRE:      Critical RE current, ~1 penalty when I_re ~ critRE.
    :param critOhm:     Critical Ohmic current, ~1 penalty when I_ohm ~ critOhm.
    :param wCQ:         Penalty for when tCQ is outside its interval.
    """
    if not isinstance(output, DREAMSimulation.Output):
        raise TypeError("output need to be a instance of DREAMSimulation.Output")

    critRE  = kwargs.get('critRE',  CRITICAL_RE_CURRENT)
    critOhm = kwargs.get('critOhm', CRITICAL_OHMIC_CURRENT)
    wCQ     = kwargs.get('wCQ',     CQ_WEIGHT)

    obj1 = output.maxRECurrent / critRE
    obj2 = output.finalOhmicCurrent / critOhm
    obj3 = sigmoid(-output.currentQuenchTime, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(output.currentQuenchTime, CQ_TIME_MAX, SLOPE_RIGHT)

    return obj1 + obj2 + wCQ * (obj3 + obj4)

def heatLossObjective(output, **kwargs):
    """
    Returns a objective function for when optimizing disruption
    simulations using DREAM that takes into account the conducted heat losses.

    :param output:      DREAMSimulation output object.

    Optional keyword arguments:
    :param critTransp:  Critical transport fraction, ~1 penalty when it is ~ critTransp.
    :param critRE:      Critical RE current, ~1 penalty when I_re ~ critRE.
    :param critOhm:     Critical Ohmic current, ~1 penalty when I_ohm ~ critOhm.
    :param wCQ:         Penalty for when tCQ is outside its interval.
    """
    if not isinstance(output, TransportSimulation.Output):
        raise TypeError("output need to be an instance of TransportSimulation.Output")

    critTransp = kwargs.get('critTransp', CRITICAL_TRANSPORTED_FRACTION)

    obj1 = baseObjective(output, **kwargs)
    obj2 = output.transportedFraction / critTransp

    return obj1 + obj2


### with step functions

def linearStep(x, xc):
    """
    Returns 0 if x<xc and (x-xc)/xc else.

    :param x:   np.array of input values.
    :param xc:  Critical value of x.
    """
    return (x - xc) * np.heaviside(x - xc, 0)

def stepObjective(output, **kwargs):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM, this using step functions.

    :param output:      DREAMSimulation output object.

    Optional keyword arguments:
    :param critRE:      Critical RE current, ~1 penalty when I_re ~ critRE.
    :param critOhm:     Critical Ohmic current, ~1 penalty when I_ohm ~ critOhm.
    :param wCQ:         Penalty for when tCQ is outside its interval.
    """
    if not isinstance(output, DREAMSimulation.Output):
        raise TypeError("output need to be a instance of DREAMSimulation.Output")

    critRE  = kwargs.get('critRE',  CRITICAL_RE_CURRENT)
    critOhm = kwargs.get('critOhm', CRITICAL_OHMIC_CURRENT)
    wCQ     = kwargs.get('wCQ',     CURRENT_QUENCH_WEIGHT)

    obj1 = linearStep(output.maxRECurrent, critRE)
    obj2 = linearStep(output.finalOhmicCurrent, critOhm)
    obj3 = sigmoid(-output.currentQuenchTime, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(output.currentQuenchTime, CQ_TIME_MAX, SLOPE_RIGHT)

    return obj1 + obj2 + wCQ * (obj3 + obj4)


def stepHeatLossObjective(output, **kwargs):
    """
    Returns a objective function for when optimizing disruption
    simulations using DREAM that takes into account the conducted heat losses,
    this using step functions.

    :param output:      DREAMSimulation output object.

    Optional keyword arguments:
    :param critRE:      Critical RE current, ~1 penalty when I_re ~ critRE.
    :param critOhm:     Critical Ohmic current, ~1 penalty when I_ohm ~ critOhm.
    :param wCQ:         Penalty for when tCQ is outside its interval.
    """
    if not isinstance(output, TransportSimulation.Output):
        raise TypeError("output need to be an instance of TransportSimulation.Output")

    critTransp = kwargs.get('critTransp', CRITICAL_TRANSPORTED_FRACTION)

    obj1 = stepObjective(output, **kwargs)
    obj2 = linearStep(output.transportedFraction, critTransp)

    return obj1 + obj2
