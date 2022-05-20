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

def _baseObjective(I_re, I_ohm, tCQ):
    """
    Returns base objective function, provided a maximum RE current, final Ohmic
    current and CQ time.

    :param I_re:    Maximum RE current (A).
    :param I_ohm:   Final Ohmic current (A).
    :param tCQ:     CQ time (s).
    """
    obj1 = I_re / CRITICAL_RE_CURRENT
    obj2 = I_ohm / CRITICAL_OHMIC_CURRENT
    obj3 = sigmoid(-tCQ, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(tCQ, CQ_TIME_MAX, SLOPE_RIGHT)
    return obj1 + obj2 + CQ_WEIGHT * (obj3 + obj4)

def baseObjective(output):
    """
    Returns base objective function, provided a DREAMSimulation output.

    :param output:      DREAMSimulation.Output object.
    """
    if not isinstance(output, DREAMSimulation.Output):
        raise TypeError("output need to be a instance of DREAMSimulation.Output")

    return _baseObjective(output.maxRECurrent, output.finalOhmicCurrent, output.currentQuenchTime)

def heatLossObjective(output):
    """
    Returns heat loss objective function, provided a TransportSimulation output.

    :param output:      TransportSimulation.Output object.
    """
    if not isinstance(output, TransportSimulation.Output):
        raise TypeError("output need to be an instance of TransportSimulation.Output")

    return baseObjective(output) + output.transportedFraction / CRITICAL_TRANSPORTED_FRACTION


### with step functions (not used...)

def linearStep(x, xc):
    """
    Returns 0 if x<xc and (x-xc)/xc else.

    :param x:   np.array of input values.
    :param xc:  Critical value of x.
    """
    return (x - xc) * np.heaviside(x - xc, 0)

def _baseObjectiveStep(I_re, I_ohm, tCQ):
    """
    Returns base objective with step functions, provided a maximum RE
    current, final Ohmic current and CQ time.

    :param I_re:    Maximum RE current (A).
    :param I_ohm:   Final Ohmic current (A).
    :param tCQ:     CQ time (s).
    """
    obj1 = linearStep(I_re, CRITICAL_RE_CURRENT)
    obj2 = linearStep(I_ohm, CRITICAL_OHMIC_CURRENT)
    obj3 = sigmoid(-tCQ, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(tCQ, CQ_TIME_MAX, SLOPE_RIGHT)
    return obj1 + obj2 + CQ_WEIGHT * (obj3 + obj4)


def baseObjectiveStep(output):
    """
    Returns base objective with step functions, provided a DREAMSimulation output.

    :param output:      DREAMSimulation.Output object.
    """
    if not isinstance(output, DREAMSimulation.Output):
        raise TypeError("output need to be a instance of DREAMSimulation.Output")

    return _baseObjectiveStep(output.maxRECurrent, output.finalOhmicCurrent, output.currentQuenchTime)

def heatLossObjectiveStep(output):
    """
    Returns heat loss objective function, provided a TransportSimulation output.

    :param output:      TransportSimulation.Output object.
    """
    if not isinstance(output, TransportSimulation.Output):
        raise TypeError("output need to be an instance of TransportSimulation.Output")

    return baseObjectiveStep(output) + linearStep(output.transportedFraction, CRITICAL_TRANSPORTED_FRACTION)
