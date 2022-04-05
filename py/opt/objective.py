import numpy as np

from sim.DREAM.DREAMSimulation import DREAMSimulation
from sim.DREAM.transport import TransportSimulation

# objective function parameters
CRITICAL_RE_CURRENT             = 150e3
CRITICAL_OHMIC_CURRENT          = 200e3
CQ_TIME_MIN                     = 50e-3
CQ_TIME_MAX                     = 150e-3
SLOPE_LEFT                      = 3e2
SLOPE_RIGHT                     = 3e2
CRITICAL_TRANSPORTED_FRACTION   = 1e-1

def sigmoid(x, x0=0, k=1):
    """
    Hyperbolic tangent sigmoid function.

    :param x:   np.ndarray of input values.
    :param x0:  Offset.
    :param k:   Sets how steep the transitions is from 0 to 1.
    """
    return 1/2 + 1/2*np.tanh(k*(x-x0))

def baseObjective(output, weight=100):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM.

    :param output:  DREAMSimulation output object.
    :param weight:  Penalty for when tCQ is outside its interval.
    """
    # if not issubclass(output, DREAMSimulation.Output):
    #     raise TypeError("output need to be a subclass of DREAMSimulation.Output")

    obj1 = output.maxRECurrent / CRITICAL_RE_CURRENT
    obj1 = output.finalOhmicCurrent / CRITICAL_OHMIC_CURRENT
    obj3 = sigmoid(-output.currentQuenchTime, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(output.currentQuenchTime, CQ_TIME_MAX, SLOPE_RIGHT)
    return obj1 + weight * (obj2 + obj3)

def heatLossObjective(output, weight1=100, weight2=20):
    """
    Returns a objective function for when optimizing disruption
    simulations using DREAM that takes into account the conducted heat losses.

    :param output:      DREAMSimulation output object.
    :param weight1:     Penalty for when tCQ is outside its interval.
    :param weight2:     Penalty for when transported fraction is too large.
    """
    # if not isinstance(output, TransportSimulation.Output):
    #     raise TypeError("output need to be an instance of TransportSimulation.Output")

    obj1 = self.baseObjective(output, weight=weight1)
    obj2 = output.transportedFraction / CRITICAL_TRANSPORTED_FRACTION
    return obj1 + weight2 * obj2


### with step functions

def linearStep(x, xc):
    """
    Returns 0 if x<xc and (x-xc)/xc else.

    :param x:   np.array of input values.
    :param xc:  Critical value of x.
    """
    return (x - xc) * np.heaviside(x - xc, 0)

def stepObjective(output, weight=100):
    """
    Returns the base objective function for when optimizing disruption
    simulations using DREAM, this using step functions.

    :param output:  DREAMSimulation output object.
    :param weight:  Penalty for when tCQ is outside its interval.
    """
    # if not issubclass(output, DREAMSimulation.Output):
    #     raise TypeError("output need to be a subclass of DREAMSimulation.Output")

    obj1 = linearStep(output.maxRECurrent, CRITICAL_RE_CURRENT)
    obj1 = linearStep(output.finalOhmicCurrent, CRITICAL_OHMIC_CURRENT)
    obj3 = sigmoid(-output.currentQuenchTime, -CQ_TIME_MIN, SLOPE_LEFT)
    obj4 = sigmoid(output.currentQuenchTime, CQ_TIME_MAX, SLOPE_RIGHT)
    return obj1 + weight * (obj2 + obj3)

def stepHeatLossObjective(output, weight1=100, weight2=20):
    """
    Returns a objective function for when optimizing disruption
    simulations using DREAM that takes into account the conducted heat losses,
    this using step functions.

    :param output:      DREAMSimulation output object.
    :param weight1:     Penalty for when tCQ is outside its interval.
    :param weight2:     Penalty for when
    """
    # if not isinstance(output, TransportSimulation.Output):
    #     raise TypeError("output need to be an instance of TransportSimulation.Output")

    obj1 = self.stepObjective(output, weight=weight1)
    obj2 = self.linearStep(output.finalOhmicCurrent, CRITICAL_OHMIC_CURRENT)
    return obj1 + weight2 * obj2
