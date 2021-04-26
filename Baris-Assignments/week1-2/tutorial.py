import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import Model, RealParameter, TimeSeriesOutcome, SequentialEvaluator
from ema_workbench.analysis import plotting, plotting_util
from scipy.integrate import odeint


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma) :
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def SIR_model(beta=0.2, gamma=0.1, I0=1, R0=0,
              N=1000, t=np.linspace(0, 160, 160)) :
    '''

    Parameters
    ----------
    beta : float
           contact rate
    gamma : float
            recovery rate
    I0 : int
         initial value infected
    R0 : int
         initial value recovered
    N : int
        population size
    t : ndarray
        points in time

    '''
    S0 = N - I0 - R0

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    return {'S' : S, 'I' : I, 'R' : R}


model = Model('SIR', function=SIR_model)

model.uncertainties = [RealParameter('beta', 0.05, 0.3),
                       RealParameter('gamma', 0.01, 0.15)]

model.outcomes = [TimeSeriesOutcome('S'),
                  TimeSeriesOutcome('I'),
                  TimeSeriesOutcome('R')]

with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=100)



for outcome in outcomes.keys():
    plotting.lines(experiments, outcomes, outcomes_to_show=outcome,
                   density=plotting_util.Density.HIST)
plt.show()