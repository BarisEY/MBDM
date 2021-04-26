import numpy as np
import matplotlib.pyplot as plt
import lakemodel_function

from ema_workbench import Model, RealParameter, TimeSeriesOutcome, SequentialEvaluator
from ema_workbench.analysis import plotting, plotting_util
from scipy.integrate import odeint

def lakemodel_model(b, q, mean, stdev, delta, I0=1, R0=0,
                    N=1000, t=np.linspace(0, 99, 100)) :

    y0 = 0, 0, 0, 0

    ret = odeint(lakemodel_function.lake_problem, y0, t, args=(b, q, mean, stdev, delta))
    phos, uti, inertia, rel = ret.T

    return {"phos": phos, "uti": uti, "inertia": inertia, "rel": rel}


model = Model('puir', function=lakemodel_model)

model.uncertainties = [RealParameter('mean', 0.01, 0.05),
                       RealParameter('stdev', 0.001, 0.005),
                       RealParameter('b', 0.1, 0.45),
                       RealParameter('q', 2, 4.5),
                       RealParameter('delta', 0.93, 0.99)]



model.outcomes = [TimeSeriesOutcome('p'),
                  TimeSeriesOutcome('u'),
                  TimeSeriesOutcome('i'),
                  TimeSeriesOutcome('r')]

with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=100)



for outcome in outcomes.keys():
    plotting.lines(experiments, outcomes, outcomes_to_show=outcome,
                   density=plotting_util.Density.HIST)
plt.show()
