import numpy as np
import matplotlib.pyplot as plt
import lakemodel_function
from ema_workbench import Model, RealParameter, SequentialEvaluator, ScalarOutcome
from scipy.integrate import odeint

def lakemodel_model(b, q, mean, stdev, delta, I0=1, R0=0,
                    N=1000, t=np.linspace(0, 99, 100)) :

    y0 = 0, 0, 0, 0

    ret = odeint(lakemodel_function.lake_problem, y0, t, args=(b, q, mean, stdev, delta))
    phos, uti, inertia, rel = ret.T

    return {"phos": phos, "uti": uti, "inertia": inertia, "rel": rel}


model = Model('lakeproblem', function = lakemodel_function.lake_problem)

model.uncertainties = [RealParameter('mean', 0.01, 0.05),
                       RealParameter('stdev', 0.001, 0.005),
                       RealParameter('b', 0.1, 0.45),
                       RealParameter('q', 2, 4.5),
                       RealParameter('delta', 0.93, 0.99)]

model.outcomes = [ScalarOutcome('max_P'),
                  ScalarOutcome('utility'),
                  ScalarOutcome('inertia'),
                  ScalarOutcome('reliability')]

with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(scenarios=100)



max_P = outcomes['max_P']
utility = outcomes['utility']
inertia = outcomes['inertia']
reliability = outcomes['reliability']

fig, axs = plt.subplots(1, 4, figsize=(20,4), sharex=True, sharey=True)

axs[0].scatter(experiments.b, experiments.q, c=max_P)
axs[0].set_title('Max_P')
axs[1].scatter(experiments.b, experiments.q, c=utility)
axs[1].set_title('Utility')
axs[2].scatter(experiments.b, experiments.q, c=inertia)
axs[2].set_title('Inertia')
axs[3].scatter(experiments.b, experiments.q, c=reliability)
axs[3].set_title('Reliability')

# axs.set_xlabel('natural removal rate')
# axs.set_ylabel('natural recycling rate')

fig.suptitle('Uncertainties: natural removal & recycling rate', size=12)
# plt.colorbar(sc)

plt.show()