from lakemodel_function import lake_problem

from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           MultiprocessingEvaluator, ema_logging,
                           Constant)
from ema_workbench.analysis import parcoords
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ema_logging.log_to_stderr(ema_logging.INFO)

#instantiate the model
lake_model = Model('lakeproblem', function=lake_problem)
lake_model.time_horizon = 100 # used to specify the number of timesteps

#specify uncertainties
lake_model.uncertainties = [RealParameter('mean', 0.01, 0.05),
                            RealParameter('stdev', 0.001, 0.005),
                            RealParameter('b', 0.1, 0.45),
                            RealParameter('q', 2.0, 4.5),
                            RealParameter('delta', 0.93, 0.99)]

# set levers, one for each time step
lake_model.levers = [RealParameter(f"l{i}", 0, 0.1) for i in
                     range(lake_model.time_horizon)] # we use time_horizon here

#specify outcomes
lake_model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                       ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome('inertia'),
                       ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

lake_model.constantcs = [Constant('alpha', 0.41),
                         Constant('reps', 150)],

if __name__ == '__main__':
    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.optimize(nfe=5000, epsilons=[0.25, 0.1, 0.1])

    outcomes = results.loc[:, ['max_P', 'utility', 'reliability']]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(outcomes.max_P, outcomes.utility, outcomes.reliability)
    ax.set_xlabel('max. P')
    ax.set_ylabel('utility')
    ax.set_zlabel('reliability')
    plt.show()

    limits = parcoords.get_limits(outcomes)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(outcomes)

    # we invert this axis so direction of desirability is the same
    axes.invert_axis('max_P')
    plt.show()

    lake_model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                           ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                           ScalarOutcome('inertia', kind=ScalarOutcome.MAXIMIZE),
                           ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.optimize(nfe=5000, epsilons=[0.5, 0.5, 0.5, 0.5])

    outcomes = results.loc[:, ['max_P', 'utility', 'reliability', 'inertia']]

    limits = parcoords.get_limits(outcomes)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(outcomes)

    # we invert this axis so direction of desirability is the same
    axes.invert_axis('max_P')
    plt.show()

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results = evaluator.optimize(nfe=5000, epsilons=[0.125, 0.05, 0.05, 0.05])

    outcomes = results.loc[:, ['max_P', 'utility', 'reliability', 'inertia']]

    limits = parcoords.get_limits(outcomes)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(outcomes)

    # we invert this axis so direction of desirability is the same
    axes.invert_axis('max_P')
    plt.show()