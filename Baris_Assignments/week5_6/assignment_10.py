import random
import numpy as np
from Week_5_6_quaq.dps_lake_model import lake_model as lake_problem

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant, SequentialEvaluator, MultiprocessingEvaluator,
                           ema_logging, Constraint, Policy, save_results)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
from ema_workbench.analysis import (parcoords, prim)
from ema_workbench.em_framework import samplers
import matplotlib.pyplot as plt
import functools
import seaborn as sns

if __name__ == '__main__':

    def robustness(direction, threshold, data) :
        if direction == SMALLER :
            return np.sum(data <= threshold) / data.shape[0]
        else :
            return np.sum(data >= threshold) / data.shape[0]


    def maxp(data) :
        return np.sum(data <= 0.75) / data.shape[0]


    SMALLER = 'SMALLER'
    LARGER = 'LARGER'

    maxp = functools.partial(robustness, SMALLER, 0.75)
    inertia = functools.partial(robustness, LARGER, 0.6)
    reliability = functools.partial(robustness, LARGER, 0.99)
    utility = functools.partial(robustness, LARGER, 0.75)

    policies = []
    for i in range(4):
        policy_dict = {"c1": random.randint(-2, 2),
                       "c2": random.randint(-2, 2),
                       "r1": random.randint(0, 2),
                       "r2": random.randint(0, 2),
                       "w1": random.randint(0, 1)}
        policies.append(Policy(f"pol_{i}", **policy_dict))



    ema_logging.log_to_stderr(ema_logging.INFO)

    lake_model = Model('lakemodel', function=lake_problem)
    lake_model.time_horizon = 100

    lake_model.uncertainties = [RealParameter("mean", 0.01, 0.05),
                                RealParameter("stdev", 0.001, 0.005),
                                RealParameter("b", 0.1, 0.45),
                                RealParameter("q", 2, 4.5),
                                RealParameter("delta", 0.93, 0.99)]

    lake_model.levers = [RealParameter("c1", -2, 2),
                         RealParameter("c2", -2, 2),
                         RealParameter("r1", 0, 2),
                         RealParameter("r2", 0, 2),
                         RealParameter("w1", 0, 1)]

    lake_model.outcomes = [ScalarOutcome("max_P", kind=ScalarOutcome.MINIMIZE, expected_range=(0, 5)),
                           ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 2)),
                           ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)),
                           ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1))]

    lake_model.constants = [Constant("alpha", 0.4),
                            Constant("nsamples", 100),
                            Constant("myears", 100)]

    n_scenario = 500

    robustness_function = [ScalarOutcome('fraction max_p', kind=ScalarOutcome.MAXIMIZE,
                             variable_name='max_P', function=maxp),
                       ScalarOutcome('fraction reliability', kind=ScalarOutcome.MAXIMIZE,
                             variable_name='reliability', function=reliability),
                       ScalarOutcome('fraction inertia', kind=ScalarOutcome.MAXIMIZE,
                             variable_name='inertia', function=inertia),
                       ScalarOutcome('fraction utility', kind=ScalarOutcome.MAXIMIZE,
                             variable_name='utility', function=utility)]
    convergence_metrics = [HyperVolume.from_outcomes(lake_model.outcomes),
                           EpsilonProgress()]

    SMALLER = "SMALLER"
    LARGER = "LARGER"
    max_p_robustness = []
    utility_robustness = []
    inertia_robustness = []
    reliability_robustness = []
    for scenarios in range(1, 500, 100):
        with MultiprocessingEvaluator(lake_model) as evaluator:
            results, outcomes = evaluator.perform_experiments(scenarios, policies=policies)

        x = samplers.MonteCarloSampler()
        y = x.sample(distribution='integer', params=(0, 500), size=100)

        max_p_outcomes = outcomes['max_P']
        utility_outcomes = outcomes['utility']
        inertia_outcomes = outcomes['inertia']
        reliability_outcomes = outcomes['reliability']
        max_p_robustness.append(robustness(SMALLER, 0.75, max_p_outcomes))
        utility_robustness.append(robustness(LARGER, 0.75, utility_outcomes))
        inertia_robustness.append(robustness(LARGER, 0.6, inertia_outcomes))
        reliability_robustness.append(robustness(LARGER, 0.99, reliability_outcomes))

    plt.plot(max_p_robustness)
    plt.plot(inertia_robustness)
    plt.plot(utility_robustness)
    plt.plot(reliability_robustness)
    plt.show()

    with MultiprocessingEvaluator(lake_model) as evaluator:
        archive, convergence = evaluator.robust_optimize(robustness_function, scenarios=n_scenario, policies=policies, nfe=10,
                                                      convergence=convergence_metrics, epsilons=[0.05, 0.05, 0.05, 0.05])

    output = archive.iloc[:, 100 : :]
    limits = parcoords.get_limits(output)
    limits.loc[0, :] = 0
    limits.loc[1, :] = 1

    axes = parcoords.ParallelAxes(limits)
    axes.plot(output)
    plt.show()

    fig, axes = plt.subplots(ncols=2, sharex=True)
    axes[0].plot(convergence.nfe, convergence.epsilon_progress)
    axes[1].plot(convergence.nfe, convergence.hypervolume)

    axes[0].set_xlabel('# nfe')
    axes[1].set_xlabel('# nfe')

    sns.despine()

    plt.show()