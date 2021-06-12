from ema_workbench import (Model, RealParameter, ScalarOutcome)
from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt
from ema_workbench import Policy
import numpy as np
import pandas as pd
from ema_workbench.analysis import parcoords
import seaborn as sns
from collections import defaultdict
from ema_workbench.analysis import prim





if __name__ == '__main__':

    from Week5_6.dps_lake_model import lake_model

    model = Model('lakeproblem', function=lake_model)

    #specify uncertainties
    model.uncertainties = [RealParameter('b', 0.1, 0.45),
                           RealParameter('q', 2.0, 4.5),
                           RealParameter('mean', 0.01, 0.05),
                           RealParameter('stdev', 0.001, 0.005),
                           RealParameter('delta', 0.93, 0.99)]

    # set levers
    model.levers = [RealParameter("c1", -2, 2),
                    RealParameter("c2", -2, 2),
                    RealParameter("r1", 0, 2),
                    RealParameter("r2", 0, 2),
                    RealParameter("w1", 0, 1)]

    #specify outcomes
    # note how we need to explicitely indicate the direction
    model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                      ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                      ScalarOutcome('inertia', kind=ScalarOutcome.MAXIMIZE),
                      ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

    ema_logging.log_to_stderr(ema_logging.INFO)

    with MultiprocessingEvaluator(model) as evaluator:
        results1 = evaluator.optimize(nfe=500, searchover='levers',
                                     epsilons=[0.1,]*len(model.outcomes))

    with MultiprocessingEvaluator(model) as evaluator:
        results2 = evaluator.optimize(nfe=500, searchover='levers',
                                     epsilons=[0.01,]*len(model.outcomes))

    data = results1.loc[:, [o.name for o in model.outcomes]]
    limits = parcoords.get_limits(data)
    limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(data)
    paraxes.invert_axis('max_P')
    # plt.show()

    data = results2.loc[:, [o.name for o in model.outcomes]]
    limits = parcoords.get_limits(data)
    limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(data)
    paraxes.invert_axis('max_P')
    # plt.show()

    from ema_workbench.em_framework.optimization import (HyperVolume,
                                                         EpsilonProgress)

    convergence_metrics = [HyperVolume(minimum=[0, 0, 0, 0], maximum=[3, 2, 1.01, 1.01]),
                           EpsilonProgress()]

    with MultiprocessingEvaluator(model) as evaluator :
        results, convergence = evaluator.optimize(nfe=100, searchover='levers',
                                                  convergence=convergence_metrics,
                                                  epsilons=[0.1, ] * len(model.outcomes))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8, 4))
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel('$\epsilon$-progress')
    ax2.plot(convergence.nfe, convergence.hypervolume)
    ax2.set_ylabel('hypervolume')

    ax1.set_xlabel('number of function evaluations')
    ax2.set_xlabel('number of function evaluations')
    # plt.show()

    convergence_metrics = [HyperVolume(minimum=[0, 0, 0, 0], maximum=[3, 2, 1.01, 1.01]),
                           EpsilonProgress()]

    with MultiprocessingEvaluator(model) as evaluator :
        results, convergence = evaluator.optimize(nfe=200, searchover='levers',
                                                  convergence=convergence_metrics,
                                                  epsilons=[0.1, ] * len(model.outcomes))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8, 4))
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel('$\epsilon$-progress')
    ax2.plot(convergence.nfe, convergence.hypervolume)
    ax2.set_ylabel('hypervolume')

    ax1.set_xlabel('number of function evaluations')
    ax2.set_xlabel('number of function evaluations')
    # plt.show()

    data = results.loc[:, [o.name for o in model.outcomes]]
    limits = parcoords.get_limits(data)
    limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(data)
    paraxes.invert_axis('max_P')
    # plt.show()

    logical = results.reliability > 0.9

    policies = results[logical]
    policies = policies.drop([o.name for o in model.outcomes], axis=1)


    policies_to_evaluate = []

    for i, policy in policies.iterrows() :
        policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))

    n_scenarios = 100
    with MultiprocessingEvaluator(model) as evaluator :
        results = evaluator.perform_experiments(n_scenarios,
                                                policies_to_evaluate)

    def s_to_n(data, direction) :
        mean = np.mean(data)
        std = np.std(data)

        if direction == ScalarOutcome.MAXIMIZE :
            return mean / std
        else :
            return mean * std


    experiments, outcomes = results

    overall_scores = {}
    for policy in np.unique(experiments['policy']) :
        scores = {}

        logical = experiments['policy'] == policy

        for outcome in model.outcomes :
            value = outcomes[outcome.name][logical]
            sn_ratio = s_to_n(value, outcome.kind)
            scores[outcome.name] = sn_ratio
        overall_scores[policy] = scores
    scores = pd.DataFrame.from_dict(overall_scores).T


    data = scores
    limits = parcoords.get_limits(data)
    limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(data)
    paraxes.invert_axis('max_P')
    # plt.show()


    def calculate_regret(data, best) :
        return np.abs(best - data)


    experiments, outcomes = results

    overall_regret = {}
    max_regret = {}
    for outcome in model.outcomes :
        policy_column = experiments['policy']

        # create a DataFrame with all the relevent information
        # i.e., policy, scenario_id, and scores
        data = pd.DataFrame({outcome.name : outcomes[outcome.name],
                             "policy" : experiments['policy'],
                             "scenario" : experiments['scenario']})

        # reorient the data by indexing with policy and scenario id
        data = data.pivot(index='scenario', columns='policy')

        # flatten the resulting hierarchical index resulting from
        # pivoting, (might be a nicer solution possible)
        data.columns = data.columns.get_level_values(1)

        # we need to control the broadcasting.
        # max returns a 1d vector across scenario id. By passing
        # np.newaxis we ensure that the shape is the same as the data
        # next we take the absolute value
        #
        # basically we take the difference of the maximum across
        # the row and the actual values in the row
        #
        outcome_regret = (data.max(axis=1)[:, np.newaxis] - data).abs()

        overall_regret[outcome.name] = outcome_regret
        max_regret[outcome.name] = outcome_regret.max()

    max_regret = pd.DataFrame(max_regret)
    sns.heatmap(max_regret / max_regret.max(), cmap='viridis', annot=True)
    # plt.show()

    colors = sns.color_palette()

    data = max_regret

    # makes it easier to identify the policy associated with each line
    # in the parcoords plot
    # data['policy'] = data.index.astype("float64")

    limits = parcoords.get_limits(data)
    limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

    paraxes = parcoords.ParallelAxes(limits)
    for i, (index, row) in enumerate(data.iterrows()) :
        paraxes.plot(row.to_frame().T, label=str(index), color=colors[i])
    paraxes.legend()

    # plt.show()

    policy_regret = defaultdict(dict)
    for key, value in overall_regret.items() :
        for policy in value :
            policy_regret[policy][key] = value[policy]

    # this generates a 2 by 2 axes grid, with a shared X and Y axis
    # accross all plots
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10),
                             sharey=True, sharex=True)

    # to ensure easy iteration over the axes grid, we turn it
    # into a list. Because there are four plots, I hard coded
    # this.
    axes = [axes[0, 0], axes[0, 1],
            axes[1, 0], ]

    # zip allows us to zip together the list of axes and the list of
    # key value pairs return by items. If we iterate over this
    # it returns a tuple of length 2. The first item is the ax
    # the second items is the key value pair.
    for ax, (policy, regret) in zip(axes, policy_regret.items()) :
        data = pd.DataFrame(regret)

        # we need to scale the regret to ensure fair visual
        # comparison. We can do that by divding by the maximum regret
        data = data / max_regret.max(axis=0)
        sns.boxplot(data=data, ax=ax)

        # removes top and left hand black outline of axes
        sns.despine()

        # ensure we know which policy the figure is for
        ax.set_title(str(policy))
    # plt.show()


    x = experiments.drop(columns=['policy', 'c1', 'c2', 'r1', 'r2', 'w1'])
    y = outcomes['utility'] < 0.35

    prim_alg = prim.Prim(x, y, threshold=0.5)
    box = prim_alg.find_box()

    box.select(3)