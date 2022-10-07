from __future__ import (unicode_literals, print_function, absolute_import,
                        division)


from ema_workbench import (Model, MultiprocessingEvaluator,
                           ScalarOutcome, IntegerParameter, optimize, Scenario, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (EpsilonProgress, HyperVolume)
from ema_workbench.util import ema_logging
from ema_workbench.analysis import parcoords

from problem_formulation import get_model_for_problem_formulation
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    model, steps = get_model_for_problem_formulation(3)
    reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
                        'discount rate 0': 3.5, 'discount rate 1': 3.5,
                        'discount rate 2': 3.5,
                        'ID flood wave shape': 4}
    scen1 = {}

    for key in model.uncertainties:
        name_split = key.name.split('_')

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **scen1)

    convergence_metrics = [EpsilonProgress()]

    #Total costs, Expected Deaths, RfR Total Costs, Evacuation Costs
    espilon = [1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e8, 1e4]

    nfe = 10000

    with MultiprocessingEvaluator(model) as evaluator:
        results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                  epsilons=espilon,
                                                  convergence=convergence_metrics,
                                                  reference=ref_scenario)

    #results.to_csv('first_optimization_results.csv')
    results.to_csv('third_optimization_results.csv', index=False)


    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    fig, ax1 = plt.subplots(ncols=1)
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_xlabel('nr. of generations')
    ax1.set_ylabel('$\epsilon$ progress')
    #ax2.plot(convergence.nfe, convergence.hypervolume)
    #ax2.set_ylabel('hypervolume')
    plt.show()
    #sns.despine()

    #Added this Code (Lucas)
    limits = limits = parcoords.get_limits(results)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(results)
    plt.show()