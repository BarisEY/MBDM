import math
import matplotlib.pyplot as plt

from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis import dimensional_stacking
import ema_workbench
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from SALib.analyze import sobol

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator, Policy, SequentialEvaluator)
from ema_workbench.em_framework.evaluators import SOBOL
from ema_workbench.em_framework import get_SALib_problem

from lakemodel_function import lake_problem
import seaborn as sns

# def analyze(results, ooi):
#     '''analyze results using SALib sobol, returns a dataframe
#
#     '''
#
#     _, outcomes = results
#
#     problem = get_SALib_problem(lake_model.uncertainties)
#     y = outcomes[ooi]
#     sobol_indices = sobol.analyze(problem, y)
#     sobol_stats = {key: sobol_indices[key] for key in ['ST', 'ST_conf', 'S1',
#                                                        'S1_conf']}
#     sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
#     sobol_stats.sort_values(by='ST', ascending=False)
#     s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
#                       columns=problem['names'])
#     s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'],
#                            columns=problem['names'])
#
#     return sobol_stats, s2, s2_conf

ema_logging.log_to_stderr(ema_logging.INFO)

# instantiate the model
lake_model = Model('lakeproblem', function=lake_problem)
lake_model.time_horizon = 100

# specify uncertainties
lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                            RealParameter('q', 2.0, 4.5),
                            RealParameter('mean', 0.01, 0.05),
                            RealParameter('stdev', 0.001, 0.005),
                            RealParameter('delta', 0.93, 0.99)]

# set levers, one for each time step
lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in
                     range(lake_model.time_horizon)]

# specify outcomes
lake_model.outcomes = [ScalarOutcome('max_P'),
                       ScalarOutcome('utility'),
                       ScalarOutcome('inertia'),
                       ScalarOutcome('reliability')]

# override some of the defaults of the model
lake_model.constants = [Constant('alpha', 0.41),
                        Constant('nsamples', 150)]

# generate sa single default no release policy
policies = [Policy('no release'  , **{'l'+str(i): 0.01 for i in range(100)}),
          Policy('some release', **{'l'+str(i): 0.05 for i in range(100)}),
          Policy('more release', **{'l'+str(i): 0.1  for i in range(100)})]

n_scenarios = 1000

if __name__ == '__main__':

    for policy in policies:
        with MultiprocessingEvaluator(lake_model) as evaluator:
            results = evaluator.perform_experiments(n_scenarios, policies=policy,
                                                    uncertainty_sampling=SOBOL)

            # sobol_stats, s2, s2_conf = analyze(results, 'reliability')
            # print(sobol_stats)
            # print(s2)
            # print(s2_conf)

        x, y = results
        x = x.iloc[:, :8]
        fs = feature_scoring.get_feature_scores_all(x, y)
        sns.heatmap(fs, cmap='viridis', annot=True)
        plt.show()