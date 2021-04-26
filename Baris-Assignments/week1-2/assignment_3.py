import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import (Model, RealParameter, TimeSeriesOutcome, perform_experiments,
                           ema_logging, SequentialEvaluator)

from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.connectors.pysd_connector import PysdModel

from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS

from ema_workbench.analysis.plotting import lines, Density

model_files_path = "model_files//"

def PredPrey(prey_birth_rate=0.025, predation_rate=0.0015, predator_efficiency=0.002,
             predator_loss_rate=0.06, initial_prey=50, initial_predators=20, dt=0.25, final_time=365, reps=1) :
    # Initial values
    predators, prey, sim_time = [np.zeros((reps, int(final_time / dt) + 1)) for _ in range(3)]

    for r in range(reps) :
        predators[r, 0] = initial_predators
        prey[r, 0] = initial_prey

        # Calculate the time series
        for t in range(0, sim_time.shape[1] - 1) :
            dx = (prey_birth_rate * prey[r, t]) - (predation_rate * prey[r, t] * predators[r, t])
            dy = (predator_efficiency * predators[r, t] * prey[r, t]) - (predator_loss_rate * predators[r, t])

            prey[r, t + 1] = max(prey[r, t] + dx * dt, 0)
            predators[r, t + 1] = max(predators[r, t] + dy * dt, 0)
            sim_time[r, t + 1] = (t + 1) * dt

    # Return outcomes
    return {'TIME' : sim_time,
            'predators' : predators,
            'prey' : prey}

og_model = Model('predprey', PredPrey)
excel_model = ExcelModel('predpreyexcel', wd='model_files', model_file="PredPrey.xlsx", default_sheet='Sheet1')
vensim_model = PysdModel('predpreyvensim', mdl_file="model_files/PredPrey.mdl")

list_of_models = [excel_model, vensim_model, og_model]

for model in list_of_models:

    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]


    if __name__ == '__main__':
        with SequentialEvaluator(model) as evaluator :
            experiments, outcomes = evaluator.perform_experiments(scenarios=50)
            print(experiments, outcomes)
