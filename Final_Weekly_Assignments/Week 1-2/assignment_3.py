import numpy as np
import matplotlib.pyplot as plt

from ema_workbench import (Model, RealParameter, TimeSeriesOutcome, perform_experiments,
                           ema_logging, SequentialEvaluator, MultiprocessingEvaluator)

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


def run_excel_model():
    model = ExcelModel('predpreyexcel', wd='model_files', model_file="PredPrey.xlsx", default_sheet='Sheet1')
    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    model.default_sheet = "Sheet1"

    # Can also use "MultiprocessingEvaluator" for this.
    return perform_experiments(model, 50, reporting_interval=0.25)

def run_vensim_model():
    model = PysdModel('predpreyvensim', mdl_file="model_files/PredPrey.mdl")
    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    return perform_experiments(model, 50)

def run_netlogo_model():
    model = NetLogoModel('predpreynl', wd='model_files', model_file="PredPrey.nlogo")
    model.run_length = 365
    model.replications = 50

    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    with MultiprocessingEvaluator(model, n_processes=4, maxtasksperchild=8) as evaluator:
        results = evaluator.perform_experiments(model.replications)

    return results

def run_python_model():
    model = Model('predpreypy', function=PredPrey)

    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=50)

    return results


ema_logging.log_to_stderr(level=ema_logging.INFO)
# ema_model = Model('predprey', PredPrey)
if __name__ == '__main__':

    x, y = run_vensim_model()
    a, b = run_python_model()
    # run_netlogo_model() #TODO: There is a problem with the package for this code, comment it out if it does not work for you.
    # run_excel_model()

q = 1