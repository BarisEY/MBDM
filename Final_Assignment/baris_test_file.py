import quaquel_github_files.problem_formulation as pf
from Lucas_Final_Assignment_code import quaquel_github_files_kopie as luc_qa
import Lucas_Final_Assignment_code.quaquel_github_files_kopie.problem_formulation as luc_pf

import ema_workbench

if __name__ == '__main__':

    y, z = luc_pf.get_model_for_problem_formulation(30)
    a,b = pf.get_model_for_problem_formulation(3)

    with ema_workbench.SequentialEvaluator(y) as evaluator:
        q, w = evaluator.perform_experiments(scenarios=10)

z = 1

# A.1Dike Investment Costs 0 not found in model output
# A.2Dike Investment Costs 0 not found in model output
# A.1Dike Investment Costs 1 not found in model output
# A.2Dike Investment Costs 1 not found in model output
# A.1Dike Investment Costs 2 not found in model output
# A.2Dike Investment Costs 2 not found in model output