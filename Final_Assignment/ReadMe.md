# Final Assignment Repository

This repository contains the codes that was used in the report of Group 30. All the relevant codes are located in the
[Final Assignments Code](Final_Assignment_code) folder. The saved files generated and used by these codes are in the
[final data](final_data) folder.
A general outline of the code is as follows:

### Optimization of Initial Model 
First, the optimization was run using the optimization file provided in the GitHub repository of EPA1361, namely: 
[optimization file](Final_Assignment_code/quaquel_github_files_kopie/dike_model_optimization.py). This file has been 
slightly adjusted to meet the specification of the client and was then run. Its output was saved as a csv in the 
[final_data](final_data) folder. The optimization file was three times and there are thus three optimization result 
files: [first optimization](final_data/first_optimization_results.csv), 
[second optimization](final_data/second_optimization_results.csv) and 
[third optimization](final_data/third_optimization_results.csv).

### Data exploration
These results were then used to explore the data, with the first result being used in [this file](Final_Assignment_code/Data_exploration.ipynb)
and the second set of results in [this file](Final_Assignment_code/Data_Exploration2.ipynb). 

This allowed for the selection of certain policies that were deemed best suited for the client.

### Policy Selection and Scenario discovery
The policy selection, which was based on the previous data exploration, together with the scenario discovery was run in 
the [python file](Final_Assignment_code/data_exploration_2.py). This was done, as the final run had to be run using the
`MultiprocessingEvaluator`, which is easier done in an IDE like Pycharm. 

The Epsilon and convergence metrics used for the last run were the same as the initial run of optimizing the initial
model. 

### Deep Uncertainty Analysis
A [deep uncertainty](Final_Assignment_code/deep_uncertainty.py) analysis was conducted on all of the results found in 
the previous step. This was done by loading the results from the previous step, creating new Policies from certain 
selected outcomes and then running the model with all the new policies with an NFE of 1000.

### Final Policy Selection
The [final policy selection file](Final_Assignment_code/final_policy_selection.py) was used to filter out the best 
policy that performed the best under the 1000 scenarios that each policy was run for. Using the regret function and 
the absolute values of the outcomes, a policy had been chosen and its performance across the 1,000 runs had been 
plotted on a parcoords plot.

### Sensitivity Analysis
Finally, a [sensitivity analysis](Final_Assignment_code/final_policy_selection.py) (at the end of the file) had been 
conducted on the final 
chosen policy, using the `SOBOL` method. The [plot sensitivity file](Final_Assignment_code/plot_sensitivity.py) was
used to plot the final sensitivity analysis in order to visualize the results.