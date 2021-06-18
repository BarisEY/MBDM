# Imports:
# External
import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 6):
    # This section plots the number of deaths
    my_data = np.genfromtxt(f"..//final_data//A.{i}_Expected Number of Deaths.csv", delimiter=',')
    _ = plt.hist(my_data, bins='auto')  # arguments are passed to np.histogram
    plt.title(f"A.{i} Expected Number of Deaths Sensitivity Analysis")
    plt.show()

    # This section plots the total costs
    my_data = np.genfromtxt(f"..//final_data//A.{i} Total Costs.csv", delimiter=',')
    _ = plt.hist(my_data, bins='auto')  # arguments are passed to np.histogram
    plt.title(f"A.{i} Total Costs Sensitivity Analysis")
    plt.show()

# This section plots the evacuation costs
my_data = np.genfromtxt(f"..//final_data//Expected Evacuation Costs.csv", delimiter=',')
_ = plt.hist(my_data, bins='auto')  # arguments are passed to np.histogram
plt.title(f"Expected Evacuation Costs")
plt.show()

# This section plots the RfR Total costs
my_data = np.genfromtxt(f"..//final_data//RfR Total Costs.csv", delimiter=',')
_ = plt.hist(my_data, bins='auto')  # arguments are passed to np.histogram
plt.title(f"RfR Total Costs")
plt.show()