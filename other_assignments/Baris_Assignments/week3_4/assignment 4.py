from ema_workbench import util
from ema_workbench.util import utilities
from ema_workbench.util.utilities import load_results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ema_workbench import analysis
from ema_workbench.analysis import prim
f = pd.read_csv('data/Bryant et al 2010.csv')
f.head()


#Selecting the data as provided in the assignment for the dependent and independent variables
x = f.iloc[:, 2:11]
y = f.iloc[:, 15]
#Run the PRIM algorithm and select the first box it generates
prim_alg = prim.Prim(x, y, threshold=0.8, peel_alpha=0.1)
box1 = prim_alg.find_box()
#Shows the default tradeoff in this box between coverage and density
box1.show_tradeoff()
# plt.show()

#Inspect the 21st element of specified box
box1.inspect(21)
box1.inspect(21, style='graph')
# plt.show()
#Shows scatterplots with limits superimposed in red area
box1.select(21)
fig = box1.show_pairs_scatter()
# plt.show()

# f2 = pd.read_csv('data/Rozenberg et al 2014.csv')
# f2.head()


# x = f2.iloc[:, 0:7] #Independent variables
# y1 = f2.iloc[:, 7]  #SSP1
# y2 = f2.iloc[:, 8]  #SSP2
# y3 = f2.iloc[:, 9]  #SSP3
# y4 = f2.iloc[:, 10] #SSP4
# y5 = f2.iloc[:, 11] #SSP5
# prim_alg1 = prim.Prim(x, y1, threshold=0.8, peel_alpha=0.1)
# prim_alg2 = prim.Prim(x, y2, threshold=0.5, peel_alpha=0.1)
# prim_alg3 = prim.Prim(x, y3, threshold=0.8, peel_alpha=0.1)
# prim_alg4 = prim.Prim(x, y4, threshold=0.8, peel_alpha=0.1)
# prim_alg5 = prim.Prim(x, y5, threshold=0.3, peel_alpha=0.1)
#
# box1 = prim_alg1.find_box()
# box2 = prim_alg2.find_box()
# box3 = prim_alg3.find_box()
# box4 = prim_alg4.find_box()
# box5 = prim_alg5.find_box()


f3, outcomes = load_results('data/Hamarat et al 2013.gz')
data = outcomes['fraction renewables']
y = data[:, 0] < data[:, -1]

for col in f3.columns:
    data_type = f3[col].dtypes.name
    if data_type != "float64":
        try:
            f3[col] = f3[col].astype(int)
        except Exception as e:
            print(f"Could not convert column '{col}' to float --> {e}")

rotated_experiments, rotation_matrix = prim.pca_preprocess(f3, y, exclude=['model', 'policy'])

prim_obj = prim.Prim(rotated_experiments, y, threshold=0.8)
prim_alg1 = prim.Prim(f3, y, threshold=0.8, peel_alpha=0.1)
box = prim_obj.find_box()

box.show_tradeoff()
box.inspect(22)
plt.show()
