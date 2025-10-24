import numpy as np
import pandas as pd


predicted_50 = pd.read_csv("predicted_csv/predicted_50.csv").to_numpy()
predicted_100 = pd.read_csv("predicted_csv/predicted_100.csv").to_numpy()

simulated_50 = pd.read_csv("simulator_output/csv_50.csv").to_numpy()
simulated_100 = pd.read_csv("simulator_output/csv_100.csv").to_numpy()

####### 50% compare
