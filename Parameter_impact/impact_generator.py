# This file is to generate the test design parameters.
# Based on the first 300 simulation outputs.
# Training set parameter range:

# Material : Z9477
# tCu:          [10, 15, 20]
# wCu:          [200, 250]
# tLamCore:     [100, 150, 250, 350]
# Nlam:         [8, 12, 16]
# AlN:          [15, 20]
# tSu8:         [2, 4, 6]



import numpy as np
import pandas as pd

tCu_train = [10, 15, 20]
wCu_train = [200, 250]
tlamCore_train = [100, 150, 250, 350]
Nlam_train = [8, 12, 16]
AlN_train = [15, 20]
tSu8_train = [2, 4, 6]


tCu_test_50 = [5, 25, 30]
tCu_test_100 = [5, 25, 30, 35, 40]

wCu_test_50 = [100, 150, 300, 350, 400]
wCu_test_100 = [50, 100, 150, 300, 350, 400, 450, 500]

tlamCore_test_50 = [50, 200, 300, 400, 450, 500]
tlamCore_test_100 = [50, 200, 300, 400, 450, 500, 550, 600, 650, 700]

Nlam_test_50 = [4, 20, 24]
Nlam_test_100 = [4, 20, 24, 28, 32]

AlN_test_50 = [10, 25, 30]
AlN_test_100 = [5, 10, 25, 30, 35, 40]

tSu8_test_50 = [8]
tSu8_test_100 = [8, 10, 12]

train_set_parameters = [tCu_train, wCu_train, tlamCore_train, Nlam_train, AlN_train, tSu8_train]
test_parameters_50 = [tCu_test_50, wCu_test_50, tlamCore_test_50, Nlam_test_50, AlN_test_50, tSu8_test_50]
test_parameters_100 = [tCu_test_100, wCu_test_100, tlamCore_test_100, Nlam_test_100, AlN_test_100, tSu8_test_100]
########## 50%

def generate_50_percent():
    test_dataframe_50 = pd.DataFrame({
        "tCu" : [],
        "wCu": [],
        "tLam": [],
        "Nlam": [],
        "AlN": [],
        "tSu8": [],
    })
    # print(test_dataframe_50)
    for i in range(6):
        for param in test_parameters_50[i]:
            for j in range(3):
                test_dataframe_50.loc[len(test_dataframe_50)] = [None]*len(test_dataframe_50.columns)
                test_dataframe_50.iloc[len(test_dataframe_50)-1, i] = param
                for k in range(6):
                    if k != i:
                        test_dataframe_50.iloc[len(test_dataframe_50)-1, k] = np.random.choice(train_set_parameters[k])

    # print(test_dataframe_50)
    test_dataframe_50.to_csv("test_parameter_50.csv", index=False)


def generate_100_percent():
    test_dataframe_100 = pd.DataFrame({
        "tCu" : [],
        "wCu": [],
        "tLam": [],
        "Nlam": [],
        "AlN": [],
        "tSu8": [],
    })
    print(test_dataframe_100)
    for i in range(6):
        for param in test_parameters_100[i]:
            for j in range(3):
                test_dataframe_100.loc[len(test_dataframe_100)] = [None]*len(test_dataframe_100.columns)
                test_dataframe_100.iloc[len(test_dataframe_100)-1, i] = param
                for k in range(6):
                    if k != i:
                        test_dataframe_100.iloc[len(test_dataframe_100)-1, k] = np.random.choice(train_set_parameters[k])

    # print(test_dataframe_50)
    test_dataframe_100.to_csv("test_parameter_100.csv", index=False)

if __name__ == '__main__':
    generate_50_percent()
    generate_100_percent()