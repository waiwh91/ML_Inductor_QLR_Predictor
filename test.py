import pandas
import numpy as np
file_path = "RLQ/Z9477_tCu10um_wCu200um_tLamCore100nm_Nlam8_tAlN20nm_tSu82nm_RLQ.csv"

csv = pandas.read_csv(file_path)
matrix = csv.to_numpy()
print(matrix[1,0])

