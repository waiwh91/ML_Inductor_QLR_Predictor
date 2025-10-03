import pandas as pd
import torch
import trainer
import pandas as ps
import numpy as np



########## Generate Random Data
tcu_choice = [5,10,15,20,25]
wcu_choice = [175,200,225,250,275]
tlam_choice = [50, 100, 150, 200, 250, 350, 400]
nlam_choice = [6, 8, 10, 12, 14, 16, 18]
aln_choice = [10, 15, 20, 25]
tsu_choice = [80, 82, 84, 86, 88]



df = pd.DataFrame({
    "tCu": np.random.choice(tcu_choice, size=20),
    "wCu": np.random.choice(wcu_choice, size=20),
    "tLam": np.random.choice(tlam_choice, size=20),
    "nlam": np.random.choice(nlam_choice, size=20),
    "aln": np.random.choice(aln_choice, size=20),
    "tsu": np.random.choice(tsu_choice, size=20),

})

df.to_csv("simulation_test/output_nofreq.csv", index=False)
data = df.copy().to_numpy()



data = np.repeat(data, 4, axis=0)

x, y = data.shape

test_data = np.zeros([x,y+1])

for i in range(1, data.shape[0] + 1):
    if i % 5 == 1:
        test_data[i-1] = np.append(data[i-1], np.array([1]), axis=0)
    elif i % 5 == 0:
        test_data[i-1] = np.append(data[i-1], np.array([100]), axis=0)
    else:
        test_data[i-1] = np.append(data[i-1], np.array([( (i-1) % 5 ) * 25.25]), axis=0)



output_pd = pd.DataFrame({"tCu": test_data[:,0],"wCu": test_data[:,1],"tLam":test_data[:,2], "nLam": test_data[:,3], "aln": test_data[:,4], "tsu": test_data[:,5], "freq": test_data[:,6]})
output_pd.to_csv("simulation_test/simulation_data.csv", index=False)