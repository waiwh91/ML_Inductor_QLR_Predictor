import os

import pandas
import pandas as pd

class data_processor:
    def __init__(self, file_path):
        self.file_path = file_path


    def process_dir(self):
        files = os.listdir(self.file_path)
        print(files)
        tCu = []
        wCu = []
        tLamCore = []
        Nlam = []
        tAlN = []
        tSu = []
        freq = []

        empty = []

        output_r = []
        output_l = []
        output_q = []
        for file in files:
            ##################### Files Name Processing
            tCu_index = file.find("tCu")
            wCu_index = file.find("wCu")
            tLamCore_index = file.find("tLamCore")
            Nlam_index = file.find("Nlam")
            tAlN_index = file.find("tAlN")
            tSu_index = file.find("tSu")



            #################### RLQ Data processing, with 5 freqs
            csv_data = pandas.read_csv(f"RLQ/{file}").to_numpy()
            for i in range(5):
                tCu.append(file[tCu_index + 3: wCu_index - 3])
                wCu.append(file[wCu_index + 3: tLamCore_index - 3])
                tLamCore.append(file[tLamCore_index + 8: Nlam_index - 3])
                Nlam.append(file[Nlam_index + 4: tAlN_index - 1])
                tAlN.append(file[tAlN_index + 4: tSu_index - 3])
                tSu.append(file[tSu_index + 3: len(file) - 10])
                freq.append(csv_data[i,0])
                output_q.append(csv_data[i,1])
                output_r.append(csv_data[i,2])
                output_l.append(csv_data[i,3])


        data_frame = pd.DataFrame({"tCu": tCu, "wCu": wCu, "tLamCore": tLamCore, "Nlam": Nlam, "AlN": tAlN, "tSu": tSu, "freq": freq, "Q":output_q, "R":output_r, "L":output_l})
        data_frame.to_csv('data.csv', index = False)
