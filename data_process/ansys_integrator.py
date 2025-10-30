import os

import pandas
import pandas as pd

class data_processor:
    def __init__(self, file_path, output_path = "data.csv"):
        self.file_path = file_path
        self.output_path = output_path

    def process_dir(self):
        files = os.listdir(self.file_path)

        # print(files)
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
            tSu8_index = file.find("tSu8")



            #################### pinn_RLQ Data processing, with 5 freqs
            csv_data = pandas.read_csv(f"{self.file_path}/{file}")
            ohm_flag = csv_data.columns[2]
            csv_data = csv_data.to_numpy()

            for i in range(5):
                tCu.append(file[tCu_index + 3: wCu_index - 3])
                wCu.append(file[wCu_index + 3: tLamCore_index - 3])
                tLamCore.append(file[tLamCore_index + 8: Nlam_index - 3])
                Nlam.append(file[Nlam_index + 4: tAlN_index - 1])
                tAlN.append(file[tAlN_index + 4: tSu8_index - 3])
                tSu.append(file[tSu8_index + 4: len(file) - 10])
                freq.append(csv_data[i,0])
                output_q.append(csv_data[i,1])
                if ohm_flag == "re(Matrix1.Z(Winding,Winding)) [mOhm]":

                    output_r.append(csv_data[i,2])
                elif ohm_flag == "re(Matrix1.Z(Winding,Winding)) [ohm]":

                    output_r.append(csv_data[i, 2]*1000)
                output_l.append(csv_data[i,3])


        data_frame = pd.DataFrame({"tCu": tCu, "wCu": wCu, "tLamCore": tLamCore, "Nlam": Nlam, "AlN": tAlN, "tSu": tSu, "freq": freq, "Q":output_q, "R":output_r, "L":output_l})
        data_frame.to_csv(self.output_path, index = False)
