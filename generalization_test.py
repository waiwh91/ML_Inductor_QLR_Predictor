import model_predict
from models.model_design import transformers_model
from data_process import ansys_integrator
from data_process.output_compare import compare

import torch
def process_data():

    data_processor = ansys_integrator.data_processor("Parameter_impact/simulation_data/02 Data/RLQ_100", "Parameter_impact/simulation_csv/100.csv")
    data_processor.process_dir()
    return data_processor.output_path

def transformers_predict_test():
    model = transformers_model.PINNTransformer()
    model.load_state_dict(torch.load("saved_models/PINNtransformers_model.pth"))
    output_path = "Parameter_impact/data_compare/predicted/Transformers_100.csv"
    input_path = "Parameter_impact/generated_parameters/freq_100.csv"

    model_predict.predict(model, input_path, output_path)

    return output_path

if __name__ == "__main__":
    real_path = "Parameter_impact/simulation_csv/csv_100.csv"
    predicted_path = transformers_predict_test()
    compare_output = "Parameter_impact/data_compare/Compare_Transformers_100.csv"
    compare(predicted_path, real_path, compare_output)