import model_predict
from model_predict import pinn_predict
from models.model_design import transformers_model
from models.model_design import transformer_inter_model
from models.model_design import pinn_model
from models.model_design import PINN_inter_model
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

    # return output_path

    real_path = "Parameter_impact/simulation_csv/csv_100.csv"

    compare_output = "Parameter_impact/data_compare/Compare_Transformers_100.csv"
    compare(output_path, real_path, compare_output)


def pinn_predict_test():
    model = pinn_model.PINN()
    model.load_state_dict(torch.load("saved_models/PINN_model.pth"))
    output_path = "Parameter_impact/data_compare/predicted/PINN_100.csv"
    input_path = "Parameter_impact/generated_parameters/freq_100.csv"

    model_predict.predict(model, input_path, output_path)

    real_path = "Parameter_impact/simulation_csv/csv_100.csv"

    compare_output = "Parameter_impact/data_compare/Compare_PINN_100.csv"
    compare(output_path, real_path, compare_output)

def transformer_inter_test():
    model = transformer_inter_model.PINNTransformer()
    model.load_state_dict(torch.load("saved_models/transformer_inter_model.pth"))
    output_path = "Parameter_impact/data_compare/predicted/Transformer_inter.csv"
    input_path = "training_csv/pinn_data.csv"

    model_predict.predict(model, input_path, output_path)

    real_path = "training_csv/pinn_data.csv"
    compare_output = "Parameter_impact/data_compare/Compare_Trans_inter_100.csv"
    compare(output_path, real_path, compare_output)


def pinn_inter_test():
    model = PINN_inter_model.PINN()
    model.load_state_dict(torch.load("saved_models/PINN_inter_model.pth"))
    output_path = "Parameter_impact/data_compare/predicted/pinn_inter.csv"
    input_path = "training_csv/pinn_data.csv"

    model_predict.predict(model, input_path, output_path)

    real_path = "training_csv/pinn_data.csv"
    compare_output = "Parameter_impact/data_compare/Compare_pinn_inter_100.csv"
    compare(output_path, real_path, compare_output)


if __name__ == "__main__":

    # transformer_inter_test()
    pinn_inter_test()