import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import data_process.ansys_integrator as data_processor
from model.model_design import pinn
from model import spliter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = data_processor.data_processor("RLQ/pinn_RLQ", "pinn_data.csv")
processor.process_dir()


