from PINN import data_processor

input_path_50 = "simulator_output/02 Data/RLQ_50"
input_path_100 = "simulator_output/02 Data/RLQ_100"
output_path_50 = "simulator_output_data/csv_50.csv"
output_path_100 = "simulator_output_data/csv_100.csv"

data_processor.data_processor(input_path_50, output_path_50).process_dir()

data_processor.data_processor(input_path_100, output_path_100).process_dir()