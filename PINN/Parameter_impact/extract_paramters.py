import data_process.ansys_integrator as ansys_integrator

input_path_50 = "simulator_output/02 Data/RLQ_50"
input_path_100 = "simulator_output/02 Data/RLQ_100"
output_path_50 = "simulator_output_data/csv_50.csv"
output_path_100 = "simulator_output_data/csv_100.csv"

ansys_integrator.data_processor(input_path_50, output_path_50).process_dir()
ansys_integrator.data_processor(input_path_100, output_path_100).process_dir()