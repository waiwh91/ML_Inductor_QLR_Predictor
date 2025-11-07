from data_process import ansys_integrator

processor = ansys_integrator.data_processor("simulation_data/02 Data/RLQ_100", "simulation_csv/new.csv")
processor.process_dir()