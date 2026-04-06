import data_process.ansys_integrator as data_processor

dp = data_processor.data_processor("RLQ/full_RLQ", "RLQ/full_dataset.csv")
dp.process_dir()