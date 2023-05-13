config = {
    "dataset": "MNIST",
    "data_folder": "data",
    "log_folder": "logs",
    "prefix": "TestingRegulariztion",
    "seed": 1234
}

model_config = {
    "batch_size": 128,
    "lr": 0.01,
    "epochs": 5,
    "testing_params": True,
    "patience": 0,
    "lambd": 0,
    "alpha": 0,
    "dropout": 0.2,
}