# import libraries
import numpy as np
import pandas as pd
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.backends import cudnn
import torchvision.transforms as T


# import helper functions
# REPO_PATH = Path().cwd()
# sys.path.append(f"{REPO_PATH}/utilities")
from utilities.utils import *
from utilities.pcaFunctions import *
from utilities.scaleFunctions import *
from utilities.pathManager import fetchPaths


def train_models(model, dataset, variation, train_data, test_data, model_setup, model_config):
    """
    Train models for a variation
    Export Last hidden layer data and apply pca
    """

    # get divice
    GPU_NAME = "cuda:0"
    device = get_device(GPU_NAME)
    torch.cuda.get_device_name(device)

    # root paths
    base = Path().cwd()
    SEED = 42

    # get essential paths
    paths = fetchPaths(base, dataset)
    path = paths[dataset.lower()]
    path_dataset = paths["dataset"]
    path_lastHiddenLayer = paths["lastHiddenLayer"]
    path_lastHiddenLayer_raw = paths["lastHiddenLayer_raw"]
    path_lastHiddenLayer_pca = paths["lastHiddenLayer_pca"]
    path_savedModels = paths["savedModels"]

    # construct full model variation to append to paths
    FILENAME = f"{dataset}_{variation}"


    # append full model variation

    # save last hidden layer - raw
    path_lastHiddenLayer_raw = path_lastHiddenLayer_raw / FILENAME
    path_lastHiddenLayer_raw.mkdir(exist_ok=True)

    # save model and logs
    path_savedModels = path_savedModels / FILENAME
    path_savedModels.mkdir(exist_ok=True)

    path_lastHiddenLayer_pca = path_lastHiddenLayer_pca / FILENAME
    path_lastHiddenLayer_pca.mkdir(exist_ok=True)

    # save last hidden layer - pca
    path_lastHiddenLayer_pca_single = path_lastHiddenLayer_pca / "Single"
    path_lastHiddenLayer_pca_single.mkdir(exist_ok=True)

    path_lastHiddenLayer_pca_classes = path_lastHiddenLayer_pca / "Classes"
    path_lastHiddenLayer_pca_classes.mkdir(exist_ok=True)

    # load dataset labels
    feature_names = get_labels(dataset)

    # skip train if model already exist
    if len(list(path_savedModels.glob('*.pth.tar'))) != 0:
        print(f'[ Model Founded: {FILENAME} ]')
    else:
        # dataloaders
        trainloader = get_dataLoader(train_data, model_config["batch_size"], True)
        testloader = get_dataLoader(test_data, model_config["batch_size"], False)

        # define model
        model_ = model(**model_setup)
        model_.to(device)

        # allow parallelism multi GPU if needed
        nn.DataParallel(model_, device_ids=[0])
        # fix comutation graph
        cudnn.benchmark = True

        # cost function
        loss_function = nn.CrossEntropyLoss()

        # optimizer - pass text 'Adam' or 'SGD'
        optimizer = getattr(torch.optim, model_config["optim"])(
            model_.parameters(), lr=model_config["lr"]
        )

        # LR Scheduler
        lr_scheduler = None

        # train if not exist
        kwargs = {
            "model": model_,
            "loss_function": loss_function,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "map_classes": None,
            "skip_classes": None,
            "device": device,
            "model_path": path_savedModels / f"{dataset}_{variation}.pth.tar",
            "trainloader": trainloader,
            "testloader": testloader,
            "config": model_config,
        }


        # train
        (
            *_, # ignore some returned values
            confusion_matrix_test,
            best_model_name,
        ) = run_training_testing(**kwargs)

        # Load best model
        model_ = model(**model_setup)
        model_.to(device)
        model_file_name = best_model_name
        load_checkpoint(model_, model_file_name)

        # export last hidden layer for each data loader
        for loader, stage in [
            (trainloader, "train"),
            (testloader, "test"),
        ]:
            export_last_hidden_layer(
                loader,
                model_,
                device,
                model_.last_hidden_neurons,
                None,
                path_lastHiddenLayer_raw,
                FILENAME,
                stage,
            )

        # save confusion matrix plot and results
        save_confusion_matrix(
            confusion_matrix_test, path_savedModels, FILENAME, "test"
        )
        confusion_matrix_test_norm = normalize_confusion_matrix(confusion_matrix_test)
        plot_confusion_matrix(
            confusion_matrix_test_norm,
            feature_names=feature_names,
            fmt=".2f",
            save_path=path_savedModels,
            prefix=FILENAME,
            stage="test",
        )
        # finish training


    # apply pca and extract neurons
    df = pd.read_csv(path_lastHiddenLayer_raw / f"{FILENAME}_train.csv")
    df_true = df[df["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)
    df_test = pd.read_csv(path_lastHiddenLayer_raw / f"{FILENAME}_test.csv")

    # PCA Single
    NUM_NEURONS = df.shape[1]-2
    CLASSES_COLUMN = "y"

    if ((path_lastHiddenLayer_pca_single / f"{FILENAME}_train.csv").is_file()
    and (path_lastHiddenLayer_pca_single / f"{FILENAME}_test.csv").is_file()):
        print(f"[ Last Hidden Layer PCA Single Founded: {FILENAME} ]")
    else:
        # fit scaler and pca
        scaler_single = fitStandardScalerSingle(df, NUM_NEURONS)
        pca_single = fitPCASingle(df, scaler_single, NUM_NEURONS)

        save_pickle(path_lastHiddenLayer_pca_single / 'scaler_single.pkl', scaler_single)
        save_pickle(path_lastHiddenLayer_pca_single / 'pca_single.pkl', pca_single)

        # apply
        df_pca_train_single = applyPCASingle(pca_single, df, scaler_single, NUM_NEURONS)
        df_pca_test_single = applyPCASingle(pca_single, df_test, scaler_single, NUM_NEURONS)
        # export data
        df_pca_train_single.to_csv(
            path_lastHiddenLayer_pca_single / f"{FILENAME}_train.csv", index=False
        )
        df_pca_test_single.to_csv(
            path_lastHiddenLayer_pca_single / f"{FILENAME}_test.csv", index=False
        )
        # export important neurons
        neurons_single = neuronsLoadingsSingle(
            pca_single, NUM_NEURONS, var_thld=0.9, loadings_thld=0.5
        )
        save_json(
            path_lastHiddenLayer_pca_single / f"{FILENAME}_neurons.json",
            neurons_single,
        )


    # PCA Classes
    if ((path_lastHiddenLayer_pca_classes / f"{FILENAME}_train.csv").is_file()
        and (path_lastHiddenLayer_pca_classes / f"{FILENAME}_test.csv").is_file()):
        print(f"[ Last Hidden Layer PCA Classes Founded: {FILENAME} ]")
    else:
        # fit scaler and pca
        scaler_classes = fitScalerClasses(df, CLASSES_COLUMN, NUM_NEURONS)
        pca_classes = fitPCAClasses(df, CLASSES_COLUMN, scaler_classes, NUM_NEURONS)

        save_pickle(path_lastHiddenLayer_pca_classes / 'scaler_classes.pkl', scaler_classes)
        save_pickle(path_lastHiddenLayer_pca_classes / 'pca_classes.pkl', pca_classes)

        # apply
        df_pca_train_classes = applyPCAClasses(
            pca_classes, df, CLASSES_COLUMN, scaler_classes
        )
        df_pca_test_classes = applyPCAClasses(
            pca_classes, df_test, CLASSES_COLUMN, scaler_classes
        )
        # export data
        df_pca_train_classes.to_csv(
            path_lastHiddenLayer_pca_classes / f"{FILENAME}_train.csv", index=False
        )
        df_pca_test_classes.to_csv(
            path_lastHiddenLayer_pca_classes / f"{FILENAME}_test.csv", index=False
        )
        # export important neurons
        neurons_classes = neuronsLoadingsClasses(
            pca_classes, NUM_NEURONS, var_thld=0.9, loadings_thld=0.5
        )
        save_json(
            path_lastHiddenLayer_pca_classes / f"{FILENAME}_neurons.json",
            neurons_classes,
        )


if __name__ == "__main__":

    REPO_PATH = Path().cwd()
    DATASET = 'MNIST'

    sys.path.append(f"{REPO_PATH}/utilities")
    sys.path.append(f"{REPO_PATH}/{DATASET}/trainingModels")

    from pathManager import fetchPaths
    # import model
    from MNIST_Model import MNIST_Model
    model = MNIST_Model

    base = Path(REPO_PATH)
    paths = fetchPaths(base, DATASET)
    path = paths[DATASET.lower()]
    path_dataset = paths["dataset"]


    # transformers
    tf_train = T.Compose([T.ToTensor(), T.Normalize((0.1307), (0.3015))])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.1307), (0.3015))])

    # get dataset
    train_data = get_dataset(DATASET, path_dataset, train=True, transform=tf_train)
    test_data = get_dataset(DATASET, path_dataset, train=False, transform=tf_test)

    # basic shared configuration for all models
    model_setup = {
        "batchnorm": True,
        "dropout": 0.0,
        "first_layer_norm": False,
    }
    model_config = {
        "optim": "Adam",
        "batch_size": 256,
        "lr": 0.01,
        "epochs": 10,
        "patience": 2,
        "L2": 0.0,
        "L1": 0.0,
    }
    # model varient variation
    models_name = f"{model_config['optim']}-{model_config['batch_size']}"
    # neurons
    lhl_neurons = [30, 60, 100]

    # # # train model for each number of neurons
    for n in lhl_neurons:
        # set last_hidden_neurons for model
        model_setup["last_hidden_neurons"] = n
        train_models(model, DATASET, f"{models_name}-{n}", train_data, test_data, model_setup, model_config)