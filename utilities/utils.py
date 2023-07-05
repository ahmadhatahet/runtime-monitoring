from fastprogress import master_bar, progress_bar
from datetime import datetime as dt
import numpy as np
from pandas import read_csv
import pickle
import json

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as T
from torchvision import datasets


from utilities.plotFunctions import *
from utilities.GTSRB_Custom import GTSRB


def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_json(filename, obj):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def write_bdd_stats(msg, path):
    with path.open("a") as f:
        f.write(f'{",".join([*map(str, msg)])}\n')


def formate_datetime(date_time):
    return dt.strftime(date_time, r"%H:%M:%S")


def csv_row(string_list):
    return ",".join(map(str, string_list)) + "\n"


def start_wrtier(model_file_path, epoch_file_path):
    csv_model_headers = [
        "optimizer",
        "scheduler",
        "regularization",
        "L2",
        "L1",
        "dropout",
        "batch_size",
        "epochs",
        "test_acc",
        "test_loss",
        "start_time",
        "end_time",
        "total_time",
    ]

    csv_epoch_headers = [
        "regularization",
        "L2",
        "L1",
        "dropout",
        "epoch",
        "iteration",
        "lr",
        "stage",
        "loss",
        "acc",
        "start_time",
        "end_time",
        "total_time",
    ]

    model_file, epoch_file = None, None

    if model_file_path is not None:
        model_file = model_file_path.open("w+", newline="")
        model_file.write(csv_row(csv_model_headers))

    if epoch_file_path is not None:
        epoch_file = epoch_file_path.open("w+", newline="")
        epoch_file.write(csv_row(csv_epoch_headers))

    return model_file, epoch_file


def close_wrtier(model_file, epoch_file, model_log_file, epoch_log_file):
    if model_log_file is not None:
        model_log_file.close()
    if epoch_log_file is not None:
        epoch_log_file.close()

    if model_file is not None:
        if model_file.with_suffix(".csv").is_file():
            model_file.with_suffix(".csv").unlink()
        model_file.rename(model_file.with_suffix(".csv"))

    if epoch_file is not None:
        if epoch_file.with_suffix(".csv").is_file():
            epoch_file.with_suffix(".csv").unlink()
        epoch_file.rename(epoch_file.with_suffix(".csv"))


def get_labels(d_name):

    d_name = d_name.lower()

    if d_name == "mnist":
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

    if d_name == "fashionmnist":
        return {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        }

    if d_name == "cifar10":
        return {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }

    if d_name == "gtsrb":
        return {
            0: "20_speed",
            1: "30_speed",
            2: "50_speed",
            3: "60_speed",
            4: "70_speed",
            5: "80_speed",
            6: "80_lifted",
            7: "100_speed",
            8: "120_speed",
            9: "no_overtaking_general",
            10: "no_overtaking_trucks",
            11: "right_of_way_crossing",
            12: "right_of_way_general",
            13: "give_way",
            14: "stop",
            15: "no_way_general",
            16: "no_way_trucks",
            17: "no_way_one_way",
            18: "attention_general",
            19: "attention_left_turn",
            20: "attention_right_turn",
            21: "attention_curvy",
            22: "attention_bumpers",
            23: "attention_slippery",
            24: "attention_bottleneck",
            25: "attention_construction",
            26: "attention_traffic_light",
            27: "attention_pedestrian",
            28: "attention_children",
            29: "attention_bikes",
            30: "attention_snowflake",
            31: "attention_deer",
            32: "lifted_general",
            33: "turn_right",
            34: "turn_left",
            35: "turn_straight",
            36: "turn_straight_right",
            37: "turn_straight_left",
            38: "turn_right_down",
            39: "turn_left_down",
            40: "turn_circle",
            41: "lifted_no_overtaking_general",
            42: "lifted_no_overtaking_trucks",
        }


def get_dataset(
    d_name,
    root="./data",
    train=True,
    download=True,
    transform=T.ToTensor(),
):

    d_name = d_name.lower()

    if d_name == "mnist":
        return datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )

    # commented GTSRB because the PyTorch version does not contain the complete train set.
    # if d_name == "gtsrb":

    #     if train:   train = "train"
    #     else:       train = "test"

    #     return datasets.GTSRB(
    #         root=root, split=train, download=download, transform=transform
    #     )

    if d_name == "gtsrb":
        if train:
            return GTSRB(root / "gtsrb_aug" / "trainingset", "training.csv", transform)
        else:
            return GTSRB(root / "gtsrb_aug" / "testset", "test.csv", transform)

    if d_name == "fashionmnist":
        return datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )

    if d_name == "cifar10":
        return datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )


def split_data(data, split_size, seed=None):
    """
    Split data into two is split_size is a list using PyTorch random_split
    Other split is to return the indecies of a small balanced subset using sklearn
    """
    if seed is None:

        if not isinstance(split_size, list):
            _, indexs, _, _ = train_test_split(
                list(range(len(data))), data, test_size=split_size, stratify=data,random_state=seed)
            return indexs

        return random_split(data, split_size)

    if not isinstance(split_size, list):
        _, indexs, _, _ = train_test_split(
            list(range(len(data))), data, test_size=split_size, stratify=data)
        return indexs

    return random_split(data, split_size, generator=torch.Generator().manual_seed(seed))


def transformer_augmentation_fn():
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomApply(
                [
                    T.RandomAffine((10, 45)),
                    T.RandomAffine(0, translate=(0.1, 0.25)),
                    T.RandomAffine(0, scale=(1.1, 1.25)),
                ],
                p=0.3,
            ),
            T.ToTensor(),
        ]
    )


transformer_augmentation = transformer_augmentation_fn()


def collat_fn_augmentation(batch):

    data = torch.stack([transformer_augmentation(item[0]) for item in batch])
    label = torch.tensor([item[1] for item in batch])

    return [data, label]


def get_device(name=None):

    if name is None:
        name = "cuda:0"

    if torch.cuda.is_available():
        return torch.device(name)

    return torch.device("cpu")


def get_dataLoader(
    data, batch_size=128, shuffle=False, sampler=None, num_workers=0, collate_fn=None
):

    if sampler is not None:
        return DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def get_mean_std(dataloader):

    mean = 0.0
    std = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return mean, std


def increase_in_metric(a, b):
    return (b - a) / (1 - a)


def save_checkpoint(
    model,
    model_path,
    epoch_test_acc,
    epoch_test_loss,
    best_test_acc,
    best_test_loss,
    config
):
    """Save checkpoint while training"""
    # save checkpoint if higher accuracy and less loss
    if best_test_acc < epoch_test_acc:  # higher accuracy
        # delete old checkpoint
        if model_path.exists():
            model_path.unlink()
        # save new checkpoint name
        model_path = (
            model_path.parent
            / f"{model_path.parent.name}-acc-{round(epoch_test_acc, 3)}-loss-{round(epoch_test_loss, 3)}.pth.tar"
        )
        # Save checkpoint
        torch.save({'model': model.state_dict()}, model_path)
        # reset patience
        config["currentPatience"] = 0

        # update best test acc and loss
        return model_path, epoch_test_acc, epoch_test_loss

    return model_path, best_test_acc, best_test_loss


def load_checkpoint(model, model_path):
    """Load checkpoint"""
    model.load_state_dict(torch.load(model_path)['model'])


def export_last_hidden_layer(
    dataloader, model, device, nn_num, map_classes, path, filename, stage, mb=None
):

    # create file to save patterns
    logits_file = path / f"{filename}_{stage}.txt"
    logits_file.touch()

    model.eval()

    if map_classes is not None:
        map_classes = {v: k for k, v in map_classes.items()}

    with open(logits_file, "w+") as f:

        # generate header
        f.write(csv_row([f"x{i}" for i in range(nn_num)] + ["y_pred", "y_true"]))

        # start evaluating
        with torch.no_grad():

            for x, y in progress_bar(dataloader, parent=mb):

                # extract last hidden layer and predicted class
                logits, y_pred = model.output_last_layer(x.to(device))

                y_pred = y_pred.argmax(dim=1)

                for xi, y_pred_i, yi in zip(logits, y_pred, y):
                    # save new tensors into dataset with correct label
                    if map_classes is not None:
                        y_pred_i = map_classes[y_pred_i.tolist()]
                        f.write(csv_row(xi.tolist() + [y_pred_i, yi.tolist()]))
                    else:
                        f.write(csv_row(xi.tolist() + [y_pred_i.tolist(), yi.tolist()]))

    # convert text to csv
    logits_file.rename(logits_file.with_suffix(".csv"))
    logits_file = logits_file.with_suffix(".csv")

    # clean data
    df = read_csv(logits_file)

    df["true"] = df["y_pred"] == df["y_true"]
    df.rename({"y_true": "y"}, axis=1, inplace=True)
    df.drop(["y_pred"], axis=1, inplace=True)

    df.to_csv(logits_file, index=False)


def skip_classes_fn(x, y, map_classes, skip_classes):

    # Change skipped classes to -1
    y.apply_(lambda x: x if x not in skip_classes else -1)

    # Filter data and label indecies
    x = x[y > -1]
    y = y[y > -1]

    # Remap classes indecies to match output layer order
    y.apply_(lambda c: map_classes[c])

    return x, y


def step_lr_scheduler(lr_scheduler, epoch_test_acc, epoch_test_loss):
    """
    Step Learning rate with extra conditions based on
    Learning Rate Scheduler"""
    if type(lr_scheduler).__name__ == "ReduceLROnPlateau":
        if lr_scheduler.state_dict()["mode"] == "max":
            lr_scheduler.step(epoch_test_acc)
        elif lr_scheduler.state_dict()["mode"] == "min":
            lr_scheduler.step(epoch_test_loss)

    # here add more conditions if needed
    # if type(lr_scheduler).__name__ == ... :

    # any other scheduler
    elif lr_scheduler is not None:
        lr_scheduler.step()


def earlyStop(epoch, lr_scheduler, optimizer, best_test_acc, epoch_test_acc, config):
    """
    increase patience based on how much increase or decrease in metrics
    and the type of learning scheduler"""

    # if no scheduler passed, means static lr
    lr_ = True

    # compare current lr with overall min lr
    # if yes then continue to normal early stop
    if type(lr_scheduler).__name__ == "ReduceLROnPlateau":
        lr_ = lr_scheduler.state_dict()["min_lrs"][0] == optimizer.param_groups[0]["lr"]

    # if current epoch less than WarmRestarts epoch number
    # no early stop
    if type(lr_scheduler).__name__ == "CosineAnnealingWarmRestarts":
        lr_ = False
        if lr_scheduler.state_dict()["T_0"] < epoch:
            lr_ = True

    # increase patience if improvment less than 5%
    thld_accuracy = config.get("thld_min_accuracy_improvment", 0.05)
    if lr_ and (increase_in_metric(best_test_acc, epoch_test_acc) < thld_accuracy):

        config["currentPatience"] += 1


def loss_regularization(model, config):
    """calcualte regularization to add to loss"""
    if config["L2"] != 0 and config["L1"] != 0:
        return model._elastic_regularization(config["L2"], config["L1"])

    if config["L2"] != 0:
        return model._l2_regularization(config["L2"])

    if config["L1"] != 0:
        return model._l1_regularization(config["L1"])

    if config["L2"] == 0 and config["L1"] == 0:
        return 0


def train_epoch(
    dataloader,
    model,
    loss_function,
    optimizer,
    map_classes,
    skip_classes,
    device,
    mb,
    config,
):

    epoch_loss = []
    epoch_correct, epoch_total = 0, 0

    for x, y in progress_bar(dataloader, parent=mb):

        mb.child.comment = "Training"

        optimizer.zero_grad()
        model.train()

        # Filter data and label indecies
        if map_classes is not None and skip_classes is not None:
            x, y = skip_classes_fn(x, y, map_classes, skip_classes)

        # Forward pass
        y_pred = model(x.to(device))

        # For calculating the accuracy, save the number of correctly classified
        # images and the total number
        epoch_correct += sum(y.to(device) == y_pred.argmax(dim=1))
        epoch_total += len(y)

        # compute loss
        loss = loss_function(y_pred, y.to(device))
        # append train loss
        epoch_loss.append(loss.item())
        # compute regularization
        reg_loss = loss_regularization(model, config)
        loss = loss + reg_loss
        # compute gradients
        loss.backward()
        # step optimizer
        optimizer.step()
    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss), float(epoch_correct / epoch_total)


def test_epoch(dataloader, model, loss_function, map_classes, skip_classes, device, mb):

    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    confusion_matrix = torch.zeros(model.num_classes, model.num_classes)

    model.eval()

    with torch.no_grad():

        for x, y in progress_bar(dataloader, parent=mb):

            mb.child.comment = "Testing"

            # Filter data and label indecies
            if map_classes is not None and skip_classes is not None:
                x, y = skip_classes_fn(x, y, map_classes, skip_classes)
            # make a prediction on validation set
            y_pred = model(x.to(device))
            # For calculating the accuracy, save the number of correctly
            # classified images and the total number
            epoch_correct += sum(y.to(device) == y_pred.argmax(dim=1))
            epoch_total += len(y)
            # Fill confusion matrix
            for (y_true, y_p) in zip(y, y_pred.argmax(dim=1)):
                confusion_matrix[int(y_true), int(y_p)] += 1
            # Compute loss
            loss = loss_function(y_pred, y.to(device))
            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return (
        np.mean(epoch_loss),
        float(epoch_correct / epoch_total),
        confusion_matrix.numpy(),
    )


def run_training_testing(
    model,
    loss_function,
    optimizer,
    lr_scheduler,
    map_classes,
    skip_classes,
    device,
    model_path,
    trainloader,
    testloader,
    config,
):
    # regularization info for logging
    if config["L2"] > 0 and config["L1"] > 0:
        config["reg_type"] = "Elastic"
    elif config["L2"] > 0:
        config["reg_type"] = "L2"
    elif config["L1"] > 0:
        config["reg_type"] = "L1"
    else:
        config["reg_type"] = "NoRegularization"

    if model.dropout_p > 0:
        config["reg_type"] += f"_Dropout"
        config["dropout"] = model.dropout_p
    else:
        config["dropout"] = 0

    # adding early stop attribuates
    config["currentPatience"] = False

    mb = master_bar(range(config["epochs"]))
    mb.main_bar.comment = "Epochs"

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    best_test_acc = float(-100)
    best_test_loss = float(100)

    config["currentPatience"] = 0

    for epoch in mb:
        # Train the model
        epoch_train_loss, epoch_train_acc = train_epoch(
            trainloader,
            model,
            loss_function,
            optimizer,
            map_classes,
            skip_classes,
            device,
            mb,
            config,
        )
        epoch_test_loss, epoch_test_acc, _ = test_epoch(
            testloader, model, loss_function, map_classes, skip_classes, device, mb
        )

        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)

        step_lr_scheduler(lr_scheduler, epoch_test_acc, epoch_test_loss)

        # no saving is needed when testing parameters
        if model_path is not None:

            # increase patience if no improvment
            earlyStop(epoch, lr_scheduler, optimizer, best_test_acc, epoch_test_acc, config)

            model_path, best_test_acc, best_test_loss = save_checkpoint(
                model,
                model_path,
                epoch_test_acc,
                epoch_test_loss,
                best_test_acc,
                best_test_loss,
                config
            )

            # early stop if no improvment
            if config["currentPatience"] == config["patience"]:
                break

        # master bar feedback
        mb.write(
            f"[Epoch: {epoch:02d}]"
            + f" Loss ({round(epoch_train_loss,3):.3f} / {round(epoch_test_loss,3):.3f})"
            + f" - Accuracy ({round(epoch_train_acc,3)*100:.1f}% / {round(epoch_test_acc,3)*100:.1f}%)"
        )

        # break if accuracy is over 98%
        if best_test_acc > 0.98:
            mb.write("Test Accuracy is over 98% => Early Stop!")
            break

    test_loss, test_acc, confusion_matrix_test = None, None, []

    if model_path is not None:
        load_checkpoint(model, model_path)

        test_loss, test_acc, confusion_matrix_test = test_epoch(
            testloader, model, loss_function, map_classes, skip_classes, device, mb
        )

    if model_path is not None:
        plot_results(f"Losses", "Loss", train_losses, test_losses, np.argmax(test_accs),
                     save_path=model_path.parent / 'epochs_losses.jpg')
        plot_results(f"Accuracy", "Accuracy", train_accs, test_accs, np.argmax(test_accs),
                    save_path=model_path.parent / 'epochs_accuracy.jpg')

    return (
        train_losses,
        test_losses,
        train_accs,
        test_accs,
        test_loss,
        test_acc,
        confusion_matrix_test,
        model_path,
    )
