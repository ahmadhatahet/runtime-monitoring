# Setup Variables

# MNIST, FashionMNIST, GTSRB, Cifar10
SEED = 42
CUDA = 0
GPU_NAME = f'cuda:{CUDA}'


import os
from pathlib import Path

base = Path().cwd()

if base.name != 'runtime-monitoring':
    os.chdir('../')
    base = Path().cwd()


# # Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from argparse import ArgumentParser
from utilities.utils import *
from utilities.pathManager import fetchPaths
from utilities.scaleFunctions import *
from utilities.pcaFunctions import *


# disable warnings
import warnings
warnings.filterwarnings('ignore')

# argparser
parser = ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, type=str)
args = parser.parse_args()
DATASET = args.dataset

# # Paths

paths = fetchPaths(base, DATASET, '', False)
path_data = paths['data']
configs = load_json(paths['configuration'])
config = configs['configuration']
model_setup = configs['model_setup']
model_config = configs['model_config']
optim_name = list(config['optimizer'].keys())[0]
optim_args = config['optimizer'][optim_name]
scheduler_name = list(config['scheduler'].keys())[0]
scheduler_args = config['scheduler'][scheduler_name]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = get_device(GPU_NAME)
torch.cuda.get_device_name(device)


# ## Load model and settings

# transformers
from models.transformers import transformers
tf_train = transformers[DATASET.lower()]['train']
tf_test = transformers[DATASET.lower()]['test']


from models.mnist_model import MNIST_Model
from models.fashionmnist_model import FashionMNIST_CNN
from models.gtsrb_model import GTSRB_CNN
from models.cifar10_dla import Cifar10_DLA
from models.cifar10_model import Cifar10_CNN

models = {
    'mnist': MNIST_Model,
    'fashionmnist': FashionMNIST_CNN,
    'gtsrb': GTSRB_CNN,
#     'cifar10': Cifar10_DLA,
    'cifar10': Cifar10_CNN
}

model_ = models[DATASET.lower()]

# # Load / Split / DataLoader
feature_names = get_labels(DATASET)

train_data = get_dataset(DATASET, path_data, train=True, transform=tf_train)
test_data = get_dataset(DATASET, path_data, train=False, transform=tf_test)

trainloader = get_dataLoader(train_data, model_config['batch_size'], True)
testloader = get_dataLoader(test_data, model_config['batch_size'], False)


# # Helper Functions
model_setup['last_hidden_neurons'] = min(config['lhl_neurons'])

model = model_(**model_setup).to(device)
model = torch.compile(model)
nn.DataParallel(model, device_ids=[CUDA])

# loss function
loss_function = nn.CrossEntropyLoss()

# optimizer and scheduler
optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=model_config['lr'], **optim_args)
scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)


# training testing attributes
kwargs = {
    'model': model,
    'loss_function': loss_function,
    'optimizer': optimizer,
    'lr_scheduler': scheduler,
    'map_classes': None,
    'skip_classes': None,
    'device': device,
    'model_path': None,
    'trainloader': trainloader,
    'testloader': testloader,
    'config': model_config
}


# ## Run Training

# train

(train_losses, train_accs,
 test_losses,test_accs,
 train_loss,train_acc
 ,test_loss,test_acc,
 confusion_matrix_train,
 confusion_matrix_test,
 best_model_name) = \
run_training_testing(**kwargs)

idx = np.argmax(test_accs)
test_loss, test_acc = test_losses[idx], test_accs[idx]

print(f'Done, Best accuracy: {test_acc*100:.2f}%, Best loss: {test_loss*100:.2f}')
