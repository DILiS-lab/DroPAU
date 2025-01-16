
import sys
sys.path.append('..')
sys.path.append('../CLUE')
import os
import random
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch
import hashlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from ds import CombinedMNISTDataset
import os
from torchvision import transforms
from torch.nn.functional import relu
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

output_dir = "./combined_mnist_dataset"

from CLUE.VAE.models import MNISTplus_recognition_resnet, MNISTplus_generator_resnet
from CLUE.VAE.train import train_VAE
from CLUE.VAE.MNISTconv_bern import MNISTplusconv_VAE_bern_net

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Create PyTorch Dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = CombinedMNISTDataset(data_dir=output_dir, transform=transform)

generator1 = torch.Generator().manual_seed(seed)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                         [175000, 25000, 50000], generator=generator1)

save_dir = "./combined_mnist_vae"
identifier = "combined_mnist"

if save_dir and not os.path.exists(f"{save_dir}/models/"):
    os.makedirs(f"{save_dir}/models/")
save_dir_vae = f"{save_dir}/models/VAE_{identifier}"
if save_dir and not os.path.exists(save_dir_vae):
    os.makedirs(save_dir_vae)

latent_dim = 16
batch_size = 256
nb_epochs = 100
print("Training VAE")
print(f"number of epochs: {nb_epochs}")
print(f"latent dim: {latent_dim}")
lr = 7e-4
early_stop = 5

cuda = torch.cuda.is_available()

# Define encoder and decoder for the VAE
encoder = MNISTplus_recognition_resnet(latent_dim)
decoder = MNISTplus_generator_resnet(latent_dim)

VAE = MNISTplusconv_VAE_bern_net(latent_dim, encoder, decoder, lr, cuda=cuda)

# Clear out old models
path = f"{save_dir_vae}_models"
if os.path.exists(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))

# Train the VAE
vlb_train, vlb_dev = train_VAE(
    VAE,
    save_dir_vae,
    batch_size,
    nb_epochs,
    train_dataset,
    val_dataset,
    cuda=cuda,
    flat_ims=False,
    train_plot=False,
    early_stop=early_stop,
)

VAE.load(f"{save_dir_vae}_models/theta_best.dat")
