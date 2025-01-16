import hashlib
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import mean_squared_error
from torchvision import datasets, transforms


output_dir = "./combined_mnist_dataset"
num_samples = 500000  # Number of combined images to create

def combine_mnist_images_with_masks(mean_image, uncertainty_image, canvas_size=(64, 64)):
    """
    Combines two MNIST images into a larger image with positions randomized to one of the four corners.
    Also generates ground truth masks for each digit.

    Args:
        mean_image (PIL.Image): The image for the mean value.
        uncertainty_image (PIL.Image): The image for the uncertainty value.
        canvas_size (tuple): Size of the output canvas (height, width).

    Returns:
        PIL.Image: Combined image.
        np.ndarray: Ground truth mask for the mean value image.
        np.ndarray: Ground truth mask for the uncertainty value image.
    """
    # Create a blank canvas
    canvas = Image.new("L", canvas_size, color=0)  # Black background

    # Convert images to numpy arrays
    mean_image_np = np.array(mean_image, dtype=np.uint8)
    uncertainty_image_np = np.array(uncertainty_image, dtype=np.uint8)

    # Scale mean_image to white and uncertainty_image to dark gray
    mean_image_scaled = (mean_image_np > 0) * 255  # Keep only non-black pixels
    uncertainty_image_scaled = (uncertainty_image_np > 0) * 150  # Keep only non-black pixels

    # Define the four corners for placement
    corners = [
        (0, 0),  # Top-left
        (canvas_size[1] - 28, 0),  # Top-right
        (0, canvas_size[0] - 28),  # Bottom-left
        (canvas_size[1] - 28, canvas_size[0] - 28)  # Bottom-right
    ]

    # Randomly select two distinct corners
    corner1, corner2 = random.sample(corners, 2)

    # Create binary masks for the two digits
    mask_mean = np.zeros(canvas_size, dtype=np.uint8)
    mask_uncertainty = np.zeros(canvas_size, dtype=np.uint8)

    # Place mean image and update its mask
    canvas_np = np.array(canvas, dtype=np.uint8)
    mean_region = canvas_np[corner1[1]:corner1[1]+28, corner1[0]:corner1[0]+28]
    mean_region[:] = np.maximum(mean_region, mean_image_scaled)  # Merge with canvas
    mask_mean[corner1[1]:corner1[1]+28, corner1[0]:corner1[0]+28] = (mean_image_np > 0)

    # Place uncertainty image and update its mask
    uncertainty_region = canvas_np[corner2[1]:corner2[1]+28, corner2[0]:corner2[0]+28]
    uncertainty_region[:] = np.maximum(uncertainty_region, uncertainty_image_scaled)  # Merge with canvas
    mask_uncertainty[corner2[1]:corner2[1]+28, corner2[0]:corner2[0]+28] = (uncertainty_image_np > 0)

    return Image.fromarray(canvas_np), mask_mean, mask_uncertainty

def get_image_hash(image):
    """
    Generate a unique hash for the combined image content.

    Args:
        image (PIL.Image): The final combined image.

    Returns:
        str: A unique hash for the combined image.
    """
    # Convert image to bytes
    image_bytes = image.tobytes()

    # Generate a hash for the image bytes
    return hashlib.md5(image_bytes).hexdigest()

def create_combined_mnist_dataset_with_masks(output_dir, num_samples=1000, canvas_size=(64, 64), seed=0):
    """
    Creates a dataset of combined MNIST images with ground truth masks and saves them to the output directory.

    Args:
        output_dir (str): Directory to save the combined images and labels.
        num_samples (int): Number of combined images to create.
        canvas_size (tuple): Size of the output canvas (height, width).
    """

    # Set random seed for reproducibility
    random.seed(seed)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Load MNIST dataset
    mnist_train = datasets.MNIST(root="./data", train=True, download=True)
    images = mnist_train.data  # Access raw images
    labels = mnist_train.targets  # Access raw labels

    combined_data = []
    generated_hashes = set()  # To track unique combined images based on hash

    # Generate combined images
    while len(combined_data) < num_samples:
        # Randomly select two images from the dataset
        idx1, idx2 = random.sample(range(len(images)), 2)
        mean_image, mean_label = images[idx1], labels[idx1]
        uncertainty_image, uncertainty_label = images[idx2], labels[idx2]

        # Convert images to PIL format
        mean_image = Image.fromarray(mean_image.numpy(), mode="L")
        uncertainty_image = Image.fromarray(uncertainty_image.numpy(), mode="L")

        # Combine the images and generate masks
        combined_image, mask_mean, mask_uncertainty = combine_mnist_images_with_masks(
            mean_image, uncertainty_image, canvas_size
        )

        # Generate a unique hash for this combined image
        image_hash = get_image_hash(combined_image)

        # Skip if this combined image has already been created
        if image_hash not in generated_hashes:
            generated_hashes.add(image_hash)

            # Save the combined image
            filename_image = f"combined_{len(combined_data):06d}.png"
            combined_image.save(os.path.join(output_dir, filename_image))

            # Save the ground truth masks
            filename_mask_mean = f"combined_{len(combined_data):06d}_mask_mean.png"
            filename_mask_uncertainty = f"combined_{len(combined_data):06d}_mask_uncertainty.png"
            Image.fromarray(mask_mean * 255).save(os.path.join(output_dir, "masks", filename_mask_mean))  # Scale mask to [0, 255]
            Image.fromarray(mask_uncertainty * 255).save(os.path.join(output_dir, "masks", filename_mask_uncertainty))  # Scale mask to [0, 255]

            # Store the data
            combined_data.append({
                "filename": filename_image,
                "mean_label": mean_label.item(),
                "uncertainty_label": uncertainty_label.item(),
                "mask_mean": filename_mask_mean,
                "mask_uncertainty": filename_mask_uncertainty,
            })
        else:
            # If the image hash has been generated before, skip it
            continue

    # Save the data to a CSV file using pandas
    df = pd.DataFrame(combined_data)
    df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

class CombinedMNISTDataset(Dataset):
    """
    A PyTorch Dataset for the combined MNIST images.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing the combined images and labels.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load the labels using pandas
        labels_path = os.path.join(data_dir, "labels.csv")
        self.data = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.data_dir, self.data.loc[idx, "filename"])
        image = Image.open(img_path).convert("L")

        mean_mask_path = os.path.join(self.data_dir, "masks", self.data.loc[idx, "mask_mean"])
        mean_mask = Image.open(mean_mask_path).convert("L")
        uncertainty_mask_path = os.path.join(self.data_dir, "masks", self.data.loc[idx, "mask_uncertainty"])
        uncertainty_mask = Image.open(uncertainty_mask_path).convert("L")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
            mean_mask = self.transform(mean_mask)
            uncertainty_mask = self.transform(uncertainty_mask)

        # Load the labels
        mean_label = self.data.loc[idx, "mean_label"]
        uncertainty_label = self.data.loc[idx, "uncertainty_label"]

        exponent = 1.5  # Choose an exponent greater than 1 for non-linear scaling
        # uncertainty_label = uncertainty_label * 0.0008 * (0.01 + (uncertainty_label ** exponent))
        mean_label = mean_label + np.random.normal(0, uncertainty_label)
    
        return image, torch.Tensor([mean_label]), torch.Tensor([uncertainty_label]), mean_mask, uncertainty_mask

# Create PyTorch Dataset
transform = transforms.Compose([transforms.ToTensor()])# , v2.RandomRotation(degrees=15)])
dataset = CombinedMNISTDataset(data_dir=output_dir, transform=transform)

# Access a sample
sample_image, sample_mean, sample_uncertainty, _, _ = dataset[0]
print("Sample Labels:", sample_mean, sample_uncertainty)
print("Sample Image Shape:", sample_image.shape)


generator1 = torch.Generator().manual_seed(0)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,  [int(0.7 * num_samples), int(0.1 * num_samples), int(0.2 * num_samples)], generator=generator1)
# train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(dataset,  [30000, 10000, 20000, 500000-60000], generator=generator1)




train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')


# Define the Model
class MeanVarianceSwitchableCNN(nn.Module):
    def __init__(self, focus=None):
        super(MeanVarianceSwitchableCNN, self).__init__()
        self.focus = focus
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.fc_log_var = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        if self.focus is None:
            return mean, log_var
        elif self.focus == "mean":
            return mean
        else:
            return log_var

# Define the Lightning Module
class SwitchableCNNModule(pl.LightningModule):
    def __init__(self, switch_epoch=15, lr=0.001, weight_decay=1e-3, clip_value=5.0):
        super(SwitchableCNNModule, self).__init__()
        self.model = MeanVarianceSwitchableCNN()
        self.switch_epoch = switch_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)

        # Determine the loss function
        current_epoch = self.current_epoch
        if current_epoch < self.switch_epoch:
            loss_function = nn.MSELoss()
            loss = loss_function(means, labels)
        else:
            loss_function = nn.GaussianNLLLoss()
            variances = torch.exp(log_vars).squeeze()
            loss = loss_function(means, labels, variances)

        # Log training metrics
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # Log Gaussian NLL as a metric regardless of the loss used
        variances = torch.exp(log_vars).squeeze()
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log("train_mse", mean_squared_error(means, labels), on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_gaussian_nll", gaussian_nll, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)
        variances = torch.exp(log_vars).squeeze()

        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log("val_mse", mean_squared_error(means, labels), on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_gaussian_nll", gaussian_nll, on_epoch=True, on_step=False, prog_bar=True)

        return gaussian_nll

    def test_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)
        variances = torch.exp(log_vars).squeeze()

        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log("test_gaussian_nll", gaussian_nll, on_epoch=True, on_step=False, prog_bar=True)

        return gaussian_nll

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, _, _, _, _ = batch
        means, log_vars = self.model(images)
        return means, log_vars

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        optimizer.step(closure=optimizer_closure)







model = SwitchableCNNModule.load_from_checkpoint("checkpoints/best_model_1736245990.ckpt").model

class Localization:
    def __init__(self, top_k=None):
        metrics = ["mean_heatmap_mean_mask", "mean_heatmap_var_mask", 
                   "var_heatmap_mean_mask", "var_heatmap_var_mask"]
        self.iou_scores = {key: [] for key in metrics}
        self.mass_accuracies = {key: [] for key in metrics}
        self.rank_accuracies = {key: [] for key in metrics}
        self.top_k = top_k

    def _compute_intersection(self, heatmap, mask):
        heatmap_flat, mask_flat = heatmap.view(-1).abs(), mask.view(-1)
        intersection = (heatmap_flat * mask_flat).sum()
        return intersection, heatmap_flat.sum(), mask_flat.sum()

    def _intersection_over_union(self, heatmap, mask):
        intersection, total_heatmap, total_mask = self._compute_intersection(heatmap, mask)
        union = total_heatmap + total_mask - intersection
        return intersection / (union + 1e-10)
    
    def _mass_accuracy(self, heatmap, mask):
        overlap_mass, total_heatmap, _ = self._compute_intersection(heatmap, mask)
        return overlap_mass / (total_heatmap + 1e-10)
    
    def _rank_accuracy(self, heatmap, mask):
        heatmap_flat, mask_flat = heatmap.view(-1).abs(), mask.view(-1)
        top_k = mask_flat.sum().to(int).item() if self.top_k is None else self.top_k
        top_indices = torch.topk(heatmap_flat, top_k).indices
        return mask_flat[top_indices].sum().item() / top_k

    def _add_metrics(self, mean_heatmap, mean_mask, var_heatmap, var_mask, metric_func, metric_storage):
        metric_storage["mean_heatmap_mean_mask"].append(metric_func(mean_heatmap, mean_mask))
        metric_storage["mean_heatmap_var_mask"].append(metric_func(mean_heatmap, var_mask))
        metric_storage["var_heatmap_mean_mask"].append(metric_func(var_heatmap, mean_mask))
        metric_storage["var_heatmap_var_mask"].append(metric_func(var_heatmap, var_mask))

    def add_sample(self, mean_heatmap, mean_mask, var_heatmap, var_mask):
        self._add_metrics(mean_heatmap, mean_mask, var_heatmap, var_mask, self._intersection_over_union, self.iou_scores)
        self._add_metrics(mean_heatmap, mean_mask, var_heatmap, var_mask, self._mass_accuracy, self.mass_accuracies)
        self._add_metrics(mean_heatmap, mean_mask, var_heatmap, var_mask, self._rank_accuracy, self.rank_accuracies)

    def calculate_acc_matrices(self):
        def compute_summary(metrics_dict):
            avg = {key: torch.tensor(values).mean().item() for key, values in metrics_dict.items()}
            std = {key: torch.tensor(values).std().item() for key, values in metrics_dict.items()}
            return avg, std
        
        iou_avg, iou_std = compute_summary(self.iou_scores)
        mass_avg, mass_std = compute_summary(self.mass_accuracies)
        rank_avg, rank_std = compute_summary(self.rank_accuracies)

        def create_matrix_and_annotation(avg, std):
            matrix = [[avg["mean_heatmap_mean_mask"], avg["mean_heatmap_var_mask"]],
                      [avg["var_heatmap_mean_mask"], avg["var_heatmap_var_mask"]]]
            annotation = [[f"{avg['mean_heatmap_mean_mask']:.4f} ± {std['mean_heatmap_mean_mask']:.4f}",
                           f"{avg['mean_heatmap_var_mask']:.4f} ± {std['mean_heatmap_var_mask']:.4f}"],
                          [f"{avg['var_heatmap_mean_mask']:.4f} ± {std['var_heatmap_mean_mask']:.4f}",
                           f"{avg['var_heatmap_var_mask']:.4f} ± {std['var_heatmap_var_mask']:.4f}"]]
            return matrix, annotation

        iou_matrix, iou_annot = create_matrix_and_annotation(iou_avg, iou_std)
        mass_matrix, mass_annot = create_matrix_and_annotation(mass_avg, mass_std)
        rank_matrix, rank_annot = create_matrix_and_annotation(rank_avg, rank_std)

        return {
            "iou_matrix": iou_matrix,
            "mass_acc_matrix": mass_matrix,
            "rank_acc_matrix": rank_matrix,
            "iou_annot": iou_annot,
            "mass_acc_annot": mass_annot,
            "rank_acc_annot": rank_annot
        }
    
# CLUE
import sys
sys.path.append('..')
from utils.experiment_utils import clue_explain_images

SKIP_VAE_TRAINING = True # already trained
CPU = True
pnn = model
pnn = pnn.to(torch.float32)
if CPU:
    pnn = pnn.cpu()
identifier="combined_mnist"
print("reducing dataset size for clue")
trainset = torch.utils.data.Subset(train_dataset, range(2*len(train_dataset)//5))
valset = torch.utils.data.Subset(val_dataset, range(2*len(val_dataset)//5))
testset = torch.utils.data.Subset(test_dataset, range(2*len(test_dataset)//5))
ind_instances_to_explain = range(len(testset))

del train_dataset, val_dataset, test_dataset

clue_explain_images(
    pnn,
    trainset,
    valset,
    testset,
    ind_instances_to_explain,
    identifier,
    save_dir="combined_mnist_vae",
    save=True,
    sort=False,
    skip_vae_training=SKIP_VAE_TRAINING,
    cpu=CPU,
)
data = np.load("combined_mnist_vae/importances/CLUE_importances_combined_mnist.npy", allow_pickle=True)
variance_explanations = np.abs(data.item()["importances_directed"])  
localization = Localization()

for i in range(len(testset)):
    image, label, uc, mean_mask, var_mask = testset[i]  # Get a sample from the dataset
    explanation = variance_explanations[i]
    mean_heatmap = np.ones_like(explanation)
    var_heatmap = explanation
    localization.add_sample(torch.Tensor(mean_heatmap), mean_mask, torch.Tensor(var_heatmap), var_mask)


localization_result = localization.calculate_acc_matrices()

# Plotting the heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# IoU heatmap
sns.heatmap(localization_result["iou_matrix"], annot=localization_result["iou_annot"], cmap="Blues", xticklabels=["Mean Heatmap", "Variance Heatmap"], yticklabels=["Mean Mask", "Variance Mask"], ax=axes[0], fmt="")
axes[0].set_title('IoU Heatmap (Avg. ± Std.)')

# Mass accuracy heatmap
sns.heatmap(localization_result["mass_acc_matrix"], annot=localization_result["mass_acc_annot"], cmap="Blues", xticklabels=["Mean Heatmap", "Variance Heatmap"], yticklabels=["Mean Mask", "Variance Mask"], ax=axes[1], fmt="")
axes[1].set_title('Mass Accuracy Heatmap (Avg. ± Std.)')

# Rank accuracy heatmap
sns.heatmap(localization_result["rank_acc_matrix"], annot=localization_result["rank_acc_annot"], cmap="Blues", xticklabels=["Mean Heatmap", "Variance Heatmap"], yticklabels=["Mean Mask", "Variance Mask"], ax=axes[2], fmt="")
axes[2].set_title('Rank Accuracy Heatmap (Avg. ± Std.)')

plt.tight_layout()
plt.savefig("localization_heatmaps_clue.png")

print("done!!!")