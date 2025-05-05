import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


def combine_mnist_images_with_masks(
    mean_image, uncertainty_image, canvas_size=(64, 64)
):
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
    uncertainty_image_scaled = (
        uncertainty_image_np > 0
    ) * 150  # Keep only non-black pixels

    # Define the four corners for placement
    corners = [
        (0, 0),  # Top-left
        (canvas_size[1] - 28, 0),  # Top-right
        (0, canvas_size[0] - 28),  # Bottom-left
        (canvas_size[1] - 28, canvas_size[0] - 28),  # Bottom-right
    ]

    # Randomly select two distinct corners
    corner1, corner2 = random.sample(corners, 2)

    # Create binary masks for the two digits
    mask_mean = np.zeros(canvas_size, dtype=np.uint8)
    mask_uncertainty = np.zeros(canvas_size, dtype=np.uint8)

    # Place mean image and update its mask
    canvas_np = np.array(canvas, dtype=np.uint8)
    mean_region = canvas_np[corner1[1] : corner1[1] + 28, corner1[0] : corner1[0] + 28]
    mean_region[:] = np.maximum(mean_region, mean_image_scaled)  # Merge with canvas
    mask_mean[corner1[1] : corner1[1] + 28, corner1[0] : corner1[0] + 28] = (
        mean_image_np > 0
    )

    # Place uncertainty image and update its mask
    uncertainty_region = canvas_np[
        corner2[1] : corner2[1] + 28, corner2[0] : corner2[0] + 28
    ]
    uncertainty_region[:] = np.maximum(
        uncertainty_region, uncertainty_image_scaled
    )  # Merge with canvas
    mask_uncertainty[corner2[1] : corner2[1] + 28, corner2[0] : corner2[0] + 28] = (
        uncertainty_image_np > 0
    )

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


def create_combined_mnist_dataset_with_masks(
    output_dir, num_samples=1000, canvas_size=(64, 64), seed=0
):
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
            filename_mask_uncertainty = (
                f"combined_{len(combined_data):06d}_mask_uncertainty.png"
            )
            Image.fromarray(mask_mean * 255).save(
                os.path.join(output_dir, "masks", filename_mask_mean)
            )  # Scale mask to [0, 255]
            Image.fromarray(mask_uncertainty * 255).save(
                os.path.join(output_dir, "masks", filename_mask_uncertainty)
            )  # Scale mask to [0, 255]

            # Store the data
            combined_data.append(
                {
                    "filename": filename_image,
                    "mean_label": mean_label.item(),
                    "uncertainty_label": uncertainty_label.item(),
                    "mask_mean": filename_mask_mean,
                    "mask_uncertainty": filename_mask_uncertainty,
                }
            )
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

    def __init__(self, data_dir, lable_file="labels.csv", transform=None):
        """
        Args:
            data_dir (str): Directory containing the combined images and labels.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load the labels using pandas
        labels_path = os.path.join(data_dir, lable_file)
        self.data = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.data_dir, self.data.loc[idx, "filename"])
        image = Image.open(img_path).convert("L")

        mean_mask_path = os.path.join(
            self.data_dir, "masks", self.data.loc[idx, "mask_mean"]
        )
        mean_mask = Image.open(mean_mask_path).convert("L")
        uncertainty_mask_path = os.path.join(
            self.data_dir, "masks", self.data.loc[idx, "mask_uncertainty"]
        )
        uncertainty_mask = Image.open(uncertainty_mask_path).convert("L")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
            mean_mask = self.transform(mean_mask)
            uncertainty_mask = self.transform(uncertainty_mask)

        # Load the labels
        mean_label = self.data.loc[idx, "mean_label"]
        uncertainty_label = self.data.loc[idx, "uncertainty_label"]

        mean_label = mean_label + np.random.normal(0, uncertainty_label)

        return (
            image,
            torch.Tensor([mean_label]),
            torch.Tensor([uncertainty_label]),
            mean_mask,
            uncertainty_mask,
        )


class CombinedMNISTDatasetInfoShap(Dataset):
    """
    A PyTorch Dataset for the combined MNIST images.
    """

    def __init__(self, data_dir, metadata, transform=None):
        """
        Args:
            data_dir (str): Directory containing the combined images and labels.
            transform (callable, optional): Transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Load the labels using pandas
        self.data = metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.data_dir, self.data.loc[idx, "filename"])
        image = Image.open(img_path).convert("L")

        mean_mask_path = os.path.join(
            self.data_dir, "masks", self.data.loc[idx, "mask_mean"]
        )
        mean_mask = Image.open(mean_mask_path).convert("L")
        uncertainty_mask_path = os.path.join(
            self.data_dir, "masks", self.data.loc[idx, "mask_uncertainty"]
        )
        uncertainty_mask = Image.open(uncertainty_mask_path).convert("L")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
            mean_mask = self.transform(mean_mask)
            uncertainty_mask = self.transform(uncertainty_mask)

        # Load the labels
        label = self.data.loc[idx, "new_label"]
        uncertainty_label = self.data.loc[idx, "uncertainty_label"]

        return (
            image,
            torch.Tensor([label]),
            torch.Tensor([uncertainty_label]),
            mean_mask,
            uncertainty_mask,
        )


def get_loaders(output_dir, num_samples, get_sets=None):
    # Create PyTorch Dataset
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )  # , v2.RandomRotation(degrees=15)])
    dataset = CombinedMNISTDataset(data_dir=output_dir, transform=transform)

    if get_sets == "full":
        return dataset

    generator1 = torch.Generator().manual_seed(0)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [int(0.7 * num_samples), int(0.1 * num_samples), int(0.2 * num_samples)],
        generator=generator1,
    )
    if get_sets == "splits":
        return train_dataset, val_dataset, test_dataset

    train_dataloader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    return train_dataloader, val_dataloader, test_dataloader
