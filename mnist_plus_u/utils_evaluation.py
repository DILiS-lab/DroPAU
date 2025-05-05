import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# Visualization function
def visualize_heatmap(image, mean_heatmap, var_heatmap, method, alpha=0.5):
    # Resize heatmaps to match image size
    heatmap_resized_mean = cv2.resize(
        mean_heatmap.cpu().numpy(), (image.shape[-1], image.shape[-2])
    )
    heatmap_resized_var = cv2.resize(
        var_heatmap.cpu().numpy(), (image.shape[-1], image.shape[-2])
    )

    # Convert image tensor to NumPy
    image_np = image.squeeze().cpu().numpy()

    # Plot the image and its heatmaps side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Image plot
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")  

    # Mean heatmap plot
    axes[1].imshow(
        heatmap_resized_mean,
        cmap="jet" if method != "lrp" else "bwr",
        alpha=alpha,
        vmin=(
            0
            if method != "lrp"
            else -np.max(
                [np.abs(heatmap_resized_mean.min()), np.abs(heatmap_resized_mean.max())]
            )
        ),
        vmax=(
            1
            if method != "lrp"
            else np.max(
                [np.abs(heatmap_resized_mean.min()), np.abs(heatmap_resized_mean.max())]
            )
        ),
    )
    axes[1].set_title("Mean Prediction Heatmap")
    axes[1].axis("off")

    # Variance heatmap plot
    axes[2].imshow(
        heatmap_resized_var,
        cmap="jet" if method != "lrp" else "bwr",
        alpha=alpha,
        vmin=(
            0
            if method != "lrp"
            else -np.max(
                [np.abs(heatmap_resized_mean.min()), np.abs(heatmap_resized_mean.max())]
            )
        ),
        vmax=(
            1
            if method != "lrp"
            else np.max(
                [np.abs(heatmap_resized_mean.min()), np.abs(heatmap_resized_mean.max())]
            )
        ),
    )
    axes[2].set_title("Variance Prediction Heatmap")
    axes[2].axis("off")

    plt.colorbar(
        axes[1].imshow(
            heatmap_resized_mean, cmap="jet" if method != "lrp" else "bwr", alpha=alpha
        ),
        ax=axes[1],
    )
    plt.colorbar(
        axes[2].imshow(
            heatmap_resized_var, cmap="jet" if method != "lrp" else "bwr", alpha=alpha
        ),
        ax=axes[2],
    )

    plt.tight_layout()
    plt.show()


class Localization:
    def __init__(self, top_k=None):
        metrics = [
            "mean_heatmap_mean_mask",
            "mean_heatmap_var_mask",
            "var_heatmap_mean_mask",
            "var_heatmap_var_mask",
        ]
        self.gt_uncertainties = []
        self.iou_scores = {key: [] for key in metrics}
        self.mass_accuracies = {key: [] for key in metrics}
        self.rank_accuracies = {key: [] for key in metrics}
        self.top_k = top_k

    def _compute_intersection(self, heatmap, mask):
        heatmap_flat, mask_flat = heatmap.view(-1).abs(), mask.view(-1)
        intersection = (heatmap_flat * mask_flat).sum()
        return intersection, heatmap_flat.sum(), mask_flat.sum()

    def _intersection_over_union(self, heatmap, mask):
        intersection, total_heatmap, total_mask = self._compute_intersection(
            heatmap, mask
        )
        union = total_heatmap + total_mask - intersection
        return (intersection / (union + 1e-10)).item()

    def _mass_accuracy(self, heatmap, mask):
        overlap_mass, total_heatmap, _ = self._compute_intersection(heatmap, mask)
        return (overlap_mass / (total_heatmap + 1e-10)).item()

    def _rank_accuracy(self, heatmap, mask):
        heatmap_flat, mask_flat = heatmap.view(-1).abs(), mask.view(-1)
        top_k = mask_flat.sum().to(int).item() if self.top_k is None else self.top_k
        top_indices = torch.topk(heatmap_flat, top_k).indices
        return mask_flat[top_indices].sum().item() / top_k

    def _add_metrics(
        self,
        mean_heatmap,
        mean_mask,
        var_heatmap,
        var_mask,
        metric_func,
        metric_storage,
    ):
        metric_storage["mean_heatmap_mean_mask"].append(
            metric_func(mean_heatmap, mean_mask)
        )
        metric_storage["mean_heatmap_var_mask"].append(
            metric_func(mean_heatmap, var_mask)
        )
        metric_storage["var_heatmap_mean_mask"].append(
            metric_func(var_heatmap, mean_mask)
        )
        metric_storage["var_heatmap_var_mask"].append(
            metric_func(var_heatmap, var_mask)
        )

    def add_sample(self, mean_heatmap, mean_mask, var_heatmap, var_mask, gt_uncertainty):
        self.gt_uncertainties.append(gt_uncertainty)
        self._add_metrics(
            mean_heatmap,
            mean_mask,
            var_heatmap,
            var_mask,
            self._intersection_over_union,
            self.iou_scores,
        )
        self._add_metrics(
            mean_heatmap,
            mean_mask,
            var_heatmap,
            var_mask,
            self._mass_accuracy,
            self.mass_accuracies,
        )
        self._add_metrics(
            mean_heatmap,
            mean_mask,
            var_heatmap,
            var_mask,
            self._rank_accuracy,
            self.rank_accuracies,
        )

    def save(self, file_name):
        with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
            json.dump({
                "gt_uncertainty": [val.item() for val in self.gt_uncertainties],
                "iou_scores": self.iou_scores,
                "mass_accuracies": self.mass_accuracies,
                "rank_accuracies": self.rank_accuracies
            }, f, ensure_ascii=False, indent=4)
        pass

    def calculate_acc_matrices(self):
        def compute_summary(metrics_dict):
            avg = {
                key: torch.tensor(values).mean().item()
                for key, values in metrics_dict.items()
            }
            std = {
                key: torch.tensor(values).std().item()
                for key, values in metrics_dict.items()
            }
            return avg, std

        iou_avg, iou_std = compute_summary(self.iou_scores)
        mass_avg, mass_std = compute_summary(self.mass_accuracies)
        rank_avg, rank_std = compute_summary(self.rank_accuracies)

        def create_matrix_and_annotation(avg, std):
            matrix = [
                [avg["mean_heatmap_mean_mask"], avg["mean_heatmap_var_mask"]],
                [avg["var_heatmap_mean_mask"], avg["var_heatmap_var_mask"]],
            ]
            annotation = [
                [
                    f"{avg['mean_heatmap_mean_mask']:.4f} ± {std['mean_heatmap_mean_mask']:.4f}",
                    f"{avg['mean_heatmap_var_mask']:.4f} ± {std['mean_heatmap_var_mask']:.4f}",
                ],
                [
                    f"{avg['var_heatmap_mean_mask']:.4f} ± {std['var_heatmap_mean_mask']:.4f}",
                    f"{avg['var_heatmap_var_mask']:.4f} ± {std['var_heatmap_var_mask']:.4f}",
                ],
            ]
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
            "rank_acc_annot": rank_annot,
        }


def plot_localization(localization_result, save_path=None):
    # Plotting the heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # IoU heatmap
    sns.heatmap(
        localization_result["iou_matrix"],
        annot=localization_result["iou_annot"],
        cmap="Blues",
        yticklabels=["Mean Heatmap", "Variance Heatmap"],
        xticklabels=["Mean Mask", "Variance Mask"],
        ax=axes[0],
        fmt="",
    )
    axes[0].set_title("IoU Heatmap (Avg. ± Std.)")

    # Mass accuracy heatmap
    sns.heatmap(
        localization_result["mass_acc_matrix"],
        annot=localization_result["mass_acc_annot"],
        cmap="Blues",
        yticklabels=["Mean Heatmap", "Variance Heatmap"],
        xticklabels=["Mean Mask", "Variance Mask"],
        ax=axes[1],
        fmt="",
    )
    axes[1].set_title("Mass Accuracy Heatmap (Avg. ± Std.)")

    # Rank accuracy heatmap
    sns.heatmap(
        localization_result["rank_acc_matrix"],
        annot=localization_result["rank_acc_annot"],
        cmap="Blues",
        yticklabels=["Mean Heatmap", "Variance Heatmap"],
        xticklabels=["Mean Mask", "Variance Mask"],
        ax=axes[2],
        fmt="",
    )
    axes[2].set_title("Rank Accuracy Heatmap (Avg. ± Std.)")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def extend_mask(mask, k):
    """
    Extends a binary mask (torch tensor) by k pixels in all directions.
    
    Parameters:
        mask (torch.Tensor): Binary mask with 1s and 0s (shape: H x W or N x H x W).
        k (int): Number of pixels to extend the mask by.
        
    Returns:
        torch.Tensor: Extended binary mask.
    """
    # Ensure the mask is a binary tensor
    mask = (mask > 0).float()
    
    # Create a kernel of size (2k+1, 2k+1) with all ones
    kernel_size = 2 * k + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    
    # Add batch and channel dimensions if missing
    if mask.ndim == 2:  # H x W
        mask = mask.unsqueeze(0).unsqueeze(0)  # Convert to N x C x H x W
    elif mask.ndim == 3:  # N x H x W
        mask = mask.unsqueeze(1)  # Convert to N x C x H x W
    
    # Perform 2D convolution to dilate the mask
    extended_mask = torch.nn.functional.conv2d(mask, kernel, padding=k)
    
    # Threshold the result to maintain binary output
    extended_mask = (extended_mask > 0).float()
    
    # Remove extra dimensions if they were added
    if extended_mask.shape[1] == 1:
        extended_mask = extended_mask.squeeze(1)
    if extended_mask.shape[0] == 1:
        extended_mask = extended_mask.squeeze(0)
    
    return extended_mask