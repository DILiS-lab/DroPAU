import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableCNNModule


device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_float32_matmul_precision("medium")

output_dir = "./combined_mnist_dataset"
num_samples = 500000

train_dataset, val_dataset, test_dataset = get_loaders(
    output_dir, num_samples, get_sets="splits"
)

l_model = SwitchableCNNModule.load_from_checkpoint(
    "checkpoints/best_model_1736245990.ckpt"
)
model = l_model.model.to(device)


# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self):
        # Average the gradients spatially
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted sum of activations
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        # Apply ReLU to focus only on positive contributions
        # heatmap = torch.relu(heatmap)
        # Normalize the heatmap

        heatmap = heatmap + heatmap.min()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        # heatmap = heatmap / (
        #     heatmap.max() + 1e-10
        # )  # Add epsilon to prevent division by zero
        return heatmap


# Instantiate the Grad-CAM object for the shared CNN layers
# We use the last convolutional layer in the shared CNN network
target_layer = model.cnn[3]  # This is the last convolutional layer before flattening

gradcam = GradCAM(model, target_layer)

# samples = np.random.choice(range(len(dataset) + 1), size=10, replace=False)
# Test Grad-CAM on a sample image

localization = Localization()

model.eval()
for i in tqdm(range(len(test_dataset))):  # Loop over 10 samples
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass
    # print(model(image.to(device)).unsqueeze(0).shape)
    mean_output, log_var_output = model(image.to(device))

    # Grad-CAM for mean
    mean_output[0].backward(retain_graph=True)  # Backpropagate for mean
    mean_heatmap = gradcam.generate_heatmap()
    # print(mean_heatmap.shape)

    # Reset gradients after mean backward pass
    model.zero_grad()

    # Grad-CAM for variance
    # print(log_var_output)
    log_var_output[0].backward(retain_graph=True)  # Backpropagate for variance
    var_heatmap = gradcam.generate_heatmap()
    # Reset gradients after mean backward pass
    model.zero_grad()

    localization.add_sample(
        mean_heatmap.cpu().detach(), extend_mask(mean_mask, 2).cpu().detach(), var_heatmap.cpu().detach(), extend_mask(var_mask, 2).cpu().detach(), uc.detach()
    )

    # Visualize the heatmaps
    # print("Variance Prediction Heatmap:")
    # print(var_heatmap.shape)
    # visualize_heatmap(image.detach(), mean_heatmap.detach(), var_heatmap.detach(), "gradcam")


localization.save(file_name="mnist_plus_gradcam_extended2")
localization_result = localization.calculate_acc_matrices()

plot_localization(localization_result, save_path="mnist_plus_gradcam_extended2.png")
