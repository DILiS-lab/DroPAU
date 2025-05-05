import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from zennit.attribution import Gradient
from zennit.composites import NameMapComposite, SpecialFirstLayerMapComposite
from zennit.image import imgify
from zennit.rules import AlphaBeta, Epsilon, Norm, Pass, ZBox, Gamma

from CLRP.clrp_lib import *
from CLRP.run import clrp, visualize
from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableDoublePathCNNModule

import numpy as np

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

l_model = SwitchableDoublePathCNNModule.load_from_checkpoint(
    "checkpoints_two_models/best_model_1736941275.ckpt"
)
mean_model = l_model.mean_model.to(device)
var_model = l_model.variance_model.to(device)

name_map = [
    (["cnn.0"], ZBox(low=0, high=1)),  # Conv2d
    (["cnn.1"], Pass()),  # ReLU
    (["cnn.2"], AlphaBeta(alpha=1, beta=0)),  # Conv2d
    (["cnn.3"], Pass()),  # ReLU
    (["cnn.4"], Norm()),  # MaxPool2d
    (["fc.0"], Epsilon(zero_params="bias")),  # Linear
    (["fc.1"], Pass()),  # ReLU
    (["fc.2"], Pass()),  # Dropout
    (["fc.3"], Epsilon(zero_params="bias")),  # Linear
    (["fc.4"], Pass()),  # ReLU
    (["fc.5"], Epsilon(zero_params="bias")),  # Linear
    (["fc.6"], Pass()),  # ReLU
    (["fc.7"], Epsilon(zero_params="bias")),  # Linear
]


def get_lrp_explanation(model, img, name_map, exp_target="mean", save_image=None):
    # model.focus = exp_target
    # create composite (the zennit way of specifying the rules)
    composite = NameMapComposite(
        name_map=name_map,
    )

    # choose a target for the attribution by multiplying the output with the identity matrix
    target_base = model(img)
    # print(model(img), model(img), model(img), model(img))
    # print(target_base)
    # appying ReLU if the target is variance
    target = target_base if exp_target == "mean" else target_base.exp()
    # target = torch.ones_like(target_base)

    # create the attributor - zennit explainer
    with Gradient(model=model, composite=composite) as attributor:
        _, attribution = attributor(img, target)

    # sum over the channels to get the pixel-wise relevance
    relevance = attribution.sum(1)
    relevance_summed = relevance.sum()
    # diff = abs(relevance_summed.item() - target.item())
    # if diff > 1:
    #     print(f"{exp_target}: {diff}")

    # print(f"{exp_target}: Relevance Summed: {relevance_summed.item()}, Target: {target.item()}")

    # create an image of the visualize attribution
    img_res = imgify(relevance.cpu(), symmetric=True, cmap="bwr")
    # convert image to RGB mode if necessary
    if img_res.mode != "RGB":
        img_res = img_res.convert("RGB")

    if save_image is not None:
        img_res.save(
            f"lrp_zennit_results/attribution_{exp_target}_{save_image}_a1_b0_extended2.png"
        )
    # model.focus = None
    # print(relevance.shape)
    return relevance.squeeze()


localization = Localization()

mean_model.eval()
var_model.eval()
for i in tqdm(range(len(test_dataset))):  
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    # print(image.shape)
    torch.save(image, f"mnist_plus_lrp_zennit_double_path_a1_b0_extended2_input_{i}.pt")

    random = np.random.randint(0, 2500)
    mean_heatmap = get_lrp_explanation(
        mean_model,
        image,
        name_map,
        exp_target="mean",
        save_image=i if random == 0 else None,
    )
    # display(mean_heatmap)
    # Reset gradients after mean backward pass
    mean_model.zero_grad()

    # Grad-CAM for variance
    var_heatmap = get_lrp_explanation(
        var_model,
        image,
        name_map,
        exp_target="variance",
        save_image=i if random == 0 else None,
    )
    # print(var_heatmap.shape)
    # print( var_mask.cpu().squeeze().numpy().shape)
    # plt.imsave("varmask.png", var_mask.cpu().squeeze().numpy())
    # plt.imsave("varmask_extendend1.png", extend_mask(var_mask, 1).cpu().squeeze().numpy())
    # plt.imsave("varmask_extendend2.png", extend_mask(var_mask, 2).cpu().squeeze().numpy())

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )

    var_model.zero_grad()


localization.save(file_name="mnist_plus_lrp_zennit_double_path_a1_b0_extended2")

localization_result = localization.calculate_acc_matrices()

plot_localization(
    localization_result,
    save_path="mnist_plus_lrp_zennit_double_path_a1_b0_extended2.png",
)
