"""
Code (show_cam_on_image and generate_visualization) adapted from https://github.com/hila-chefer/Transformer-Explainability

Modifications and additions for variance feature attribution
"""

import argparse
import os
import cv2

import torch
from mivolo.data.dataset.age_gender_dataset import AgeGenderDataset
from mivolo.data.dataset.age_gender_loader import create_loader
from mivolo.model.explanation_generator import CAM
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Training Adaption")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        default="hiresCAM",
        choices=["gradCAM", "hiresCAM"],
        help="Explanation method to be used",
    )

    return parser


def main():
    parser = get_parser()
    user_args = parser.parse_args()

    args_dict = {
        "output": "output",
        "detector_weights": "models/yolov8x_person_face.pt",
        "checkpoint": f"models/{user_args.checkpoint}.pth.tar",
        "with_persons": False,
        "disable_faces": False,
        "draw": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "half": False,
    }

    args = argparse.Namespace(**args_dict)

    setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    # Loading the dataset
    test_dataset = AgeGenderDataset(
        "mivolo/data/dataset/images",
        "mivolo/data/dataset/annotations",
        name="test",
        split="test",
        use_persons=False,
        model_with_persons=False,
        is_training=False,
        min_age=predictor.age_gender_model.meta.min_age,
        max_age=predictor.age_gender_model.meta.max_age,
    )

    # Create dataloader
    test_loader = create_loader(
        test_dataset,
        (3, 224, 224),
        1,
        num_workers=8,
        crop_pct=None,
        crop_mode=None,
        pin_memory=True,
        img_dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        persistent_workers=True,
        worker_seeding="all",
        target_type=torch.float,
    )

    #  Prepare model
    model = predictor.age_gender_model.model
    model.eval()

    cam_generator = CAM(model)

    # Prepare for explanations
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    def generate_visualization(original_image, method="hiresCAM", index=3):
        if method == "gradCAM":
            transformer_attribution = cam_generator.generate_grad_cam_attn(
                original_image, index=index
            ).detach()
        elif method == "hiresCAM":
            transformer_attribution = cam_generator.generate_hires_cam_attn(
                original_image, index=index
            ).detach()

        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(
            transformer_attribution, scale_factor=16, mode="bilinear"
        )

        transformer_attribution = transformer_attribution.reshape(224, 224)
        transformer_attribution = transformer_attribution.data.cpu().numpy()
        transformer_attribution = (
            transformer_attribution - transformer_attribution.min()
        ) / (transformer_attribution.max() - transformer_attribution.min())

        normalization_shape = (1, 3, 1, 1)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        mean = torch.tensor([x * 255 for x in mean], device="cuda").view(
            normalization_shape
        )
        std = torch.tensor([x * 255 for x in std], device="cuda").view(
            normalization_shape
        )

        image_transformer_attribution = (
            original_image.mul_(std)
            .add_(mean)
            .div_(255)
            .relu()
            .squeeze()
            .permute(1, 2, 0)
            .data.cpu()
            .numpy()
        )

        vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis

    # Generate explanations
    filenames = test_loader.dataset.filenames()
    os.makedirs(f"xai_images_{user_args.method}_{user_args.checkpoint}", exist_ok=True)
    for idx, (inputs, labels) in tqdm(enumerate(test_loader), total=len(filenames)):
        explanation = generate_visualization(inputs, method=user_args.method)
        plt.axis("off")
        plt.imshow(explanation)
        plt.savefig(
            f'xai_images_{user_args.method}_{user_args.checkpoint}/{filenames[idx].replace("/", "_")}.png',
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()


if __name__ == "__main__":
    main()
