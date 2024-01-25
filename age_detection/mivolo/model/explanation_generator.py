"""
Code adapted from https://github.com/hila-chefer/Transformer-Explainability

Modifications and additions for variance feature attribution
"""

from numpy import *


class CAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_grad_cam_attn(self, input, index=None):
        output = self.model(input, register_hook=True)

        self.model.zero_grad()
        output[0, index].exp().backward(retain_graph=True)
        #################### attn
        grad = self.model.post_network[-1].attn.get_attn_gradients()
        cam = self.model.post_network[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)

        grad = grad.mean(dim=[1, 2], keepdim=True)

        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_hires_cam_attn(self, input, index=None):
        output = self.model(input, register_hook=True)

        self.model.zero_grad()
        output[0, index].exp().backward(retain_graph=True)
        #################### attn
        grad = self.model.post_network[-1].attn.get_attn_gradients()
        cam = self.model.post_network[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)

        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn
