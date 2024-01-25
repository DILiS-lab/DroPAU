""" Vision OutLOoker (VOLO) implementation

Paper: `VOLO: Vision Outlooker for Visual Recognition` - https://arxiv.org/abs/2106.13112

Code adapted from official impl at https://github.com/sail-sg/volo, original copyright in comment below

Modifications and additions for timm by / Copyright 2022, Ross Wightman

Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for variance feature attribution in ClassAttention class
"""
# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, Mlp, to_2tuple, to_ntuple, trunc_normal_


__all__ = ["VOLO"]  # model_registry will add each entrypoint fn to this


class OutlookAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=3,
        padding=1,
        stride=1,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = (
            self.unfold(v)
            .reshape(
                B,
                self.num_heads,
                C // self.num_heads,
                self.kernel_size * self.kernel_size,
                h * w,
            )
            .permute(0, 1, 4, 3, 2)
        )  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = (
            self.attn(attn)
            .reshape(
                B,
                h * w,
                self.num_heads,
                self.kernel_size * self.kernel_size,
                self.kernel_size * self.kernel_size,
            )
            .permute(0, 2, 1, 3, 4)
        )  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .permute(0, 1, 4, 3, 2)
            .reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        )
        x = F.fold(
            x,
            output_size=(H, W),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        padding,
        stride=1,
        num_heads=1,
        mlp_ratio=3.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        qkv_bias=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(
            dim,
            num_heads,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, self.head_dim * self.num_heads * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (
            (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        )

        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(
            self.attn(self.norm1(x), register_hook=register_hook)
        )
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


def get_block(block_type, **kargs):
    if block_type == "ca":
        return ClassBlock(**kargs)


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = (W * cut_rat).astype(int)
    cut_h = (H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(
        self,
        img_size=224,
        stem_conv=False,
        stem_stride=1,
        patch_size=8,
        in_chans=3,
        hidden_dim=64,
        embed_dim=384,
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    hidden_dim,
                    kernel_size=7,
                    stride=stem_stride,
                    padding=3,
                    bias=False,
                ),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = None

        self.proj = nn.Conv2d(
            hidden_dim,
            embed_dim,
            kernel_size=patch_size // stem_stride,
            stride=patch_size // stem_stride,
        )
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """Image to Patch Embedding, downsampling between stage1 and stage2"""

    def __init__(self, in_embed_dim, out_embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(
            in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def outlooker_blocks(
    block_fn,
    index,
    dim,
    layers,
    num_heads=1,
    kernel_size=3,
    padding=1,
    stride=2,
    mlp_ratio=3.0,
    qkv_bias=False,
    attn_drop=0,
    drop_path_rate=0.0,
    **kwargs,
):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            block_fn(
                dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                drop_path=block_dpr,
            )
        )
    return nn.Sequential(*blocks)


def transformer_blocks(
    block_fn,
    index,
    dim,
    layers,
    num_heads,
    mlp_ratio=3.0,
    qkv_bias=False,
    attn_drop=0,
    drop_path_rate=0.0,
    **kwargs,
):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            block_fn(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                drop_path=block_dpr,
            )
        )
    return nn.Sequential(*blocks)


class VOLO(nn.Module):
    """
    Vision Outlooker, the main class of our model
    """

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        use_mix_token=False,
        pooling_scale=2,
    ):
        super().__init__()
        num_layers = len(layers)
        mlp_ratio = to_ntuple(num_layers)(mlp_ratio)
        img_size = to_2tuple(img_size)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.mix_token = use_mix_token
        self.pooling_scale = pooling_scale
        self.num_features = embed_dims[-1]
        if use_mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert global_pool == "token", "return all tokens if mix_token is enabled"
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        # inital positional encoding, we add positional encoding after outlooker blocks
        patch_grid = (
            img_size[0] // patch_size // pooling_scale,
            img_size[1] // patch_size // pooling_scale,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, patch_grid[0], patch_grid[1], embed_dims[-1])
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = outlooker_blocks(
                    Outlooker,
                    i,
                    embed_dims[i],
                    layers,
                    num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
            else:
                # stage 2
                stage = transformer_blocks(
                    Transformer,
                    i,
                    embed_dims[i],
                    layers,
                    num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    drop_path_rate=drop_path_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
            network.append(stage)
            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.ModuleList(network)

        # set post block, for example, class attention layers
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.ModuleList(
                [
                    get_block(
                        post_layers[i],
                        dim=embed_dims[-1],
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio[-1],
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop_rate,
                        drop_path=0.0,
                        norm_layer=norm_layer,
                    )
                    for i in range(len(post_layers))
                ]
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
            trunc_normal_(self.cls_token, std=0.02)

        # set output type
        if use_aux_head:
            self.aux_head = (
                nn.Linear(self.num_features, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            self.aux_head = None
        self.norm = norm_layer(self.num_features)

        # Classifier head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[
                (r"^network\.(\d+)\.(\d+)", None),
                (r"^network\.(\d+)", (0,)),
            ],
            blocks2=[
                (r"^cls_token", (0,)),
                (r"^post_network\.(\d+)", None),
                (r"^norm", (99999,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        if self.aux_head is not None:
            self.aux_head = (
                nn.Linear(self.num_features, num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            if idx == 2:
                # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x)
            else:
                x = block(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward_cls(self, x, register_hook=False):
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for block in self.post_network:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, register_hook=register_hook)
            else:
                x = block(x, register_hook=register_hook)
        return x

    def forward_features(self, x, register_hook=False):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        # step2: tokens learning in the two stages
        x = self.forward_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x, register_hook=register_hook)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        x = self.head_drop(x)
        if pre_logits:
            return out
        out = self.head(out)
        if self.aux_head is not None:
            # generate classes in all feature tokens, see token labeling
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]
        return out

    def forward(self, x, register_hook=False):
        """simplified forward (without mix token training)"""
        x = self.forward_features(x, register_hook=register_hook)
        x = self.forward_head(x)
        return x
