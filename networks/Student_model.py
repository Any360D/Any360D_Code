import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

from .dpt import DPT_DINOv2

from safetensors.torch import save_file
from safetensors import safe_open
from torch.nn.parameter import Parameter

import torchvision
import cv2
import copy

from .utils import LoRA_Depth_Anything
from losses import *

from argparse import Namespace
from .models import register

from .zoedepth_layers.attractor import AttractorLayer, AttractorLayerUnnormed
from .zoedepth_layers.dist_layers import ConditionalLogBinomial
from .zoedepth_layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)


def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Student_Model(nn.Module):
    def __init__(self, args):
        """Depth Any 360 model with one branch and bin prediction
        """
        super().__init__()
        
        midas_model_type = args.midas_model_type
        min_depth = args.min_depth
        max_depth = args.max_depth
        lora = args.lora
        train_decoder = args.train_decoder
        btlnck_features = args.btlnck_features

        # conf = get_config("zoedepth", "train")
        # self.backbone = build_model(conf)

        # Pre-defined setting of the model
        if midas_model_type == 'vits':
            depth_anything = DPT_DINOv2(encoder=midas_model_type, features=64, out_channels=[48, 96, 192, 384], use_clstoken=False)
        elif midas_model_type == 'vitb':
            depth_anything = DPT_DINOv2(encoder=midas_model_type, features=128, out_channels=[96, 192, 384, 768], use_clstoken=False)
        elif midas_model_type == 'vitl':
            depth_anything = DPT_DINOv2(encoder=midas_model_type, features=256, out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        else:
            raise NotImplementedError
        
        # Load the pretrained model of depth anything
        depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{midas_model_type}14.pth'))

        # Apply LoRA to the model for erp branch
        if lora:
            self.core = depth_anything
            LoRA_Depth_Anything(depth_anything, 4)
            if not train_decoder:
                for param in self.core.depth_head.parameters():
                    param.requires_grad = False
        else:
            self.core = depth_anything
            if not train_decoder:
                for param in self.core.parameters():
                    param.requires_grad = False

        self.core_out = {}
        self.handles = []
        self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.set_fetch_features(True)

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv
        SeedBinRegressorLayer = SeedBinRegressorUnnormed
        Attractor = AttractorLayerUnnormed
        n_bins = 64
        bin_embedding_dim = 128
        num_out_features = [btlnck_features, btlnck_features, btlnck_features, btlnck_features]
        n_attractors = [16, 8, 4, 1]
        attractor_alpha = 300
        attractor_gamma = 2
        attractor_kind = 'sum'
        attractor_type = 'exp'
        N_MIDAS_OUT = 32
        min_temp = 5
        max_temp = 50
        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def set_fetch_features(self, fetch_features):
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self
    
    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(midas.depth_head.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self
    
    def forward(self, image):

        rel_depth, _ = self.core(image)
        out = [self.core_out[k] for k in self.layer_names]

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        outputs = {}
        outputs["pred_depth"] = out

        return outputs
    
@register('student_model')
def make_model(midas_model_type='vits', min_depth=0.1, max_depth=10.0, lora=True, train_decoder=True, btlnck_features=256):
    args = Namespace()
    args.midas_model_type = midas_model_type
    args.min_depth = min_depth
    args.max_depth = max_depth
    args.lora = lora
    args.train_decoder = train_decoder
    args.btlnck_features = btlnck_features
    return Student_Model(args)