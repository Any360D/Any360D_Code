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


def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class Student_Model_Conv(nn.Module):
    def __init__(self, args):
        """Depth Any 360 model with one branch
        """
        super().__init__()
        
        midas_model_type = args.midas_model_type
        min_depth = args.min_depth
        max_depth = args.max_depth
        lora = args.lora
        train_decoder = args.train_decoder

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

        self.sigmoid = nn.Sigmoid()

        self.max_depth = nn.Parameter(torch.tensor(10.0), requires_grad=False)

        self.convert_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.convert_2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    
    def erp_forward(self, image):
        erp_pred, _ = self.core(image)
        convert_feats = self.convert_1(erp_pred)
        convert_pred = self.convert_2(convert_feats)
        return convert_pred
    
    def forward(self, image):
        b, c, h, w = image.shape

        # Forward of erp image
        erp_pred = self.erp_forward(image)
        erp_pred = self.max_depth * self.sigmoid(erp_pred)
      
        outputs = {}
        outputs["pred_depth"] = erp_pred

        return outputs
    
@register('student_model_conv')
def make_model(midas_model_type='vits', min_depth=0.1, max_depth=10.0, lora=True, train_decoder=True):
    args = Namespace()
    args.midas_model_type = midas_model_type
    args.min_depth = min_depth
    args.max_depth = max_depth
    args.lora = lora
    args.train_decoder = train_decoder
    return Student_Model_Conv(args)