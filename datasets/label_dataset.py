from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import Compose

from PIL import Image, ImageOps, ImageFilter
import torch.nn.functional as F
from einops import rearrange

def read_list(list_file):
    rgb_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_list.append(line.strip())
    return rgb_list


class Label_Dataset(data.Dataset):
    """Label for Unlabel Dataset"""

    def __init__(self, root_dir, list_file, height=504, width=1008):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)

        self.w = width
        self.h = height

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):

        # Read and process the image file
        rgb_name = self.rgb_depth_list[idx]
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        rgb = self.to_tensor(rgb.copy())

        # Conduct output
        inputs = {}

        inputs["rgb"] = self.normalize(rgb)
        inputs["rgb_name"] = rgb_name

        return inputs