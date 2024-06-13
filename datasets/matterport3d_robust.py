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

from networks.tangent_utils import equi2pers, pers2equi
from networks.utils import img2windows
from .util import Equirec2Cube

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Matterport3D_Robust(data.Dataset):
    """The Matterport3D Dataset with different representations for evaluation"""

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

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0
        self.min_depth_meters = 0.01


        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.rgb_depth_list = read_list(list_file)

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):

        # Read and process the image file
        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # Read and process the depth file
        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float)/4000
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        # Disparity
        gt_disp = gt_depth.copy()
        gt_disp[gt_disp > 0] = 1.0 / gt_disp[gt_disp > 0]
        gt_disp = gt_disp.astype(np.float32)

        # ERP
        erp_rgb = self.to_tensor(rgb.copy())
        # Cube Map
        cube_rgb = self.e2c.run(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb)

        gt_disp = torch.from_numpy(np.expand_dims(gt_disp, axis=0)).to(torch.float32)

        # Conduct output
        inputs = {}

        inputs["erp_rgb"] = self.normalize(erp_rgb)
        inputs["cube_rgb"] = self.normalize(cube_rgb)
        inputs["gt_depth"] = gt_disp
        inputs["val_mask"] = ((gt_disp > 0) & ~torch.isnan(gt_disp))

        return inputs