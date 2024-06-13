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

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

class Zind(data.Dataset):
    """The Zind Dataset with Label from Teacher Model"""

    def __init__(self, root_dir, list_file, height=504, width=1008, color_augmentation=True,
                 LR_filp_augmentation=True, yaw_rotation_augmentation=True, repeat=1, is_training=False):
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
        self.rgb_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0
        self.min_depth_meters = 0.01

        self.color_augmentation = color_augmentation
        self.LR_filp_augmentation = LR_filp_augmentation
        self.yaw_rotation_augmentation = yaw_rotation_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.is_training = is_training

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):

        # Read and process the image file
        rgb_name = self.rgb_list[idx]
        rgb = cv2.imread(rgb_name)
        if rgb is None:
            print(rgb_name)
            assert False
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        # Read and process the depth file
        depth_name = rgb_name.replace( \
            "./Dataset", \
            "./unlabel_depth") \
            .replace("pano_", "depth_").replace(".jpg", ".npy")
        try:
            gt_depth = np.load(depth_name)
        except:
            print(depth_name)
            assert False
        # gt_depth = np.load(depth_name)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(float)
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1
        # from matplotlib import pyplot as plt
        # plt.imsave("test.png", gt_depth)
        # assert False
        
        raw_rgb = rgb.copy()

        if self.is_training and self.color_augmentation:
            strong_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(raw_rgb)))
        else:
            strong_rgb = raw_rgb.copy()

        raw_rgb = self.to_tensor(raw_rgb.copy())
        strong_rgb = self.to_tensor(strong_rgb.copy())

        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)

        # Conduct output
        inputs = {}

        inputs["raw_rgb"] = self.normalize(raw_rgb)
        inputs["strong_rgb"] = self.normalize(strong_rgb)
        inputs["gt_depth"] = gt_depth
        inputs["val_mask"] = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters)
                                & ~torch.isnan(gt_depth))

        return inputs