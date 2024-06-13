from __future__ import absolute_import, division, print_function
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import tqdm
import yaml
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import datasets
from metrics_disp import Evaluator
from networks.models import *
from saver import Saver

import matplotlib.pyplot as plot
from einops import rearrange

from networks.layers import Cube2Equirec
from networks.tangent_utils import equi2pers, pers2equi
from networks.utils import img2windows, windows2img

from networks.dpt import DPT_DINOv2


def main(config):
    midas_model_type = config['midas_model_type']
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

    # data
    datasets_dict = {"matterport3d": datasets.Matterport3D_Robust}
    cf_test = config['test_dataset']
    dataset = datasets_dict[cf_test['name']]

    test_dataset = dataset(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'])
    test_loader = DataLoader(test_dataset, 
                            cf_test['batch_size'], 
                            False,
                            num_workers=cf_test['num_workers'], 
                            pin_memory=True, 
                            drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // cf_test['batch_size']
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    model = depth_anything
    model.cuda()
    model.eval()

    evaluator_erp = Evaluator(config['median_align'])
    evaluator_erp.reset_eval_metrics()
    evaluator_cube = Evaluator(config['median_align'])
    evaluator_cube.reset_eval_metrics()
    evaluator_tangent = Evaluator(config['median_align'])
    evaluator_tangent.reset_eval_metrics()
    evaluator_horizon = Evaluator(config['median_align'])
    evaluator_horizon.reset_eval_metrics()
    evaluator_vertical = Evaluator(config['median_align'])
    evaluator_vertical.reset_eval_metrics()
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    cube_h = 504 // 2
    equi_h = 504
    equi_w = 1008

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            batch_size = inputs["erp_rgb"].shape[0]
            equi_inputs = inputs["erp_rgb"].cuda()
            equi_pred = model(equi_inputs)[0].detach().cpu()

            cube_inputs = inputs["cube_rgb"].cuda()
            cube_inputs = torch.cat(torch.split(cube_inputs, 504 // 2, dim=-1), dim=0)
            cube_pred = model(cube_inputs)[0].detach().cpu()
            cube_pred = torch.cat(torch.split(cube_pred, batch_size, dim=0), dim=-1)
            cube_pred = Cube2Equirec(cube_h, equi_h, equi_w)(cube_pred)

            tangent_inputs = equi2pers(inputs["erp_rgb"].cuda(), (80, 80), nrows=4, patch_size=(126, 126))[0]
            tangent_inputs = rearrange(tangent_inputs, 'b c h w n -> (b n) c h w')
            tangent_pred = model(tangent_inputs)[0].detach().cpu()
            tangent_pred = rearrange(tangent_pred, '(b n) c h w -> b c h w n', b=batch_size)
            tangent_pred = pers2equi(tangent_pred, fov=(80, 80), patch_size=(126, 126), nrows=4, erp_size=(504, 1008), layer_name='b')

            horizon_inputs = img2windows(inputs["erp_rgb"].cuda(), equi_h // 4, equi_w)
            horizon_inputs = rearrange(horizon_inputs, 'b h w c -> b c h w')
            horizon_pred = model(horizon_inputs)[0].detach().cpu()
            horizon_pred = rearrange(horizon_pred, 'b c h w -> b h w c')
            horizon_pred = windows2img(horizon_pred, equi_h // 4, equi_w, equi_h, equi_w)
            horizon_pred = rearrange(horizon_pred, 'b h w c -> b c h w')
            
            vertical_inputs = img2windows(inputs["erp_rgb"].cuda(), equi_h, equi_w // 4)
            vertical_inputs = rearrange(vertical_inputs, 'b h w c -> b c h w')
            vertical_pred = model(vertical_inputs)[0].detach().cpu()
            vertical_pred = rearrange(vertical_pred, 'b c h w -> b h w c')
            vertical_pred = windows2img(vertical_pred, equi_h, equi_w // 4, equi_h, equi_w)
            vertical_pred = rearrange(vertical_pred, 'b h w c -> b c h w')
        
            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]
            for i in range(gt_depth.shape[0]):
                evaluator_erp.compute_eval_metrics(gt_depth[i:i + 1], equi_pred[i:i + 1], mask[i:i + 1])
                evaluator_cube.compute_eval_metrics(gt_depth[i:i + 1], cube_pred[i:i + 1], mask[i:i + 1])
                evaluator_tangent.compute_eval_metrics(gt_depth[i:i + 1], tangent_pred[i:i + 1], mask[i:i + 1])
                evaluator_horizon.compute_eval_metrics(gt_depth[i:i + 1], horizon_pred[i:i + 1], mask[i:i + 1])
                evaluator_vertical.compute_eval_metrics(gt_depth[i:i + 1], vertical_pred[i:i + 1], mask[i:i + 1])

    # evaluator.print(config["load_weights_dir"])
    print("ERP")
    evaluator_erp.print()
    print("Cube")
    evaluator_cube.print()
    print("Tangent")
    evaluator_tangent.print()
    print("Horizon")
    evaluator_horizon.print()
    print("Vertical")
    evaluator_vertical.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config)