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
from metrics import Evaluator
from networks.models import *
from saver import Saver

import matplotlib.pyplot as plot


def main(config):
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    model_dict = torch.load(model_path)

    # data
    datasets_dict = {"label": datasets.Label_Dataset}
    cf_test = config['test_dataset']
    dataset = datasets_dict[cf_test['name']]

    test_dataset = dataset(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'], )
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
    model = make(config['model'])
    # model = nn.parallel.DataParallel(model)
    model.cuda()

    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["rgb"].cuda()
            
            outputs = model(equi_inputs)

            pred_depth = outputs['pred_depth']
            pred_depth =pred_depth.detach().cpu()

            raw_path = inputs["rgb_name"][0]
            save_path = raw_path.replace("Dataset", "unlabel_depth")
            save_path = save_path.replace("pano_", "depth_").replace(".jpg", ".npy")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pred_depth = (pred_depth.numpy()[0, 0])
            np.save(save_path, pred_depth)
            # cv2.imwrite(save_path, pred_depth)

            # depth = F.interpolate(pred_depth, (512, 1024), mode='bilinear', align_corners=False)
            depth = pred_depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            visual_path = save_path.replace("depth_", "visual_").replace(".npy", ".png")
            cv2.imwrite(visual_path, depth)

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