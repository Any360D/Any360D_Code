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
    datasets_dict = {"matterport3d": datasets.Matterport3D}
    cf_test = config['test_dataset']
    dataset = datasets_dict[cf_test['name']]

    test_dataset = dataset(cf_test['root_path'], 
                            cf_test['list_path'],
                            cf_test['args']['height'],
                            cf_test['args']['width'], 
                            cf_test['args']['augment_color'],
                            cf_test['args']['augment_flip'],
                            cf_test['args']['augment_rotation'],
                            cf_test['args']['repeat'],
                            is_training=False)
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

    evaluator = Evaluator(config['median_align'])
    evaluator.reset_eval_metrics()
    saver = Saver(config["load_weights_dir"])
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")

    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            equi_inputs = inputs["rgb"].cuda()

            # from thop import profile
            # from thop import clever_format
            # macs, params = profile(model, inputs=(equi_inputs, cube_inputs))
            # print(macs, params) 
            # macs, params = clever_format([macs, params], "%.3f")
            # print(macs, params) 
            # assert False
            
            outputs = model(equi_inputs)

            pred_depth = outputs['pred_depth']
            pred_depth =pred_depth.detach().cpu()
            # outputs = F.interpolate(outputs[None], (512, 1024), mode='bilinear', align_corners=False)
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.detach().cpu().numpy().astype(np.uint8)[0, 0]
            # print(depth.shape)
            # depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # cv2.imwrite('real_world_pred.png', depth)

            gt_depth = inputs["gt_depth"]
            mask = inputs["val_mask"]
            for i in range(gt_depth.shape[0]):
                evaluator.compute_eval_metrics(gt_depth[i:i + 1], pred_depth[i:i + 1], mask[i:i + 1])
    evaluator.print()


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