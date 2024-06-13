from __future__ import absolute_import, division, print_function
import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
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
from torchvision.transforms import Compose

import matplotlib.pyplot as plot

from depth_anything_utils import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':
    config_path = './configs/test/student_bin.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = config["load_weights_dir"]
    model_dict = torch.load(model_path)

    # network
    model = make(config['model'])
    # model = nn.parallel.DataParallel(model)
    model.cuda()

    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    transform = Compose([
        Resize(
            width=504,
            height=504,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = os.listdir(args.img_path)
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    print(filenames)
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = model(image)["pred_depth"]
        
        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        filename = os.path.basename(filename)
        
        if args.pred_only:
            # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_rgb.png'), raw_image)
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)