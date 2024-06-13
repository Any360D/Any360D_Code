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
from torchvision.transforms import Compose

import matplotlib.pyplot as plot

from depth_anything_utils import Resize, NormalizeImage, PrepareForNet

from networks.dpt import DPT_DINOv2


if __name__ == '__main__':
    config_path = './configs/test/student_conv.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='./demo_3.mp4')
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    args = parser.parse_args()

    midas_model_type = 'vits'
    # Pre-defined setting of the model
    if midas_model_type == 'vits':
        depth_anything = DPT_DINOv2(encoder=midas_model_type, features=64, out_channels=[48, 96, 192, 384], use_clstoken=False)
    elif midas_model_type == 'vitb':
        depth_anything = DPT_DINOv2(encoder=midas_model_type, features=128, out_channels=[96, 192, 384, 768], use_clstoken=False)
    elif midas_model_type == 'vitl':
        depth_anything = DPT_DINOv2(encoder=midas_model_type, features=256, out_channels=[256, 512, 1024, 1024], use_clstoken=False)
    else:
        raise NotImplementedError
    
    depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{midas_model_type}14.pth'))
    
    margin_width = 20

    caption_height = 70
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 4

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(config["load_weights_dir"], 'model.pth')
    model_dict = torch.load(model_path)

    # network
    model = make(config['model'])
    # model = nn.parallel.DataParallel(model)
    model.cuda()

    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()

    dam = depth_anything
    dam.cuda()
    dam.eval()

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

    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 3 + 2 * margin_width
        
        filename = os.path.basename(filename)
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height+caption_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
            
            frame = transform({'image': frame})['image']
            frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                depth = model(frame)["pred_depth"]
                depth_dam = dam(frame)[0]

            depth = F.interpolate(depth, (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

            depth_dam = F.interpolate(depth_dam, (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
            depth_dam = (depth_dam - depth_dam.min()) / (depth_dam.max() - depth_dam.min()) * 255.0
            depth_dam = depth_dam.cpu().numpy().astype(np.uint8)
            depth_dam_color = cv2.applyColorMap(depth_dam, cv2.COLORMAP_INFERNO)
            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_dam_color, split_region, depth_color])

            caption_space = np.ones((caption_height, combined_frame.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything (Disparity)', 'Ours (Depth)']
            segment_width = frame_width + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (frame_width - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 50), font, font_scale, (0, 0, 0), font_thickness)

            final_result = cv2.vconcat([caption_space, combined_frame])
            
            out.write(final_result)
        
        raw_video.release()
        out.release()