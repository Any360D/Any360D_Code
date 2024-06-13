from __future__ import absolute_import, division, print_function
#Successful! Best!#
import json
import os
import time
import random
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

torch.manual_seed(100)
torch.cuda.manual_seed(100)

import datasets
from networks.models import *
from metrics import compute_depth_metrics, Evaluator
from losses import *

from mobius_utils import make_coord, warp_mobius_image, get_random_mobius


class Semi_Trainer:
    def __init__(self, config_, save_path_):
        self.config = config_
        self.save_path = save_path_
        self.best_abs = 100

        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

        # data
        datasets_dict = {"matterport3d": datasets.Matterport3D,
                         "zind": datasets.Zind,}
        
        cf_label = self.config['label_dataset']
        self.label_dataset = datasets_dict[cf_label['name']]
        label_dataset = self.label_dataset(cf_label['root_path'], 
                                     cf_label['list_path'],
                                     cf_label['args']['height'],
                                     cf_label['args']['width'], 
                                     cf_label['args']['augment_color'],
                                     cf_label['args']['augment_flip'],
                                     cf_label['args']['augment_rotation'],
                                     cf_label['args']['repeat'],
                                     is_training=True)
        self.label_loader = DataLoader(label_dataset, 
                                       cf_label['batch_size'], 
                                       True,
                                       num_workers=cf_label['num_workers'], 
                                       pin_memory=True, 
                                       drop_last=True)
           
        cf_unlabel = self.config['unlabel_dataset']
        self.unlabel_dataset = datasets_dict[cf_unlabel['name']]
        unlabel_dataset = self.unlabel_dataset(cf_unlabel['root_path'], 
                                     cf_unlabel['list_path'],
                                     cf_unlabel['args']['height'],
                                     cf_unlabel['args']['width'], 
                                     cf_unlabel['args']['augment_color'],
                                     cf_unlabel['args']['augment_flip'],
                                     cf_unlabel['args']['augment_rotation'],
                                     cf_unlabel['args']['repeat'],
                                     is_training=True)
        self.unlabel_loader = DataLoader(unlabel_dataset, 
                                       cf_unlabel['batch_size'], 
                                       True,
                                       num_workers=cf_unlabel['num_workers'], 
                                       pin_memory=True, 
                                       drop_last=True)

        cf_val = self.config['val_dataset']
        self.val_dataset = datasets_dict[cf_val['name']]
        val_dataset = self.val_dataset(cf_val['root_path'], 
                                     cf_val['list_path'],
                                     cf_val['args']['height'],
                                     cf_val['args']['width'], 
                                     cf_val['args']['augment_color'],
                                     cf_val['args']['augment_flip'],
                                     cf_val['args']['augment_rotation'],
                                     cf_val['args']['repeat'],
                                     is_training=False)
        self.val_loader = DataLoader(val_dataset, 
                                     cf_val['batch_size'], 
                                     False,
                                     num_workers=cf_val['num_workers'], 
                                     pin_memory=True, 
                                     drop_last=True)
        
        num_train_samples = len(label_dataset) + len(unlabel_dataset)
        self.num_total_steps = num_train_samples // (cf_label['batch_size'] + cf_unlabel['batch_size']) * self.config['epoch_max']

        # network
        self.model = make(self.config['model'])
        self.model.cuda()

        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.config['optimizer']['lr'])

        if self.config.get('load_weights_dir') is not None:
            self.load_model()
        
        losses_dict = {"berhu": BerhuLoss(),
                       "silog": Silog_Loss(),
                       "rmselog": RMSELog(),
                       'scale_invariant': ScaleAndShiftInvariantLoss(),
                       'affine_invariant': Affine_Invariant_Loss(),
                       'cosine': CosLoss(),
                       'l1': L1Loss()}
        self.basic_loss = losses_dict[self.config['loss'][0]]
        if len(self.config['loss']) > 1:
            self.refine_loss = losses_dict[self.config['loss'][1]]
       
        self.evaluator = Evaluator()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.save_path, mode))

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        # self.validate()
        
        for self.epoch in range(self.config['epoch_max']):
            self.train_one_epoch()
            if (self.epoch + 1) % self.config['epoch_save'] == 0:
                self.save_model(if_best=False)
            self.validate()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(zip(self.label_loader, self.unlabel_loader))
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):
            label_inputs, unlabel_inputs = inputs
            outputs, losses = self.process_batch(label_inputs, unlabel_inputs, val=False)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.config['log_frequency'] == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                b = label_inputs["gt_depth"].size(0)
                pred_depth = outputs['pred_depth'][:b].detach()
                gt_depth = label_inputs["gt_depth"]
                mask = label_inputs["val_mask"]

                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                self.log("train", label_inputs, outputs, losses)

            self.step += 1

    def process_batch(self, label_inputs, unlabel_inputs, val):
        for key, ipt in label_inputs.items():
            label_inputs[key] = ipt.cuda()

        for key, ipt in unlabel_inputs.items():
            unlabel_inputs[key] = ipt.cuda()

        losses = {}

        if self.config['mobius']['conduct'] and not val:
            b, c, h, w = label_inputs["gt_depth"].shape
            # Supervised Learning
            outputs_1 = self.model(label_inputs["rgb"])
            loss_label = self.basic_loss(label_inputs["gt_depth"], outputs_1['pred_depth'], label_inputs["val_mask"])
            # Pseudo-labeling Learning
            outputs_2 = self.model(unlabel_inputs["raw_rgb"])
            loss_psuedo = self.basic_loss(unlabel_inputs["gt_depth"], outputs_2['pred_depth'], unlabel_inputs["val_mask"])
            # Consistency Learning for Color Augmentation
            outputs_3 = self.model(unlabel_inputs["strong_rgb"])
            loss_cons_color = self.basic_loss(outputs_2['pred_depth'], outputs_3['pred_depth'], unlabel_inputs["val_mask"])
            # Consistency Learning for Weak Augmentation
            M = get_random_mobius(self.config['mobius']['vertical_res'], self.config['mobius']['zoom_res']).cuda()
            coord_hr = make_coord([h, w], flatten=True).unsqueeze(0)
            mobius_inputs = warp_mobius_image(unlabel_inputs["raw_rgb"], M, coord_hr, pole='Equator')
            outputs_4 = self.model(mobius_inputs)
            mobius_gt = warp_mobius_image(outputs_2['pred_depth'], M, coord_hr, pole='Equator')
            mobius_mask = ((mobius_gt > 0) & (mobius_gt <= 10.0) & ~torch.isnan(mobius_gt))
            loss_cons_mobius = self.basic_loss(mobius_gt, outputs_4['pred_depth'], mobius_mask)
            losses["loss"] = self.config['loss_weights'][0] * loss_label + self.config['loss_weights'][1] * loss_psuedo + \
                            self.config['loss_weights'][2] * loss_cons_color + self.config['loss_weights'][3] * loss_cons_mobius
            # print(loss_label, loss_psuedo, loss_cons_color, loss_cons_mobius)
        else:
            b, c, h, w = label_inputs["gt_depth"].shape
            # Supervised Learning
            outputs_1 = self.model(label_inputs["rgb"])
            loss_label = self.basic_loss(label_inputs["gt_depth"], outputs_1['pred_depth'], label_inputs["val_mask"])
            # Pseudo-labeling Learning
            outputs_2 = self.model(unlabel_inputs["raw_rgb"])
            loss_psuedo = self.basic_loss(unlabel_inputs["gt_depth"], outputs_2['pred_depth'], unlabel_inputs["val_mask"])
            # Consistency Learning for Strong Augmentation
            outputs_3 = self.model(unlabel_inputs["strong_rgb"])
            loss_cons_strong = self.basic_loss(outputs_2['pred_depth'], outputs_3['pred_depth'], unlabel_inputs["val_mask"])
            # Total Loss
            losses["loss"] = self.config['loss_weights'][0] * loss_label + self.config['loss_weights'][1] * loss_psuedo + \
                            self.config['loss_weights'][2] * loss_cons_strong
            # print(loss_label, loss_psuedo, loss_cons_strong, loss_cons_weak)
            
        return outputs_1, losses
    
    def process_batch_val(self, inputs, val):
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()

        losses = {}

        equi_inputs = inputs["rgb"]
        gt_depth = inputs["gt_depth"]

        outputs = self.model(equi_inputs)
        loss_basic = self.basic_loss(gt_depth, outputs['pred_depth'], inputs["val_mask"])
        losses["loss"] = loss_basic

        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch_val(inputs, val=True)
                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                # gt_depth = rearrange(inputs["gt_depth"], 'b n c h w -> (b n) c h w')
                # mask = rearrange(inputs["val_mask"], 'b n c h w -> (b n) c h w')
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth, mask)

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        
        abs = losses['err/rms']
        if abs < self.best_abs:
            self.best_abs = abs
            self.save_model(if_best=True)

        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses=None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        # for l, v in losses.items():
        #     writer.add_scalar("{}".format(l), v, self.step)

        for j in range(1):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            writer.add_image("gt_depth/{}".format(j),
                             inputs["gt_depth"][j].data/inputs["gt_depth"][j].data.max(), self.step)
            # writer.add_image("coarse_depth/{}".format(j),
            #                  outputs["coarse_depth"][j].data/outputs["coarse_depth"][j].data.max(), self.step)
            writer.add_image("pred_depth/{}".format(j),
                             outputs["pred_depth"][j].data/outputs["pred_depth"][j].data.max(), self.step)

    def save_model(self, if_best=False):
        """Save model weights to disk _withoutVT
        """
        if not if_best:
            save_folder = os.path.join(self.save_path, "weights_{}".format(self.epoch))
        else:
            save_folder = os.path.join(self.save_path, "best")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.evaluator.print(save_folder)
        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        # save resnet layers - these are needed at prediction time
        # save the input sizes
        # save the dataset to train on
        to_save['epoch'] = self.epoch
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        load_weights_dir = os.path.expanduser(os.path.expanduser(self.config['load_weights_dir']))

        assert os.path.isdir(load_weights_dir), \
            "Cannot find folder {}".format(load_weights_dir)
        print("loading model from folder {}".format(load_weights_dir))

        path = os.path.join(load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
