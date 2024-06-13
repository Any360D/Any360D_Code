from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(10)
torch.cuda.manual_seed(10)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        valid_mask = (target > 0).detach()

        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = target - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()
        return loss


class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
    
class Silog_Loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(Silog_Loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, target, pred, mask=None):
        d = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
    
class RMSELog(nn.Module):
    def __init__(self):
        super(RMSELog, self).__init__()

    def forward(self, target, pred, mask=None):
        #assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()
        target = target[valid_mask]
        pred = pred[valid_mask]
        log_error = torch.abs(torch.log(target / (pred + 1e-12)))
        loss = torch.sqrt(torch.mean(log_error**2))
        return loss
    
class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()
    
    def forward(self, target_feat, pred_feat, valid_mask=None):
        loss = 0
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        for i in range(len(target_feat)):
            similarity = cos(target_feat[i], pred_feat[i])
            mask = (similarity < 0.85)
            loss += torch.mean(1 - similarity[mask])
            # loss += torch.mean(1 - similarity)
        return loss
    
class Affine_Invariant_Loss(nn.Module):
    def __init__(self):
        super(Affine_Invariant_Loss, self).__init__()

    def align(self, x):
        t_x = torch.median(x)
        s_x = torch.mean(torch.abs(x - t_x))
        return (x - t_x) / (s_x + 1e-3)
    
    def noarmalize(self, x):
        # max_disp = torch.max(x, dim=0, keepdim=True)[0]
        max_disp = x.max()
        min_disp = x[x > 0].min()
        x[x > 0] = (x[x > 0] - min_disp) / (max_disp - min_disp + 0.001)
        # x = (x - min_disp) / (max_disp - min_disp + 0.001)
        return x

    def forward(self, target, pred, mask=None):
        if mask is None:
            mask = target > 0

        target = self.noarmalize(target)

        target = target[mask]
        pred = pred[mask]
        
        target = self.align(target)
        pred = self.align(pred)
        return torch.mean(torch.abs(target - pred))
    
def compute_scale_and_shift(target, prediction, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
    
class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, target, prediction, mask, interpolate=True, return_interpolated=False):
        
        _, _, h_i, w_i = prediction.shape
        _, _, h_t, w_t = target.shape
    
        if h_i != h_t or w_i != w_t:
            prediction = F.interpolate(prediction, (h_t, w_t), mode='bilinear', align_corners=True)

        prediction, target, mask = prediction.squeeze(1), target.squeeze(1), mask.squeeze(1)
        
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        return loss
