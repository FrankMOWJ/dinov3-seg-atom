"""The foreground Dice + weighted CE objective used by the original Model."""

import torch
from torch import nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, class_weights=(0.2, 1.0, 1.0, 2.0), ce_weight=0.5,
                 dice_weight=0.5, label_smoothing=0.05, ignore_index=255):
        super().__init__()
        if len(class_weights) != 4 or min(class_weights) <= 0:
            raise ValueError('Expected four positive class weights: background, Mo, S, V1S')
        if min(ce_weight, dice_weight) < 0 or ce_weight + dice_weight <= 0:
            raise ValueError('Loss weights must be nonnegative and not both zero')
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.float()
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0
        ce = F.cross_entropy(logits, targets, weight=self.class_weights,
                             label_smoothing=self.label_smoothing, ignore_index=self.ignore_index)
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, 4).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid.unsqueeze(1)
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        return self.ce_weight * ce + self.dice_weight * (1 - dice[1:].mean())
