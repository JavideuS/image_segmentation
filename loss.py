import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


class ModifiedJaccardLoss(nn.Module):
    def __init__(self):
        super(ModifiedJaccardLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.view(60, -1)
        targets = targets.view(60, -1)

        intersection = (inputs * targets).sum(dim=1)
        total = (inputs + targets).sum(dim=1)
        union = total - intersection

        jaccard_index = intersection / union

        jaccard_loss = 1 - jaccard_index.mean()

        return jaccard_loss
