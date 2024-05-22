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

def tr_loading(target_path, ind_path, batch_s, batch_e):
    tr_labels = {}
    tr_ds = []
    inds = os.listdir(ind_path)
    targs = os.listdir(target_path)

    for i in targs[batch_s:batch_e]:
        tr_labels[i] = []
        for j in os.listdir(target_path + "/" + i):
            path = target_path + "/" + i + "/" + j
            image = Image.open(path)
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            image_2d_tensor = image_tensor.squeeze(0)
            tr_labels[i].append(image_2d_tensor[0, :, :])

    label = {key: torch.stack(value, dim=0) for key, value in tr_labels.items()}
    tr_labels = []
    for i in label.keys():
        dims = label[i].shape
        zero_matrices = torch.zeros(60 - dims[0], dims[1], dims[2])
        tr_labels.append(torch.cat((label[i], zero_matrices), dim=0))
    tr_labels = torch.stack(tr_labels, dim=0)

    for i in inds[batch_s:batch_e]:
        image = Image.open(ind_path + i)
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        image_tensor = image_tensor
        tr_ds.append(image_tensor)
    tr_ds = torch.stack(tr_ds, dim=0)

    return tr_labels, tr_ds
