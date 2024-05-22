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
from model_zoo import ZeroNeurons
from training_data_loading import tr_loading
from loss import ModifiedJaccardLoss

def train(model, loss_fn, optimizer, num_epochs, batch_size,
          target_path="C:/Users/fabia/Downloads/DATASET_5R/LABELS/MASK/TRAIN",
          ind_path="C:/Users/fabia/Downloads/DATASET_5R/IMAGES/TRAIN"):
    lossi = []
    losse = []

    batches = [i for i in range(0 + batch_size, len(os.listdir(ind_path)) + batch_size, batch_size)]

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for b in batches:
            tr_labels, tr_set = tr_loading(target_path=target_path, ind_path=ind_path, batch_s=b - batch_size,
                                           batch_e=b)

            outputs = model(tr_set)
            loss = loss_fn(outputs, tr_labels)

            # Backward pass
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            epoch_loss += loss.item()
            epoch_loss /= 2
            lossi.append(loss.item())
            losse.append(epoch_loss)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{b}] ----> Loss: {epoch_loss}')

    return lossi, losse
