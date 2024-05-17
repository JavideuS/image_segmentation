import torch
import torch.nn.functional as f
import numpy
from torchvision import transforms
from PIL import Image
import os
import json

def tr_loading(path_input):
    tr_labels = {}
    tr_labels_path = path_input
    for i in os.listdir(tr_labels_path)[:1]:
        tr_labels[i] = []

    for i in os.listdir(tr_labels_path)[:1]:
        for j in os.listdir(tr_labels_path + "/" + i):
            path = tr_labels_path + "/" + i + "/" + j
            image = Image.open(path)
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            image_2d_tensor = image_tensor.squeeze(0)
            tr_labels[i].append(image_2d_tensor)

    im = []
    for i in os.listdir(tr_labels_path):
        x=0
        for j in os.listdir(tr_labels_path + "/" + i):
           x+=1
        im.append(x)
    print(max(im))
    #labels = {key: [tensor.tolist() for tensor in tensor_list] for key, tensor_list in tr_labels.items()}

    #with open('labels.json', 'w') as file:
    #    json.dump(labels, file, indent=4)

    label = {key : torch.stack(value, dim=0) for key , value in tr_labels.items()}


    return label

x = tr_loading(path_input="C:/Users/fabia/Downloads/DATASET_5R/LABELS/MASK/TRAIN")
print(x['20240321_150920'])