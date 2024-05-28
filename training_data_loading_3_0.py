import torch
from torchvision import transforms
from PIL import Image
import os
from adjust_3_0 import adjust



def tr_loading(target_path, ind_path, batch_s, batch_e):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_labels = {}
    tr_ds = []
    inds = os.listdir(ind_path)
    targs = os.listdir(target_path)

    for i in targs[batch_s:batch_e]:
        tr_labels[i] = []
        for j in os.listdir(target_path + "/" + i):
            path = target_path + "/" + i + "/" + j
            image = Image.open(path)
            im_s = image.size
            transform = transforms.ToTensor()
            image_tensor = transform(image).to(device)
            image_2d_tensor = image_tensor.squeeze(0).to(device)

            if image.size[0] != 2208 or image.size[1] != 1242:
                image_2d_tensor = adjust(img_size=im_s, img=image_2d_tensor).to(device)

            # print('lab:' , i+j , image_2d_tensor.shape)
            if len(image_2d_tensor.shape) != 3:
                tr_labels[i].append(image_2d_tensor[:, :])
            else:
                tr_labels[i].append(image_2d_tensor[0, :, :])

    label = {key: torch.stack(value, dim=0) for key, value in tr_labels.items()}
    tr_labels = []
    for i in label.keys():
        dims = label[i].shape
        zero_matrices = torch.zeros(60 - dims[0], dims[1], dims[2]).to(device)
        tr_labels.append(torch.cat((label[i], zero_matrices), dim=0))
    tr_labels = torch.stack(tr_labels, dim=0).to(device)

    for i in inds[batch_s:batch_e]:
        image = Image.open(ind_path + "/" + i)
        im_s = image.size
        transform = transforms.ToTensor()
        image_tensor = transform(image).to(device)
        image_tensor = image_tensor
        if image.size[0] != 2208 or image.size[1] != 1242:
            image_tensor = adjust(img_size=im_s, img=image_tensor).to(device)
        tr_ds.append(image_tensor)
    tr_ds = torch.stack(tr_ds, dim=0).to(device)

    return tr_labels, tr_ds
