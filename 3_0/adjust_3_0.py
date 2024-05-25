import torch.nn.functional as F
from torchvision import transforms


def adjust(width=2208, height=1242, img_size=None, img=None):
    img_s = []
    for i in img.shape:
        if i == img_size[1]:
            img_s.insert(0, i)
        if i == img_size[0]:
            img_s.insert(1, i)

    # print('size' , img_s)

    if height < img_s[1] or width < img_s[0]:
        # print('precrop:' , img.shape)
        img = transforms.functional.center_crop(img, (min(img_s[1], height), min(img_s[0], width)))
        img_s = img.shape[1:3]
        # print('postcrop' , img_s)

    if height > img_s[0] or width > img_s[1]:
        padding_left = max((width - img_s[1]) // 2, 0)
        padding_top = max((height - img_s[0]) // 2, 0)
        padding_right = max(width - img_s[1] - padding_left, 0)
        padding_bottom = max(height - img_s[0] - padding_top, 0)
        # print(padding_left, padding_top, padding_right, padding_bottom)
        # print('prepad:' , img.shape)
        img = F.pad(img, (padding_left, padding_right, padding_top, padding_bottom))
        # print('postpad' , img.shape)

    return img