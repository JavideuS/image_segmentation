from PIL import Image
import torch
import torchvision.transforms as transforms
from model_zoo import Enc , Dec , Upscaling


image = Image.open("/home/javideus/programming/Cerveras_5R/image_segmentation/DATASET_5R/IMAGES/TRAIN/20240321_150920_defaultImage.png")
transform = transforms.ToTensor()
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.shape)

e = Enc(in_chan=4, out_chan=10)
y = e.forward(image_tensor)
print(y.shape)

d = Dec(in_chan=10 , out_chan=60)
z = d.forward(y)
print(z.shape)

up = Upscaling()
x = up.forward(z , o_shape1=image_tensor.shape[2] , o_shape2=image_tensor.shape[3])
print(x.shape)


# Pensar estructura
# Crear la funcion para entrenar
# Crear la funcion para loddear la data
# Data augmentation

