import torch
import torchvision.transforms as transforms
import torch 
import os
from PIL import Image
import numpy as np


def image_load(image_name, imsize):
    """
    Args:
    =====
    image_name: Path to the image 
    imsize: scalar denoting image size. Asumming image is 1:1 h:w
    """
    loader = transforms.Compose([
        transforms.Resize((imsize , imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name)
    image = torch.tensor(loader(image),requires_grad=True)
    image = image.unsqueeze(0) #make it a 4d for convnet
    return image 

def image_loader_gray(image_name, imsize ):
    loader = transforms.Compose([
        transforms.Resize((imsize,imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('L')
    image = np.asarray(image)
    image = np.asarray([image, image, image]) #(3,H,W)
    image = Image.fromarray(np.uint8(image).transpose(1,2,0)) #unit8 turn into range 0,255
    image = torch.tensor(loader(image),requires_grad=True)
    image = image.unsqueeze(0)
    return image

def save_image(tensor, size, input_size, fname=None):
    unloader = transforms.ToPILImage() #reconvert into PIL image

    image = tensor.clone().cpu() #we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)

    out_path = os.path.join('transferred',fname)
    if not os.path.exists('transferred'):
        os.mkdir('tansferred')
    
    image.save(out_path)

    return 