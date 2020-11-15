from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os 
from PIL import Image
import numpy as np

import torchvision.transforms as transforms 
import torchvision.models as models

import copy
import argparse

from loss import ContentLoss,GramMatrix, StyleLoss
from utils import image_loader, image_loader_gray, save_image

parser = argparse.ArgumentParser()
parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=500, help='The weight of style loss')
parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise? elif initialize with content image')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and args.cuda

dtype = torch.cudaFloatTensor if use_cuda else torch.FloatTensor

#desired size of the output image
imsize = 512 if use_cuda else 128 #use small size if no gpu

style_img = image_loader(args.style,imsize).type(dtype)
content_img = image_loader_gray(args.content,imsize).type(dtype)

if args.initialize_noise:
    input_img = torch.randn(content_img.data.size(),requires_grad=True).type(dtype)
else:
    input_img = image_loader_gray(args.content, imsize).type(dtype)

input_size = Image.open(args.content).size

assert style_img.size() == content_img.size(),\
    "we need to import syle and content image of the same size"

cnn = models.vgg19(pretrained=True).features #this load all the weights and bias for all layer

#move the features to gpu is possible
if use_cuda:
    cnn = cnn.cuda()

#desired depth layers to compute style/content losses:
content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3', 'conv_4', 'conv_5' ]





def get_style_model_and_losses(cnn,style_img,content_img,style_weight=1000,
                               content_weight=1,
                                content_layers=content_layers_default,
                                style_layers = style_layers_default ):
    cnn = copy.deepcopy(cnn)

    #just in order to have iterable acces to or list of content/style
    content_losses = []
    syle_losses  = []
    model = nn.Sequential() # the new Sequential module network
    gram = GramMatrix() #we need a gram module in order to compute syle target

    #move these modules to GPU if possible
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()
    
    i = 1 
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            #add the layer to the nn.Seq
            model.add_module(name,layer)
            
            if name in content_layers:
                #add content loss:
                target = model(content_img).clone()
                
            




    return 

def get_input_param_optimizer():
    return

def run_style_transfer():
    return 