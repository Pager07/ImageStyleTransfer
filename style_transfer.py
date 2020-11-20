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
    style_losses  = []
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
            
            if name in content_layers: #content_layers list with layer names to get F
                #add content loss:
                target = model(content_img).clone() #get activations 
                content_loss = ContentLoss(target, content_weight)
                #we add the content_loss layer to model 
                model.add_module("content_loss_"+ str(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                #add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                #we add the style_loss layer to the model
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        if isinstance(layer,nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name,layer)

            if name in content_layers:
                #add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target,content_weight)
                model.add_module("content_loss_"+str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram , style_weight)
                model.add_module("style_loss_"+str(i)+style_loss)
                style_losses.append(style_losses)
            
            i +=1
        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name,layer)
    return model, style_losses , content_losses

def get_input_param_optimizer(input_img):
    #this line to show that input is a parameter tha requires a graident 
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LGBFS([input_param])
    return input_param, optimizer

def run_style_transfer(cnn,content_img,style_img,input_img,num_steps=300,
                       style_weight=1000, content_weight=1):
    """Run the style tranfer"""
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,content_img,style_img,content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)
    

    return 