import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data

parser = argparse.ArgumentParser (description = "Prediction of flower name from an image")
parser.add_argument ('image_dir', help = 'Path to image', type = str)
parser.add_argument ('load_dir', help = 'Path to checkpoint', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names', type = str)
parser.add_argument ('--GPU', help = "GPU or CPU", type = str)

def loading_model (file_path):
    checkpoint = torch.load (file_path) 
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']
    for param in model.parameters():
        param.requires_grad = False 
    return model

def process_image(image):
    ''' Scales, crops, and normalizes an image, returns an Numpy array
    '''
    im = Image.open (image)
    width, height = im.size

    if width > height:
        height = 256
        im.thumbnail ((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        im.thumbnail ((width,50000), Image.ANTIALIAS)

    width, height = im.size
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + 224
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))
    np_image = np.array (im)/255
    np_image -= np.array ([0.485, 0.456, 0.406])
    np_image /= np.array ([0.229, 0.224, 0.225])
    np_image= np_image.transpose ((2,0,1))
    return np_image

def predict(image_path, model, topk_l, device):
    ''' Predict the classes of an image
    '''
    image = process_image (image_path) 

    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)
    im = im.unsqueeze (dim = 0) 
    model.to (device)
    im.to (device)

    with torch.no_grad ():
        output = model.forward (im)
    output_prob = torch.exp (output)
    prob, indece = output_prob.topk (topk_l)
    prob = prob.cpu ()
    indece = indece.cpu ()
    prob = prob.numpy () 
    indece = indece.numpy ()
    prob = prob.tolist () [0]
    indece = indece.tolist () [0]

    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping [item] for item in indece]
    classes = np.array (classes) 
    return prob, classes

args = parser.parse_args ()
file_path = args.image_dir

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

model = loading_model (args.load_dir)

if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1
probs, classes = predict (file_path, model, nm_cl, device)
class_names = [cat_to_name [item] for item in classes]

for l in range (nm_cl):
     print("Number: {}/{}.. ".format(l+1, nm_cl),"Name of the Class: {}.. ".format(class_names [l]),"Probability: {:.3f}..% ".format(probs [l]*100),) 