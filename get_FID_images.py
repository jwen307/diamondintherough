#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:55:10 2021

@author: jeff

main.py



"""


import torch
import torchvision
import torch.optim as optim
import torch.nn as nn


import os
import pandas as pd
import numpy as np
import math
import argparse

import sys
sys.path.append('.')
sys.path.append('./functions')

from functions import utils, gan_utils


#%%
'''
pretrained_dir = 'model_types/pretrained/'
model_type = 'pgan'
dataset = 'celeba'
category = 963
num_samples = 2000
minibatch_size = 5
proto_dir = 'results/pgan_celeba_trial5/pgan_celeba_protomean.pt'
boundary_dir = None
'''


#%%

if __name__ == '__main__':
    
    #Get arguments from the user
    parser = argparse.ArgumentParser(description='Run latent space improvement and produce images for FID')
    parser.add_argument('--model_type', default='pgan', help='Model type. Options: pgan, wgangp, biggan')
    parser.add_argument('--dataset',default='celeba', help='Dataset that model was trained on. Options: celeba, celebaHQ512, celebaHQ1024, church, train, imagenet')
    parser.add_argument('--category', default=963, type = int, help = 'BigGAN category to run')
    parser.add_argument('--num_samples', default=10000, type = int, help = 'Number of images to produce')
    parser.add_argument('--proto_dir', help = 'Location of the protomean.pt file (results/some_trial/model_dataset_protomean.pt)')
    parser.add_argument('--minibatch_size', default = 5, type = int, help = 'Number of images in each minibatch')
    parser.add_argument('--boundary_dir', default = None, help = 'Location of the boundary.npy file (for comparison with Shen2020)')
    args = parser.parse_args()
    
    pretrained_dir = 'model_types/pretrained/'
    model_type = args.model_type
    dataset = args.dataset
    category = args.category
    num_samples = args.num_samples
    minibatch_size = args.minibatch_size
    proto_dir = args.proto_dir
    boundary_dir = None
    
    
    #Set the device
    if torch.cuda.is_available():
        print('GPU Available')
        device = torch.device('cuda:0')
        torch.cuda.device(device)
        print('Using ' + str(device))
        
    #Directory to save the results
    save_dir = 'results/{0}_{1}_FID_Images'.format(model_type,dataset)   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    #Get the GAN model
    model = gan_utils.get_model(pretrained_dir, ganType = model_type, dataset = dataset)
#%%    
    #Larger images need to be split up into batches
    num_in_batch = 1000
    batches = int(num_samples/num_in_batch)
    for batch in range(0,batches):
    
        #Generate random images and the latent vectors
        gen_imgs, gen_latents, y_shared, y = gan_utils.fake_samples_gen(num_in_batch, model, model_type, category, minibatch_size)
        gen_latents = utils.normalize(gen_latents)
        
        #Save the latents
        torch.save(gen_latents, save_dir + '/random_latents{0}.pt'.format(batch))
        for i in range(num_in_batch):
            utils.save_img(gen_imgs[i],(batch*num_in_batch)+i,save_dir + '/RandomImages/RandomImages/')
        
    
    #%%    
        #Get the protolatents
        protoLatents = torch.load(proto_dir)
        protoMean = utils.normalize(protoLatents.mean(dim=0).unsqueeze(0))
        
        for j in range(1,5):
            
            for k in range(2):
                #Fraction of distance to move
                if k == 0:
                    a = j*0.01
                else:
                    a = j*0.1
            
                #Find a point in between the vector between the original image and band image
                betterLatents = (a*protoMean.detach().cpu()) + (1-a)*gen_latents.detach().cpu()
                betterLatents = utils.normalize(betterLatents)
                
                #Get the images for this alpha
                newImgs = gan_utils.get_images(betterLatents, y_shared, model, model_type, minibatch_size).detach().cpu()
                
                #Save the images
                for i in range(num_in_batch):
                    utils.save_img(newImgs[i],(batch*num_in_batch)+i,save_dir + '/ours{0:.2f}/ours{0:.2f}/'.format(a))
                
            
    #%% Run the same thing for the boundary direction
        if boundary_dir is not None:
            #Get the boundary direction
            boundary = torch.Tensor(np.load('boundary.npy'))
            
            for j in range(1,5):
                
                for k in range(2):
                    #Fraction of distance to move
                    if k == 1:
                        a = j*1.0
                    else:
                        a = j*0.1
                    
                    #Get latents in the boundary direction
                    betterLatents = (k*boundary.detach().cpu()) + gen_latents.detach().cpu()
                    betterLatents = utils.normalize(betterLatents)
                    
                    #Get the images for this alpha
                    newImgs = gan_utils.get_images(betterLatents, y_shared, model, model_type, minibatch_size).detach().cpu()
                    
                    #Save the images
                    for i in range(num_in_batch):
                        utils.save_img(newImgs[i],(batch*num_in_batch)+i,save_dir + '/boundary{0:.2f}/boundary{0:.2f}/'.format(k))
                        

