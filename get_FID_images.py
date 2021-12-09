#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:55:10 2021

get_FID_images.py
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
dataset = 'church'
category = 963
num_samples = 10000
minibatch_size = 5
#proto_dir = 'results/pgan_celeba_trial8/pgan_celeba_protomean.pt'
proto_dir = 'results/pgan_church_trial0/pgan_church_protomean.pt'
boundary_dir = None
trunc=True
'''


#%%

if __name__ == '__main__':
    
    #Get arguments from the user
    parser = argparse.ArgumentParser(description='Run latent space improvement and produce images for FID')
    parser.add_argument('--model_type', default='pgan', help='Model type. Options: pgan, wgangp, biggan')
    parser.add_argument('--dataset',default='celeba', help='Dataset that model was trained on. Options: celeba, celebaHQ512, celebaHQ1024, church, train, imagenet')
    parser.add_argument('--category', default=963, type = int, help = 'BigGAN category to run')
    parser.add_argument('--num_samples', default=10000, type = int, help = 'Number of images to produce')
    parser.add_argument('--proto_dir', help = 'Location of the protomean.pt or protolatents.pt file (results/some_trial/model_dataset_protomean.pt)')
    parser.add_argument('--latent_dir', default=None, help = 'Location of the genlatents.pt file (results/some_trial/model_dataset_genlatents.pt)')
    parser.add_argument('--minibatch_size', default = 5, type = int, help = 'Number of images in each minibatch')
    parser.add_argument('--boundary_dir', default = None, help = 'Location of the boundary.npy file (for comparison with Shen2020)')
    parser.add_argument('--trunc', action='store_true', help = 'Also include images from using the truncation trick')
    args = parser.parse_args()
    
    pretrained_dir = 'model_types/pretrained/'
    model_type = args.model_type
    dataset = args.dataset
    category = args.category
    num_samples = args.num_samples
    minibatch_size = args.minibatch_size
    proto_dir = args.proto_dir
    latent_dir = args.latent_dir
    boundary_dir = args.boundary_dir
    trunc = args.trunc
    
    
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
        
    #Get a new folder to save results to
    new = False
    trial = 0
    while not new:
        #Directory to save the results
        save_dir = 'results/{0}_{1}_FID_Images{2}'.format(model_type,dataset,trial)   
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            new = True
        else:
            trial += 1
    
    
    #Get the GAN model
    model = gan_utils.get_model(pretrained_dir, ganType = model_type, dataset = dataset)
#%%    
    #Can be the mean protolatent or individual protolatents
    protoLatents = torch.load(proto_dir)    
    
    if latent_dir is not None:
        gen_latents_all = torch.load(latent_dir)

    #Larger images need to be split up into batches
    num_in_batch = 1000
    batches = int(num_samples/num_in_batch)
    for batch in range(0,batches):
    
        #If you already have generated latent vectors, use those
        if latent_dir is not None:
            gen_latents = gen_latents_all[batch*num_in_batch: (batch+1)*num_in_batch]
            y_shared, y = None, None
            gen_imgs = gan_utils.get_images(gen_latents, None, model, model_type, minibatch_size)

        
        else:
            #Generate random images and the latent vectors
            gen_imgs, gen_latents, y_shared, y = gan_utils.fake_samples_gen(num_in_batch, model, model_type, category, minibatch_size)
            gen_latents = utils.normalize(gen_latents)
            
            #Save the latents
            torch.save(gen_latents, save_dir + '/random_latents{0}.pt'.format(batch))
            
        #Save the generated images
        for i in range(num_in_batch):
            utils.save_img(gen_imgs[i],(batch*num_in_batch)+i,save_dir + '/RandomImages/RandomImages/')
            
    
    #%%    
        #Get the protolatents

        if len(protoLatents) > 1:
            batch_proto = protoLatents[batch*num_in_batch: (batch+1)*num_in_batch]
        
        for j in range(1,6):
            
            for k in range(2):
                #Fraction of distance to move
                if k == 0:
                    a = j*0.01
                else:
                    a = j*0.1
            
                #Find a point in between the vector between the original image and band image
                betterLatents = (a*batch_proto.detach().cpu()) + (1-a)*gen_latents.detach().cpu()
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
            
            for j in range(1,6):
                
                for k in range(2):
                    #Fraction of distance to move
                    if k == 1:
                        a = j*1.0
                    else:
                        a = j*0.1
                    
                    #Get latents in the boundary direction
                    betterLatents = (a*boundary.detach().cpu()) + gen_latents.detach().cpu()
                    betterLatents = utils.normalize(betterLatents)
                    
                    #Get the images for this alpha
                    newImgs = gan_utils.get_images(betterLatents, y_shared, model, model_type, minibatch_size).detach().cpu()
                    
                    #Save the images
                    for i in range(num_in_batch):
                        utils.save_img(newImgs[i],(batch*num_in_batch)+i,save_dir + '/boundary{0:.2f}/boundary{0:.2f}/'.format(a))



        if trunc:
            
            radii = [1.0, 0.75, 0.50, 0.25, 0.10]
            
            for radius in radii:

                #Apply the truncation trick
                trunc_latents = utils.truncation_trick(gen_latents,radius)
                
                #Get the images
                trunc_imgs = gan_utils.get_images(trunc_latents, None, model, model_type, minibatch_size).detach().cpu()
                
                #Save the images
                for i in range(num_in_batch):
                    utils.save_img(trunc_imgs[i],(batch*num_in_batch)+i,save_dir + '/trunc{0:.2f}/trunc{0:.2f}/'.format(radius))