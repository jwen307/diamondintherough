#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:55:10 2021

DitR_main.py



"""


import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from functions import utils
from functions import gan_utils
from functions.get_improved_latents import improve_latents
from functions import inverse_gan_optim
from functions.proto_optim import proto_optim
import os
import pandas as pd
import argparse

import sys



#%%

if __name__ == '__main__':
    
    pretrained_dir = 'model_types/pretrained/'
    model_type = 'pgan'
    dataset = 'celeba'
    category = 963
    num_samples = 1000
    minibatch_size = 5
    #encoder_file = pretrained_dir + 'epoch_100.pth.tar'
    encoder_file = None
    rec_lr = 0.1
    alpha = 3.0
    beta = 1.0
    
    proto_lr = 0.1
    Lambda = 10.0
    
    min_sigma = 0.6
    max_sigma = 0.8
    impr_lr = 0.01
    startSigma = 0.7
    # parser = argparse.ArgumentParser(description='Run latent space improvement')
    # parser.add_argument('--model-type', default='pgan', help='Model type. Options: pgan, wgangp, biggan')
    # parser.add_argument('--dataset',default='celeba', help='Dataset that model was trained on. Options: celeba, celebahq512, celebahq1024, church, train, imagenet')
    # args = parser.parse_args()
    
    
    #Set the device
    if torch.cuda.is_available():
        print('GPU Available')
        device = torch.device('cuda:0')
        torch.cuda.device(device)
        print('Using ' + str(device))
        
    #Directory to save the results
    save_dir = 'results/'   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 

    #Get the GAN model
    model = gan_utils.get_model(pretrained_dir, ganType = model_type, dataset = dataset)
    

#%%    
    #Generate random images and the latent vectors
    gen_imgs, gen_latents, y_shared, y = gan_utils.fake_samples_gen(num_samples, model, model_type, category, minibatch_size)
    gen_latents = utils.normalize(gen_latents)
    
    #Get the initialization for the GAN inversion
    init, features = inverse_gan_optim.get_init(gen_imgs, model, model_type, y, category, minibatch_size, encoder_file)
#%%    
    #Invert the generator to recover the latent vectors
    recoveredLatents = inverse_gan_optim.get_latent_vector(gen_imgs, y, y_shared, features, init.to(device), model, model_type, device, minibatch_size, rec_lr, alpha, beta, max_iter = 501, epsilon = 0.0001)
    recoveredLatents = utils.normalize(recoveredLatents)
    
#%% 
    
    #Optimize to get the band images
    protoLatents = proto_optim(gen_latents, recoveredLatents, model, model_type, device, proto_lr, Lambda, epsilon = 0.0001, minibatch_size = minibatch_size)
    
    #Get a new folder to save results to
    new = False
    trial = 0
    while not new:
        img_dir = save_dir + '{0}_{1}_trial{2}/'.format(model_type,dataset, trial)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            new = True
        else:
            trial += 1
    
    #Get the mean proto latent
    protoMean = utils.normalize(protoLatents.mean(dim=0).unsqueeze(0))
    
    #Save the mean proto latent
    torch.save(protoMean, img_dir + '{0}_{1}_protomean.pt'.format(model_type, dataset))
    