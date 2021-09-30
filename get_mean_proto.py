#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:55:10 2021

get_mean_proto.py



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
    
    
    #Get arguments from the user
    parser = argparse.ArgumentParser(description='Run latent space improvement')
    parser.add_argument('--model_type', default='pgan', help='Model type. Options: pgan, wgangp, biggan')
    parser.add_argument('--dataset',default='celeba', help='Dataset that model was trained on. Options: celeba, celebaHQ512, celebaHQ1024, church, train, imagenet')
    parser.add_argument('--category', default=963, type = int, help = 'BigGAN category to run')
    parser.add_argument('--num_samples', default=1000, type = int, help = 'Number of images to produce')
    parser.add_argument('--minibatch_size', default=5, type = int, help = 'Number of samples in each minibatch')
    parser.add_argument('--encoder_file', default=None, help = 'Location of encoder network weights (Ex: encoder.pth.tar)')
    parser.add_argument('--non_default_config', action='store_true', help = 'Use your own hyperparmenter values')
    
    parser.add_argument('--rec_lr', default = 0.1, type = float, help = 'Learning rate for the GAN inversion optimization')
    parser.add_argument('--alpha', default = 2.0, type = float,help = 'Weight for the discriminator feature space loss term')
    parser.add_argument('--beta', default = 1.0, type = float,help = 'Weight for the loss term keeping the search on the hypersphere')
    
    parser.add_argument('--proto_lr', default = 0.1, type = float,help = 'Learning rate for protoimage optimization')
    parser.add_argument('--Lambda', default = 3.0, type = float,help = 'Weight for the cosine similarity term')
    
    
    args = parser.parse_args()
    
    pretrained_dir = 'model_types/pretrained/'
    
    #Set the values of constants and hyperparameters. If you don't want to run from a command line, comment out the parser and enter values here.
    model_type = args.model_type
    dataset = args.dataset
    category = args.category
    num_samples = args.num_samples
    minibatch_size = args.minibatch_size
    

    if args.non_default_config:
        #encoder_file = pretrained_dir + 'epoch_100.pth.tar'
        encoder_file = args.encoder_file
        rec_lr = args.rec_lr
        alpha = args.alpha
        beta = args.beta
        
        proto_lr = args.proto_lr
        Lambda = args.Lambda

        
    else:
        print('Using Default Configuration')
        config = utils.read_json('configs/{0}_{1}.json'.format(model_type,dataset))
        if config['encoder_file'] == 'None':
            encoder_file = None
        else:
            encoder_file = config['encoder_file']
        rec_lr = config['rec_lr']
        alpha = config['alpha']
        beta = config['beta']
        
        proto_lr = config['proto_lr']
        Lambda = config['Lambda']
        
        min_sigma = config['min_sigma']
        max_sigma = config['max_sigma']
        impr_lr = config['impr_lr']
        startSigma = config['startSigma']
    
    
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
    
