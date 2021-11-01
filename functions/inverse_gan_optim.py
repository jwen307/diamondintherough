#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:24:48 2021

inverse_gan_optim.py
"""

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import math

import sys
sys.path.append('.')
sys.path.append('./model_types/ganseeing')
sys.path.append('./model_types')

from model_types.ganseeing.seeing import encoder_net, nethook

from . import utils
from . import gan_utils
from tqdm import tqdm


#Find the latent vector corresponding to the input image
def get_latent_vector(imgs, ys, ys_shared, features, init, model, model_type, device, mini_batch_size, lr=0.1, alpha = 2.0, beta = 1.0, max_iter = 501, epsilon = 0.0001):
    
    print('Recovering latent vectors')
    
    #Find the number of images in this batch
    num_imgs,_, image_size, _= imgs.size()
    

    #Split into mini batches
    num_batches = int(math.ceil(num_imgs / mini_batch_size))
    
    #Get the dimensionality of the features
    _, num_dims = init.size()
    features = features.to(device)
    
    #Store the latent vectors when done optimizing
    recoveredLatents = torch.zeros(num_imgs,num_dims)

    for batch_num in tqdm(range(num_batches)):
        
        #Check if there are fewer than mini_batch_size images left
        if (batch_num + 1) * mini_batch_size > num_imgs:
            end = num_imgs + 1
        else:
            end = (batch_num + 1) * mini_batch_size
        
        
        #Put the image on the GPU
        img = imgs[batch_num * mini_batch_size: end].to(device)
        
        #Put the imagenet labels on the device and get the generator and discriminator
        if model_type == 'biggan':
            y = ys[batch_num * mini_batch_size: end].to(device)
            y_shared = ys_shared[batch_num * mini_batch_size: end].to(device)
            G = model[0]
            D = model[1]
        
        else:
            G = model.netG
            D = model.netD
        
        
        
        #Initialize latent vector 
        noise = init[batch_num * mini_batch_size: end]
        
        #Set the latent vector to track the gradient
        zp = noise.clone().detach().requires_grad_(True)
        
        #Define the optimizer with z as the parameters to optimize
        #Recommended 0.1 for PGAN, 0.01 for WGAN
        opt = optim.Adam([zp], lr = 0.1)
        
        #Learning rate scheduler
        sched = optim.lr_scheduler.StepLR(optimizer = opt, step_size = 200, gamma = 0.1)

        #Define the loss function
        lossfn = nn.L1Loss()
        mse = nn.MSELoss()
        
        oldLoss = 0
        
        
        #Recommended: 201 for PGAN
        for i in range(max_iter):
            
            #Zero the gradients
            opt.zero_grad()
            
            #Put the noise samples into the GAN to generate images
            if model_type == 'biggan':
                gen_img = G(zp,y_shared)
            else:
                gen_img = G(zp)
            
            #Put the generated image into the discriminator
            if model_type == 'biggan':
                gen_features = D(gen_img, y, getFeature=True)
            elif model_type == 'pgan':
                gen_features = D(gen_img, feature = 3)
            else:
                gen_features = D(gen_img, getFeature=True)
            
            
            #Find the loss
            loss = lossfn(gen_features,features[batch_num * mini_batch_size: end]) + alpha * lossfn(gen_img,img) + beta*mse(zp.norm(dim=1).cuda(),torch.tensor([math.sqrt(num_dims) for _ in range(len(zp))]).cuda())

            
            #Backpropagate the error
            loss.backward()
            
            #Take a step with the optimizer
            opt.step()
            
            #Update the learning rate
            sched.step()
            
            #If the change between epochs is less than epsilon, stop
            if abs(loss - oldLoss) < epsilon:
                break
            else:
                oldLoss = loss
            
            #print('Iteration: {0}     Loss: {1}'.format(i,loss))

        #print("Image: " + str(batch_num*mini_batch_size) + ", Loss: " + str(loss))
                
        recoveredLatents[batch_num * mini_batch_size: end] = zp.detach().cpu()
           
    return recoveredLatents      



#Get good initialization
def get_init(imgs, model, model_type, y, category, minibatch_size, encoder_file = None):
    
    num_samples,_,_,_ = imgs.size()
    
    #Get the size of the latent space
    _, test_latents, _, _ = gan_utils.fake_samples_gen(1,model, model_type, category, minibatch_size)
    _,num_dims = test_latents.size()
    
    #Get the features for the images
    flat_imgs, feats = gan_utils.discrim_features(imgs, model, model_type,y)
    
    if encoder_file is not None:
        #Get the trained encoder
        encoder = nethook.InstrumentedModel(encoder_net.HybridLayerNormEncoder())
        encoder.load_state_dict(torch.load(encoder_file)['state_dict'])
        encoder.eval()
        
        print('Getting Initializations from Encoder')  
        init = torch.zeros(num_samples,num_dims)
        
        with torch.no_grad():
            #Break into batches so can fit on GPU
            sample_batches = int(math.ceil(num_samples/minibatch_size))
        
            for i in range(sample_batches):
                if (i+1)*minibatch_size > num_samples:
                    end = num_samples+1
                else:
                    end = (i+1) * minibatch_size
                    
                init[i*minibatch_size:end] = encoder(imgs[i*minibatch_size:end]).squeeze(3).squeeze(2)
                torch.cuda.empty_cache()
                print('Encoder Batch: ' + str(i))

        
    else:
        
        print('Initialization from Nearest Neighbor Random Vector')
        
        #Generate 1000 latent vectors as potential latent vectors
        pot_init_imgs, pot_init_latents, _, _ = gan_utils.fake_samples_gen(1000,model, model_type, category, minibatch_size)
        
        #Get the discriminator features
        pot_init_flat, pot_init_feats = gan_utils.discrim_features(pot_init_imgs, model, model_type, y)
        
        #Find the initial vectors
        init, _ = utils.find_NN(feats, pot_init_feats, pot_init_latents)
        
    return init.cpu(), feats.cpu()
        
        
        
