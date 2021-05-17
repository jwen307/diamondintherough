#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:24:48 2021

@author: jeff

proto_optim.py
    - Find the protoimage vectors

"""

import torch
import torch.optim as optim
import torch.nn as nn
import math
from tqdm import tqdm



from . import utils



#Optimization to get the protoimages
def proto_optim(og_latents,recovered_latents, model, model_type, device, lr = 0.1, Lambda = 3.0, epsilon = 0.0001, minibatch_size = 16):
    '''
    Lambda: weight for the cosine similarity loss (3.0 for celeba_pgan, 5.0 for church_pgan, 10.0 for celebaHQ_pgan)

    '''
    
    print('Searching for the protoimage latent vector...')
    
    #Get the number of samples and the dimensionality
    num_samples, num_dims = og_latents.size()
    
    #Number of batches needed
    num_batches = int(math.ceil(num_samples/minibatch_size))
    
    #Vector for the found protoimage latent vectors
    protoLatents = torch.zeros(num_samples,num_dims)
    
    for batch_num in tqdm(range(num_batches)):
        
        #Check if there are fewer than minibatch_size images left
        if (batch_num + 1) * minibatch_size > num_samples:
            end = num_samples + 1
        else:
            end = (batch_num + 1) * minibatch_size
            
        #Original latent vectors
        x = og_latents[batch_num * minibatch_size: end].to(device)
        batch_size,_ = x.size()
        
        #Recovered latent vectors
        y = recovered_latents[batch_num * minibatch_size: end].to(device)
        
        #Put both on the device
        x = x.detach().requires_grad_(False).to(device)
        y = y.detach().requires_grad_(False).to(device)
        
        og_x = x * 1.0
    
        alpha = torch.ones(batch_size,512,device=device) 
        alpha = alpha.requires_grad_(True)
        
        #Initial direction
        diff = y - x
    
        opt = optim.Adam([alpha], lr = lr)
        
        cosSim = nn.CosineSimilarity()
        
        #Learning rate scheduler
        sched = optim.lr_scheduler.StepLR(optimizer = opt, step_size = 200, gamma = 0.1)
        oldLoss = 0
        
        
        for i in range(501):
            
            #Zero the gradients
            opt.zero_grad()
            
            #Move the direction of the difference vector
            ynew = y + (torch.mm(alpha,diff.t()).diagonal().unsqueeze(1) * ((diff / (diff.norm(dim=1).unsqueeze(1)**2))))
            ynew = utils.normalize(ynew)
            
            #Get the images of the current latent vectors
            currImgs = model.netG(ynew)
            
            #Get the discriminator score
            discrimScore = model.netD(currImgs,getFeature = False)
            
            #Calculate the loss
            if model_type == 'wgangp':
                loss = discrimScore.mean() + 0.2*cosSim(ynew,og_x).mean() + 1.0*discrimScore.std() + 3.0*cosSim(ynew,og_x).std()
            else:
                loss = discrimScore.mean() + Lambda*cosSim(ynew,og_x).mean()
            
            #Backpropagate the error
            loss.backward()
            
            #Take a step with the optimizer
            opt.step()
            sched.step()
        
            #Early stopping condition
            if abs(loss-oldLoss) < epsilon:
                break
            else:
                oldLoss = loss
            
            x = y * 1.0
            y = ynew.detach()
            
            diff = y - x
            
            #Show the progress
            # if i % 1 == 0:
            #     print('Iterations: ' + str(i))
            #     print('Loss: ' + str(loss))
            
        protoLatents[batch_num * minibatch_size: end] = ynew.detach().cpu()
    
    return protoLatents