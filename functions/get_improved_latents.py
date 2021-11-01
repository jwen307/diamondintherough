#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:24:48 2021

proto_optim.py
    - Find the protoimage vectors

"""

import torch
import torch.optim as optim
import torch.nn as nn
import math

import sys
sys.path.append('../../Data/OtherPackages')



from . import utils



#Optimization to get the improved images along the geodesic between the original image and the protoimages
def improve_latents(og_latents,protoLatents, model, model_type, device, lr = 0.1, startSigma = 0.7, epsilon = 0.0001, min_sigma = 0.5, max_sigma = 0.8, minibatch_size = 16):
    '''


    '''
    print('Finding the improved latent...')
    
    #Get the number of samples and the dimensionality
    num_samples, num_dims = og_latents.size()
    
    #Number of batches needed
    num_batches = int(math.ceil(num_samples/minibatch_size))
    
    #Vector for the found protoimage latent vectors
    improvedLatents = torch.zeros(num_samples,num_dims)
    finalSigmas = torch.zeros(num_samples,1)
    
    for batch_num in range(num_batches):
        
        #Check if there are fewer than minibatch_size images left
        if (batch_num + 1) * minibatch_size > num_samples:
            end = num_samples + 1
        else:
            end = (batch_num + 1) * minibatch_size
            
    
        #Original latent vectors
        x = og_latents[batch_num * minibatch_size: end].to(device)
        batch_size,_ = x.size()
        
        #Recovered latent vectors
        y = protoLatents[batch_num * minibatch_size: end].to(device)
        
        #Put both on the device
        x = x.detach().requires_grad_(False).to(device)
        y = y.detach().requires_grad_(False).to(device)
        
        
        #Initialize the latent vectors
        sigmas = torch.ones(batch_size,1,device=device) * startSigma
        sigmas = sigmas.requires_grad_(True)
        
        #Initialize optimizer
        opt = optim.Adam([sigmas], lr = lr)
        
        lossfn = nn.MSELoss()
        
        #Learning rate scheduler
        sched = optim.lr_scheduler.StepLR(optimizer = opt, step_size = 50, gamma = 0.1)
        oldLoss = 0
        
        
        for i in range(100):
            
            #Zero the gradients
            opt.zero_grad()
            
            #Get the latents for the current value of alpha
            xnew = (sigmas*y)+(1-sigmas)*x
            
            xnew = utils.normalize(xnew)
            
            currImgs = model.netG(xnew)
            
            #Get the discriminator score
            discrimScore = model.netD(currImgs,getFeature = False)
            
            #Find loss
            if model_type == 'wgangp':
                loss = lossfn(discrimScore,torch.ones(num_samples,1).to(device)*-1.0) + 1.0*discrimScore.var()
            else:
                loss = lossfn(discrimScore,torch.zeros(batch_size,1).to(device))
            
            #Backpropagate loss
            loss.backward()
            
            #Take a step
            opt.step()
            sched.step()
        
            #Early stopping criterion
            if abs(loss-oldLoss) < epsilon:
                break
            else:
                oldLoss = loss
            
            # if i % 1 == 0:
            #     print('Iterations: ' + str(i))
            #     print('Loss: ' + str(loss))
                
            #Project back into space if leaves the bounds   
            with torch.no_grad():
                for k,sigma in enumerate(sigmas):
                    if sigma > max_sigma:
                        sigmas[k] = max_sigma
                        
                    if sigma < min_sigma:
                        sigmas[k] = min_sigma
                        
            

        improvedLatents[batch_num * minibatch_size: end] = xnew.cpu()
        finalSigmas[batch_num * minibatch_size: end] = sigmas.detach().cpu()
        torch.cuda.empty_cache()
    
    return improvedLatents, finalSigmas
