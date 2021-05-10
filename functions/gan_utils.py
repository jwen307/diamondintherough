#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:09:18 2021

@author: jeff

gan_utils.py
    - Common functions for the GAN

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
sys.path.append('./model_types/pytorch_GAN_zoo')
sys.path.append('./model_types')

from model_types.pytorch_GAN_zoo.models.progressive_gan import ProgressiveGAN
from model_types.pytorch_GAN_zoo.models.DCGAN import DCGAN
from model_types.pytorch_GAN_zoo import models
from model_types import pytorch_GAN_zoo

from . import utils

import model_types.BigGANPyTorch
from model_types.BigGANPyTorch import utils as bigUtils
from model_types.BigGANPyTorch import BigGAN as bgmodel





#Function to get the model
def get_model(pretrained_dir, ganType='pgan', dataset='celeba'):
    
    
    if ganType == 'pgan':
        
        #Indicate if the pretrained model is from the Tensorflow implementation
        fromTF = False
        
        #Load the pretrained network
        if dataset == 'celeba':
            modelPath = pretrained_dir + 'celeba128_pretrained.pt'
        elif dataset == 'celebaHQ512':
            modelPath = pretrained_dir + 'celebaHQ512_pretrained.pt'
        elif dataset == 'celebaHQ1024':
            modelPath = pretrained_dir + 'celebaHQ1024_pretrained.pt'
            fromTF = True
        elif dataset == 'church':
            modelPath = pretrained_dir + 'church_pretrained.pt'
            fromTF = True
        elif dataset == 'train':
            modelPath = pretrained_dir + 'train_pretrained.pt'
            fromTF = True
        else:
            raise ValueError('Not a valid pretrained network for PGAN')
            
        #Load a generic PGAN
        model = ProgressiveGAN(miniBatchStdDev = True, fromTF = fromTF)
  
        #Load the pretrained model
        state_dict = torch.load(modelPath)
        model.load_state_dict(state_dict)
        
        return model
        
    elif ganType == 'wgangp':
        #Load a DCGAN
        model = DCGAN()
        
        #Load the pretrained network
        if dataset == 'celeba':
            modelPath = pretrained_dir + 'celeba_cropped_wgan.pt'
            
        else:
            raise ValueError('Not a valid pretrained network for WGAN-GP')
            
        #Load the pretrained model
        state_dict = torch.load(modelPath)
        model.load_state_dict(state_dict)
        
        return model
        
         
    elif ganType == 'biggan':
        model_folder = pretrained_dir + '100k'
        
        # Prepare state dict, which holds things like epoch # and itr #
        state_dict = torch.load(model_folder + '/state_dict.pth')
        
        config = state_dict['config']
        
        #Initialize the generator and discriminator
        G = bgmodel.Generator(**config).cuda()
        D = bgmodel.Discriminator(**config).cuda()
        
        bigUtils.load_weights(G,D, state_dict, model_folder, '')
        
        G.eval()
        D.eval()
        
        return G,D
    
    else:
        raise ValueError('Not a valid network')
        
    


#Function to get the images from latent vector
def get_images(latent_vector, y_shared, model, model_type, minibatch_size):
    num_samples, _ = latent_vector.size()

    #Test run to get the dimensions
    if model_type == 'biggan':
        test_img = model[0](latent_vector[0:2],y_shared[0:2])
    else:
        test_img = model.netG(latent_vector[0:2])
    
    
    _, num_channels, image_size, image_size = test_img.size()
    
    #Vector for the images
    gen_find_imgs = torch.zeros(num_samples,num_channels,image_size,image_size)
    
    #If the number of samples is greater than 20, we need to split it up into batches so it doesn't fill up the GPU memory
    if num_samples > minibatch_size:
        sample_batch = int(math.ceil(num_samples/minibatch_size))
        for i in range(0,sample_batch):
            with torch.no_grad():
                #Check if there are fewer than minibatch_size images left
                if (i + 1) * minibatch_size > num_samples:
                    end = num_samples + 1
                else:
                    end = (i + 1) * minibatch_size
                    
                #Get all the generated images
                if model_type == 'biggan':
                    gen_imgs_gpu = model[0](latent_vector[(i*minibatch_size):end], y_shared[(i*minibatch_size):end])
                else:
                    gen_imgs_gpu = model.netG(latent_vector[(i*minibatch_size):end])
                
                gen_find_imgs[(i*minibatch_size):end] = gen_imgs_gpu.cpu()
                
                del gen_imgs_gpu
                torch.cuda.empty_cache()
    else:
        with torch.no_grad():
            
            if model_type == 'biggan':
                gen_find_imgs = model[0](latent_vector[0:num_samples], y_shared[0:num_samples])    
            else:
                gen_find_imgs = model.netG(latent_vector[0:num_samples])        
            
    return gen_find_imgs.detach().cpu()



#Function to show the images of latent vectors
def show_img_from_latents(latent_vector, y_shared, model, model_type, minibatch_size, nrow=5):
    #Add a dimension if needed
    if len(latent_vector.shape) < 2:
        latent_vector = latent_vector.unsqueeze(0)
    
    imgs = get_images(latent_vector, y_shared, model, model_type, minibatch_size)
    
    utils.show_imgs(imgs,nrow = nrow)




#Function to generate a set of fake samples
def fake_samples_gen(num_samples, model, model_type, category, minibatch_size):
    
    if model_type == 'biggan':
        #Generate noise samples to put into the GAN
        y_test = torch.ones([1]).long() * category
        z,_ = bigUtils.prepare_z_y(1,120,1000) 
        
        #Define the generator
        G = model[0]
        
        
        #Put the noise samples into the GAN to generate images
        with torch.no_grad():
            y_test_shared = G.shared(y_test.cuda())
            gen_test = G(z.cuda(),y_test_shared.cuda())
            
        num_channels, image_size, _ = gen_test[0].size()
        
        #Store all the generated images
        gen_imgs = torch.zeros(num_samples,num_channels,image_size,image_size, device="cpu")
        
        #Latent vectors of generated images
        z,_ = bigUtils.prepare_z_y(num_samples,120,1000) 
        y = torch.ones([num_samples]).long() * category
        y_shared = G.shared(y.cuda()).detach()
        
        
        #If the number of samples is greater than 20, we need to split it up into batches so it doesn't fill up the GPU memory
        if num_samples > minibatch_size:
            sample_batch = int(math.ceil(num_samples/minibatch_size))
            for i in range(0,sample_batch):
                with torch.no_grad():
                    
                    #Check if there are fewer than minibatch_size images left
                    if (i + 1) * minibatch_size > num_samples:
                        end = num_samples + 1
                    else:
                        end = (i + 1) * minibatch_size
                        
                    #Get all the generated images
                    gen_imgs_gpu = G(z[(i*minibatch_size):end].cuda(), y_shared[(i*minibatch_size):end])
                    gen_imgs[(i*minibatch_size):end] = gen_imgs_gpu.cpu()
                    
                    del gen_imgs_gpu
                    torch.cuda.empty_cache
                    
        else:
            with torch.no_grad():
                #Get all the generated images
                gen_imgs_gpu = G(z.cuda(), y_shared)
                gen_imgs = gen_imgs_gpu.cpu()
                
                del gen_imgs_gpu
                torch.cuda.empty_cache
    
        
        #Return the cpu images
        return gen_imgs, z, y_shared, y   
        
    else:
        
        #Generate noise samples to put into the GAN
        noiseTest, _ = model.buildNoiseData(1)
        
        #Put the noise samples into the GAN to generate images
        with torch.no_grad():
            gen_test = model.test(noiseTest)
            
        num_channels, image_size, _ = gen_test[0].size()
        
        #Store all the generated images
        gen_imgs = torch.zeros(num_samples,num_channels,image_size,image_size, device="cpu")
        
        #Latent vectors of generated images
        gen_latent, _ = model.buildNoiseData(num_samples)
        
        
        #If the number of samples is greater than 20, we need to split it up into batches so it doesn't fill up the GPU memory
        if num_samples > minibatch_size:
            sample_batch = int(math.ceil(num_samples/minibatch_size))
            for i in range(0,sample_batch):
                with torch.no_grad():
                    
                    #Check if there are fewer than minibatch_size images left
                    if (i + 1) * minibatch_size > num_samples:
                        end = num_samples + 1
                    else:
                        end = (i + 1) * minibatch_size
                        
                    #Get all the generated images
                    gen_imgs_gpu = model.test(gen_latent[(i*minibatch_size):end])
                    gen_imgs[(i*minibatch_size):end] = gen_imgs_gpu.cpu()
                    
                    del gen_imgs_gpu
                    torch.cuda.empty_cache
                    
        else:
            with torch.no_grad():
                #Get all the generated images
                gen_imgs_gpu = model.test(gen_latent)
                gen_imgs = gen_imgs_gpu.cpu()
                
                del gen_imgs_gpu
                torch.cuda.empty_cache
    
        
        #Return the cpu images
        return gen_imgs,gen_latent, None, None   




#Function to transform images to the feature space of the discriminator
def discrim_features(imgs, model, model_type, y):

    #Find the number of features 
    if model_type == 'wgangp':
        test_feat = model.netD(imgs[0:2],getFeature = True)
    elif model_type == 'pgan':
        test_feat = model.netD(imgs[0:2],feature=3)
    else:
        D = model[1]
        test_feat = D(imgs[0:2].cuda(), y[0:2].cuda(), getFeature = True)
        
    _, dim_feats = test_feat.size()

    #Number of samples
    num_samples, num_channels, image_size, image_size = imgs.size()

    #Matrix for the features on the cpu
    features_cpu = torch.zeros(num_samples,dim_feats, device="cpu")
    
    with torch.no_grad():
            
        #Pass each image of the batch into the discriminator by itself with a copy of itself
        for i in range(num_samples):
            
            img_copy = torch.zeros(2, num_channels, image_size, image_size)
            img_copy[0] = imgs[i]
            img_copy[1] = imgs[i]
            
            #Pass in the image with its copy to the discriminator
            if model_type == 'wgangp':
                feat = model.netD(img_copy,getFeature=True)
            elif model_type == 'pgan':
                feat = model.netD(img_copy,feature=3)
            else:
                feat = D(img_copy.cuda(),y.cuda(),getFeature=True)
            
            #Store the layers
            features_cpu[i] = feat[0].cpu()
            
            del feat
            torch.cuda.empty_cache()
        
    #Flatten the entire batch so each row is one sample 
    flat_samples = imgs.view(num_samples, num_channels*image_size*image_size)
    
    #Return the features on the cpu
    return flat_samples, features_cpu


def progression_look(og_latents, recovered_latents, ys, ys_shared, model, model_type, num_rows, save_dir, minibatch_size):

    num_samples, _ = og_latents.size()
    
    #If the number of samples is greater than 20, we need to split it up into batches so it doesn't fill up the GPU memory
    sample_batch = int(math.ceil(num_samples/minibatch_size))
    
    for i in range(sample_batch):
        
        #Check if there are fewer than minibatch_size images left
        if (i + 1) * minibatch_size > num_samples:
            end = num_samples + 1
        else:
            end = (i + 1) * minibatch_size

        #Original latent vectors
        x = og_latents[(i*minibatch_size):end]
        
        #Recovered latent vectors
        y = recovered_latents[(i*minibatch_size):end]
        
        num_this_batch, _ = x.size()
        
        
        #Difference vector
        diff = y.cpu() - x.cpu() 
    
        #Get the sizes of the images
        if model_type == 'biggan':
            test = model[0](x[0:1],ys_shared[0:1])
        else:
            test = model.netG(x[0:1])
            
        _,num_channels,image_size,_ = test.size()
        
        imgs = torch.zeros((num_rows+1) * (num_this_batch), num_channels, image_size, image_size)
        
        old = x * 1.0
        
        #Add the original images
        if model_type == 'biggan':
            imgs[0 : num_this_batch] = model[0](old.cuda(),ys_shared[:num_this_batch]).detach().cpu()
        else:
            imgs[0 : num_this_batch] = model.netG(old).detach().cpu()
        
        #Show the progression of subtracting distance vector
        for j in range(num_rows):
        
            #Move in the difference vector direction
            new = old.cpu() + diff
            new = utils.normalize(new)
            
            #Get the new difference direction
            diff = new.cpu() - old.cpu()
            
            old = new * 1.0
            
            #Pass the new latents in and get the images
            if model_type == 'biggan':
                imgs[(j+1)*num_this_batch : (j+2)*num_this_batch] = model[0](new.cuda(),ys_shared[:num_this_batch]).detach().cpu()
            else:
                imgs[(j+1)*num_this_batch : (j+2)*num_this_batch] = model.netG(new).detach().cpu()
            
    
        utils.save_grid(imgs[0:num_this_batch*num_rows], save_dir + 'grid{0}.png'.format(i), nrow=num_this_batch)

    
    
    
    
def move_in_dir(og_latents, direction, model):
    num_prog_imgs = 5

    #Original latent vectors
    x = og_latents[0:num_prog_imgs]
    
    
    pairwise = nn.PairwiseDistance()
    cosine_similarity = nn.CosineSimilarity()
    
    #Difference vector
    diff = direction

    #Number of rows to show
    num_rows = 10
    
    test = model.netG(x[0:1])
    _,num_channels,image_size,_ = test.size()
    
    imgs = torch.zeros(num_rows * (num_prog_imgs+1), num_channels, image_size, image_size)
    
    old = x * 1.0
    
    scores = torch.zeros(num_rows, num_prog_imgs)
    
    imgs[0 : num_prog_imgs] = model.netG(old).detach().cpu()
    
    #Show the progression of subtracting distance vector
    for i in range(num_rows):
    
        #Move in the difference vector direction
        new = old.cpu() + diff
        new = utils.normalize(new)
        
        diff = new.cpu() - old.cpu()
        
        old = new * 1.0
        
        
        print('Distances for ' + str(i) + ' times diff')
        print(pairwise(x.cpu(),new))
        print(cosine_similarity(x.cpu(),new))
        imgs[(i+1)*num_prog_imgs : (i+2)*num_prog_imgs] = model.netG(new).detach().cpu()
        
        with torch.no_grad():
            score, _, _, _ = model.netD(imgs[i*num_prog_imgs : (i+1)*num_prog_imgs])
            scores[i] = score.t()
            
    print('Discriminator Score')
    print(scores)
        
    
    utils.show_imgs(imgs[0:num_prog_imgs*num_rows], nrow=num_prog_imgs)
    #show_imgs(imgs[100:200],nrow = num_prog_imgs)