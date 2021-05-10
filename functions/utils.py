#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 07:54:34 2021

@author: jeff

utils.py
    - Commonly used functions

"""

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as image
import os
import json




#Show multiple images in a grid
def show_imgs(plot_imgs,nrow = 7):
    
    #Put the images in a grid and show them
    grid = torchvision.utils.make_grid(plot_imgs.clamp(min=-1, max=1), nrow = int(nrow), scale_each=True, normalize=True)
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    
    
    
#Function to normalize vectors onto the hypersphere of sqrt(n) where n is the number of dimensions
def normalize(x):
    x = x * (((x**2).mean(dim=1, keepdim=True) + 1e-8).rsqrt())
    
    return x




#This function takes two matrices. Each matrix should have each row as one sample. The function will find the distance matrix which has the Euclidean distance from each sample of one matrix to one sample of the other.
def distance_matrix(samples_matrix_1, samples_matrix_2, device):
    X = samples_matrix_1.to(device)
    Y = samples_matrix_2.to(device)

    X_dim, _ = X.size()
    Y_dim, _ = Y.size()
    
    #Calculates the distance matrix: diag(X_t*X) - 2*Y_t*X + diag(Y_t*Y)
    #Transpose is swapped because the matrix has the samples as rows instead of columns
    diag_X = torch.diag(torch.mm(X,torch.t(X)))
    diag_Y = torch.diag(torch.mm(Y,torch.t(Y)))
    X_Y = torch.mm(Y,torch.t(X))
    mDist = diag_X.expand(Y_dim,X_dim) - 2*X_Y + torch.t(diag_Y.expand(X_dim,Y_dim))

    return mDist



#Cosine similarity
#This function takes two matrices. Each matrix should have each row as one sample. Similarity matrix will have the cosine similarity of each vector in X and each vector in Y
def cos_similarity_matrix(samples_matrix_1, samples_matrix_2, device):
    X = samples_matrix_1.to(device)
    Y = samples_matrix_2.to(device)

    #Find the inner product of all the vectors in X with all the vectors in Y
    numerator = torch.mm(X,Y.t())
    
    #Find the multiplication between the norms of all the vectors of X and Y
    denominator = torch.mm(X.norm(dim=1).unsqueeze(1), Y.norm(dim=1).unsqueeze(0))
    
    #Find the similarity matrix
    similarity_matrix = numerator / denominator
    
    return similarity_matrix



#Function to apply the truncation trick
def truncation_trick(latents, radius):
    num_samples,num_dims = latents.size()
    
    for i in range(num_samples):
        for j in range(num_dims):
            #Check if the value is outside of the truncation threshold
            while abs(latents[i][j]) > radius:
                latents[i][j] = torch.randn(1)
    
    return latents  



#Find the latent variable of the NN 
def find_NN(data_features, gen_features, latent_vectors):
    
    if torch.cuda.is_available():
        #Set device to be the GPU
        device = torch.device("cuda:0")
       
        #Calculate the distance_matrix for this batch
        mDist = distance_matrix(gen_features,data_features,device=device)
        
        
        #Get the index of the smallest distance along each row
        NN_dist, NN_idx = torch.min(mDist, dim=1)
    
        num_imgs,num_features = data_features.shape
        _, latent_dim = latent_vectors.shape
    
        #Store the latent vector of the NN for each image
        NN_latent = torch.zeros([num_imgs,latent_dim])
    
        for i in range(num_imgs):
            NN_latent[i] = latent_vectors[NN_idx[i]]
        
    return NN_latent,mDist   



#Save a single image
def save_img(gen_img, img_count, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Make it so it does not need gradient
    gen_img = gen_img.clone().detach().requires_grad_(False)
    
    #Save the image
    torchvision.utils.save_image(gen_img.clamp(min=-1, max=1), save_dir + '%05d.png' % img_count, scale_each=True, normalize=True)
    
#Save a single image
def save_img_ex(gen_img, save_dir):
    
    #Make it so it does not need gradient
    gen_img = gen_img.clone().detach().requires_grad_(False)
    
    #Save the image
    torchvision.utils.save_image(gen_img.clamp(min=-1, max=1), save_dir, scale_each=True, normalize=True)
    
#Save a grid of images
def save_grid(plot_imgs, save_dir, nrow = 5):
    
    #Put the images in a grid 
    grid = torchvision.utils.make_grid(plot_imgs.clamp(min=-1, max=1), nrow = int(nrow), scale_each=True, normalize=True)
    
    # f = plt.figure()
    # f.set_figheight(15)
    # f.set_figwidth(15)
    # plt.imshow(grid.permute(1, 2, 0).numpy())
    
    #Save the image
    torchvision.utils.save_image(grid,save_dir)
    
    
#Read a json config file
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)