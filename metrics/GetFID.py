#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 07:36:11 2020

@author: jeff
`

"""
import tensorflow as tf
import numpy as np
import os
import torch
import torchvision

from collections import OrderedDict



import TTUR
import TTUR.fid as fid
from tqdm import tqdm


#%% 
        
dataset_dir = None
dataset_name = None
dataset_stats_dir = 'dataset_fid_stats/' 
dataset_stats_name = 'celebastats.npz'
dataset_stats_file = dataset_stats_dir + dataset_stats_name
fid_img_dir = '../results/pgan_celeba_FID_Images/'

if __name__ == '__main__':
    
    if dataset_stats_name is None:
        
        if not os.path.exists(dataset_stats_dir):
            os.makedirs(dataset_stats_dir)
        
        #Find the statistics for the dataset
        inception_path = fid.check_or_download_inception(None)
        fid.create_inception_graph(str(inception_path))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mu, sigma = fid._handle_path(dataset_dir, sess, low_profile=True)
            
        dataset_stats_file= dataset_stats_dir+'{0}.npz'.format(dataset_name)
            
        #Save the dataset statistics
        np.savez(dataset_stats_file,mu=mu,sigma=sigma)
        
    #Keep track of the FID scores
    fid_score = OrderedDict()
    paths = [dataset_stats_file]
    paths.append(' ')
    
    for folder in os.listdir(fid_img_dir):
        
        #Skip if it is not a folder of images
        if not os.path.isdir(fid_img_dir + folder):
            continue
    
        if not os.path.exists(fid_img_dir+'{0}.npz'.format(folder)):
            #Find the statistics for the dataset
            inception_path = fid.check_or_download_inception(None)
            fid.create_inception_graph(str(inception_path))
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mu, sigma = fid._handle_path(fid_img_dir + '{0}/{0}'.format(folder), sess, low_profile=True)
                
            #Save the dataset statistics
            np.savez(fid_img_dir +'{0}.npz'.format(folder),mu=mu,sigma=sigma)
            
        
        paths[1] = fid_img_dir +'{0}.npz'.format(folder)
        
        #Calculate the fid score
        fid_score[folder] = fid.calculate_fid_given_paths(paths, None,low_profile=True)
    

    
    for key,value in fid_score.items():
        print(key,value)
    
