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
import argparse
import pandas as pd



import TTUR
import TTUR.fid as fid
from tqdm import tqdm


#%% 
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Find the FID of the images in the folders')
    parser.add_argument('--dataset_stats_dir', default='dataset_fid_stats/', help='Location of the dataset statistics folder')
    parser.add_argument('--dataset_stats_name',  help='Name of the datset statistics file (Ex: celebastats.npz)')
    parser.add_argument('--fid_img_dir', help='Location of the FID image folder (Ex:../results/pgan_celeba_FID_Images/')
    parser.add_argument('--dataset_dir', default = None, help='Location of dataset directory')
    parser.add_argument('--dataset_name', help='Name of the dataset')
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    dataset_stats_dir = args.dataset_stats_dir
    dataset_stats_name = args.dataset_stats_name
    dataset_stats_file = dataset_stats_dir + dataset_stats_name
    fid_img_dir = args.fid_img_dir
    
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
        
    #Save the results
    fid_pd = pd.Dataframe.from_dict(fid_score.items())
    fid_pd.to_csv(fid_img_dir + 'fid_score.csv')
    
