#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:32:39 2020

@author: xiaoyangchen
"""
import h5py
import numpy as np
import SimpleITK as sitk

subject_list = [80]

for i in range(len(subject_list)):
    subject_id = subject_list[i]

    # subject8
    lmks = np.array([[179., 131., 334.],
                     [426., 158., 329.],
                     [204., 110., 281.],
                     [404., 136., 274.],
                     [228., 107., 265.],
                     [376., 127., 256.],
                     [138., 261., 258.],
                     [163., 157., 284.],
                     [440., 293., 259.],
                     [438., 190., 275.],
                     [297.,  49., 148.],
                     [320.,  51., 150.],
                     [250.,  79.,  61.],
                     [358.,  98.,  60.],
                     [177., 171., 245.],
                     [170., 208., 206.],
                     [416., 202., 242.],
                     [415., 240., 200.]])
    
    lmks = lmks.astype(np.int32)
    
    label = sitk.ReadImage('/Volumes/XYCHEN/label_nii/subject{0}.nii.gz'.format(subject_id))
    label = sitk.GetArrayFromImage(label) * 1000
    
    for idx in range(lmks.shape[0]):
        print('idx: ', idx)
        landmark_i = lmks[idx] - 1
        
        if not np.all(landmark_i == np.array([-1, -1, -1])):
            x0, y0, z0 = landmark_i.astype(np.int32)
            radius = 3
            
            for x in range(x0-radius, x0+radius+1):
                for y in range(y0-radius, y0+radius+1):
                    for z in range(z0-radius, z0+radius+1):
                        #label[(z-2):(z+5), (y-2):(y+5), (x-2):(x+5)] = idx+1
                        if np.sqrt(np.sum(np.square(np.array([x, y, z]) - np.array([x0, y0, z0])))) <= radius:
                            label[z][y][x] = idx+1
    
    new_label = label
    new_label = sitk.GetImageFromArray(new_label)
    sitk.WriteImage(new_label, '/Volumes/XYCHEN/lmk_on_label/subject{0}_phase2.nii.gz'.format(subject_id))



