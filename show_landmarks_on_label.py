#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:26:46 2021

@author: xiaoyangchen
"""

import h5py
import numpy as np
import SimpleITK as sitk

subject_list = [64]

for i in range(len(subject_list)):
    subject_index = subject_list[i]
    
    label = sitk.ReadImage('./case_{0}_cbct_patient/seg.mha'.format(subject_index))
    spacing = label.GetSpacing()[::-1]
    label = ( sitk.GetArrayFromImage(label) > 0).astype(np.float32) * 2000

#    # subject8
#    lmks = np.loadtxt('./case_{0}_ct_normal_pred.txt'.format(subject_index), delimiter=',')
#    lmks = np.round( lmks/np.array(spacing) )
#    lmks = lmks.astype(np.int32)
#    print(lmks, '\n')
    
    # subject8
    with h5py.File('./case_{0}_cbct_patient.hdf5'.format(subject_index), 'r') as f:
        lmks = f['landmark_i'][()][:, ::-1]*1.6
    lmks = np.round( lmks/np.array(spacing) )
    lmks = lmks.astype(np.int32)
    print(lmks, '\n')
    
    for idx in range(lmks.shape[0]):
        
        landmark_i = lmks[idx] #- 1
        print('idx: ', idx, landmark_i)
        
        if not np.all(landmark_i == np.array([0, 0, 0])):
            z0, y0, x0 = landmark_i.astype(np.int32)
            radius = 3
            
            for x in range(x0-radius, x0+radius+1):
                for y in range(y0-radius, y0+radius+1):
                    for z in range(z0-radius, z0+radius+1):
                        #label[(z-2):(z+5), (y-2):(y+5), (x-2):(x+5)] = idx+1
                        if np.sqrt(np.sum(np.square(np.array([x, y, z]) - np.array([x0, y0, z0])))) <= radius:
                            label[z][y][x] = idx+1
    
    new_label = label
    new_label = sitk.GetImageFromArray(new_label)
    sitk.WriteImage(new_label, './case_{0}_cbct_patient/lmks_show_on_label_gt.mha'.format(subject_index))