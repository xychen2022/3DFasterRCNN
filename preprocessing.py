#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 18:56:47 2021

@author: xiaoyangchen
"""

import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as snd

data_path = './TeethSeg/'
processed_data_path = './TeethSeg_processed/'

c = os.listdir(data_path)

for index in range(len(c)):
    print('Processing ', c[index])
    
    img = sitk.ReadImage(data_path + c[index] + '/org.nii.gz')
    print('Image orientation: ', img.GetDirection())
    
    original_spacing = img.GetSpacing()[::-1]
    print('original_spacing: ', original_spacing)
    common_spacing = np.array([1.6, 1.6, 1.6])
        
    array = sitk.GetArrayFromImage(img)
    
    array = snd.zoom(array, zoom=original_spacing/common_spacing, order=1)

    image = np.clip(array, -400, 3000)
    
    image = ( image - np.min(image) ) / ( np.max(image) - np.min(image) )
    
    print('Image shape: ', image.shape, 'Max: ', np.max(image), 'Min: ', np.min(image))
    
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(common_spacing[::-1])
    
    if not os.path.exists(processed_data_path + c[index]):
        os.makedirs(processed_data_path + c[index])
    
    sitk.WriteImage(image, processed_data_path + c[index] + '/img.nii.gz')