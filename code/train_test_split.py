#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:02:57 2020

@author: gardar
"""

import sys
import os

CURRENT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT)

import import_images as ds

# Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"

# Retrieve image and label data
X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)