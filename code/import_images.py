#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:09:17 2020
Module to import .jpg images from celeba dataset using the imageio Python lib
Images are 178 x 218 px

@author: gardar
"""

# Import os module to work with paths etc.
import os
# Import imageio Python library to read image data into numpy arrays
import imageio
# Import numpy package for greater mathematical support
import numpy as np

# Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
data_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"

# Define a list to hold image data
X = []
y = []

# Let's perform this action for every single image in our directory
for file in os.listdir(data_path):
    if file.endswith('.jpg'):
        X.append(imageio.imread(data_path + file))

# We'll now convert our list to a numpy array
X = np.array(X)

# We can return the size of the array to verify that all images were read
print(X.shape)