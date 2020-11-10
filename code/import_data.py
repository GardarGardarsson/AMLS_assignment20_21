#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:09:17 2020
Module to import .jpg images from celeba dataset using the imageio Python lib
Images are 178 x 218 px

@author: gardar
"""

# Import os module to work with paths, directories etc.
import os

# Import numpy package for mathematical support
import numpy as np
# Import imageio Python library to read image data into numpy arrays
import imageio
# Import matplotlib for visualisation purposes
import matplotlib.pyplot as plt
# Import Pandas for convenient storage of data in DataFrames
import pandas as pd

"""
C O N F I G U R E   M A T P L O T L I B
"""
# Define font sizes
SMALL = 12
MEDIUM = 14
LARGE = 18
HUGE = 22

# Set fontsizes for figures
plt.rc('font', size=SMALL)          # Controls default text sizes
plt.rc('axes', titlesize=LARGE)     # Fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # Fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)    # Fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL)    # Fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL)    # Legend fontsize
plt.rc('figure', titlesize=HUGE)    # Fontsize of the figure title

"""
I M A G E   I M P O R T 
"""

def importImages(img_path,surpress=False):
    
    print('Importing image data...')
    
    # Define a list to hold image data
    X = [] # Our feature vector
    
    # Let's perform this action for every single image in our directory
    for i in range(len(os.listdir(img_path))):
        # We'll devise a filename to keep things in order
        filename = str(i) + '.jpg'
        # We append each 178 x 218 px and 3 channel (RGB) to the list
        X.append(imageio.imread(img_path + filename))
    
    # We'll now convert our list to a numpy array
    X = np.array(X)
    
    # As there are 5000 images, the dimensions of our array should be 5000 x 3 x 178 x 218
    # We can return the size of the array to verify that all images were read 
    # We may choose to surpress outputs
    if not surpress:
        print("Image data is stored in numpy array 'X' of size: {}".format(X.shape))
    else:
        pass
    
    return X

"""
L A B E L   I M P O R T 
"""

def importLabels(label_path,surpress=False):
    
    print('Importing labels...')
    
    # Define a list to hold label data
    y = [] # Label vector
    
    # Scan for .csv's on label_path
    entriesOnPath = os.listdir(label_path)
    
    # Extract the name of the label file
    label_file = [file for file in entriesOnPath if file.endswith(".csv")]
    
    # Let us now load the labels from the .csv file
    y = pd.read_csv(label_path + label_file[0],sep="\t")
    
    # We may choose to surpress outputs
    if not surpress:
        # Let's print size of label df to indicate everything went smoothly
        print("Label data is stored in Pandas DataFrame 'y' with dimensions: {}".format(y.shape))
    else:
        pass
    return y

"""
I M P O R T   D A T A   &   V I S U A L I S E
"""

def dataImport(img_path,label_path,surpress=False,return_img_indices=False):    
    
    # We load image and label data to vectors X and y respectively
    X = importImages(img_path,surpress)
    y = importLabels(label_path,surpress)
    
    # Let's print a couple of random images from our imported data with corresponding labels
    random_img = np.random.randint(5000, size=(2,3))
    
    # We may choose to surpress outputs in our function call
    if not surpress:
        # Displaying random image data...
        print('Displaying random images and corresponding labels from set...')
        
        # Define a tuple to iterate over
        row, col = random_img.shape
        
        # Define layout and size of subplot
        fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(12,8))
        
        # Let's populate our 2x3 subplot
        for i in range(row) :
            for j in range(col):
                # We extract a 4D array from our image library
                extract = X[(random_img[i,j]),:,:,:]
                # Get rid of the 4th dimension, i.e. squeeze back to 3D
                img = np.squeeze(extract)
                
                # We can then plot the image using matplotlib's .imshow() function
                ax[i,j].imshow(img)
                # Turn off plot axes
                ax[i,j].axis("off")
                
                # Set the title of each image as the corresponding labels
                gender = y.loc[random_img[i,j],'gender']
                smiling = y.loc[random_img[i,j],'smiling']
                title = "Gender: {} \n Smiling: {}".format(("Female" if gender == -1 else "Male"),
                                                          ("No" if smiling == -1 else "Yes"))
                ax[i,j].set_title(title)
                
        # Set tight layout and display the plot
        plt.suptitle('Randomly chosen images and corresponding labels from set')
        plt.tight_layout()
        plt.show()
        
    # If outputs were surpressed, do nothing
    else:
        pass
    
    # If user has requested to retrieve indices used to show random images we return them with X and y
    if return_img_indices:
        return X,y,random_img
    
    # In other cases we just return the feature vector X and label vector y
    else:
        return X,y

if __name__ == '__main__':
    
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"
    
    # Retrieve image and label data
    X , y , random_img = dataImport(img_path,label_path,surpress=False,return_img_indices=True)
