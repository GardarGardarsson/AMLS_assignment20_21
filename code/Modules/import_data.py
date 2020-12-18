#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:09:17 2020
Module to import .jpg and .png images from celeba and cartoon dataset 
using the imageio Python library. 


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

# The keras utilities module contains a function to transform data to
# one hot vector
from keras.utils import to_categorical

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

# Image import function for dataset
def importImages(img_path, img_type='.jpg', surpress=False):
    
    print('Importing image data...')
    
    # Define a list to hold image data
    X = [] # Our feature vector
    
    # Navigate to the subfolder that was handed to us
    dir_path = os.getcwd()
    full_path = dir_path + img_path
    
    # Let's perform this action for every single image in our directory
    for i in range(len(os.listdir(full_path))):
        # We'll devise a filename to keep things in order
        filename = str(i) + img_type
        # We append each 178 x 218 px and 3 channel (RGB) to the list
        X.append(imageio.imread(full_path + filename))
    
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

# Label import function for dataset
def importLabels(label_path,surpress=False):
    
    print('Importing labels...')
    
    # Define a list to hold label data
    y = [] # Label vector
    
    # Navigate to the subfolder that was handed to us
    dir_path = os.getcwd()
    full_path = dir_path + label_path
    
    # Scan for .csv's on label_path
    entriesOnPath = os.listdir(full_path)
    
    # Extract the name of the label file
    label_file = [file for file in entriesOnPath if file.endswith(".csv")]
    
    # Let us now load the labels from the .csv file
    y = pd.read_csv(full_path + label_file[0],sep="\t")
    
    # We may choose to surpress outputs
    if not surpress:
        # Let's print size of label df to indicate everything went smoothly
        print("Label data is stored in Pandas DataFrame 'y' with dimensions: {}".format(y.shape))
    else:
        pass
    return y

# Import function that uses importImages() and importLabels() and plots samples from data
def dataImport(img_path,label_path,img_type='.jpg',task='A',surpress=False,return_img_indices=False):
    
    # We load image and label data to vectors X and y respectively
    X = importImages(img_path,img_type,surpress)
    y = importLabels(label_path,surpress)

    """
    P L O T   T A S K   A
    """
    # Display images from the celebrity dataset
    if task == 'A':
        
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
                plt.suptitle('Randomly selected images and corresponding labels from set')
                plt.tight_layout()
                plt.show()
  
        # If outputs were surpressed, do nothing
        else:
            pass
    
    """
    P L O T   T A S K   B
    """
    
    # If the task is 'B' or other, we display images from the cartoon dataset
    if task == 'B':
        
        # Let's print a couple of random images from our imported data with corresponding labels
        random_img = np.random.randint(10000, size=(2,3))
        
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
                    face_shape = y.loc[random_img[i,j],'face_shape']
                    eye_color = y.loc[random_img[i,j],'eye_color']
                    title = "Face shape: {} \n Eye colour: {}".format((face_shape),(eye_color))
                    ax[i,j].set_title(title)
                    
            # Set tight layout and display the plot
            plt.suptitle('Randomly selected images and corresponding labels from set')
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

# Convert a category (column) of a Pandas Dataframe to one hot vector
def to_one_hot_vector(label_data,category):
    
    # Transform a column of a label dataframe to one hot vector
    labels = label_data.loc[:,category].copy()
    one_hot_vec = to_categorical(labels)
    
    return one_hot_vec

# Convert -1 / 1 categorised data to 0 / 1 binary values
def to_binary(label_data,category):

    # Transform -1 | 1 categorised data to 0 | 1 values
    labels = label_data.loc[:,category].copy()
    labels[labels<0] = 0
    
    return labels

# Extract column from dataframe
def get_category(label_data,category):

    # Extract -1, 1 labels from 
    labels = label_data.loc[:,category].copy()
    
    return labels

# Put this function together last-minute to use further throughout code.
def plot_celeba(img_arr,labels, nrows, ncols):
    
    # Let's print a couple of random images from our imported data with corresponding labels
    random_img = np.random.randint(len(labels), size=(nrows,ncols))
    
    # Define a tuple to iterate over
    row, col = random_img.shape
    
    # Define layout and size of subplot
    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(20,30))
    
    labels = labels.reset_index(drop=True)
    print(random_img)
    # Let's populate our 2x3 subplot
    for i in range(row) :
        for j in range(col):
            # We extract a 4D array from our image library
            extract = img_arr[(random_img[i,j]),:,:,:]
            # Get rid of the 4th dimension, i.e. squeeze back to 3D
            img = np.squeeze(extract)
            
            # We can then plot the image using matplotlib's .imshow() function
            ax[i,j].imshow(img)
            # Turn off plot axes
            ax[i,j].axis("off")
            
            # Set the title of each image as the corresponding labels
            gender = labels.loc[random_img[i,j],'gender']
            smiling = labels.loc[random_img[i,j],'smiling']
            img_name = labels.loc[random_img[i,j],'img_name']
            title = "Gender: {} \n Smiling: {}\nFilename: {}".format(("Female" if gender == -1 else "Male"),
                                                      ("No" if smiling == -1 else "Yes"),img_name)
            ax[i,j].set_title(title)
            
    # Set tight layout and display the plot
    plt.suptitle('Some misclassified samples')
    plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"
    
    # Retrieve image and label data
    X , y , random_img = dataImport(img_path,label_path,surpress=False,return_img_indices=True)
