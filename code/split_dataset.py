#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:02:57 2020

@author: gardar
"""
# To modularise the program, we need to add the current directory to sys path
import os
import sys

# Grab the current file path...
currentPath = os.path.dirname(os.path.abspath(__file__))

# ... and append it to the system path
sys.path.append(currentPath)

# Now we can import our own modules into our script.
import import_data as ds

# Matplotlib for visualisation
import matplotlib.pyplot as plt
# Scikit learn for splitting the datasets into train, validation and test folds
from sklearn.model_selection import train_test_split

def split_dataset(X,y,test_size=0.2,val_size=0.2,surpress=False):
    # Split data to train and test sets
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
        
    # Split train set to train end validation sets
    Xtrain,Xval,ytrain,yval = train_test_split(Xtrain,ytrain,test_size=0.2)
    
    # We may choose to surpress outputs of this function
    if not surpress:
        # Displaying segmentation of dataset
        print('Plotting the dataset split...')
        
        # Display the splits in a pie chart
        sizes = [(len(Xtrain)/len(X)*100),(len(Xval)/len(X)*100),(len(Xtest)/len(X)*100)]
        labels = ['Train: '+str(len(Xtrain)),'Validation: '+str(len(Xval)),'Test: '+str(len(Xtest))]
            
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.suptitle('Dataset split')
        plt.show()
        
    else:
        pass 
    
    return Xtrain,Xval,Xtest,ytrain,yval,ytest
    
    
if __name__ == '__main__':
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    # Split data to train and test sets
    Xtrain,Xval,Xtest,ytrain,yval,ytest = split_dataset(X,y,test_size=0.2,val_size=0.2)
 