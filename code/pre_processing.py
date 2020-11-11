#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:03:34 2020

@author: gardar
"""
import numpy as np
import matplotlib.pyplot as plt

def reduceRGB(X):
    """    
    Convert RGB image array to grayscale using Rec. 601 encoding
    
    Parameters
    ----------
    X : N x h x w x RGB array
    Where:
        N = number of images
        h = image height in pixels
        w = image width in pixels
        RGB = RGB channels (always = 3)

    Returns
    -------
    X : back as a grayscale image using Rec. 601 encoding

    """
    print("Encoding RGB channels to Rec. 601 grayscale...")
    
    # Define RGB weights according to Rec. 601
    RGB_weights = [0.3, 0.6, 0.11]
    # Flatten the RGB channels with weights.
    X = np.dot(X[...,:3], RGB_weights)
    
    return X

def centerImg(X, surpress=False):
    """
    Center image as deviation from mean of image set

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    Xcenter = A centered image set on the mean
    
    C A L C U L A T E   M E A N
    """
    
    print("Centering images...")
    
    mean = X.mean(axis=0) # We then calculate the mean of the 2nd and 3rd dimensions
    
    if not surpress:
        # Display an image of the "average" face from the dataset
        print("Displaying the average face from the data...")
        plt.figure()
        plt.imshow(mean,cmap=plt.get_cmap("gray"))
        plt.axis("off")
        plt.title("Mean Image from Data")
        plt.show()
    else:
        pass
    
    """
    C E N T E R   I M A G E S
    """
    
    # X is a three dimensional array, i.e. library of 2D grayscale images
    N,h,w = X.shape 
    
    # We want to reshape it by unstacking the 2nd and 3rd dimensions, i.e. width by height    
    Xflat = np.reshape(X, (N,h * w)) 
    
    # Let's flatten the mean image from 218 x 178 to 1 x 38804:
    mean = mean.flatten()
    
    # Let's load the centered images to a list using a comprehension
    Xcenter = np.array([Xflat[img,...] - mean for img in range(len(Xflat))])
    
    # As we may well obtain negative values here, we constrain the centered image array to the interval 0...255 
    Xcenter = np.interp(Xcenter, (Xcenter.min(), Xcenter.max()), (0, 255))
    
    return Xcenter

def imgProcessing(X,surpress=False):
    """ 
    X is an unprocessed N x h x w x RGB array 
    It is encoded to grayscale and then centered around the mean image from the data

    Parameters
    ----------
    X : N x h x w x RGB
    Where:
        N = number of images
        h = image height in pixels
        w = image width in pixels
        RGB = RGB channels (always = 3)

    Returns
    -------
    Xcenter: Centered imagedata

    """
    # Grab dimensions of image data
    N,h,w,RGB = X.shape 
    
    # Convert to grayscale images
    Xgry = reduceRGB(X)
    
    # Center images on average image
    Xcenter = centerImg(Xgry,surpress)
    
    if not surpress:
        fig,ax = plt.subplots(1,3)
        # Plot original image example
        ax[0].set_title("Original")
        ax[0].imshow(X[0])
        ax[0].axis("off")
        
        # Plot grayscale image example
        ax[1].set_title("Rec.601")
        ax[1].imshow(Xgry[0].reshape(h,w),cmap=plt.get_cmap("gray"))
        ax[1].axis("off")
        
        # Show centered image
        ax[2].set_title("Centered Image")
        ax[2].imshow(Xcenter[0].reshape(h,w),cmap=plt.get_cmap("gray"))
        ax[2].axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        pass
    
    return Xcenter
    
def PCA_w_SVD(Xcenter,surpress=False):
    """
    Here we perform principal component analysis on an imageset.
    We use singular value decomposition on the centered data,
    to extract the eigenvalues and -vectors from the covariance matrix

    Parameters
    ----------
    X_centered : Expects centered imagedata

    Returns
    -------
    Sigma: Eigenvalues
    WT: Eigenvector

    """
    
    print("Performing PCA with Singular Value Decomposition...")
    
    # Normalise values, i.e. bind to the interval X € [0,1]
    Xnorm = Xcenter / 255.0
    
    # Perform Singular Value Decomposition to retrieve eigenvalues and -vectors
    U, Sigma, WT = np.linalg.svd(Xnorm, full_matrices=False)
    
    # How many features do we really need to accurately describe our imaged data?
    # This PCA Scree Plot may help us decide.
    if not surpress:
        # Scree plot 
        print("Displaying scree plot...")
        features = 100 # Number of components to plot on X-axis
        plt.figure(figsize=(16,7))
        plt.plot(range(features),Sigma[:features],'o')
        plt.plot(range(features),Sigma[:features])
        plt.xlabel("$W_i$")
        plt.ylabel("$\Sigma_{x}$")
        plt.title("Eigenvalues")
        plt.grid("on")
        plt.show()
    else:
        pass 
    
    return Sigma, WT

def showEigenfaces(WT,X):
    """
    Plotting function to display 10 eigenfaces

    Parameters
    ----------
    WT : Eigen vector
    X : Original imageset X (to grab image shape)

    """
    # Show eigenfaces
    print("Displaying 10 first eigenfaces...")
    
    N,h,w,RGB = X.shape
    
    # Set up a 2 x 5 subplot
    fig,ax = plt.subplots(2,5,figsize=(16,6))
    
    # Grab shape to iterate over all axes
    row,col = ax.shape

    # Initialise an iterator to set image titles.
    k = 1 # 1D iterator

    for i in range(row):
        for j in range(col):
            ax[i,j].imshow(np.reshape(WT[k,:], (h,w)),cmap=plt.get_cmap("gray"))
            ax[i,j].axis("off")
            ax[i,j].set_title("#{}".format(k))
            k+=1 
    
    plt.suptitle("Eigenfaces")
    plt.tight_layout()
    plt.show()
    
    return 

def fitPCA(WT,X,n_components=30):
    """
    Fits image array to an existing eigenvector base
    Returns configurable amounts of components to keep
    It's a good idea to see how many should be retained using the scree plot

    Parameters
    ----------
    WT : Eigenvectors
    X : Array to be fitted
    n_components : int, optional
        Number of components to retain. The default is 30.

    Returns
    -------
    Xfitted : A PCA fitted array

    """
    # Fitting to PCA
    print("Fitting to PCA base")
    
    # Calculate "weights"
    Xfitted = X * WT[:X.shape[0]] 
    
    # Reduce dimensions to n number of components
    Xfitted = Xfitted[:,:n_components]
    
    return Xfitted
