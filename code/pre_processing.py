#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 21:03:34 2020

@author: gardar
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Convert RGB image array to grayscale using Rec. 601 encoding
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

# Center image as deviation from mean of image set
def centerImg(X, surpress=False,mean=None):
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
    
    # We perform the first pre-processing action on the training imageset.
    # When we center our validation and test images, we must center them
    # on the "mean image", calculated from the training set.
    # Hence, during the first pass, i.e. with the training data, 
    # we calculate the mean, and then pass the mean into the centering
    # function again, for pre-processing of the test and validation data.
    
    # If we have the training data, no mean is assumed
    if mean is None:
        # We calculate the mean of the 2nd and 3rd dimensions
        mean = X.mean(axis=0) 
        
        # We flatten the mean image from 218 x 178 to 1 x 38804, as we perform
        # centering using this later
        mean = mean.flatten()
    # If we're processing for the validation or test sets, the mean is already
    # known, and we route to the else condition.
    else:
        pass # ... and do nothing
    
    # We can choose to surpress outputs
    if not surpress:
        # Display an image of the "average" face from the dataset
        print("Displaying the average face from the data...")
        plt.figure()
        
        # We grab the dimensions of the images we're working with
        N,h,w = X.shape
        
        # And use it to restructure the previously flattened mean image
        plt.imshow(np.reshape(mean,(h,w)),cmap=plt.get_cmap("gray"))
        
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
    
    # Let's load the centered images to a list using a comprehension
    Xcenter = np.array([Xflat[img,...] - mean for img in range(len(Xflat))])
    
    # As we may well obtain negative values here, we constrain the centered image array to the interval 0...255 
    Xcenter = np.interp(Xcenter, (Xcenter.min(), Xcenter.max()), (0, 255))
    
    return Xcenter, mean

# Uses reduceRGB(X) and centerImg(X,mean) to pre-process images for PCA
def imgProcessing(X,surpress=False,mean=None):
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
    Xcenter,mean = centerImg(Xgry,surpress)
    
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
    
    return Xcenter,mean

# A hand-crafted PCA function using singular value decomposition    
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

# Displays eigenfaces
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

# Fits an image array to a previously calulated eigenvector base and returns 
# n number of principal components
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
    Xfitted = np.dot(X, WT.T)
    
    # Reduce dimensions to n number of components
    Xfitted = Xfitted[:,:n_components]
    
    return Xfitted

# Crop an image array given final height, width and crop offsets
def crop(img_arr,H,W,ver_off,hor_off):
    """
    Given an image array, this function returns a region of interest supplied
    by pixelwise height, width and horizontal- and vertical offsets from the
    original images
    
    Parameters
    ----------
    img_arr: A library RGB(a) images
    H: height of final images
    W: width of final images
    ver_off: vertical offset to crop from (V offset + height)
    hor_off: horizontal offset to crop from (H offset + width)
    
    Returns
    -------
    cropped : An array of cropped images
    
    """
    
    print("Cropping images to final size: {} x {} px...".format(H,W))
    
    # Initialise an empty array that can accomodate as many instances
    # as the number of images we wish to crop, in the required final dimensions
    cropped = np.empty((img_arr.shape[0],
                        H, # Final height in px
                        W, # Final width in px
                        img_arr.shape[3]))
    
    # For each of the images provided
    for i,img in enumerate(img_arr):
        # We store a cropped version, that crops a H x W px rectangle
        # from a supplied region with horizontal and vertical offsets
        cropped[i] = img[ver_off:ver_off + H,hor_off:hor_off + W]
        
    return cropped

# Apply Sobel filter for edge detection
def sobel(img_arr, surpress=False):
    
    """
    Given an image array, this function applies a Sobel-Feldman operator to the
    pixels of the image. The Sobel-Feldman operator is a computationally in-
    expensive isotropic gradient operator, namely a discrete differentation 
    operator and is useful for edge detection in images, in the presence of 
    little noise.
    
    Parameters
    ----------
    img_arr: A library RGB(a) images
    surpress: An option to turn off outputs
    
    Returns
    -------
    sobel : An array of Sobel filtered images
    
    """
    
    print("Applying Sobel filter...")
    
    # The sobel function assumes the dimensions of the image array it is passed
    sobel = np.empty(img_arr.shape)
    
    for i,img in enumerate(img_arr):
        
        # x-directional SOBEL gradient operator
        # mode = 'constant' fills values beyond edges with a constant value cval
        s_x = ndimage.sobel(img, axis=0, mode='constant',cval = 0.0)
        
        # y-directional SOBEL gradient operator
        s_y = ndimage.sobel(img, axis=1, mode='constant',cval = 0.0)
        
        # Our combined filter response is the hypotenuse of the x- and y-components
        sobel[i] = np.hypot(s_x, s_y)
    
    # We may want to display the output of our function
    if not surpress:
        
        random = np.random.randint(len(img_arr))
        
        fig,ax = plt.subplots(1,2,figsize=(6,6))
        # Plot original
        ax[0].imshow(img_arr[random], cmap=plt.get_cmap("gray"))
        ax[0].set_title("Passed sample image")
        ax[0].axis("off")
        
        ax[1].imshow(sobel[random], cmap=plt.get_cmap("gray"))
        ax[1].set_title("Sobel filtered")
        ax[1].axis("off")
        
    return sobel

# SURF feature descriptor
def surf(img_arr, n_keypoints, hessianThreshold, upright=True, surpress = False):
    
    """
    Given an image array, this function performs SURF feature extraction
    SURF is a faster alternative to SIFT that yields similar results
    
    Parameters
    ----------
    img_arr: A library RGB(a) images
    surpress: An option to turn off outputs
    
    Returns
    -------
    sobel : An array of Sobel filtered images
    
    """
    print("Extracting {} SURF feature keypoints from images...".format(n_keypoints))
        
    surf_des = []
    
    # Create a SURF detector with a defined hessianThreshold and directionality
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, 
                                       #nOctaves=None, 
                                       #nOctaveLayers=None,
                                       #extended=None, 
                                       upright=upright) # Rotation (in/)variant application    
    for img in img_arr: 
        # Calculate SURF
        kp,des = surf.detectAndCompute(img,None)
        # Smallest keypoint array depends on the hessian threshold.
        # A higher hessian threshold is correlated with fewer but more salient features.
        # This was found as ~20 for the hessian threshold 400, i.e. we can
        # retain only 20 keypoints from our images if we are to treat the
        # imageset without discrimination. This suggests we should skip PCA.
        surf_des.append(des[:n_keypoints])
    
    surf_des = np.array(surf_des)
    
    print(np.array(des[:n_keypoints]).shape)
    print(surf_des.shape)
    
    # We may want to display the output of our function
    if not surpress:
        
        # Always apply on last img in array to save on computation
        # i.e. I'm not saving the keypoints in an object, only the descriptors
        kp_img = cv2.drawKeypoints(img,kp[:n_keypoints],None,(255,0,0),4)
            
        fig,ax = plt.subplots(1,2,figsize=(6,6))
        # Plot original
        ax[0].imshow(img, cmap=plt.get_cmap("gray"))
        ax[0].set_title("Passed image")
        ax[0].axis("off")
        
        ax[1].imshow(kp_img)
        ax[1].set_title("{} SURF keypoints".format(n_keypoints))
        ax[1].axis("off")
        
    return surf_des

# Flatten an image array of:  N x h x w (x Channels) ->  to  ->  N x dim
def flatten(img_arr):
    """
    Flattens a library of grayscale images to N-samples of 1 dimension
    
    Parameters
    ----------
    img_arr: A library RGB(a) images
    
    Returns
    -------
    flat_arr : An array of N x dim dimensions
    
    """
    
    print("Flattening image array...")
    
    dim = img_arr.ndim - 1 # 1st dimension is N-number of samples 
    flat = 1 # Initalise the final dimension to be folded to
    
    # Flatten all but the 1st dimension
    for i in range(dim):
        flat *= img_arr.shape[i+1]
    
    # Reshape image array from N,h,w,channels, to N,(h * w)
    flat_arr = np.reshape(img_arr,(img_arr.shape[0],flat))
    
    # Return the library of flattened objects
    return flat_arr