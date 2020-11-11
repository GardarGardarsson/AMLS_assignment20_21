# To modularise the program, we need to add the current directory to sys path
import os
import sys

# Grab the current file path...
currentPath = os.path.dirname(os.path.abspath(__file__))

# ... and append it to the system path
sys.path.append(currentPath)

# Now we can import our own modules into our script.
import import_data as ds
import split_dataset as sd 
import pre_processing as prp

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"
    
    """
    L O A D   D A T A 
    """
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xval,Xtest,ytrain,yval,ytest = sd.split_dataset(X,y,test_size=0.2,val_size=0.2,surpress=False)
    
    """
    P R E - P R O C E S S   D A T A 
    """
    # Center images on mean
    Xtrain = prp.imgProcessing(Xtrain,surpress=False)
    Xval = prp.imgProcessing(Xval,surpress=True)
    Xtest = prp.imgProcessing(Xtest,surpress=True)
    
    # Perform PCA on test data to obtain eigenvalues and vectors
    Sigma,WT = prp.PCA_w_SVD(Xtrain,surpress=False)
    
    """
    V I S U A L I S E   E I G E N F A C E S 
    """
    # Let's check what these spooky eigenfaces look like
    prp.showEigenfaces(WT,X)
    
    """
    F I T   V A L I D A T I O N   A N D   T E S T   T O   P C A 
    """
    Xtrain = prp.fitPCA(WT,Xtrain,n_components=20)
    Xval = prp.fitPCA(WT,Xval,n_components=20)
    Xtest = prp.fitPCA(WT,Xtest,n_components=20)
    
    