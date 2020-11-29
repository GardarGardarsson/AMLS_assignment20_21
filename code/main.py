# %% Import libraries...
"""
L I B R A R Y   I M P O R T
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
import split_dataset as sd
import pre_processing as prp

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt
# Pandas dataframes for enhanced data storage
import pandas as pd


# %% Load and split data...
if __name__ == '__main__':
    
    """
    L O A D   D A T A
    """
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Datasets/celeba/img/"
    label_path = "/Datasets/celeba/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xval,Xtest,ytrain,yval,ytest = sd.split_dataset(X,y,test_size=0.2,val_size=0.2,surpress=False)
    
    # %% Augment data...
    """
    D A T A   A U G M E N T A T I O N
    """
    
    # We may want to perform data augmentation ...
    # For e.g. flips, mirroring, concealment.
    # I'll leave this cell empty for later implementation.
    
    
    #%% Crop images...
    """
    P R E - P R O C E S S :   C R O P
    """
    
    def crop(img_arr):
        # Initialise an empty array that can accomodate as many instances
        # as the number of images we wish to crop, in the required final dimensions
        cropped = np.empty((img_arr.shape[0],
                            img_arr.shape[1]-40, # Subtract 40px from the height
                            img_arr.shape[2],
                            img_arr.shape[3]),int)
        
        # For each of the images provided
        for i,img in enumerate(img_arr):
            # We store a cropped version, that crops a 20 px banner from the
            # top and bottom of the image
            cropped[i] = img[20:20+178]
            
        return cropped
    
    Xtrain_C = crop(Xtrain)
    Xval_C = crop(Xval)
    Xtest_C = crop(Xtest)
    
    #%% Grayscale images...
    """
    P R E - P R O C E S S :   G R A Y S C A L E
    """
    
    # Reduce the RGB channels to a singular gray channel using Rec. 601 encoding
    Xtrain_gry = prp.reduceRGB(Xtrain_C).astype(np.uint16)
    Xval_gry = prp.reduceRGB(Xval_C).astype(np.uint16)
    Xtest_gry = prp.reduceRGB(Xtest_C).astype(np.uint16)
    
    #%% Perform histogram equalization...
    """
    P R E - P R O C E S S :   H I S T O G R A M   E Q U A L I S A T I O N
    """
    
    print("Checkpoint: Start Histogram Equalisation")
    
    # To perform Contrast Limited Adaptive Histogram Equalisation (CLAHE)
    # we import a scikit-image package
    from skimage.exposure import equalize_adapthist
    
    # We equalize all the image data using an adaptive contrast equaliser
    Xtrain_eq = equalize_adapthist(Xtrain_gry)
    Xval_eq = equalize_adapthist(Xval_gry)
    Xtest_eq = equalize_adapthist(Xtest_gry)
    
    print("Checkpoint: End Histogram Equalisation")
    
    #%% Edge detection with Sobel-Feldman operator...
    """
    P R E - P R O C E S S :   S O B E L
    """
    
    # We now proceed by processing the images with a Sobel filter
    # The Sobel-Feldman operator provides powerful means of edge detection
    # especially, on equalised images
    from scipy import ndimage
    
    def sobel(img_arr, surpress=False):
        
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
        
        return sobel
    
    Xtrain_sob = sobel(Xtrain_eq)
    Xval_sob = sobel(Xval_eq)
    Xtest_sob = sobel(Xtest_eq)
    
    # %% Flatten image data to 1D feature vector...
    """
    P R E - P R O C E S S :   F L A T T E N   T O  1 D - F E A T U R E   V E C T O R
    """
    
    def flatten(img_arr):
        N,h,w = img_arr.shape
        flattened = np.reshape(img_arr,(N,(h*w)))
        return flattened
    
    Xtrain_flat = flatten(Xtrain_sob)
    Xval_flat = flatten(Xval_sob)
    Xtest_flat = flatten(Xtest_sob)
    
    # %%
    """
    P R E - P R O C E S S :   S C A L E   D A T A
    """
    
    # Import a standard scaler
    from sklearn.preprocessing import StandardScaler
    
    # Fit the scaler to the training data
    scaler = StandardScaler().fit(Xtrain_flat)
    
    # Apply the transformation to the training, validation and testing data
    Xtrain_scl  =   scaler.transform(Xtrain_flat)
    Xval_scl    =   scaler.transform(Xval_flat)
    Xtest_scl   =   scaler.transform(Xtest_flat)
    
    
    
    # %% Prepare label data...
    """
    P R E P A R E   L A B E L   D A T A
    """
    
    # The keras utilities module contains a function to transform data to
    # one hot vector
    from keras.utils import to_categorical
    
    def to_one_hot_vector(label_data,category):
        
        # Transform a column of a label dataframe to one hot vector
        labels = label_data.loc[:,category].copy()
        one_hot_vec = to_categorical(labels)
        
        return one_hot_vec
    
    def to_binary(label_data,category):
    
        # Transform -1 | 1 categorised data to 0 | 1 values
        labels = label_data.loc[:,category].copy()
        labels[labels<0] = 0
        
        return labels
    
    # Transform the 'gender' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A1 =  to_binary(ytrain,'gender')
    yval_A1   =  to_binary(yval,'gender')
    ytest_A1  =  to_binary(ytest,'gender')
    
    # Transform the 'smiling' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A2 =  to_one_hot_vector(ytrain,'smiling')
    yval_A2   =  to_one_hot_vector(yval,'smiling')
    ytest_A2  =  to_one_hot_vector(ytest,'smiling')
    
    # %% Perform Cross-Validation...
    """
    B U I L D   P I P E L I N E :   P C A + S V M
    """

    # Let us now import an unsupervised classifier for the
    # dimensionality reduction of the data, namely that of PCA
    from sklearn.decomposition import PCA
    
    # We will also use the SVM classifier of the scikit-learn library
    from sklearn.svm import SVC
    
    # To optimise the number of components, and parameters of our SVM,
    # we shall construct a pipeline, and perform a gridsearch to find
    # both the optimal number of principal components and hyperparameters for the SVM.
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    
    # So we don't have to perform the CV numerous times
    import joblib
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile('CV_object.pkl'):
            
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use a linear SVC
        svm = SVC()
        
        # Combine PCA and SVC to a pipeline
        pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
        
        # Number of Principal components to try
        n_components = [100,500,1000]
        
        # Parameters to try for the SVM and PCA
        params_grid = {
        'svm__C': [0.01, 0.1, 1],
        'svm__kernel': ['linear','rbf','poly'],
        'svm__degree' : [2],
        'svm__gamma': [0.9, 0.5, 0.1],
        'pca__n_components': n_components,
        }
        
        # Initialise the gridsearch cross validation
        estimator = GridSearchCV(pipe,                      # Use the PCA + SVM pipeline
                                 params_grid,               # For these SVM hyperparams (and components)
                                 scoring='accuracy',        # Score on validation accuracy
                                 n_jobs = 2,                # Use 2 processors
                                 return_train_score=True,   # Record the scores for later use
                                 verbose = 10)              # Print the process as it runs
        
        # Fit the model on the training data
        estimator.fit(Xtrain_scl, ytrain_A1)
        
        # Output the best parameters and the score they achieved
        print(estimator.best_params_, estimator.best_score_)
        
        # Pickle the results for later consumption
        joblib.dump(estimator, 'CV_object.pkl')
        
    # %%
    """
    T A S K   B :   C A R T O O N   D A T A S E T
    """
    
    
    # %%
    """
    L O A D   D A T A
    """
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Datasets/cartoon_set/img/"
    label_path = "/Datasets/cartoon_set/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,
                                       label_path,
                                       img_type='.png',
                                       task='B',
                                       surpress=False,
                                       return_img_indices=True)
    
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xval,Xtest,ytrain,yval,ytest = sd.split_dataset(X,y,test_size=0.2,val_size=0.2,surpress=True)
    
    
    # %% Crop images...
    """
    P R E - P R O C E S S :   C R O P
    """
    
    def crop(img_arr):
        # Initialise an empty array that can accomodate as many instances
        # as the number of images we wish to crop, in the required final dimensions
        cropped = np.empty((img_arr.shape[0],
                            img_arr.shape[1]-425, # Subtract 40px from the height
                            img_arr.shape[2]-300,
                            img_arr.shape[3]),int)
        
        # For each of the images provided
        for i,img in enumerate(img_arr):
            # We store a cropped version, that crops a 20 px banner from the
            # top and bottom of the image
            cropped[i] = img[225:225+75,150:150+200]
            
        return cropped
    
    Xtrain_C = crop(Xtrain[:300])
    
    plt.imshow(Xtrain_C[101])
    print(Xtrain_C[101].shape)
    # %%
