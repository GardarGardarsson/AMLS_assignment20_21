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
import user_interface as ui

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt
# Pandas dataframes for enhanced data storage
import pandas as pd

# %% Load data...

if __name__ == '__main__':
    
    """
    L O A D   D A T A
    """
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Datasets/celeba/img/"
    label_path = "/Datasets/celeba/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=True,return_img_indices=True)
    
    
    #%% Crop images...
    
    """
    P R E - P R O C E S S :   C R O P
    """
    X_C =  prp.crop(X, H=178, W=178, ver_off=20,hor_off=0)
    
    
    #%% Grayscale images...
    
    """
    P R E - P R O C E S S :   G R A Y S C A L E
    """
    
    # Reduce the RGB channels to a singular gray channel using Rec. 601 encoding
    X_gry =  prp.reduceRGB(X_C).astype(np.uint16)
    
    # %% Split data to training, validation and test folds...
    
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xval,Xtest,ytrain,yval,ytest = sd.split_dataset(X_gry,y,
                                                           test_size=0.2,
                                                           val_size=0.25,
                                                           surpress=True)
    
    
    # %% Augment data...
    
    """
    T R A I N I N G   D A T A   A U G M E N T A T I O N
    """
    
    # We may want to perform data augmentation ...
    # For e.g. flips, mirroring, concealment.
    # I'll leave this cell empty for later implementation.
    
    # Flip an image array horizontally
    def flip_horizontal(array):
        return np.flip(array,axis = 2)
    
    print("Enriching training sample space with horizontally mirrored images...")
    
    # Enrich the sample space with augmented images (horizontally flip)
    Xtrain = np.concatenate((Xtrain,flip_horizontal(Xtrain)))
    # Appending labels to label set
    ytrain = pd.concat([ytrain,ytrain])
    ytrain.reset_index(drop=True, inplace=True)
    
    
    # %% User selection: Pre-processing method...
    
    """
    P R E - P R O C E S S :   U S E R   S E L E C T   M E T H O D
    """
    
    message = 'Please select a pre-processing method: '
    options = {'1': 'Histogram of Oriented Gradients (Feature Descriptor) - Recommended',
               '2': 'Sobel-Feldman Operator (Edge Detector)',
               '3': 'Speeded Up Robust Features (Feature Descriptor)',
               '4': 'None (Continue with only CLAHE equalisation and PCA)'}
    
    selection = ui.selection_menu(message, options)
    
    print("\nYou've selected: \n{}. {}".format(selection, options[str(selection)]))
    
    
    # %% Feature description with HOG...
    
    """
    P R E - P R O C E S S :   H O G
    """
    
    # If user chose option 1 - HOG pre-processing
    if selection == 1: 
        
        print("Generating HOG feature descriptors...")
        
        # Import the Histogram of Oriented Gradients feature descriptor
        from skimage.feature import hog
        
        # Define the settings for our HOG descriptor
        settings = {'orientations'      :   12,
                    'pixels_per_cell'   :   (8, 8),
                    'cells_per_block'   :   (1, 1), 
                    'block_norm'        :   'L2-Hys', 
                    'visualize'         :   False, 
                    'transform_sqrt'    :   True, # Global normalisation
                    'feature_vector'    :   True,
                    'multichannel'      :   False }
        
        # This could perhaps be parallelised using the joblib library 
        # Will implement if I have time
        Xtrain_hog_d    = np.array([ hog(img , **settings ) for img in Xtrain ])  
        Xval_hog_d      = np.array([ hog(img , **settings ) for img in Xval   ])  
        Xtest_hog_d     = np.array([ hog(img , **settings ) for img in Xtest  ])  
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_hog_d
        Xval_pre    =   Xval_hog_d
        Xtest_pre   =   Xtest_hog_d
        
        
    #%% Perform histogram equalization...
    
    """
    P R E - P R O C E S S :   H I S T O G R A M   E Q U A L I S A T I O N
    """
    
    # Most pre-processing methods benefit from image equalisation to reduce
    # the influence of illumination effects on the result.
    # This method proved very helpful for clean edge detection using the Sobel operator. 
    # The HOG feature descriptor allows for global normalisation equalisation 
    
    # We use CLAHE with Sobel (2), SURF (3) and when no feature extr. (4) was chosen
    if selection == 2 or selection == 3 or selection == 4: 
        
        # To perform Contrast Limited Adaptive Histogram Equalisation (CLAHE)
        # we import a scikit-image package
        from skimage.exposure import equalize_adapthist
        
        print("Performing CLAHE, Contrast Limited Adaptive Histogram Equalisation, this may take a while...")
        
        # We equalize all the image data using an adaptive contrast equaliser
        Xtrain_eq   =  equalize_adapthist(Xtrain)
        Xval_eq     =  equalize_adapthist(Xval)
        Xtest_eq    =  equalize_adapthist(Xtest)
        
    #%% Edge detection with Sobel-Feldman operator...
    
    """
    P R E - P R O C E S S :   S O B E L
    """
    # We now proceed by processing the images with a Sobel filter
    # The Sobel-Feldman operator provides powerful means of edge detection
    # especially, on equalised images
    
    if selection == 2: 
        
        Xtrain_sob  =  prp.sobel(Xtrain_eq , surpress=True)
        Xval_sob    =  prp.sobel(Xval_eq   , surpress=True)
        Xtest_sob   =  prp.sobel(Xtest_eq  , surpress=True)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_sob
        Xval_pre    =   Xval_sob
        Xtest_pre   =   Xtest_sob
        
    
    #%% Feature description with SURF...
    
    """
    P R E - P R O C E S S :   S U R F 
    """
    
    if selection == 3: 
        
        # The OpenCV SURF feature detector requires single channel 0-255 UINT8 images
        # Scale images from 0...1 floating point to 0...255 unsigned 8-bit integer
        Xtrain_imgs = np.array([( img * 255).astype(np.uint8) for img in Xtrain_eq ]) 
        Xval_imgs   = np.array([( img * 255).astype(np.uint8) for img in Xval_eq   ]) 
        Xtest_imgs  = np.array([( img * 255).astype(np.uint8) for img in Xtest_eq  ]) 
        
        Xtrain_surf = prp.surf(Xtrain_imgs, n_keypoints = 20, hessianThreshold = 400, 
                          upright = False, surpress = True)
        
        Xval_surf = prp.surf(Xval_imgs, n_keypoints = 20, hessianThreshold = 400, 
                  upright = False, surpress = True)
        
        Xtest_surf = prp.surf(Xtest_imgs, n_keypoints = 20, hessianThreshold = 400, 
                  upright = False, surpress = True)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_surf
        Xval_pre    =   Xval_surf
        Xtest_pre   =   Xtest_surf
        

    #%% No further pre-processing, return equalised images...
    
    """
    S K I P   P R E - P R O C E S S I N G
    """
    
    if selection == 4: 
        
        # Passing solely equalised object to placeholder for classification task
        Xtrain_pre  =   Xtrain_eq
        Xval_pre    =   Xval_eq
        Xtest_pre   =   Xtest_eq

    # %% Flatten image data to 1D feature vector...
    
    """
    P R E - P R O C E S S :   F L A T T E N   T O  1 D - F E A T U R E   V E C T O R
    """
    
    # HOG does not require flattening of an image array as it only carries a 1D descriptor vector
    if selection != 1: 
        Xtrain_flat =  prp.flatten(Xtrain_pre)
        Xval_flat   =  prp.flatten(Xval_pre)
        Xtest_flat  =  prp.flatten(Xtest_pre)
    else: 
        Xtrain_flat =  Xtrain_pre
        Xval_flat   =  Xval_pre
        Xtest_flat  =  Xtest_pre
    
    # %% Scale data...
    
    """
    P R E - P R O C E S S :   S C A L E   D A T A
    """
    
    # Import a standard scaler
    from sklearn.preprocessing import StandardScaler
    
    print("Performing feature standardisation, scaling to zero mean with unit variance...")
    
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
    
    print("Preparing labels...")
    
    # Transform the 'gender' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A1 =  ds.to_binary(ytrain,'gender')
    yval_A1   =  ds.to_binary(yval,'gender')
    ytest_A1  =  ds.to_binary(ytest,'gender')
    
    # Transform the 'smiling' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A2 =  ds.to_binary(ytrain,'smiling')
    yval_A2   =  ds.to_binary(yval,'smiling')
    ytest_A2  =  ds.to_binary(ytest,'smiling')
    
    
    # %% Filenames for pickled Cross-Validation records...
    """
    C V   R E C O R D   M A N A G E M E N T
    """
    
    # Filenames for Task A1 - Gender classification
    cv_filename_A1 = {'1': 'CV_HOG_A1.pkl',
                      '2': 'CV_Sobel_A1.pkl',
                      '3': 'CV_SURF_A1.pkl',
                      '4': 'CV_none_A1.pkl'}
    
    # Filenames for Task A2 - Smile classification
    cv_filename_A2 = {'1': 'CV_HOG_A2.pkl',
                      '2': 'CV_Sobel_A2.pkl',
                      '3': 'CV_SURF_A2.pkl',
                      '4': 'CV_none_A2.pkl'}
    
    # The filenames to be indexed depend on user's selection of pre-processing method
    selection_filenames = [cv_filename_A1[str(selection)],
                           cv_filename_A2[str(selection)]]

    # %% Perform Cross-Validation...
    """
    T A S K   A 1 :   B U I L D   P I P E L I N E :   P C A + S V M
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
    
    # So we don't have to perform the CV numerous times, i.e. can save results
    import joblib
    
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile(  selection_filenames[0]  ):
            
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use a linear SVC
        svm = SVC()
        
        # Combine PCA and SVC to a pipeline
        pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
        
        # If user chose HOG, Sobel or no pre-processing
        if selection != 3:
            # Number of Principal components to try
            n_components = [300,500,1000]
        else: 
            # SURF carries so few keypoints that we must narrow the scope
            n_components = [10,20]
        
        # Parameters to try for the SVM and PCA
        params_grid = {
        'svm__C': [100,10,1],
        'svm__kernel': ['rbf'],
        'svm__degree': [2],
        'svm__gamma': ['scale','auto'],
        'pca__n_components': n_components,
        }
        
        # Initialise the gridsearch cross validation
        estimator = GridSearchCV(pipe,                      # Use the PCA + SVM pipeline
                                 params_grid,               # For these SVM hyperparams (and components)
                                 scoring='accuracy',        # Score on validation accuracy
                                 n_jobs = 4,                # Use 2 processors
                                 return_train_score=True,   # Record the scores for later use
                                 verbose = 10)              # Print the process as it runs
        
        # Fit the model on the training data
        estimator.fit(Xtrain_scl, ytrain_A1)
        
        # Output the best parameters and the score they achieved
        print(estimator.best_params_, estimator.best_score_)
        
        # Pickle the results for later consumption
        joblib.dump(estimator, selection_filenames[0])
    
    # If CV record exists
    else: 
        # We load it
        estimator = joblib.load(selection_filenames[0])
    
    # Store the best parameters for the SVM...
    best_svm = {'kernel': estimator.best_params_['svm__kernel'],
                'C'     : estimator.best_params_['svm__C'],
                'gamma' : estimator.best_params_['svm__gamma']}
    
    # ... and number of components for PCA
    best_n_components = estimator.best_params_['pca__n_components']
    
    # Then redefine our classifiers, as their "ideal" version for the task
    svm = SVC(**best_svm)
    pca = PCA(n_components=best_n_components)
    
    # %% Perform Cross-Validation...
    """
    T A S K   A 2 :   B U I L D   P I P E L I N E :   P C A + S V M
    """
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile(  selection_filenames[1]  ):
            
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use a linear SVC
        svm = SVC()
        
        # Combine PCA and SVC to a pipeline
        pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
        
        # If user chose HOG, Sobel or no pre-processing
        if selection != 3:
            # Number of Principal components to try
            n_components = [300,500,1000]
        else: 
            # SURF carries so few keypoints that we must narrow the scope
            n_components = [10,20]
        
        # Parameters to try for the SVM and PCA
        params_grid = {
        'svm__C': [100,10,1],
        'svm__kernel': ['rbf'],
        'svm__degree' : [2],
        'svm__gamma': ['scale','auto'],
        'pca__n_components': n_components,
        }
        
        # Initialise the gridsearch cross validation
        estimator = GridSearchCV(pipe,                      # Use the PCA + SVM pipeline
                                 params_grid,               # For these SVM hyperparams (and components)
                                 scoring='accuracy',        # Score on validation accuracy
                                 n_jobs = 1,                # Use 2 processors
                                 return_train_score=True,   # Record the scores for later use
                                 verbose = 10)              # Print the process as it runs
        
        # Fit the model on the training data
        estimator.fit(Xtrain_scl, ytrain_A1)
        
        # Output the best parameters and the score they achieved
        print(estimator.best_params_, estimator.best_score_)
        
        # Pickle the results for later consumption
        joblib.dump(estimator, selection_filenames[1])
    
    # If CV record exists
    else: 
        # We load it
        estimator = joblib.load(selection_filenames[1])
    
    # Store the best parameters for the SVM...
    best_svm = {'kernel': estimator.best_params_['svm__kernel'],
                'C'     : estimator.best_params_['svm__C'],
                'gamma' : estimator.best_params_['svm__gamma']}
    
    # ... and number of components for PCA
    best_n_components = estimator.best_params_['pca__n_components']
    
    # Then redefine our classifiers, as their "ideal" version for the task
    svm = SVC(**best_svm)
    pca = PCA(n_components=best_n_components)
    
    # %%
    print("Finished section A")
    print("------------------")
    # Output the best parameters and the score they achieved
    print(estimator.best_params_, estimator.best_score_)
