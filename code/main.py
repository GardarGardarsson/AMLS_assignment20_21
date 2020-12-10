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
import performance_analysis as pa

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
    img_path = "/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/Assignment/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/Assignment/dataset_AMLS_20-21/celeba/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    
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
    
    
    # %% Split data to training and test folds...
    """
    S P L I T   D A T A
    """
    
    # Split dataset into train-, validation- and test folds
    Xtrain,Xtest,ytrain,ytest = sd.split_dataset(X_gry,y,test_size=0.2,surpress=False)
        
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
        Xtest_hog_d     = np.array([ hog(img , **settings ) for img in Xtest  ])  
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_hog_d
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
        Xtest_eq    =  equalize_adapthist(Xtest)
        
    #%% Edge detection with Sobel-Feldman operator...
    
    """
    P R E - P R O C E S S :   S O B E L
    """
    # We now proceed by processing the images with a Sobel filter
    # The Sobel-Feldman operator provides powerful means of edge detection
    # especially, on equalised images
    
    if selection == 2: 
        
        Xtrain_sob  =  prp.sobel(Xtrain_eq , surpress=False)
        Xtest_sob   =  prp.sobel(Xtest_eq  , surpress=True)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_sob
        Xtest_pre   =   Xtest_sob
        
    
    #%% Feature description with SURF...
    
    """
    P R E - P R O C E S S :   S U R F 
    """
    
    if selection == 3: 
        
        # The OpenCV SURF feature detector requires single channel 0-255 UINT8 images
        # Scale images from 0...1 floating point to 0...255 unsigned 8-bit integer
        Xtrain_imgs = np.array([( img * 255).astype(np.uint8) for img in Xtrain_eq ]) 
        Xtest_imgs  = np.array([( img * 255).astype(np.uint8) for img in Xtest_eq  ]) 
        
        Xtrain_surf = prp.surf(Xtrain_imgs, n_keypoints = 20, hessianThreshold = 400, 
                          upright = False, surpress = False)
        
        Xtest_surf = prp.surf(Xtest_imgs, n_keypoints = 20, hessianThreshold = 400, 
                  upright = False, surpress = True)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_surf
        Xtest_pre   =   Xtest_surf
        

    #%% No further pre-processing, return equalised images...
    
    """
    S K I P   P R E - P R O C E S S I N G
    """
    
    if selection == 4: 
        
        # Passing solely equalised object to placeholder for classification task
        Xtrain_pre  =   Xtrain_eq
        Xtest_pre   =   Xtest_eq

    # %% Flatten image data to 1D feature vector...
    
    """
    P R E - P R O C E S S :   F L A T T E N   T O  1 D - F E A T U R E   V E C T O R
    """
    
    # HOG does not require flattening of an image array as it only carries a 1D descriptor vector
    if selection != 1: 
        Xtrain_flat =  prp.flatten(Xtrain_pre)
        Xtest_flat  =  prp.flatten(Xtest_pre)
    else: 
        Xtrain_flat =  Xtrain_pre
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
    Xtest_scl   =   scaler.transform(Xtest_flat) 
    
    
    # %% Prepare label data...
    
    """
    P R E P A R E   L A B E L   D A T A
    """
    
    print("Preparing labels...")
    
    # Transform the 'gender' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A1 =  ds.get_category(ytrain,'gender')
    ytest_A1  =  ds.get_category(ytest,'gender')
    
    # Transform the 'smiling' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_A2 =  ds.get_category(ytrain,'smiling')
    ytest_A2  =  ds.get_category(ytest,'smiling')
    
    
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

    # %% Task A1 - Perform Cross-Validation...
    """
    T A S K   A 1 :   B U I L D   P I P E L I N E :   P C A + S V M
    """
    # Let us now import an unsupervised classifier for the
    # dimensionality reduction of the data, namely that of PCA
    from sklearn.decomposition import PCA
    
    # We will also use the SVM classifier of the scikit-learn library
    from sklearn.svm import LinearSVC
    
    # To optimise the number of components, and parameters of our SVM,
    # we shall construct a pipeline, and perform a gridsearch to find
    # both the optimal number of principal components and hyperparameters for the SVM.
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    
    # So we don't have to perform the CV numerous times, i.e. can save results
    import joblib
    
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile(  selection_filenames[0]  ):
        print("CV Records not found in directory, performing CV")
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use a linear SVC
        svm = LinearSVC()
        
        # Combine PCA and SVC to a pipeline
        pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
        
        # If user chose HOG, Sobel or no pre-processing
        if selection != 3:
            # Number of Principal components to try
            n_components = [400,5808]
        else: 
            # SURF carries so few keypoints that we must narrow the scope
            n_components = [10,20]
        
        # Parameters to try for the SVM and PCA
        params_grid = {
        'svm__C': [5e-4],
        'svm__penalty':['l1','l2'],
        'svm__loss':['hinge','squared_hinge'],
        'svm__dual':[False],
        'pca__n_components': n_components
        }
        
        # Initialise the gridsearch cross validation
        estimator = GridSearchCV(pipe,                      # Use the PCA + SVM pipeline
                                 params_grid,               # For these SVM hyperparams (and components)
                                 scoring='accuracy',        # Score on validation accuracy
                                 n_jobs = -1,                # Use 4 processors
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
        print("CV Records found, loading from directory")
        # We load it
        estimator = joblib.load(selection_filenames[0])
    
    # Store the best parameters for the SVM...
    best_svm = {'C'       : estimator.best_params_['svm__C'],
                'penalty' : estimator.best_params_['svm__penalty'],
                'loss'    : estimator.best_params_['svm__loss'],
                'dual'    : estimator.best_params_['svm__dual']
                }
    
    # ... and number of components for PCA
    best_n_components = estimator.best_params_['pca__n_components']
    
    # Then redefine our classifiers, as their "ideal" version for the task
    svm = LinearSVC(**best_svm)
    pca = PCA(n_components=best_n_components)
    # %% Task A1 - Plot learning curves...
    """
    T A S K   A 1 :   P L O T   L E A R N I N G   C U R V E S
    """    
    
    from sklearn.model_selection import ShuffleSplit
    
    # Fit the PCA classifier to the scaled training data
    pca.fit(Xtrain_scl)
    
    # Transform the training-, validation- and test folds to 'n' principal components
    Xtrain_pc   =   pca.transform(Xtrain_scl)
    Xtest_pc    =   pca.transform(Xtest_scl)
    
    # Fit SVM
    svm.fit(Xtrain_pc, ytrain_A1)
    
    prep_method  =  {'1': 'HOG & PCA',
                     '2': 'Sobel Operator & PCA',
                     '3': 'SURF & PCA',
                     '4': 'CLAHE & PCA'}
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    title = "Task A1 - {} w/ {} components\nSVM, linear kernel, $C={}$\nLearning Curves".format(prep_method[str(selection)], pca.n_components,svm.C)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    pa.plot_learning_curve(svm, title, Xtrain_pc, ytrain_A1, axes=axes, ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)
    plt.tight_layout()
    plt.show()
    # %% Task A1 - Predict & report...
    """
    T A S K   A 1 :   P R E D I C T   &   R E P O R T
    """    
    
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    
    # Predict on the training and unseen data
    yp_tr_A1 = svm.predict(Xtrain_pc)
    yp_te_A1 = svm.predict(Xtest_pc)
    
    # Store accuracy scores
    acc_A1_train =  accuracy_score(y_true = ytrain_A1, y_pred = yp_tr_A1)
    acc_A1_test =   accuracy_score(y_true = ytest_A1 , y_pred = yp_te_A1)
    
    # Print results:
    print("Task A1:\n"+"-"*12+"\nTrain: {:.1f}% \nTest:  {:.1f}%".format(acc_A1_train*100,acc_A1_test*100))
    
    # Print classification report
    print("\n"*2+"Classification Report:\n"+"-"*54+"\n",classification_report(ytest_A1, yp_te_A1))
    
    # Plot confusion matrix
    disp = plot_confusion_matrix(svm,Xtest_pc,ytest_A1,normalize='true', display_labels=['Female','Male'],cmap='Blues')
    disp.figure_.set_dpi(400)
    disp.ax_.set_title("Task A1 - {} w/ {} components\nSVM, linear kernel, $C={}$\nConfusion matrix".format(prep_method[str(selection)], pca.n_components, svm.C))
    plt.tight_layout()
    plt.show()
    
    # %% Check out misclassified instances...
    
    # We may want to inspect if there is any commonality in the misclassified examples
    
    misclassified = np.where(ytest_A1 != yp_te_A1)    
    miscl_labels = ytest.iloc[misclassified[0]]
    miscl_imgs = X[miscl_labels.index]
    
    ds.plot_celeba(miscl_imgs,miscl_labels,5,6)
    
    # %% Task A2 - Perform Cross-Validation...
    """
    T A S K   A 2 :   B U I L D   P I P E L I N E :   P C A + S V M
    """
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile(  selection_filenames[1]  ):
        print("CV Records not found in directory, performing CV")
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use a linear SVC
        svm = SVC()
        
        # Combine PCA and SVC to a pipeline
        pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])
        
        # If user chose HOG, Sobel or no pre-processing
        if selection != 3:
            # Number of Principal components to try
            n_components = [100,500,1000]
        else: 
            # SURF carries so few keypoints that we must narrow the scope
            n_components = [10,20]
        
        # Parameters to try for the SVM and PCA
        params_grid = {
        'svm__C': [0.1 , 0.01],
        'svm__kernel': ['linear','rbf','poly'],
        'svm__degree' : [2],
        'svm__gamma': [0.1 , 0.01],
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
        print("CV Records found, loading from directory")
        # We load it
        estimator = joblib.load(selection_filenames[1])
        
        # Output the best parameters and the score they achieved
        print(estimator.best_params_, estimator.best_score_)
    
    # Store the best parameters for the SVM...
    best_svm = {'kernel': estimator.best_params_['svm__kernel'],
                'C'     : estimator.best_params_['svm__C'],
                'gamma' : estimator.best_params_['svm__gamma']}
    
    # ... and number of components for PCA
    best_n_components = estimator.best_params_['pca__n_components']
    
    # Then redefine our classifiers, as their "ideal" version for the task
    svm = SVC(**best_svm)
    pca = PCA(n_components=best_n_components)
    
    # %% Task A2 - Plot learning curves...
    """
    T A S K   A 2 :   P L O T   L E A R N I N G   C U R V E S
    """    
    
    from sklearn.model_selection import ShuffleSplit
    
    # Fit the PCA classifier to the scaled training data
    pca.fit(Xtrain_scl)
    
    # Transform the training-, validation- and test folds to 'n' principal components
    Xtrain_pc   =   pca.transform(Xtrain_scl)
    Xtest_pc    =   pca.transform(Xtest_scl)
    
    # Fit SVM
    svm.fit(Xtrain_pc, ytrain_A2)
    
    prep_method  =  {'1': 'HOG & PCA',
                     '2': 'Sobel Operator & PCA',
                     '3': 'SURF & PCA',
                     '4': 'CLAHE & PCA'}
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    title = "Task A2 - {} w/ {} components\nSVM, {}-kernel,$\gamma = {}$, $C={}$\nLearning Curves".format(prep_method[str(selection)], pca.n_components,svm.kernel, svm.gamma,svm.C)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    pa.plot_learning_curve(svm, title, Xtrain_pc, ytrain_A2, axes=axes, ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)
    plt.show()
    
    # %% Task A2 - Predict & report...
    """
    T A S K   A 2 :   P R E D I C T   &   R E P O R T
    """    
    
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import accuracy_score
    
    # Predict on the training and unseen data
    yp_tr_A2 = svm.predict(Xtrain_pc)
    yp_te_A2 = svm.predict(Xtest_pc)
    
    # Store accuracy scores
    acc_A2_train =  accuracy_score(y_true = ytrain_A2, y_pred = yp_tr_A2)
    acc_A2_test =   accuracy_score(y_true = ytest_A2 , y_pred = yp_te_A2)
    
    # Print results:
    print("Task A2:\n"+"-"*12+"\nTrain: {:.1f}% \nTest:  {:.1f}%".format(acc_A2_train*100,acc_A2_test*100))
    disp = plot_confusion_matrix(svm,Xtest_pc,ytest_A2,normalize='true', display_labels=['No smile','Smile'],cmap='Blues')
    disp.figure_.set_dpi(400.0)
    disp.ax_.set_title("Task A2 - {} w/ {} components\nSVM, {}-kernel,$\gamma = {}$, $C={}$\nConfusion matrix".format(prep_method[str(selection)], pca.n_components,svm.kernel, svm.gamma,svm.C))
    plt.show()
    
    # %%
    
    """
    T A S K   B :   C A R T O O N   D A T A S E T
    """

    # Delete all variables
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
            
    # Import garbage collection library 
    import gc
    
    # Explicitly free memory
    gc.collect()
    
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
    
    # %% Load data...
    """
    L O A D   D A T A
    """
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/Assignment/dataset_AMLS_20-21/cartoon_set/img/"
    label_path = "/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/Assignment/dataset_AMLS_20-21/cartoon_set/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,
                                       label_path,
                                       img_type='.png',
                                       task='B',
                                       surpress=False,
                                       return_img_indices=True)
    # %% Split data...
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xval,Xtest,ytrain,yval,ytest = sd.split_dataset(X,y,test_size=0.2,val_size=0.2,surpress=False)
    
    
    # %% Crop images...
    """
    P R E - P R O C E S S :   C R O P   R E G I O N S   O F   I N T E R E S T
    """
    
    def crop(img_arr,H,W,ver_off,hor_off):
        # Initialise an empty array that can accomodate as many instances
        # as the number of images we wish to crop, in the required final dimensions
        cropped = np.empty((img_arr.shape[0],
                            H, # Final height in px
                            W, # Final width in px
                            img_arr.shape[3]),int)
        
        # For each of the images provided
        for i,img in enumerate(img_arr):
            # We store a cropped version, that crops a H x W px rectangle
            # from a supplied region with horizontal and vertical offsets
            cropped[i] = img[ver_off:ver_off + H,hor_off:hor_off + W]
            
        return cropped
    
    # We crop the training, validation and test data to the face region for task B1
    Xtrain_C_B1 = crop(Xtrain, H = 250, W = 220, ver_off = 150, hor_off = 140)
    Xval_C_B1 = crop(Xval, H = 250, W = 220, ver_off = 150, hor_off = 140)
    Xtest_C_B1 = crop(Xtest, H = 250, W = 220, ver_off = 150, hor_off = 140)
    
    # We crop the training, validation and test data to the eye region for task B2
    Xtrain_C_B2 = crop(Xtrain, H = 35, W = 55, ver_off = 245, hor_off = 180)
    Xval_C_B2 = crop(Xval, H = 35, W = 55, ver_off = 245, hor_off = 180)
    Xtest_C_B2 = crop(Xtest, H = 35, W = 55, ver_off = 245, hor_off = 180)
    
    # %% Grayscale images...
    """
    P R E - P R O C E S S :   G R A Y S C A L E 
    """
    
    # Grayscale face images for task B1 as shape classification is colour 
    # independent
    Xtrain_gry_B1 = prp.reduceRGB(Xtrain_C_B1)
    Xval_gry_B1 = prp.reduceRGB(Xval_C_B1)
    Xtest_gry_B1 = prp.reduceRGB(Xtest_C_B1)
  
    # Grayscale face images for task B2, for further processing to detect
    # shaded eyeglasses
    Xtrain_gry_B2 = prp.reduceRGB(Xtrain_C_B2)
    Xval_gry_B2 = prp.reduceRGB(Xval_C_B2)
    Xtest_gry_B2 = prp.reduceRGB(Xtest_C_B2)
    
        
    # %% Prepare label data...
    """
    P R E P A R E   L A B E L   D A T A
    """
    
    # Transform the 'gender' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_B1 =  pd.Series.to_numpy(ytrain.loc[:,'face_shape'].copy())
    yval_B1   =  pd.Series.to_numpy(yval.loc[:,'face_shape'].copy())
    ytest_B1  =  pd.Series.to_numpy(ytest.loc[:,'face_shape'].copy())
    
    # Transform the 'smiling' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_B2 =  pd.Series.to_numpy(ytrain.loc[:,'eye_color'].copy())
    yval_B2   =  pd.Series.to_numpy(yval.loc[:,'eye_color'].copy())
    ytest_B2  =  pd.Series.to_numpy(ytest.loc[:,'eye_color'].copy())
    
    # %%
    
    """
    T A S K   B 1 :   F A C E   S H A P E   C L A S S I F I C A T I O N
    """
    
    # %% Flatten data for task B1...
    
    # Flattens a library of grayscale images to N-samples of 1 dimension
    def flatten(img_arr):
        
        dim = img_arr.ndim - 1 # 1st dimension is N-number of samples 
        flat = 1 # Initalise the final dimension to be folded to
        
        # Flatten all but the 1st dimension
        for i in range(dim):
            flat *= img_arr.shape[i+1]
        
        # Reshape image array from N,h,w,channels, to N,(h * w)
        flat_arr = np.reshape(img_arr,(img_arr.shape[0],flat))
        
        # Return the library of flattened objects
        return flat_arr
    
    # Flatten the training, validation and test data to N-samples X 1D vector
    Xtrain_B1 = flatten(Xtrain_gry_B1)
    Xval_B1 = flatten(Xval_gry_B1)
    Xtest_B1 = flatten(Xtest_gry_B1)
    
    # %% Task B1 with Logistic Regression
    """
    T A S K   B 1 :   F A C E   S H A P E   W/   L O G I S T I C   R E G R E S S I O N
    """
    from sklearn.linear_model import LogisticRegression
    
    # sklearn functions implementation
    def logRegrPredict(Xtrain,ytrain,Xtest, solver='lbfgs'):
        # Build Logistic Regression Model
        logreg = LogisticRegression(solver=solver,
                                    C=5.0,
                                    penalty='l2', # <--- Uses Ridge Regression as cost-function
                                    multi_class='multinomial',
                                    max_iter = 100) 
        
        # Train the model using the training sets
        logreg.fit(Xtrain, ytrain)
        
        ypred= logreg.predict(Xtest)
        
        return ypred

    ypred_B1 = logRegrPredict(Xtrain_B1, ytrain_B1, Xtest_B1)
    print(accuracy_score(ytest_B1,ypred_B1)) 

    # %%
    
    """
    T A S K   B 2 :   E Y E   C O L O R   C L A S S I F I C A T I O N
    """

    # %% Detect dark eyeglasses...
    
    """
    P R E - P R O C E S S :   D E T E C T   D A R K   E Y E G L A S S E S
    """
    
    # Import the Canny Edge Filter from the scikit-image library
    from skimage.feature import canny
    
    # Define a function for dark glasses detection using Canny filter on grayscale
    # eye-region imagery
    def detect_dark_glasses(img_arr, surpress=False):
        
        """
        A function that applies a Canny filter to a grayscale eye region image.
        Edges are stored as binary values in a 35 x 55 (H x W in px) array
        By summing the values of the array, a feature descriptor is realised.
        A low scoring descriptor denotes a feature poor eye region which 
        indicates the eye features are hidden.
        
        The Canny filter, tuned with a low sigma parameter, performs well even with 
        semi-shaded glasses, as the half-transparent alpha channel gets interpreted 
        as noise. This noise is translated to more edges in the feature descriptor 
        and hence still provides an excellent classifier for the discrimination 
        of dark shades.
        """
        
        # Initialise a container for the Canny edge images
        canny_img = np.empty(img_arr.shape)
        
        # Initialise a container for the feature description values
        canny_values = np.empty(len(img_arr))
        
        # For each of the supplied images
        for i,img in enumerate(img_arr):
            
            # Apply a Canny filter
            edges = canny(img)
            
            # Calculate a feature description value:
            value = edges.astype(int).sum()
            
            # Store the Canny image
            canny_img[i] = edges
            
            # Store the feature description value: 
            canny_values[i] = value
        
        # We may choose to turn of plotting
        if not surpress: 
            
            # Find a random amount of images
            rand_img = np.random.randint(0,len(img_arr),size=(3,4))
            row, col = rand_img.shape
            
            fig,ax = plt.subplots(nrows=row,ncols=col)
            for i in range(row):
                for j in range(col):
                    ax[i][j].imshow(canny_img[rand_img[i][j]],cmap=plt.get_cmap("gray"))
                    ax[i][j].axis("off")
                    ax[i][j].set_title("{}".format(canny_values[rand_img[i][j]]))
            
            plt.suptitle("Eye region edges with pixelwise values")
            plt.tight_layout()
            plt.show()
            
        return canny_img, canny_values
    
    # Retrieve eye region imagery and feature values
    Xtrain_CE_img , Xtrain_CE_val = detect_dark_glasses(Xtrain_gry_B2)
    Xval_CE_img , Xval_CE_val = detect_dark_glasses(Xval_gry_B2, surpress = True)
    Xtest_CE_img , Xtest_CE_val = detect_dark_glasses(Xtest_gry_B2, surpress = True)
    
    # %%
    
    """
    V I S U A L I S E   T H E   B O U N D A R Y
    """
    
    # To get a better view of the apparent split in the data, we can
    # plot the Canny feature descriptor values. The boundary is placed
    # between the lowest scoring eye (198) and highest ranking glasses (175)
    
    plt.figure(figsize=(6,2),dpi=400)
    
    shades = Xtrain_CE_val[Xtrain_CE_val < 185]
    no_shades = Xtrain_CE_val[Xtrain_CE_val > 185]
    
    plt.scatter(no_shades , np.zeros_like(no_shades),color="green")
    plt.scatter(shades , np.zeros_like(shades))
    plt.axvline(187,color='orange',linestyle = '--')
    plt.scatter(187,0,marker='x',color='red', label = "Sunglasses boundary")
    
    plt.suptitle("Canny eye feature values")
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
    # %% Grabbing the indices where glasses were detected in dataset...
    """
    I N D E X   D A R K   G L A S S E S
    """
    Xtrain_glasses = np.where(Xtrain_CE_val < 187)[0]
    Xtrain_no_glasses = np.where(Xtrain_CE_val >= 187)[0]
    
    Xval_glasses = np.where(Xval_CE_val < 187)[0]
    Xval_no_glasses = np.where(Xval_CE_val >= 187)[0]
    
    Xtest_glasses = np.where(Xtest_CE_val < 187)[0]
    Xtest_no_glasses = np.where(Xtest_CE_val >= 187)[0]

    # %% Image prep method 1...
    
    # Only flatten the training, validation and test data to N-samples X 1D vector
    Xtrain_B2 = flatten(Xtrain_C_B2)
    Xval_B2 = flatten(Xval_C_B2)
    Xtest_B2 = flatten(Xtest_C_B2)

    # %% Image prep method 2...
    """
    P R E P A R E   I M A G E S   E X C L U D I N G   D A R K   G L A S S E S 
    """
   
    # Define a function capable of filtering out label and image data given 
    # a set of indices
    def only_keep(where_no_glasses, img_arr_to_filter, label_arr_to_filter):
        
        # Create container for image data to keep, will be an array of 
        # N samples, where N = number of indices where no glasses were detected
        filtered_img_arr = np.empty((where_no_glasses.shape[0],
                                     img_arr_to_filter.shape[1],
                                     img_arr_to_filter.shape[2],
                                     img_arr_to_filter.shape[3]))
        # We must remove the corresponding label data as well, create container
        filtered_label_arr = np.empty(where_no_glasses.shape[0])
        
        # For each of the image indices passed to us
        for i,index in enumerate(where_no_glasses):
            filtered_img_arr[i] = img_arr_to_filter[index]
            filtered_label_arr[i] = label_arr_to_filter[index]
            
        return filtered_img_arr, filtered_label_arr
    
    
    Xtrain_B2,ytrain_B2 = only_keep(Xtrain_no_glasses, Xtrain_C_B2, ytrain_B2)
    Xval_B2,yval_B2 = only_keep(Xval_no_glasses, Xval_C_B2, yval_B2)
    Xtest_B2,ytest_B2 = only_keep(Xtest_no_glasses, Xtest_C_B2, ytest_B2)
    

    
    # %% Flatten data for task B2...
        
    # Flatten the training, validation and test data to N-samples X 1D vector
    Xtrain_B2 = flatten(Xtrain_B2)
    Xval_B2 = flatten(Xval_B2)
    Xtest_B2 = flatten(Xtest_B2)
   
    # %% Task B2 with Logistic Regression
    """
    T A S K   B 2 :   E Y E   C O L O R   W/   L O G I S T I C   R E G R E S S I O N
    """
    
    # sklearn functions implementation
    def logRegrPredict(Xtrain,ytrain,Xtest, solver='lbfgs'):
        # Build Logistic Regression Model
        logreg = LogisticRegression(solver=solver,
                                    C=5.0,
                                    penalty='l2',
                                    multi_class='multinomial',
                                    max_iter = 100) 
        
        # Train the model using the training sets
        logreg.fit(Xtrain, ytrain)
        
        ypred= logreg.predict(Xtest)
        
        return ypred

    ypred_B2 = logRegrPredict(Xtrain_B2, ytrain_B2, Xtest_B2)
    print(accuracy_score(ytest_B2,ypred_B2))
