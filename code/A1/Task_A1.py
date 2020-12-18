# %% Import libraries...
"""
L I B R A R Y   I M P O R T
"""

# To modularise the program, we need to add the current directory to sys path
import os
import sys

# Browse up one directory...
os.chdir(os.path.dirname(os.getcwd()))
currentPath = os.getcwd()

# ... and make that the system path
sys.path.append(currentPath)

# Now we can import our own modules into our script.
import Modules.import_data as ds
import Modules.split_dataset as sd
import Modules.pre_processing as prp
import Modules.user_interface as ui
import Modules.performance_analysis as pa

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt

# Pandas dataframes for enhanced data storage
import pandas as pd

# To perform Contrast Limited Adaptive Histogram Equalisation (CLAHE)
# we import a scikit-image package
from skimage.exposure import equalize_adapthist
# Import the Histogram of Oriented Gradients feature descriptor
from skimage.feature import hog
# Import a standard scaler
from sklearn.preprocessing import StandardScaler
# Let us now import an unsupervised classifier for the
# dimensionality reduction of the data, namely that of PCA
from sklearn.decomposition import PCA
# We will also use the SVM classifier of the scikit-learn library
from sklearn.svm import SVC        # Different kernels were tested but linear one's yielded suprisingly high accuracy scores
from sklearn.svm import LinearSVC  # For this sake, CV was performed again with LinearSVC, to be able to try 
                                   # out different regularisation methods (L1 or L2 penalty), and loss functions 
                                   # (hinge or squared-hinge)
                                   
# To optimise the number of components, and parameters of our SVM,
# we shall construct a pipeline, and perform a gridsearch to find
# both the optimal number of principal components and hyperparameters for the SVM.
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit

# So we don't have to perform the CV numerous times, i.e. can save results, we import joblib
import joblib
# Scikit-Learn metrics
from sklearn.metrics import plot_confusion_matrix,accuracy_score,classification_report

# %% Load data...

if __name__ == '__main__':
    
    """
    L O A D   D A T A
    """
    # Define a path to the data
    img_path = "/Datasets/celeba/img/"
    label_path = "/Datasets/celeba/"
    img_ts_path = "/Datasets/celeba_test/img/"
    label_ts_path = "/Datasets/celeba_test/"
    
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    # Test set
    Xts , yts  = ds.dataImport(img_ts_path,label_ts_path,surpress=True,return_img_indices=False)
    
    #%% Crop images...
    
    """
    P R E - P R O C E S S :   C R O P
    """
    X_C =  prp.crop(X, H=176, W=176, ver_off=21,hor_off=0)
    
    # Test set
    X_C_ts =  prp.crop(Xts, H=176, W=176, ver_off=21,hor_off=0)
    
    #%% Grayscale images...
    
    """
    P R E - P R O C E S S :   G R A Y S C A L E
    """
    
    # Reduce the RGB channels to a singular gray channel using Rec. 601 encoding
    X_gry =  prp.reduceRGB(X_C).astype(np.uint16)
    
    # Test set 
    X_gry_ts = prp.reduceRGB(X_C_ts).astype(np.uint16)
    
    #%% Perform histogram equalization...
    
    """
    P R E - P R O C E S S :   H I S T O G R A M   E Q U A L I S A T I O N
    """
    
    # Most pre-processing methods benefit from image equalisation to reduce
    # the influence of illumination effects on the result.
    # This method proved very helpful for clean edge detection using the Sobel operator. 
    # The HOG feature descriptor allows for global normalisation equalisation in it's function call
    # CLAHE however seemed to yield less noise around edges and was hence chosen in it's favour
    
    print("Performing CLAHE, Contrast Limited Adaptive Histogram Equalisation, this may take a while...")
    
    # We equalize all the image data using an adaptive contrast equaliser
    X_eq   =  equalize_adapthist(X_gry)    
    
    # Test set
    X_eq_ts   =  equalize_adapthist(X_gry_ts)
    
    # %% Split data to training and test folds...
    """
    S P L I T   D A T A
    """
    
    # We split the data here, as we intend to augment only the training image space
    
    # Split dataset into train- and test folds
    Xtrain,Xtest,ytrain,ytest = sd.split_dataset(X_eq,y,test_size=0.2,surpress=False)

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
    
    # Enrich the sample space with augmented images (horizontally flipped)
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
        
        # Define the settings for our HOG descriptor
        settings = {'orientations'      :   8,
                    'pixels_per_cell'   :   (8, 8),
                    'cells_per_block'   :   (1, 1), 
                    'block_norm'        :   'L2-Hys', 
                    'visualize'         :   False, 
                    'transform_sqrt'    :   False, # Global normalisation
                    'feature_vector'    :   True,
                    'multichannel'      :   False }
        
        # This could perhaps be parallelised using the joblib library 
        # Will implement if I have time
        Xtrain_hog_d    = np.array([ hog(img , **settings ) for img in Xtrain ])  
        Xtest_hog_d     = np.array([ hog(img , **settings ) for img in Xtest  ])  
        
        # Test set
        Xts_hog_d       = np.array([ hog(img , **settings ) for img in X_eq_ts])  
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_hog_d
        Xtest_pre   =   Xtest_hog_d
        
        # Test set
        Xts_pre     =   Xts_hog_d
    
    #%% Edge detection with Sobel-Feldman operator...
    
    """
    P R E - P R O C E S S :   S O B E L
    """
    # We now proceed by processing the images with a Sobel filter
    # The Sobel-Feldman operator provides powerful means of edge detection
    # especially, on equalised images
    
    if selection == 2: 
        
        Xtrain_sob  =  prp.sobel(Xtrain , surpress=False)
        Xtest_sob   =  prp.sobel(Xtest  , surpress=True)
        
        # Test set
        Xts_sob     =  prp.sobel(X_eq_ts)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_sob
        Xtest_pre   =   Xtest_sob
        
        # Test set
        Xts_pre     =   Xts_sob
    
    #%% Feature description with SURF...
    
    """
    P R E - P R O C E S S :   S U R F 
    """
    
    if selection == 3: 
        
        # The OpenCV SURF feature detector requires single channel 0-255 UINT8 images
        # Scale images from 0...1 floating point to 0...255 unsigned 8-bit integer
        Xtrain_imgs = np.array([( img * 255).astype(np.uint8) for img in Xtrain ]) 
        Xtest_imgs  = np.array([( img * 255).astype(np.uint8) for img in Xtest  ]) 
        # Test set
        Xts_imgs    = np.array([( img * 255).astype(np.uint8) for img in X_eq_ts  ]) 
        
        # Extract SURF keypoints
        Xtrain_surf = prp.surf(Xtrain_imgs, n_keypoints = 20, hessianThreshold = 400,upright = False, surpress = False)
        Xtest_surf  = prp.surf(Xtest_imgs, n_keypoints = 20, hessianThreshold = 400,upright = False, surpress = True)
        # Test set
        Xts_surf    = prp.surf(Xts_imgs, n_keypoints = 20, hessianThreshold = 400,upright = False, surpress = True)
        
        # Passing pre-processed object to placeholder for classification task
        Xtrain_pre  =   Xtrain_surf
        Xtest_pre   =   Xtest_surf
        
        # Test set
        Xts_pre     =   Xts_surf

    #%% No further pre-processing, return equalised images...
    
    """
    S K I P   P R E - P R O C E S S I N G
    """
    
    if selection == 4: 
        
        # Passing solely equalised object to placeholder for classification task
        Xtrain_pre  =   Xtrain
        Xtest_pre   =   Xtest
        
        # Test set
        Xts_pre     =   X_eq_ts

    # %% Flatten image data to 1D feature vector...
    
    """
    P R E - P R O C E S S :   F L A T T E N   T O  1 D - F E A T U R E   V E C T O R
    """
    
    # HOG does not require flattening of an image array as it only carries a 1D descriptor vector
    if selection != 1: 
        Xtrain_flat =  prp.flatten(Xtrain_pre)
        Xtest_flat  =  prp.flatten(Xtest_pre)
        Xts_flat    =  prp.flatten(Xts_pre)
    else: 
        Xtrain_flat =  Xtrain_pre
        Xtest_flat  =  Xtest_pre
        Xts_flat    =  Xts_pre
    
    # %% Scale data...
    
    """
    P R E - P R O C E S S :   S C A L E   D A T A
    """
        
    print("Performing feature standardisation, scaling to zero mean with unit variance...")
    
    # Fit the scaler to the training data
    scaler = StandardScaler()
    scaler.fit(Xtrain_flat)
    
    # Apply the transformation to the training, validation and testing data
    Xtrain_scl  =   scaler.transform(Xtrain_flat)
    Xtest_scl   =   scaler.transform(Xtest_flat) 

    # Test set
    Xts_scl = scaler.transform(Xts_flat)
    
    # %% Prepare label data...
    
    """
    P R E P A R E   L A B E L   D A T A
    """
    
    print("Preparing labels...")
    
    # Extract the 'gender' column of the ytrain-, validation and test
    ytrain_A1 =  ds.get_category(ytrain,'gender')
    ytest_A1  =  ds.get_category(ytest,'gender')
    
    # Extract the 'smiling' column of the ytrain-, validation and test
    ytrain_A2 =  ds.get_category(ytrain,'smiling')
    ytest_A2  =  ds.get_category(ytest,'smiling')
    
    # Test set
    yts_A1  =  ds.get_category(yts,'gender')
    yts_A2  =  ds.get_category(yts,'smiling')
    
    # %% Filenames for pickled Cross-Validation records...
    
    """
    C V   R E C O R D   M A N A G E M E N T
    """
    
    # Filenames for Task A1 - Gender classification
    cv_filename_A1 = {'1': 'CV_Records/CV_HOG_A1.pkl',
                      '2': 'CV_Records/CV_Sobel_A1.pkl',
                      '3': 'CV_Records/CV_SURF_A1.pkl',
                      '4': 'CV_Records/CV_none_A1.pkl'}
    
    # Filenames for Task A2 - Smile classification
    cv_filename_A2 = {'1': 'CV_Records/CV_HOG_A2.pkl',
                      '2': 'CV_Records/CV_Sobel_A2.pkl',
                      '3': 'CV_Records/CV_SURF_A2.pkl',
                      '4': 'CV_Records/CV_none_A2.pkl'}
    
    # The filenames to be indexed depend on user's selection of pre-processing method
    selection_filenames = [cv_filename_A1[str(selection)],
                           cv_filename_A2[str(selection)]]
    
    # %% Task A1 - Perform Cross-Validation...
    """
    T A S K   A 1 :   B U I L D   P I P E L I N E :   P C A + S V M
    """
    
    # If we don't have a record of previous cross-validation
    if not os.path.isfile(  selection_filenames[0]  ):
        print("CV records not found in directory, performing cross-validation...")
        # We initialise an empty PCA model
        pca = PCA()
        
        # Use an SVC
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
        'svm__C':      [100,10,1,0.1],
        'svm__kernel': ['rbf','poly','linear'],
        'svm__degree': [2],
        'svm__gamma':  ['scale','auto'],
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
        
        # Pickle the results for later consumption
        joblib.dump(estimator, selection_filenames[0])
    
    # If CV record exists
    else: 
        print("CV records found in directory, loading...")
        # We load it
        estimator = joblib.load(selection_filenames[0])
    
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
    
    # %%
    """
    T A S K   A 1 :   F I T   C L A S S I F I E R S
    """    
    
    print("\nFitting classifiers...") 
    
    # Fit the PCA classifier to the scaled training data
    pca.fit(Xtrain_scl)
    
    # Transform the training-, validation- and test folds to 'n' principal components
    Xtrain_pc   =   pca.transform(Xtrain_scl)
    Xtest_pc    =   pca.transform(Xtest_scl)
    
    # Test set
    Xts_pc = pca.transform(Xts_scl)
    
    # Fit SVM
    svm.fit(Xtrain_pc,ytrain_A1)

    # %% Task A1 - Plot learning curves...
    """
    T A S K   A 1 :   P L O T   L E A R N I N G   C U R V E S
    """    
    # Titles for learning curve plots
    prep_method  =  {'1': 'HOG & PCA',
                     '2': 'Sobel Operator & PCA',
                     '3': 'SURF & PCA',
                     '4': 'CLAHE & PCA'}
    
    # As this operation can take a while it can be skipped    
    if ui.yes_no_menu("\nDo you want to plot the learning curves for Task A1 ?\n(This may take a while) [y] / [n] "):
        
        print("Plotting learning curves, this may take a while...")
             
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
       
        title = "Task A1 - {0} w/ {1} components\nSVM, {2} kernel, $C={3}$, $\gamma={4}$\nLearning curves".format(prep_method[str(selection)], pca.n_components, svm.kernel,svm.C,svm.gamma)
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        pa.plot_learning_curve(svm, title, Xtrain_pc, ytrain_A1, axes=axes, ylim=(0.7, 1.01),cv=cv, n_jobs=4)
        plt.tight_layout()
        plt.show()
    
    # %% Task A1 - Predict & report...
    """
    T A S K   A 1 :   P R E D I C T   &   R E P O R T
    """    
    
    # Predict on the training and unseen data
    yp_tr_A1 = svm.predict(Xtrain_pc)
    yp_te_A1 = svm.predict(Xtest_pc)
    
    # Test set
    yp_ts_A1 = svm.predict(Xts_pc)
    
    # Store accuracy scores
    acc_A1_train =  accuracy_score(y_true = ytrain_A1, y_pred = yp_tr_A1)
    acc_A1_test  =  accuracy_score(y_true = ytest_A1 , y_pred = yp_te_A1)
    acc_A1_ts    =  accuracy_score(y_true = yts_A1, y_pred = yp_ts_A1)
    
    # Print results:
    print("Task A1:\n"+"-"*12+"\nTrain: {:.1f}% \nTest:  {:.1f}%".format(acc_A1_train*100,acc_A1_test*100))
    
    # Test set:
    print("Unseen Test Set: {:.1f}%".format(acc_A1_ts*100))
    
    # Print classification report
    print("\n"*2+"Classification Report:\n"+"-"*54+"\n",classification_report(ytest_A1, yp_te_A1))
    
    if ui.yes_no_menu("Plot confusion matrix for task A1? [y] / [n]"):
        # Plot confusion matrix
        disp = plot_confusion_matrix(svm,Xtest_pc,ytest_A1,normalize='true', display_labels=['Female','Male'],cmap='Blues')
        disp.figure_.set_dpi(400)
        disp.ax_.set_title("Task A1 - {0} w/ {1} components\nSVM, {2} kernel, $C={3}$, $\gamma={4}$\nConfusion Matrix".format(prep_method[str(selection)], pca.n_components, svm.kernel,svm.C,svm.gamma))
        plt.tight_layout()
        plt.show()
    
    
    # %% Check out misclassified instances...
    """
    T A S K   A 1 :   M I S C L A S S I F I E D
    """ 
    
    # We may want to inspect if there is any commonality in the misclassified examples
    if ui.yes_no_menu("Show a sample of misclassified images? [y]/[n]"):
        misclassified = np.where(ytest_A1 != yp_te_A1)    
        miscl_labels  = ytest.iloc[misclassified[0]]
        miscl_imgs    = X[miscl_labels.index]
        
        ds.plot_celeba(miscl_imgs,miscl_labels,5,6)