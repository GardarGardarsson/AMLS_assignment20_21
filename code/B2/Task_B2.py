# %% Import libraries...
"""
L I B R A R Y   I M P O R T
"""

# To modularise the program, we need to add the current directory to sys path
import os
import sys

## Browse up one directory...
os.chdir(os.path.dirname(os.getcwd()))
currentPath = os.getcwd()

# ... and make that the system path
sys.path.append(currentPath)

# Now we can import our own modules into our script.
import Modules.import_data as ds
import Modules.user_interface as ui
import Modules.split_dataset as sd
import Modules.pre_processing as prp
import Modules.performance_analysis as pa # Used for generating learning curves, now deleted from code.

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt
# Pandas dataframes for enhanced data storage
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit # Used for generating learning curves, now deleted from code.

 # %% Load data...
if __name__ == '__main__':
        
    """
    L O A D   D A T A
    """
    # Define a path to the data
    img_path = "/Datasets/cartoon_set/img/"
    label_path = "/Datasets/cartoon_set/"
    
    # Test set
    img_ts_path = "/Datasets/cartoon_set_test/img/"
    label_ts_path = "/Datasets/cartoon_set_test/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,
                                       label_path,
                                       img_type='.png',
                                       task='B',
                                       surpress=False,
                                       return_img_indices=True)
    # Test set
    Xts , yts  = ds.dataImport(img_ts_path,label_ts_path,img_type='.png',task='B',surpress=True,return_img_indices=False)
    
    # %% Split data...
    """
    S P L I T   D A T A
    """
    # Split dataset into train-, validation- and test folds
    Xtrain,Xtest,ytrain,ytest = sd.split_dataset(X,y,test_size=0.2,surpress=False)
    
    # %% Crop images...
    """
    P R E - P R O C E S S :   C R O P   R E G I O N S   O F   I N T E R E S T
    """
        
    # We crop the training, validation and test data to the eye region for task B2
    Xtrain_C_B2 = prp.crop(Xtrain,  H = 35, W = 55, ver_off = 245, hor_off = 180)
    Xtest_C_B2  = prp.crop(Xtest,   H = 35, W = 55, ver_off = 245, hor_off = 180)
    Xts_C_B2    = prp.crop(Xts,     H = 35, W = 55, ver_off = 245, hor_off = 180)
    
    # %% Grayscale images...
    """
    P R E - P R O C E S S :   G R A Y S C A L E 
    """
      
    # Grayscale face images for task B2, for further processing to detect
    # shaded eyeglasses
    Xtrain_gry_B2 = prp.reduceRGB(Xtrain_C_B2)
    Xtest_gry_B2  = prp.reduceRGB(Xtest_C_B2)
    Xts_gry_B2    = prp.reduceRGB(Xts_C_B2)
        
    # %% Prepare label data...
    """
    P R E P A R E   L A B E L   D A T A
    """
        
    # Transform the 'smiling' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_B2 =  pd.Series.to_numpy(ytrain.loc[:,'eye_color'].copy())
    ytest_B2  =  pd.Series.to_numpy(ytest.loc[:,'eye_color'].copy())
    yts_B2    =  pd.Series.to_numpy(yts.loc[:,'eye_color'].copy())
        
    # %%
    
    """
    T A S K   B 2 :   E Y E   C O L O R   C L A S S I F I C A T I O N
    """
    
    message = 'Please select an option for processing task B2: '
    options = {'1': 'Remove dark-shaded glasses wearing subjects from data',
               '2': 'Keep dark-shaded glasses wearing subjects'}
    
    selection = ui.selection_menu(message, options)
    

    # %% Detect dark eyeglasses...
    
    """
    P R E - P R O C E S S :   D E T E C T   D A R K   E Y E G L A S S E S
    """
    
    if selection == 1:
    
        print("Removing dark-shaded glasses wearing subjects from data...")
        
        # Retrieve eye region imagery and feature values
        Xtrain_CE_img , Xtrain_CE_val = prp.detect_dark_glasses(Xtrain_gry_B2, surpress = False)
        Xtest_CE_img  , Xtest_CE_val  = prp.detect_dark_glasses(Xtest_gry_B2 , surpress = True)
        Xts_CE_img    , Xts_CE_val    = prp.detect_dark_glasses(Xts_gry_B2   , surpress = True)
        
        """
        I N D E X   D A R K   G L A S S E S
        """
        Xtrain_glasses    = np.where(Xtrain_CE_val <  187)[0]
        Xtrain_no_glasses = np.where(Xtrain_CE_val >= 187)[0]
        
        Xtest_glasses     = np.where(Xtest_CE_val <  187)[0]
        Xtest_no_glasses  = np.where(Xtest_CE_val >= 187)[0]
    
        Xts_glasses       = np.where(Xts_CE_val  <   187)[0]
        Xts_no_glasses    = np.where(Xts_CE_val  >=  187)[0] 
        
        """
        P R E P A R E   I M A G E S   E X C L U D I N G   D A R K   G L A S S E S 
        """    
        
        Xtrain_B2, ytrain_B2 = prp.only_keep(Xtrain_no_glasses, Xtrain_C_B2, ytrain_B2)
        Xtest_B2 , ytest_B2  = prp.only_keep(Xtest_no_glasses , Xtest_C_B2 , ytest_B2 )
        Xts_B2   , yts_B2    = prp.only_keep(Xts_no_glasses   , Xts_C_B2   , yts_B2   ) 
        
        # Flatten the training, validation and test data to N-samples X 1D vector
        Xtrain_B2 = prp.flatten(Xtrain_B2)
        Xtest_B2  = prp.flatten(Xtest_B2)
        Xts_B2    = prp.flatten(Xts_B2)
        
    # %% Image prep method 2...
    """
    P R E - P R O C E S S :   S K I P   D A R K   G L A S S E S   D E T E C T I O N 
    """
    if selection == 2:
        # Only flatten the training, validation and test data to N-samples X 1D vector
        Xtrain_B2 = prp.flatten(Xtrain_C_B2)
        Xtest_B2  = prp.flatten(Xtest_C_B2)
        Xts_B2    = prp.flatten(Xts_C_B2)

    # %% Scale
    
    print("Scaling data to the interval € [0,1]...")
    
    # Scale data to 0...1 range
    scaler = MinMaxScaler()
    scaler.fit(Xtrain_B2)
    
    Xtrain_B2 = scaler.transform(Xtrain_B2)
    Xtest_B2  = scaler.transform(Xtest_B2)
    Xts_B2    = scaler.transform(Xts_B2)

    # %% Task B2 with Logistic Regression
    """
    T A S K   B 2 :   E Y E   C O L O R   W/   S O F T M A X   R E G R E S S I O N
    """    

    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs', # <--- SAGA solver is compatible with all penalty types: l1, l2 and elastinet, we use lbfgs
                                C=5,            # <--- Higher C translates to lower regularisation strength
                                penalty='l2',   # <--- Uses Ridge Regression as cost-function, if using LASSO (l1) we're promoting sparsity
                                multi_class='multinomial', # <--- Softmax (multinomial logistic regression) not OvR 
                                max_iter = 100) 
        
    # Train the model using the training sets
    logreg.fit(Xtrain_B2, ytrain_B2)
                
    # Predict on the training and unseen data
    yp_tr_B2 = logreg.predict(Xtrain_B2)
    yp_te_B2 = logreg.predict(Xtest_B2)
    yp_ts_B2 = logreg.predict(Xts_B2)
    
    # Store accuracy scores
    acc_B2_train =  accuracy_score(y_true = ytrain_B2, y_pred = yp_tr_B2)
    acc_B2_test  =  accuracy_score(y_true = ytest_B2 , y_pred = yp_te_B2)
    acc_B2_ts    =  accuracy_score(y_true = yts_B2   , y_pred = yp_ts_B2)
    
    # Print results:
    print("Task B2:\n"+"-"*12+"\nTrain: {:.1f}% \nTest:  {:.1f}%\nUnseen: {:.1f}".format(acc_B2_train*100,acc_B2_test*100,acc_B2_ts*100))
    
    if ui.yes_no_menu("Show confusion matrix? [y]/[n]"):
        disp = plot_confusion_matrix(logreg,Xtest_B2,ytest_B2,normalize='true',cmap='Blues')
        disp.figure_.set_dpi(400.0)
        disp.ax_.set_title("Task B2\nLogistic Regression, $C={}$, penalty={}\nConfusion matrix".format(logreg.C, logreg.penalty))
        plt.show()  
