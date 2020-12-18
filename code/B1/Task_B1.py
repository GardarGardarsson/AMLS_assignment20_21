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
    
    # We crop the training, validation and test data to the face region for task B1
    Xtrain_C_B1 = prp.crop(Xtrain,  H = 250, W = 220, ver_off = 150, hor_off = 140)
    Xtest_C_B1  = prp.crop(Xtest,   H = 250, W = 220, ver_off = 150, hor_off = 140)
    Xts_C_B1    = prp.crop(Xts,     H = 250, W = 220, ver_off = 150, hor_off = 140)
    
    # %% Grayscale images...
    """
    P R E - P R O C E S S :   G R A Y S C A L E 
    """
    
    # Grayscale face images for task B1 as shape classification is colour 
    # independent
    Xtrain_gry_B1 = prp.reduceRGB(Xtrain_C_B1)
    Xtest_gry_B1  = prp.reduceRGB(Xtest_C_B1)
    Xts_gry_B1    = prp.reduceRGB(Xts_C_B1)
        
    # %% Prepare label data...
    """
    P R E P A R E   L A B E L   D A T A
    """
    
    # Transform the 'gender' column of the ytrain-, validation and test
    # labels to one hot vector
    ytrain_B1 =  pd.Series.to_numpy(ytrain.loc[:,'face_shape'].copy())
    ytest_B1  =  pd.Series.to_numpy(ytest.loc[:,'face_shape'].copy())
    yts_B1    =  pd.Series.to_numpy(yts.loc[:,'face_shape'].copy())
    
    # %%
    
    """
    T A S K   B 1 :   F A C E   S H A P E   C L A S S I F I C A T I O N
    """
    
    # %% Flatten data for task B1...
        
    # Flatten the training, validation and test data to N-samples X 1D vector
    Xtrain_B1 = prp.flatten(Xtrain_gry_B1)
    Xtest_B1  = prp.flatten(Xtest_gry_B1)
    Xts_B1    = prp.flatten(Xts_gry_B1)
    
    print("Scaling data to the interval € [0,1]")
    
    # Scale data to 0...1 range
    scaler = MinMaxScaler()
    scaler.fit(Xtrain_B1)
    
    Xtrain_B1 = scaler.transform(Xtrain_B1)
    Xtest_B1  = scaler.transform(Xtest_B1)
    Xts_B1    = scaler.transform(Xts_B1)
    
    # %% Task B1 with Logistic Regression
    """
    T A S K   B 1 :   F A C E   S H A P E   W/   S O F T M A X   R E G R E S S I O N
    """    

    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs',# <--- SAGA solver is compatible with all penalty types: l1, l2 and elastinet, we use lbfgs
                                C=5,           # <--- Higher C translates to lower regularisation strength
                                penalty='l2',  # <--- Uses Ridge Regression as cost-function, if using LASSO (l1) we're promoting sparsity
                                multi_class='multinomial', # <--- Softmax (multinomial logistic regression) not OvR 
                                max_iter = 100) 
        
    # Train the model using the training sets
    logreg.fit(Xtrain_B1, ytrain_B1)
                
    # Predict on the training and unseen data
    yp_tr_B1 = logreg.predict(Xtrain_B1)
    yp_te_B1 = logreg.predict(Xtest_B1)
    yp_ts_B1 = logreg.predict(Xts_B1)
    
    # Store accuracy scores
    acc_B1_train =  accuracy_score(y_true = ytrain_B1, y_pred = yp_tr_B1)
    acc_B1_test  =  accuracy_score(y_true = ytest_B1 , y_pred = yp_te_B1)
    acc_B1_ts    =  accuracy_score(y_true = yts_B1   , y_pred = yp_ts_B1)
    
    # Print results:
    print("Task B1:\n"+"-"*12+"\nTrain: {:.1f}% \nTest:  {:.1f}%\nUnseen:  {:.1f}%".format(acc_B1_train*100,acc_B1_test*100, acc_B1_ts*100))
  
    if ui.yes_no_menu("Show confusion matrix? [y]/[n]"):
        disp = plot_confusion_matrix(logreg,Xtest_B1,ytest_B1,normalize='true',cmap='Blues')
        disp.figure_.set_dpi(400.0)
        disp.ax_.set_title("Task B1\nLogistic Regression, $C={}$, penalty={}\nConfusion matrix".format(logreg.C, logreg.penalty))
        plt.show()   
