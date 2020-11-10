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

# numpy for enhanced mathematical support
import numpy as np
# Matplotlib for visualisation
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define a path to the data - REMEMBER TO RESET THIS BEFORE TURNING IN
    img_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/img/"
    label_path = "/Users/gardar/Documents/UCL/ELEC0134 MLS-I Applied Machine Learning Systems/Assignments/dataset_AMLS_20-21/celeba/"
    
    # Load image and label data with the novel 'import_data' module
    X , y , random_img = ds.dataImport(img_path,label_path,surpress=False,return_img_indices=True)
    
    # Split data to train, validation and test sets
    Xtrain,Xtest,Xval,ytrain,ytest,yval = sd.split_dataset(X,y,
                                                           test_size=0.2,
                                                           val_size=0.2,
                                                           surpress=False) 
    print(ytrain.loc[:,'gender'])
    
    """ ... TO PLOT FROM TRAINING SET ...
    # Choose a couple of random var's from Xtrain
    random_Xtrain = np.random.randint(len(Xtrain), size=(2,3))
    
    # Define a tuple to iterate over
    row, col = random_Xtrain.shape
    
    # Define layout and size of subplot
    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(16,8))
    
    # Let's populate our 2x3 subplot
    for i in range(row) :
        for j in range(col):
            # We extract a 4D array from our image library
            extract = X[(random_Xtrain[i,j]),:,:,:]
            # Get rid of the 4th dimension, i.e. squeeze back to 3D
            img = np.squeeze(extract)
            
            # We can then plot the image using matplotlib's .imshow() function
            ax[i,j].imshow(img)
            # Turn off plot axes
            ax[i,j].axis("off")
            
            # Set the title of each image as the corresponding labels
            gender = y.loc[random_Xtrain[i,j],'gender']
            smiling = y.loc[random_Xtrain[i,j],'smiling']
            title = "Gender: {} \n Smiling: {}".format(("Female" if gender == -1 else "Male"),
                                                      ("No" if smiling == -1 else "Yes"))
            ax[i,j].set_title(title)
                
    # Set tight layout and display the plot
    plt.suptitle('Randomly chosen images and corresponding labels from training set')
    plt.tight_layout()
    plt.show()
    """