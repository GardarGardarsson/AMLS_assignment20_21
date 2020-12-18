# README

Remark: The Anaconda environment needed to run the python program is in the /Conda_Environment directory. Please install it using:

		conda env create -f env_sn20167036.yml
		
	This includes all the conda- and pip-installed packages.
	Once setup is complete you can do: 
	
		conda env list
	
	or alternatively:
	
		conda info --envs
		
	and you should see the environment "env_sn20167036" there.
	Don't forget to activate the environment using:
		
		conda activate env_sn20167036
		
	before executing the script.
	The .yml can also be opened in a text editor to see the packages used (if a similar env. happens to exists on your client) but correct execution is understandably not guaranteed.
	
	More info on managing conda environments here:
	
		--- M A N A G I N G   C O N D A   E N V I R O N M E N T S ---
	
		https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

Remark: All the code was initially developed in the main.py file. This is the bread and butter of this project. The code was then broken down into each of the tasks and stored in the A1...B2 folders.
	I will only describe the code briefly here.

Remark: Sorry if some plot renderings are very big during runtime - I think the high DPI setting I was using to generate graphics for my report may have caused that - this isn't an issue when running the code in the spyder IDE.

+++++++++++++
Directory: A1
+++++++++++++

Contents: 
	- Task_A1.py

Description:
	- Contains the program code for task A1.
	- The code uses the OS module to browse up one directory, so that modules, cross-validation records, images and labels can be accessed.
	- User has a choice of computing one of 4 pre-processing methods (HOG+PCA, Sobel+PCA, SURF or none+PCA).
	- A training pipeline is constructed (PCA+SVM) and cross-validation performed only if the program does not find a *.pkl file for the chosen pre-processing method in the \CV_Records directory. (See details in CV_Records below)
	- Uses a gaussian kernel SVM for classification.
	- Prints out training, validation and testing accuracy (called, train, test, unseen in program)
	- Offers user to plot learning curves and confusion matrix.

... A2 is very similar to A1
... B1 is very similar to B2

+++++++++++++
Directory: B2
+++++++++++++

Contents:
	- Task_B2.py

Description:
	- Contains the program code for task B2
	- The program offers the user to perform classification on the data with or without subjects wearing dark tinted sunglasses.
	- Posts training, validation and testing accuracy scores.
	- Offers user to plot confusion matrix.
	- Code snippets to plot the learning curves were removed from code before submission.

+++++++++++++++++++++
Directory: CV_Records
+++++++++++++++++++++

Contents:
	- CV_HOG_A1.pkl
	- CV_HOG_A2.pkl
	- CV_SURF_A2.pkl
	- CV_SURF_A2.pkl
	
Description:
	- Contains pickled cross-validation results that was performed on UCL's GPU servers.
	- Each pre-processing method has it's own cross-validation record.
	- Histogram of Oriented Gradients (HOG) is the best performing feature descriptor.
	- Only HOG and SURF was delivered in Github repo.
	- The others can be obtained from here:
	
	----- GOOGLE DRIVE LINK FOR MORE CV RECORDS -----
	https://drive.google.com/drive/folders/1K-X1RD9iGfQfEPSTcUi3oh_89OI16ujT?usp=sharing
	
	- CV_none_A1.pkl  266 MB (CV records for only PCA for A1)
	- CV_none_A2.pkl  266 MB (CV records for only PCA for A2)
	- CV_Sobel_A1.pkl 272 MB (CV records for the Sobel-Feldman edge detector for A1)
	- CV_Sobel_A2.pkl 272 MB (CV records for the Sobel-Feldman edge detector for A2)

+++++++++++++++++++++
Directory: CV_Records
+++++++++++++++++++++

Contents:
	- CV_HOG_A1.pkl
	- CV_HOG_A2.pkl
	- CV_SURF_A2.pkl
	- CV_SURF_A2.pkl
	
Description:
	- Contains pickled cross-validation results that was performed on UCL's GPU servers.
	- Each pre-processing method has it's own cross-validation record.
	- Histogram of Oriented Gradients (HOG) is the best performing feature descriptor.
	- Only HOG and SURF was delivered in Github repo.
	- The others can be obtained from h

++++++++++++++++++++++++++++
Directory: Conda_Environment
++++++++++++++++++++++++++++

Contents:
	- env_sn20167036.yml
	
Description:
	- The Anaconda environment needed to execute the program, see the topmost "Remark" of this README for further guidance.

+++++++++++++
File: main.py
+++++++++++++
	
Description:
	- Executes all the tasks, offering a choice of a single pre-processing method for both tasks A1 and A2 (Histogram of Oriented Gradients is recommended for both as they yield highest accuracy and the necessary CV records are delivered in this repo, see *Directory: CV_Records).
	- Uses some standard libraries and handcrafted Modules from the \Modules directory.
	- I hope you don't find it messy... I find the structure should make it decently readable.
	
+++++++++++++
File: Modules
+++++++++++++

Contents:
	- import_data.py 
	- performance_analysis.py
	- pre_processing.py
	- split_dataset.py
	- user_interface.py
	- utilities.py
	
Description:
	- Hosts handcrafted modules that were built to complete the tasks at hand, and additional functions that were developed and experimented during it's solution.
	
	*** import_data.py
	Loads images and labels, displays a set of random images loaded along with their labels.
	Accommodates both the data of task A and task B in a single function.
	Default import settings are ".jpg" and "Task_A" but is changed to ".png" and "Task_B" for, well you guessed it Task B.
	
	*** performance_analysis.py
	Uses an example from scikit-learn's website to plot learning curves. A link to the sample is provided in the header.
	
	*** pre_processing.py
	A collection of functions that were constructed during development of this program.
		- reduceRGB(): grayscales images according to the Rec. 601 standard
		
		- centerImg(): Centers an image as a deviation from mean of image set (part of a homemade PCA function that unfortunately wasn't used as I ended up using a pipeline to optimise the SVM+PCA symbiosis, and hence wanted rather to rely on library PCA)
		
		- imgProcessing(): Collective function that gray-scales and centers an image on the image set mean. The results from this novel PCA function can actually be viewed in: 
		
				-> root\notebooks folder 
					-> Pre-Processing Benchmark Testing - PCA.ipynb
					
		- PCA_w_SVD(): Principal Component Analysis using Singular Value Decomposition.
		
		- showEigenfaces(): Function that displays Eigen Faces of PCA processed images. See notebooks for results if interested.
		
		- fitPCA(): Projects image array to the found eigenvector base of the training samples
		
		- crop(): Crops an image to a W x H px size from a given pixel offset (vertical and horizontal)
		
		- sobel(): Uses scipy functions to calculate a Sobel-Feldman filter for an image for edge detection.
		
		- surf(): Uses Open-CV functions to perform Speeded Up Robust Features on images. Results found to be rather poor as there was much variation in the images and not much localisation was done (i.e. I did not detect and extract faces), so the SURF could only use some ~20 keypoints at most (since that was the lowest in the data). 
		
		- LBP(): Local Binary Pattern of an image, I was experimenting with fusing the LBP output to the HOG feature descriptor but think I may have messed up the number of bins in the LBP and it didn't return any improvement of the classification.
		
		- detect_dark_glasses(): Detect outliers in the cartoon_set. Function that takes in a cropped frame from the cartoon_set (of an eye region) and applies a Canny edge detection filter to it. A pixelwise sum of this frame is then calculated. The sum indicates whether or not an eye is in the frame. Eye features are feature rich, and hence register a high pixelwise sum. Dark shades are feature poor and hence register a low sum. A desolate boundary separating the two was found and subjects below the threshold were indexed and, by user choice, can be rejected.
		
		- only_keep(): Used to clean the glasses wearing subjects from the data after running detect_dark_glasses().
	
	
	
	