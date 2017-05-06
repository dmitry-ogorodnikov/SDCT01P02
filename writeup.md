#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist_train_labels.png "HistogramLabels"
[image2]: ./examples/grayscaling.png "Grayscaling"
[image3]: ./examples/equalization.png "Equalization"
[image4]: ./examples/hist_train_extended_labels.png "HistLabelsExtendedDataset"
[image5]: ./examples/augmented_img.png "AugmentedImage"
[image6]: ./examples/convnet_architecture.png "ConvNetArchitecture"
[image7]: ./examples/traffic_sign_0.png "Traffic Sign 0"
[image8]: ./examples/traffic_sign_1.png "Traffic Sign 1"
[image9]: ./examples/traffic_sign_2.png "Traffic Sign 2"
[image10]: ./examples/traffic_sign_3.png "Traffic Sign 3"
[image11]: ./examples/traffic_sign_4.png "Traffic Sign 4"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32, 3]
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per each label. It can be seen that some traffic signs have more examples than others.

![HistogramLabels][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth, fifth code cells of the IPython notebook.

My preprocessing pipeline consisted of 2 steps.

Step 1. Convert the images to grayscale (function "ConvertToGrayscale"). This allows you to reduce the size of the data set and to work only with the intensity. As Pierre Sermanet and Yann LeCun mentioned in their paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks", using color channels didn’t seem to improve things a lot.
Here is an example of a traffic sign image before and after grayscaling.
![Grayscaling][image2]

Step 2. Apply localized histogram equalization and scaling of pixel values to [0, 1]. Function "HistogramEqualization". I equalized the image data because the images significantly in terms of contrast and brightness. This should noticeably improve feature extraction. I scaled pixel values to [0, 1] that there is no numerical overflow.
Here is an example of a traffic sign image before and after localized histogram equalization.
![Equalization][image3]

Function PreprocessImages() contains the implementation of the pipeline.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I left the validation and testing data unchanged after loading from the files in the first code cell of the IPython notebook.
My final training set had 827997 number of images. My validation set and test set had 4410 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the amount of data we have is not sufficient for a model to generalise well. Signs are also fairly unbalanced, and some classes are represented to significantly lower extent than the others. The number of examples for each sign is about 20000. 

It is a histogram showing the number of images per each label after augmentation of training dataset.

![HistLabelsExtendedDataset][image4]

ConvNets architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set. To add more data to the the data set, I used flipping, rotation and projection transformation. 

Here is an example of an original image (after preprocessing) and an augmented image:

![AugmentedImage][image5]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth, tenth cells of the ipython notebook. 

It is a diagram of my final model:
 
![ConvNetArchitecture][image6]

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 11th, 12th cells of the ipython notebook. 

To train the model, I used the following functions and parameters:
1. "softmax_cross_entropy_with_logits" - computes softmax cross entropy between logits and labels.
2. "AdamOptimizer" - computes gradients for a loss and applies gradients to variables.
3. BATCH_SIZE = 128
4. EPOCHS = 10
5. learning_rate = 0.001
6. keep_probs for regularization by droupout: 0.9, 0.8, 0.7, 0.5

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.982 
* test set accuracy of 0.973

If a well known architecture was chosen:
* What architecture was chosen?
I use multi-scale convolutional neural network as model, which is described in "http://navoshta.com/traffic-signs-classification/" by Alex Staravoitau. Model based on "Traffic Sign Recognition with Multi-Scale Convolutional Networks" paper.

* Why did you believe it would be relevant to the traffic sign application?
Since this model was implemented for the classification of traffic signs and showed a good result (accuracy 99.33%), I decided to use this model.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 It seems to me that the model is working well if the final model's accuracy on the training, validation and test set have values greater than 0.93. And the accuracy values should not be very different.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

!["Traffic Sign 0"][image7] !["Traffic Sign 1"][image8] !["Traffic Sign 2"][image9] 
!["Traffic Sign 3"][image10] !["Traffic Sign 4"][image11]

The first image might be difficult to classify because the image is very dark.
The second image might be difficult to classify because the image has two signs with a slight overlap. In addition, it is difficult to classify sign with the naked eye.
The third image might be difficult to classify because the image is very dark. In addition, it is difficult to classify sign with the naked eye.
The fourth image might be difficult to classify because the image of poor quality and it is difficult to classify sign with the naked eye.
The fifth image might be difficult to classify because the image of poor quality, the sign is partially overlaped by the object and it is difficult to classify sign with the naked eye.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

Here are the results of the prediction:

| Image	       		                            |     Prediction	        				   | 
|:---------------------------------------------:|:--------------------------------------------:| 
| Yield sign      		                        | Yield sign   								   | 
| Right-of-way at the next intersection         | Right-of-way at the next intersection		   |
| No passing for vehicles over 3.5 metric tons	| No passing for vehicles over 3.5 metric tons |
| Speed limit (80km/h)	      					| Speed limit (80km/h) 		 				   |
| Speed limit (60km/h)							| Speed limit (60km/h)  					   |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making softmax probabilities for predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a yield sign (probability of 1), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         				|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Yield sign   									| 
| 1.05078595e-21     				| Speed limit (80km/h)							|
| 9.76452079e-22					| Priority road									|
| 4.44673842e-25	      			| Beware of ice/snow			 				|
| 7.20038184e-27				    | Dangerous curve to the right					|


For the second image, the model is relatively sure that this is a right-of-way at the next intersection sign (probability of 1), and the image does contain a right-of-way at the next intersection. The top five soft max probabilities were

| Probability         				|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| Right-of-way at the next intersection			| 
| 8.94472682e-35    				| Beware of ice/snow							|
| 3.62979866e-37					| Pedestrians									|
| 3.37250501e-41	      			| Roundabout mandatory			 				|
| 1.67455166e-42				    | Priority road 								|


For the third image, the model is relatively sure that this is a no passing for vehicles over 3.5 metric tons sign (probability of 1), and the image does contain a no passing for vehicles over 3.5 metric tons. The top five soft max probabilities were

| Probability         				|     Prediction	        						| 
|:---------------------------------:|:-------------------------------------------------:| 
| 1.00000000e+00         			| No passing for vehicles over 3.5 metric tons		| 
| 2.12289475e-23     				| No passing										|
| 1.77916628e-23					| End of no passing by vehicles over 3.5 metric tons|
| 1.18841225e-24	      			| Speed limit (80km/h)			 					|
| 9.06942792e-32				    | Dangerous curve to the right 						|
 

 For the fourth image, the model is relatively sure that this is a speed limit (80km/h) sign (probability of 0.96), and the image does contain a speed limit (80km/h). The top five soft max probabilities were

| Probability         				|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| 9.69035029e-01         			| Speed limit (80km/h)							| 
| 1.52533660e-02     				| Speed limit (100km/h)							|
| 3.87175637e-03					| Speed limit (30km/h)							|
| 3.45783588e-03	      			| Speed limit (60km/h)			 				|
| 3.41215124e-03				    | Speed limit (50km/h) 							|


For the fifth image, the model is relatively sure that this is a speed limit (60km/h) sign (probability of 0.63), and the image does contain a speed limit (60km/h). The top five soft max probabilities were

| Probability         				|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| 6.34504974e-01         			| Speed limit (60km/h)							| 
| 1.44834086e-01     				| Speed limit (80km/h)							|
| 1.20748892e-01					| Speed limit (50km/h)							|
| 6.71594962e-02	      			| Speed limit (30km/h)			 				|
| 9.00623854e-03				    | Speed limit (20km/h)		 					|

