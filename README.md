# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/hist.png "Visualization"
[image2]: ./examples/train_image.png "Training dataset"
[image3]: ./examples/test_image.png "Test image"
[image4]: ./examples/acc.png "training acc"
[image5]: ./examples/test_image_validation.png "Test image result"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ruyi-tsai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 27839
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 27839

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data
![training dataset][image2]
![visualization][image1]

### Design and Test a Model Architecture



#### 1. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|


| Fully connected		| input 400, outputs = 120       									|
| RELU					|												|
| Fully connected		| input 120, outputs = 84       									|
| RELU					|												|
| Fully connected		| input 84, outputs = 43       									|
| Softmax				|         									|

 


#### 2. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an cross_entropy to optimize my model
```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

#### 3. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Test Accuracy = 0.852
* train Accuracy = 0.976
* validation Accuracy = 0.950

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![test image][image3]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (50km/h)    			| Speed limit (50km/h) 										|
| Bumpy road				| Bumpy road										|
| Speed limit (20km/h)	      		| Speed limit (20km/h)				 				|
| Speed limit (30km/h)			|Speed limit (30km/h)      							|
| Speed limit (60km/h)			|Speed limit (60km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .53         			| Stop sign    									| 
| .2.1675653e-05    				| Speed limit (50km/h) 	 										|
| .99					| Bumpy road													|
| .0	      			| Speed limit (20km/h)				 				|
| .9				    | Speed limit (30km/h)     							|
| 1.0			    | Speed limit (60km/h)     							|

INFO:tensorflow:Restoring parameters from .\lenet
[[5.3014201e-01 4.6933591e-01 5.2100507e-04 1.0354216e-06 3.4842436e-09]
 [9.7496051e-01 1.4051526e-02 1.0535956e-02 4.3005211e-04 2.1675653e-05]
 [9.9924314e-01 6.2494935e-04 1.3203399e-04 2.2313480e-09 1.0302140e-18]
 [5.1201493e-01 4.8798504e-01 4.3405543e-11 2.4774509e-11 6.1084411e-14]
 [1.0000000e+00 2.3983929e-13 1.0448375e-15 7.9142087e-17 5.5983243e-19]
 [9.9730992e-01 2.6703512e-03 1.9688114e-05 2.3572751e-13 6.2075338e-14]]
[[14  8  7  2  1]
 [33 42 28  3  2]
 [22 12 26 25 13]
 [ 5  3  1  2 40]
 [ 1  2  0  5  6]
 [ 3  2  1  5 35]]
New Images Test Accuracy = 80.0%

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![Udacity - predict image][image4]
