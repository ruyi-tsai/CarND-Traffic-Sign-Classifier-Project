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

[image1]: ./examples/hist1.png "Visualization"
[image2]: ./examples/train_image1.png "Training dataset"
[image3]: ./examples/test_image1.png "Test image"
[image4]: ./examples/acc1.png "training acc"
[image5]: ./examples/test_image_validation1.png "Test image result"
[image6]: ./examples/test_image_validation2.png "Test image result"
[image7]: ./examples/test_image_validation3.png "Test image result"
[image8]: ./examples/test_image_validation4.png "Test image result"
[image9]: ./examples/test_image_validation5.png "Test image result"


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

#### 0.Pre-process the Data Set (normalization, grayscale, etc.)
I using normalize image to gray image for training set.
step 1: All image scaling to  0~1.0 
```
def minMaxScalingNormalization(X):
    a = 0
    b = 1.0
    return a + X * (b-a) / 255
```


step 2: then standardize
```

def standardize(X):
    # zero-center
    X -= np.mean(X)
    
    # normalize
    X /= np.std(X) 
    
    return (X)
```
#### 1. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
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
* Test Accuracy = 0.942
* train Accuracy = 1.000
* validation Accuracy = 0.945

![training][image4]
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![test image][image3]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		| Speed limit (60km/h)   									| 
| Yield    			| Yield 										|
| Keep right				| Keep right										|
| Roundabout mandatory	      		| Roundabout mandatory				 				|
| Speed limit (30km/h)			|Speed limit (30km/h)      							|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were
image1 prediction
Top 5 prediction probabilities: [[1.0000000e+00 7.4788867e-17 5.4530958e-20 6.0629548e-21 5.3650680e-22]]

Top 5 prediction ids: [[ 5  6 38 10 25]]

![image1 prediction][image5]


image2 prediction
Top 5 prediction probabilities: [[1.0000000e+00 1.8768512e-10 3.6470696e-12 1.4914224e-12 1.1456215e-13]]

Top 5 prediction ids: [[40  1 20  2 12]]


![image2 prediction][image6]


image3 prediction
Top 5 prediction probabilities: [[9.9992561e-01 7.0059883e-05 4.3119339e-06 4.6060009e-13 1.4078725e-14]]

Top 5 prediction ids: [[25 23 21 11 31]]

![image3 prediction][image7]



image4 prediction
Top 5 prediction probabilities: [[9.9999988e-01 1.6485251e-07 4.0969379e-08 3.6969006e-08 4.4289821e-09]]

Top 5 prediction ids: [[34 39 40  3  9]]


![image4 prediction][image8]


image5 prediction
Top 5 prediction probabilities: [[1.0000000e+00 6.7018640e-15 1.3108409e-21 5.2212903e-22 8.3676543e-24]]

Top 5 prediction ids: [[13 10 33  9 29]]


![image4 prediction][image9]


