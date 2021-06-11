The Project
---
The goals / steps of this project are the following:
step 1 : Load the data set
---
In this steps, the provided data is loaded using the pickle library. The images have labels to recognize what they represent. The labels are numbers, but there is a .csv file containing the mapping between the labels and a text name of the image to make it more human-friendly.

step 2:  Explore, summarize and visualize the data set
---
Here the data set is explored. First, show some general numbers about it:
* Number of training examples = 27839
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 27839

step 3: Design, train and test a model architecture
---
Convolution layer 1. The output shape should be 28x28x6.

Activation 1. Your choice of activation function.

Pooling layer 1. The output shape should be 14x14x6.

Convolution layer 2. The output shape should be 10x10x16.

Activation 2. relu activation function.

Pooling layer 2. The output shape should be 5x5x16.

Flatten layer. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Fully connected layer 1. This should have 120 outputs.

Activation 3. relu activation function.

Fully connected layer 2. This should have 84 outputs.

Activation 4. relu activation function.

Fully connected layer 3. This should have 10 outputs.



step 4. Use the model to make predictions on new images
---
[![Udacity - predict image](https://github.com/ruyi-tsai/CarND-Traffic-Sign-Classifier-Project/blob/master/result.png)






