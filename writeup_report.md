#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Sample.png "Sample"
[image2]: ./images/Brightness.png "Brightness"
[image3]: ./images/Flip.png "Flip"
[image4]: ./images/Shift.png "Shift"
[image5]: ./images/Cropped.png "Cropped"
[image6]: ./images/nvidia.png "Nvidia"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.py for recording the run
* video.mp4 recorded on track #1
* video2.mp4 recorded on track #2
* images containing training dataset samples

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model used for this task is the one described by the self driving car team  from NVIDIA in their article: [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

It consists of 3 convolution layers with 5x5 filter sizes and 2x2 strides and depths between 24 and 48 (model.py lines 130-132), followed by 2 convolution layers with 3x3 filter sizes and depth size 64. 

There are also 3 fully connected layers with sizes between 100 and 10 (model.py lines 136-138), linked to a single output, representing the estimated steering angle. 

I used RELU on all the convolution layers and on the first two fully connected layers to introduce nonlinearity (model.py lines 130-137), and the data is normalized in the model using a Keras lambda layer (model.py lines 123-124).

I also cropped of the top 70 pixels (representing the area above the horizon) and the bottom 25 pixels (representing the bonnet of the car) using the Keras Cropping2D layer (model.py line 125)

Here is an example of a cropped image:

![alt text][image5]


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 114-115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used the images from all 3 cameras and from both tracks. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to test the NVIDIA model architecture on the provided sample dataset. I thought this model might be appropriate because the data on which it was trained is similar to the data from the simulator (wider images) and it was also tested in real life scenarios.

As the results on the sample dataset seemed promising, I stuck to this model and focused on the data acquisition and augmentation process.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.This implied that the model was overfitting. 

To combat the overfitting, I collected the training data by driving the car, as close as possible to the center of the road, around the first track for 3 laps in counter clockwise direction (approx. 11k images), and for one lap around the second track (approx. 7k images).

As I found difficult to acquire data from recovery scenarios, I decided to simulate these events synthetically (inspired by this post: [post](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.a0mmedkp1)). For more details on simulating the recovery events, see section "3. Creation of the Training Set & Training Process"

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I increased the batch_size to 256, used a 0.1 correction value for the left & right camera images and increased the number of epochs to 10.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 130-139) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image     						| 
| Normalization         | outputs 160x320x3 RGB image     				|
| Cropping 	            | outputs 65x320x3 RGB image     				| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 31x155x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 14x76x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 5x36x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x34x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 1x32x64 	|
| RELU					|												|
| Flatten   	      	| input 1x32x64,  outputs 2048   				|
| Fully connected		| outputs 100  									|
| RELU          		|												|
| Fully connected		| outputs 50  									|
| RELU          		|												|
| Fully connected		| outputs 10  									|
| Fully connected		| outputs 1  									|
| MSE   				|												|


Here is a visualization of the architecture

![alt text][image6]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. To simulate recovery driving, I also used the images from the left and right cameras, with corrected angles Here is an example image of center lane driving, with samples from all three cameras:

![alt text][image1]

To overcome the fact that I drove the car only in counterclockwise direction, I flipped the images and angles. For example, here is an image that has been flipped:

![alt text][image3]

Then I repeated this process on track two (one lap) in order to get more data points.

To further augment the dataset, I also performed random brightness changess (code lines 46-53) in order to simulate different lighting conditions. Here is an example of an image whose brightness was altered:

![alt text][image2]

In order to simulate the car being at different positions on the road, I also performed random horizontal and vertical shifting and angle correction, as can be seen in the following example:

![alt text][image4]

All of these augmentation steps are performed randomly, on-the-fly, in the training data generator, which, practically, could yield an infinite number of training images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

I trained the model for 10 epochs, with batch size 256 and 20000 samples per epoch, so the model saw a total of 200.000 images (lines 142-143). I used an adam optimizer so that manually training the learning rate wasn't necessary.
