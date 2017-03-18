import csv
import cv2
import numpy as np
import os
import sklearn

data_root = './data'
data_dirs = os.listdir(data_root)
correction = .1

samples = []

# Read the csv files
for data_dir in data_dirs:
	if os.path.isdir(os.path.join(data_root, data_dir)):
		csvname = os.path.join(data_root, data_dir, 'driving_log.csv')

		with open(csvname) as csvfile:
			reader = csv.reader(csvfile)

			for line in reader:
				samples.append(line)

# Slice out 20% of the samples for validation
from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples, test_size=.2)

# Read the next image
def get_next_image(index, img_type, samples):
	sample = samples[index]
	angle = float(sample[3])
	img_path = os.path.join(data_root,
		'/'.join(sample[img_type].strip().split('/')[-3:]))
	image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

	# if the image is from the left camera,
	# add the correction value to the angle
	if img_type == 1: angle += correction

	# if the image is from right camera,
	# subtract the correction value from the angle
	elif img_type == 2: angle -= correction

	return image, angle

# Perform random brightness augmentation
def brightness_augmentation(image):
	img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	random_bright = np.random.uniform() + .5
	img_hsv[:,:,2] = img_hsv[:,:,2] * random_bright
	img_hsv[:,:,2][np.where(img_hsv[:,:,2] > 255)] = 255

	return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

# Perform random flip augmentation
def flip_augmentation(image, angle):
	flip = np.random.randint(2)

	if flip == 0:
		image, angle = cv2.flip(image, 1), -angle

	return image, angle

# Perform random horizontal and vertical shifts
def shift_augmentation(image, angle, shift_range):
	shift_x = np.random.uniform() * shift_range - shift_range * .5
	angle += shift_x * .4 / shift_range
	shift_y = np.random.uniform() * 40.0 - 20.0
	shift_M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
	rows, cols = image.shape[:2]
	image = cv2.warpAffine(image, shift_M, (cols, rows))

	return image, angle


# Generator for the training dataset
# it randomly reads batch_size images, applies the augmentation
# techniques described above and yields the result
def get_train_generator(samples, batch_size=32):
	num_samples = len(samples)

	while 1:
		sklearn.utils.shuffle(samples)
		images, angles = [], []

		for i in range(batch_size):
			index = np.random.randint(num_samples)
			img_type = np.random.randint(3)

			image, angle = get_next_image(index, img_type, samples)
			image = brightness_augmentation(image)
			image, angle = flip_augmentation(image, angle)
			image, angle = shift_augmentation(image, angle, 100)

			images.append(image)
			angles.append(angle)

		X_train = np.array(images)
		y_train = np.array(angles)

		yield sklearn.utils.shuffle(X_train, y_train)

def get_valid_set(samples):
	images, angles = [], []

	for i, sample in enumerate(samples):
		image, angle = get_next_image(i, 0, samples)

		images.append(image)
		angles.append(angle)

	return np.array(images), np.array(angles)

train_generator = get_train_generator(train_samples, batch_size=256)
valid_set = get_valid_set(valid_samples)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - .5,
	input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

#####################
##  NVIDIA model  ###
#####################
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=20000,
	validation_data=valid_set, nb_epoch=10)

model.save('model.h5')
