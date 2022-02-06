

###-----------MODEL DESRIPTION-----------###
# Take spectrograms at (overlapping?) 5 second rolling intervals? try 10 second intervals every 5?
# input of train and not-train of 1 and 0s for every second

import os
import pandas as pd
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.fftpack import ifft
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import scipy
from scipy.signal import welch
import librosa
import librosa.display
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from model_01_functions import *

#import data
input_wav = 'train_recordings/hour0101.wav'
trains_array = find_train_trainbar_array(judged_timings_hour01())
data = rolling_spectrograms_with_labels(input_wav=input_wav, trains_array=trains_array, samplerate=1000, interval=10, lag=5)

#data = pd.concat([data[data['train?']==1], data[data['train?']==0].sample(n=70).reset_index(drop=True)])

#randomise rows
data = data.sample(frac=1).reset_index(drop=True)

#convert features and labels into numpy arrays
X = np.array(data.array.tolist())
y = np.array(data['train?'].tolist())

#encode the labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

##########          model architecture          ##########

num_rows = 40
num_columns = 63
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

##########          compiling the model             ##########

# Compile the model

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=SGD(learning_rate=0.01))

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy) 

#########           training            #########

num_epochs = 100
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', verbose=1, save_best_only=True)


model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)






# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
a = model.evaluate
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


#save model
model.save('model_continuous_1.h5')
model.save_weights('model_continuous_1_weights.h5')
print('saved model to disk')