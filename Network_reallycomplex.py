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
import keras
from keras.models import Sequential
from keras import Input, Model, layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn import metrics 
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from model_02_functions import find_train_trainbar_array_datetime, rolling_spectrograms_with_labels_02
import keras_resnet
import random

#import data
store = pd.HDFStore('initial_datasets/spectro3.h5')
data = store['first']

#output counts of data entries
input_shape = (len(data.array[0]),len(data.array[0][0]))
print('individual input feature size: {} x {}'.format(input_shape[0], input_shape[1]))
"""print('No. trains: {}'.format(data[data.train==1].array.count()))
print('No. cen_eas trains: {}'.format(data[data.train1==1].array.count()))
print('No. cen_wes trains: {}'.format(data[data.train2==1].array.count()))"""

#finalised dataset for training from full dataset
data = pd.concat([data[data.no_train==0],data[data.no_train==1].sample(300)])

#randomise rows
data = data.sample(frac=1).reset_index(drop=True)

#convert features and labels into numpy arrays
X = np.array(data.array.tolist())
y0 = np.array(data.no_train.tolist())
y1 = np.array(data.cen_eas.tolist())
y2 = np.array(data.cen_wes.tolist())
y3 = np.array(data.bak_sou.tolist())
y4 = np.array(data.bak_nor.tolist())
y5 = np.array(data.vic_sou.tolist())
y6 = np.array(data.vic_nor.tolist())
#encode the labels
le = LabelEncoder()
yy0 = le.fit_transform(y0)
yy1 = le.fit_transform(y1)
yy2 = le.fit_transform(y2)
yy3 = le.fit_transform(y3)
yy4 = le.fit_transform(y4)
yy5 = le.fit_transform(y5)
yy6 = le.fit_transform(y6)
#split the dataset
#x_train, x_test, y1_train, y1_test, y2_train, y2_test,  = train_test_split(X, yy1, yy2, test_size=0.2, random_state=42)

#set input shape
num_channels = 1
inputs = Input(shape=(input_shape[0],input_shape[1],num_channels), name='spectrogram')

###########################################################################

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=256, kernel_size=2, activation="relu",padding='causal')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(256)(x)
x = layers.GlobalAveragePooling2D()(x)

output_cen_eas = layers.Dense(1, activation='sigmoid', name='cen_eas')(x)
output_cen_wes = layers.Dense(1, activation='sigmoid', name='cen_wes')(x)
output_bak_sou = layers.Dense(1, activation='sigmoid', name='bak_sou')(x)
output_bak_nor = layers.Dense(1, activation='sigmoid', name='bak_nor')(x)
output_vic_sou = layers.Dense(1, activation='sigmoid', name='vic_sou')(x)
output_vic_nor = layers.Dense(1, activation='sigmoid', name='vic_nor')(x)
output_no_train = layers.Dense(1, activation='sigmoid', name='no_train')(x)

##########################################################################

model = Model(inputs=inputs, outputs=[output_cen_eas,output_cen_wes,output_bak_sou,output_bak_nor,output_vic_sou,output_vic_nor,output_no_train])

model.summary()

keras.utils.plot_model(model, 'networkcomplex_model.png', show_shapes=True)

model.compile(
    optimizer='adam',
    metrics='acc',
    loss='binary_crossentropy'
)

model.fit(
    X,
    {"cen_eas": y1, "cen_wes": y2, 'bak_sou':y3, 'bak_nor':y4, 'vic_sou':y5, 'vic_nor':y6,'no_train': y0},
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

#test_scores = model.evaluate(x_test, y_test, verbose=0)
#print("Test accuracy:", test_scores)

model.save('networkcomplex_model.h5')
model.save_weights('networkcomplex_weights.h5')