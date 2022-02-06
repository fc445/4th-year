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

#import data

trains_array_cen_eas = find_train_trainbar_array_datetime('judged_timetables/judged_windows_cen_eas.csv')
trains_array_cen_wes = find_train_trainbar_array_datetime('judged_timetables/judged_windows_cen_wes.csv')
trains_array_all = find_train_trainbar_array_datetime('judged_timetables/judged_timetable_all_rough.csv')
hours=['00','01','02','03','04','05','06','07','08','09','10','11','12','13']
data = rolling_spectrograms_with_labels_02(hours=hours, trains_array=[trains_array_all,trains_array_cen_eas,trains_array_cen_wes], samplerate=1000, interval=10, lag=5)

store = pd.HDFStore('dataset1_attempt7_upto13.h5')
store['first'] = data
"""
data = store['first']
print(data[data.train==1].array.count())
print(data[data.train1==1].array.count())
print(data[data.train2==1].array.count())

data = pd.concat([data[data.train1==1], data[data.train2==1], data[data.train==0].sample(n=3000).reset_index(drop=True), data[data.train==1].sample(n=3000).reset_index(drop=True)])#, data[data.train==1].sample(n=50).reset_index(drop=True)])

#randomise rows
#data = data.sample(frac=1).reset_index(drop=True)

#convert features and labels into numpy arrays
X = np.array(data.array.tolist())
y = np.array(data.train.tolist())
y1 = np.array(data.train1.tolist())
y2 = np.array(data.train2.tolist())
#encode the labels
le = LabelEncoder()
yy = le.fit_transform(y)
yy1 = le.fit_transform(y1)
yy2 = le.fit_transform(y2)
#split the dataset
x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, yy, yy1, yy2, test_size=0.2, random_state=42)
##########          model architecture          ##########

inputs = Input(shape=(125,49,1), name='mfcc')

###########################################################################

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(256)(x)
x = layers.GlobalAveragePooling2D()(x)
output_train = layers.Dense(1, activation='sigmoid', name='train')(x)

##########################################################################

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(256)(x)
x = layers.GlobalAveragePooling2D()(x)
output_cen_eas = layers.Dense(1, activation='sigmoid', name='cen_eas')(x)

############################################################################

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(256)(x)
x = layers.GlobalAveragePooling2D()(x)
output_cen_wes = layers.Dense(1, activation='sigmoid', name='cen_wes')(x)

##############################################################################

model = Model(inputs=inputs, outputs=[output_train, output_cen_eas, output_cen_wes])

model.summary()

keras.utils.plot_model(model, 'something.png', show_shapes=True)

model.compile(
    optimizer='adam',
    metrics='acc',
    loss=keras.losses.BinaryCrossentropy(from_logits=True)
)

model.fit(
    x_train,
    {"train": y_train, "cen_eas": y1_train, "cen_wes": y2_train},
    epochs=72,
    batch_size=32,
)

test_scores = model.evaluate(x_test, y1_test, verbose=0)
print("Test accuracy:", test_scores[1])

test_scores = model.evaluate(x_test, y2_test, verbose=0)
print("Test accuracy:", test_scores[1])


model.save('model2.h5')
model.save_weights('model2_weights.h5')

"""