import os
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy
from tqdm import tqdm
import librosa
from scipy.signal import welch, spectrogram
from expected_train_times import *
from octave_band import *

#funtion to take train intervals and output array with train or not-train (1,0) every second.
def find_train_trainbar_array(train_times):
    train_times_list = sorted(np.concatenate(np.array(train_times), axis=0))
    trains = np.linspace(1,3600,3600)
    flag = 0
    for time in range(3600):
        if time in train_times_list:
            if flag==0:
                flag=1
            else:
                flag=0
        trains[time] = flag
    return trains

#train_times = pd.read_csv('timings/hour0001_all_trains.csv')
#data1, samplerate1 = librosa.load('train_recordings/hour0101.wav', sr=500)

###---------------TIMINGS-------------###
def judged_timings_hour01():
    judged_cen_east = [(188,210),(498,516),(806,824),(1463,1481)]
    judged_cen_west = [(87,100),(303,319),(650,677),(904,925),(1307,1320),(1494,1506)]
    judged_bak_south = [(286,302),(493,508),(752,780),(1055,1073),(1157,1173),(1510,1522)]
    judged_bak_north = [(350,370),(956,972),(1540,1562)]
    judged_vic_south = [(140,175),(716,730),(1357,1373)]
    train_times = judged_cen_east + judged_cen_west + judged_bak_south + judged_bak_north + judged_vic_south
    return train_times
"""plt.plot(np.linspace(1,3600,3600),0.005*find_train_trainbar_array(train_times))

###----------PLOTTING 125 OCTAVE---------###

flower, fupper = find_octave_band(125)
b, a = scipy.signal.butter( N=4, Wn=np.array([flower, fupper])/(samplerate1/2), btype='bandpass', analog=False, output='ba')
filtered = abs(scipy.signal.filtfilt(b, a, data1))
time = np.linspace(0,len(filtered)/samplerate1, len(filtered))

plt.plot(time, filtered)
plt.show()"""

###---------FUNCTION FOR ROLLING SPECTROGRAMS-------###

def rolling_spectrograms_with_labels(input_wav, trains_array, samplerate=1000, interval=10, lag=5):
################################
###INPUT
#   input_wav - wav file
#   samplerate = desired sample rate
#   interval - length of spectrogram
#   lag - time between spectrograms
###OUTPUT
#   returns t and f for plotting
#   return array of spectrograms
#################################

    #load wav file
    data, samplerate = librosa.load(input_wav, sr=samplerate)

    #empty array for spectrograms and whether a train
    output = []
    train = []

    #find start times for spectrograms
    starts = range(0,samplerate*(3600-interval),samplerate*lag)
    #perform spectrograms
    for start in tqdm(starts):
        end = start + interval*samplerate
        mfcc = librosa.feature.mfcc(y=data[start:end], sr=samplerate, n_mfcc=40, n_mels=40, hop_length=160, fmin=0, fmax=None, htk=False)
        output.append(mfcc)
        #print(mfcc.shape)
        #if more than half a train, append as train
        mean = np.mean(trains_array[int(start/1000):int(end/1000)])
        if mean > 0.5:
            train.append(1)
        else:
            train.append(0)

    print(mfcc.shape)

    #join spectrograms to labels
    output = pd.Series(output, name='array')
    output = pd.concat((output, pd.Series(train, name='train?')),axis=1)

    return output    

def rolling_spectrograms(input_wav, samplerate=1000, interval=10, lag=5):
################################
###INPUT
#   input_wav - wav file
#   samplerate = desired sample rate
#   interval - length of spectrogram
#   lag - time between spectrograms
###OUTPUT
#   returns t and f for plotting
#   return array of spectrograms
#################################

    #load wav file
    data, samplerate = librosa.load(input_wav, sr=samplerate)

    #empty array for spectrograms and whether a train
    output = []

    #find start times for spectrograms
    starts = range(0,samplerate*(3600-interval),samplerate*lag)
    #perform spectrograms
    for start in tqdm(starts):
        end = start + interval*samplerate
        mfcc = librosa.feature.mfcc(y=data[start:end], sr=samplerate, n_mfcc=40, n_mels=40, hop_length=160, fmin=0, fmax=None, htk=False)
        output.append(mfcc)

    #make pandas array
    output = np.array(output)

    return output    