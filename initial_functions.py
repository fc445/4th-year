import pandas as pd
import numpy as np 
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import spectrogram
import datetime
from features import *
from tqdm import tqdm

def normalise(arrs):
    """
    scales array to between 0 and 1
    """
    #THIS IS NOT USEABLE
    summed = 0
    summed_squared = 0
    for arr in arrs:
        summed += np.mean(np.array(arr).flatten())
        summed_squared += np.mean([i**2 for i in np.array(arr).flatten()])
    mean = summed / len(arrs)
    var = (summed_squared / len(arrs)) - mean**2

    for arr in arrs:
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j] = (arr[i][j]-mean)/(np.sqrt(var))
    
    return arrs

def normalisei(arr):
    """
    scales array to between 0 and 1
    """
    mean = np.mean(np.array(arr).flatten())
    square_mean = np.mean([i**2 for i in np.array(arr).flatten()])
    var = square_mean - mean**2

    for i in range(len(arr)):
        arr[i] = (arr[i]-mean)/(np.sqrt(var))
    
    return arr

def rolling_output_with_labels_02(hours, trains_array, samplerate=3200, interval=10, lag=5, ste_octave=None):
    #empty array for spectrograms and whether a train
    zcr = []
    ste = []
    prom = []
    rms = []
    #pulsedur = []
    peak = []
    cent = []
    rolloff = []

    train = {}

    #find start times for spectrograms
    starts = range(0,samplerate*(3600-interval),samplerate*lag)

    #loop through trains_array
    #feature = librosa.feature.mfcc(y=data[start:end], sr=samplerate, n_mfcc=40, n_mels=40, hop_length=160, fmin=0, fmax=None, htk=False)
    for hour in hours:
    #load wav file
        data, samplerate = librosa.load('train_recordings/hour'+hour+'01.wav', sr=samplerate)  
        for start in tqdm(starts):
            end = start + interval*samplerate    
            zcr.append(zero_crossing_rate(data[start:end], samplerate=samplerate, frame_length=samplerate*interval)[0])
            ste.append(short_time_energy(data[start:end], samplerate=samplerate, frame_length=samplerate*interval)[0])
            prom.append(prominent_frequency(data[start:end], samplerate=samplerate, frame_length=samplerate*interval))
            rms.append(RMS(data[start:end], frame_length=samplerate*interval)[0])
            #pulsedur.append(pulse_duration(data[start:end], frame_length=samplerate*interval))
            peak.append(peak_value(data[start:end], samplerate=samplerate, frame_length=samplerate*interval)[0])
            cent.append(spectral_centroid(data[start:end], samplerate=samplerate, frame_length=samplerate*interval)[0])
            rolloff.append(spectral_roll_off(data[start:end], samplerate=samplerate, frame_length=samplerate*interval)[0])
            
            for train_line in trains_array:
                train0 = []
                #loop through hours
                for train_data in trains_array[train_line]:
                    #perform
                    for start in starts:
                        end = start + interval*samplerate
                        #if more than half a train, append as train
                        mean = np.mean(train_data[int(start/samplerate):int(end/samplerate)])
                        if mean > 0.5:
                            train0.append(1)
                        else:
                            train0.append(0)

                train[train_line] = train0

    #join spectrograms to labels

    output = pd.Series(normalisei(zcr), name='zcr')
    output = pd.concat([output, pd.Series(normalisei(ste), name='ste')],axis=1)
    output = pd.concat([output, pd.Series(normalisei(prom), name='prom')],axis=1)
    output = pd.concat([output, pd.Series(normalisei(rms), name='rms')],axis=1)
    #output = pd.concat((output, pd.Series(normalisei(pulsedur), name='pulsedur')),axis=1)
    output = pd.concat([output, pd.Series(normalisei(peak), name='peak')],axis=1)
    output = pd.concat([output, pd.Series(normalisei(cent), name='cent')],axis=1)
    output = pd.concat([output, pd.Series(normalisei(rolloff), name='rolloff')],axis=1)
    for train_line in trains_array:
        output = pd.concat([output, pd.Series(train[train_line], name=train_line)],axis=1)

    print(output.shape)

    return output