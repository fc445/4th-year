import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from octave_band import plot_octave, moving_average, find_octave_band
from wavfile_manipulations import extract_wavfile_and_date
from train_period_checker import max_pooling
import datetime
from tqdm import tqdm
import scipy
import librosa
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from features import short_time_energy
#function to allow input of train timings (rough single time input)
def input_train_times(wavfile, samplerate=1000, octave_band=125, interval_size=600, save_to='judged_timetables/judged_timetable_all_rough.csv'):
    #load csv if exists
    try:
        previous_csv = pd.read_csv(save_to)
        train_starts = list(pd.to_datetime(previous_csv.start, format='%Y-%m-%d %H:%M:%S'))
        train_ends = list(pd.to_datetime(previous_csv.end, format='%Y-%m-%d %H:%M:%S'))
    except:
        train_starts = []
        train_ends = []


    interval_starts = range(0,3600-interval_size, interval_size)

    if isinstance(wavfile, list):
        for i in wavfile:
            data, time = extract_wavfile_and_date('train_recordings/hour'+str(i)+'01.wav')

            try:
                if train_ends[-1] > (time + datetime.timedelta(hours=1)):
                    print('Skipping hour '+i)
                    continue
            except:
                pass
            
            for start in interval_starts:
                current_datetime = time + datetime.timedelta(seconds=start)
                try:
                    if train_ends[-1] > current_datetime:
                        print('Skipping interval: '+str(start)+' to '+str(start+interval_size))
                        continue
                except:
                    pass

                #plot octave filtered signal
                plot_octave(data[start*samplerate:samplerate*(start+interval_size)], octave_band, samplerate=samplerate, ma=1000, ticks=True, show_type='document')

                #request input for times of trains
                status = 0
                count = 0
                while status != 1:
                    train_start = input('Please enter train start time: ')
                    train_end = input('Please enter train end time: ')
                    #attempt to convert to int
                    try:
                        train_start = int(train_start)
                        train_end = int(train_end)
                    except:
                        pass
                    if isinstance(train_start, int) and isinstance(train_end, int):
                        date_of_train_start = current_datetime + datetime.timedelta(seconds=(train_start))
                        date_of_train_end = current_datetime + datetime.timedelta(seconds=(train_end))
                        train_starts.append(date_of_train_start)
                        train_ends.append(date_of_train_end)
                        count += 1
                    elif (train_start == 'done') or (train_end == 'done') or (train_start == 'exit') or (train_end == 'exit'):
                        status = 1

                if (train_start == 'exit') or (train_end == 'exit'):
                    break

            if (train_start == 'exit') or (train_end == 'exit'):
                break
    else:   
        print('incorrect wavfile format')
        return 
    
    #create csv
    pd.DataFrame(zip(train_starts[:-count], train_ends[:-count]), columns=['start','end']).to_csv(save_to, index=False)
    print('Published to: '+save_to)

#function to extract datetime intervals when 125hz octave band is above threshold
def threshold_extract(threshold, hours=['00','01','02','03','04','05','06','07','08','09','10','11','12','13'], ma=1000):
    """
    INPUT - hours - array of hour numbers to use

    OUTPUT - csv with train intervals
    """
    
    #empty arrays of starts and ends
    starts = []
    ends = []

    #set bands
    flower, fupper = find_octave_band(125)
    b, a = scipy.signal.butter( N=4, Wn=np.array([flower, fupper])/(1000/2), btype='bandpass', analog=False, output='ba')
    flag = 0
    #iterate through wavfiles
    for hour in tqdm(hours):
        data, time = extract_wavfile_and_date('train_recordings/hour'+hour+'01.wav')
        #apply moving average
        filtered = abs(scipy.signal.filtfilt(b, a, data))
        data = moving_average(filtered, ma)

        for t in range(3600):
            if data[t*1000] > threshold:
                if flag == 0:
                    flag = 1
                    starts.append(time + datetime.timedelta(seconds=t))
            if data[t*1000] < threshold:
                if flag == 1:
                    flag = 0
                    ends.append(time + datetime.timedelta(seconds=t))

    #ensure same length at end
    if len(starts) > len(ends):
        ends.append(time + datetime.timedelta(hours=1))
    
    #output as csv
    pd.DataFrame(zip(starts, ends), columns=['start', 'end']).to_csv('judged_timetables/judged_timetable_all_rough.csv', index=False)


#function to take datetimes of train windows and output continuous array of 1s and 0s (like find_train_trainbar_array - model 1)
def find_train_trainbar_array_datetime(train_times, up_to_hour=14):
    """
    INPUT -     train times (datetime-csv format)
    OUTPUT -    array of 1s and 0s (continuous)
    """
    train_times_list = pd.read_csv(train_times)
    train_times_list.start = pd.to_datetime(train_times_list.start)
    train_times_list.end = pd.to_datetime(train_times_list.end)
    hours = ['_060913_220655','_070913_000657','_070913_010731','_070913_020731','_070913_030731','_070913_040731','_070913_050731','_070913_060731','_070913_070731','_070913_080731','_070913_090731','_070913_100730','_070913_110730','_070913_120730','_070913_130730','_070913_140730','_070913_150730','_070913_160730','_070913_170730','_070913_180730','_070913_190730','_070913_200730','_070913_210730','_070913_220730','_070913_230730','_080913_000730','_080913_010730','_080913_020730','_080913_030730','_080913_040730','_080913_050730','_080913_060730','_080913_070730','_080913_080730','_080913_090729','_080913_100729','_080913_110729','_080913_120729','_080913_130729','_080913_140729','_080913_150729','_080913_160729','_080913_170729','_080913_180729','_080913_190729','_080913_200729','_080913_210729','_080913_220729','_080913_230729','_090913_000729','_090913_010729','_090913_020729','_090913_030729','_090913_040729']
    
    output = []
    for hour in hours[:up_to_hour]:
        hour_start = datetime.datetime.strptime(hour, '_%d%m%y_%H%M%S')
        out = [0]*3600
        starts = train_times_list.start[(train_times_list.start - hour_start < datetime.timedelta(hours=1)) & (train_times_list.start > hour_start)]
        ends = train_times_list.end[(train_times_list.end - hour_start < datetime.timedelta(hours=1)) & (train_times_list.start > hour_start)]
        for start, end in zip(starts, ends):
            start_seconds = int((start - hour_start).total_seconds())
            end_seconds = int((end - hour_start).total_seconds())

            for i in range(start_seconds,end_seconds):
                out[i] = 1
            
        output.append(out)
    
    return output

                
# a function to take the first hours and output data and training arrays
def model_02_values(hours, train_input):
    """
    ENSURE TRAIN_INPUT COVERS LONGER THAN HOURS
    """
    #load timetables
    timetables = []
    for i in train_input:
        timetables.append(find_train_trainbar_array_datetime(i))

    data = []
    #for i in hours:
    #    data.append(librosa.load())

def normalise(arrs):
    """
    scales array to between 0 and 1
    """
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

def rolling_spectrograms_with_labels_02(hours, trains_array, samplerate=1000, interval=10, lag=5):
    #empty array for spectrograms and whether a train
    output = []
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
            f0, t0, feature = scipy.signal.spectrogram(data[start:end], fs=samplerate, window='hamming', nperseg=39, noverlap=0, nfft=1000)
            feature = max_pooling(feature, pool_size=(3,3), pool_overlap=(1,1))
            lim = [(50*len(feature))//(samplerate/2), (300*len(feature))//(samplerate/2)]
            feature = feature[int(lim[0]):int(lim[1])]
            feature = [i[:125] for i in feature]
            #feature = [i[:128] for i in librosa.feature.melspectrogram(y=data[start:end], sr=samplerate,n_fft=100,hop_length=78)]
            #feature = librosa.feature.mfcc(y=data[start:end],sr=samplerate,n_mfcc=100, n_mels=100, hop_length=101, fmin=0, fmax=None, htk=False)
            output.append(feature)

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

    #print(len(output[0]))
    #print(len(output[0][0]))
    #join spectrograms to labels
    output = pd.Series(normalise(output), name='array')
    for train_line in trains_array:
        output = pd.concat([output, pd.Series(train[train_line], name=train_line)],axis=1)

    #print(output.shape)
    #print(len(output.array[0]))
    #print(len(output.array[0][0]))

    return output

def rolling_spectrograms_no_labels_02(hours, samplerate=1000, interval=10, lag=5):
    #empty array for spectrograms and whether a train
    output = []
    train = []

    #find start times for spectrograms
    starts = range(0,samplerate*(3600-interval),samplerate*lag)

    for hour in hours:
    #load wav file
        data, samplerate = librosa.load('train_recordings/hour'+hour+'01.wav', sr=samplerate)  
        for start in tqdm(starts):
            end = start + interval*samplerate    
            f0, t0, feature = scipy.signal.spectrogram(data[start:end], fs=samplerate, window='hamming', nperseg=39, noverlap=0, nfft=1000)
            feature = max_pooling(feature, pool_size=(3,3), pool_overlap=(1,1))
            lim = [(50*len(feature))//(samplerate/2), (300*len(feature))//(samplerate/2)]
            feature = feature[int(lim[0]):int(lim[1])]
            feature = [i[:125] for i in feature]
            #feature = [i[:128] for i in librosa.feature.melspectrogram(y=data[start:end], sr=samplerate,n_fft=100,hop_length=78)]
            #feature = librosa.feature.mfcc(y=data[start:end],sr=samplerate,n_mfcc=100, n_mels=100, hop_length=101, fmin=0, fmax=None, htk=False)
            output.append(feature)

    #join spectrograms to labels
    output = pd.Series(normalise(output), name='array')

    print(output.shape)
    print(len(output.array[0]))
    print(len(output.array[0][0]))

    return output

def rolling_power_with_labels_02(hours, trains_array, octave_bands=[125], samplerate=3200, interval=10, lag=5):
    #empty array for spectrograms and whether a train
    output = []
    train = []
    features = []

    #find start times for spectrograms
    starts = range(0,samplerate*(3600-interval),samplerate*lag)

    #loop through trains_array
    #feature = librosa.feature.mfcc(y=data[start:end], sr=samplerate, n_mfcc=40, n_mels=40, hop_length=160, fmin=0, fmax=None, htk=False)
    for hour, train_data in tqdm(zip(hours, trains_array)):
    #load wav file
        data, samplerate = librosa.load('train_recordings/hour'+hour+'01.wav', sr=samplerate)    
        for start in starts:
            end = start + interval*samplerate
            for octave_band in octave_bands:    
                feature = short_time_energy(data[start:end],octave_band=octave_band,samplerate=samplerate,frame_length=len(data[start:end]),hop_length=samplerate//2)[0]
                features.append(feature)

            #if more than half a train, append as train
            mean = np.mean(train_data[int(start/samplerate):int(end/samplerate)])
            if mean > 0.5:
                train.append(1)
            else:
                train.append(0)

    for i,j in zip(features, train):
        output.append([i,j])

    #print(output.shape)

    return output

