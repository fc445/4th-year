import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
import pandas as pd
import os
import librosa
from tqdm import tqdm
import datetime

#this tests for data vectors if same length and outputs a new array with all equal length, intended for use before feature extraction
def same_length(input_dataframe, time_limit=15, samplerate=1000):
    #for hour 1, setting all trains to 15 seconds
    #cut off ends to make centre of each train time_limit seconds
    #split non-trains into time_limit segments if enough
    #if train is less than time_limit then delete

    #new dataframe to edit
    dataframe = pd.DataFrame([[]])
    for row_num, row in input_dataframe.iterrows():
        recording = row['recording']
        train = row['train?']

        #find length of recording
        length = len(recording)

        #first delete all small recordings
        if length < time_limit * samplerate: #too short
            #do nothing
            continue

        #next take larger recording segments
        elif length > time_limit * samplerate: #too long
            #if train, cut down to size
            if train == 1:
                #find amount to cut from either end
                to_cut = (length - time_limit * samplerate) // 2
                #cut
                dataframe_to_add = pd.DataFrame([[recording[to_cut:length-to_cut], 1]], columns=['recording', 'train?'])
                dataframe = dataframe.append(dataframe_to_add)

            #if not train, cut to as many as possible
            else:
                #max we can split into
                max_split = length // (time_limit * samplerate)
                #find amount to cut from either end
                to_cut = (length - time_limit * samplerate * max_split) // 2
                #take the middle of the array
                middle = recording[to_cut:length-to_cut]
                #cut array to time_limit seconds and append to dataframe
                for i in range(max_split):
                    new_vector = middle[i*samplerate*time_limit : (i+1)*samplerate*time_limit]
                    dataframe_to_add = pd.DataFrame([[new_vector, 0]], columns=['recording', 'train?'])
                    dataframe = dataframe.append(dataframe_to_add, ignore_index=True)
    
    dataframe = dataframe[:][1:].reset_index(drop=True)

    return dataframe

def feature_extraction(dataframe, sr=1000):
    features = pd.DataFrame([[]])
    
    #loop through dataframe
    for row_index, row in dataframe.iterrows():
        data = row['recording']
        label = row['train?']

        #find mfccs
        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, n_mels=40, hop_length=160, fmin=0, fmax=None, htk=False)
        #set feature vector
        feature_element = pd.DataFrame([[mfccs, label]], columns=['feature', 'label'])
        #append features
        features = features.append(feature_element)
    
    features = features[1:]
    features.reset_index(drop=True, inplace=True)


    return features



#can't remember what this does but similar to the two below
"""def output_train_recordings(train_times, recording_times, wav_directory, sr=1000): #this also downsamples to 1kHz
    #input times as pandas dataframe, no header

    #create directory for new wav files (folder called trains_only in current directory)
    path = os.path.join(wav_directory, 'trains_only_1kHz')
    if not os.path.exists(path):
        os.mkdir(path)

    #train times csv format: [train_id, loco, day, tunnel, direction, T1, T2, t1(s), t2(s), dt(s)] - lowercase times from midnight friday/saturday
    #recording times csv format: [start time (_ddmmyy_hhmmss.), hour number, start time (s) from midnight friday/saturday] - all durations 3600s

    #create list of channel strings
    #channels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
    channels = ['01']

    #iterate through train times
    for train_index, train_row in train_times.iterrows():
        train_start = train_row[7]
        train_end = train_row[8]

        #iterate through recording times
        for recording_index, recording_row in recording_times.iterrows():
            recording_start = recording_row[2]
            recording_end = recording_start + 3600
            
            #if train is fully contained in this hour, save as new train-only wav file
            if (train_start > recording_start) and (train_end < recording_end):
                #create list of wav files where this train is recorded, and corrosponding output files
                wavfile_list = []
                new_wavfile_list = []
                for i in channels:
                    wavfile_list.append(wav_directory + '\\hour' + recording_row[1][-2:] + i + '.wav')
                    new_wavfile_list.append(path + '\\train' + str(train_row[0]) + '_channel' + i + '.wav')
                
                #trim wav files to only contain a train
                for old_path, new_path in zip(wavfile_list, new_wavfile_list):
                    start = train_start - recording_start
                    end = start + train_end - train_start
                    trim_wav(old_path, new_path, start, end, sr=1000)
            
            elif (train_start > recording_start) and (train_end > recording_end):
                continue
                #print('train recording overlaps recording files: train ' + str(train_row[0]))
                
            else: continue"""

#this takes all train times and train recordings and extracts train-only recordings as wav files
def output_train_recordings(train_times, recording_times, wav_directory, sr=1000): #this also downsamples to 1kHz
    #input times as pandas dataframe, no header

    #create directory for new wav files (folder called trains_only in current directory)
    path = os.path.join(wav_directory, 'trains_only_1kHz')
    if not os.path.exists(path):
        os.mkdir(path)

    #train times csv format: [train_id, loco, day, tunnel, direction, T1, T2, t1(s), t2(s), dt(s)] - lowercase times from midnight friday/saturday
    #recording times csv format: [start time (_ddmmyy_hhmmss.), hour number, start time (s) from midnight friday/saturday] - all durations 3600s

    #create list of channel strings
    #channels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
    channels = ['01']

    #iterate through train times
    for train_index, train_row in train_times.iterrows():
        train_start = train_row[7]
        train_end = train_row[8]

        #iterate through recording times
        for recording_index, recording_row in recording_times.iterrows():
            recording_start = recording_row[2]
            recording_end = recording_start + 3600
            
            #if train is fully contained in this hour, save as new train-only wav file
            if (train_start > recording_start) and (train_start < recording_end):
                #create list of wav files where this train is recorded, and corrosponding output files
                wavfile_list = []
                new_wavfile_list = []
                
                for i in channels:
                    print(train_row[0])
                    wavfile_list.append(wav_directory + '\\hour' + recording_row[1][-2:] + i + '.wav')
                    new_wavfile_list.append(path + '\\train' + str(train_row[0]) + '_channel' + i + '.wav')

                #trim wav files to only contain a train
                for old_path, new_path in zip(wavfile_list, new_wavfile_list):
                    start = train_start - recording_start
                    end = start + train_end - train_start
                    trim_wav(old_path, new_path, start, end, sr=1000)
            
            """#elif train is partially contained in this hour - save what you can from the single hour
            elif (train_start > recording_start) and (train_start < recording_end) and (train_end > recording_end):"""

#this takes all train times and train recordings and outputs a single csv with all train/non-train data vectors in chronological order
def output_train_recordings_as_csv(train_times, recording_times, wav_directory, limit=2, sr=1000):
    #input times as pandas dataframe, no header

    #create directory for new wav files (folder called trains_only in current directory)
    """path = os.path.join(wav_directory, 'trains_only')
    if not os.path.exists(path):
        os.mkdir(path)"""

    #train times csv format: [train_id, loco, day, tunnel, direction, T1, T2, t1(s), t2(s), dt(s)] - lowercase times from midnight friday/saturday
    #recording times csv format: [start time (_ddmmyy_hhmmss.), hour number, start time (s) from midnight friday/saturday] - all durations 3600s

    #create list of channel strings
    #channels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
    channels = ['01']

    data = []

    #set count to limit output
    count = 0


    
    #iterate through train times
    for train_index, train_row in train_times.iterrows():
        train_start = train_row[7]
        train_end = train_row[8]
        
        if count < limit:
            #iterate through recording times
            for recording_index, recording_row in recording_times.iterrows():
                recording_start = recording_row[2]
                recording_end = recording_start + 3600
                
                #if train is contained in this hour, save as new train-only wav file
                if (train_start > recording_start) and (train_end < recording_end):
                    #create list of wav files where this train is recorded, and corrosponding output files
                    wavfile_list = []
                    new_wavfile_list = []
                    for i in channels:
                        wavfile_list.append(wav_directory + '\\hour' + recording_row[1][-2:] + i + '.wav')
                        new_wavfile_list.append('train' + str(train_row[0]) + '_channel' + i)
                    
                    #set previous recording start to nil originally (must be before any train times - check!)
                    previous_recording_end = -5000000
                    
                    #trim wav files to only contain a train
                    for old_path, new_path in zip(wavfile_list, new_wavfile_list):
                        start_time = train_start - recording_start
                        end_time = start_time + train_end - train_start
                        
                        #read wav file
                        wav_array, samplerate = librosa.load(old_path, sr=sr)

                        #trim to train only
                        train_start_in_recording = int(start_time * samplerate)
                        train_end_in_recording = int(end_time * samplerate)

                        #cut train and no-train times from data
                        ind_train_array = wav_array[train_start_in_recording:train_end_in_recording]
                        no_train_array_1 = wav_array[:train_start_in_recording]
                        no_train_array_2 = wav_array[train_end_in_recording:]

                        #need to check if last wav array 2 finished after the train to stop trains which are in an hour with another train from being recorded as no train
                        if train_start < previous_recording_end:
                            #must remove previous parts of recording that contain train and labelled as no-train (0)
                            
                            #set last no-train array to a variable
                            last_no_train_data = data[-1][0]
                            
                            #need to find parts of array we need to cut - difference between the previous train end and new train start
                            time_to_start_cut = samplerate*(train_start - previous_train_end)
                            #need to check train doesn't end beyond recording --- TO DO
                            time_to_end_cut = samplerate*(train_end - previous_train_end)
                            #cut down previous data to not contain any train
                            new_no_train_data_1 = last_no_train_data[:time_to_start_cut]
                            new_no_train_data_2 = last_no_train_data[time_to_end_cut:]
                            #now fix last input to database as 3 new entries, no-train/train/no-train
                            data[-1][0] = new_no_train_data_1
                            data.append([ind_train_array, 1, new_path])
                            data.append([new_no_train_data_2, 0, np.nan])
                        
                        else:
                            #append data to databases
                            data.append([no_train_array_1, 0, np.nan])
                            data.append([ind_train_array, 1, new_path])
                            data.append([no_train_array_2, 0, np.nan])

                
                
                elif (train_start > recording_start) and (train_end > recording_end):
                    
                    print('train recording overlaps recording files: train ' + str(train_row[0]))
                    
                else: 
                    continue

                previous_recording_end = recording_end

            previous_train_end = train_end
            count += 1 
        
        else: break

    return pd.DataFrame(data)

#this takes the ARRAY of a single hour and outputs the train/no-train times in chronological order as a pickle file
def single_hour_extraction_to_pickle(train_times_array, data_array, output_file):
    #must insert train times as csv with start and end only, headings as start and end
    #takes single hour at a time
    #outputs as a pickle
    #train is 1, no_train is 0

    #load the train times in the first hour
    train_times = pd.read_csv(train_times_array)

    #read the first hour - 1kHz samplerate
    time_signal, samplerate = librosa.load(data_array, sr=1000)

    #empty array which will contain the time spectrum and whether a train or not
    data = [[]]

    #set initial upper bound to nil
    previous_upper = 0

    #loop through train times and save them individually
    for start, end in zip(train_times['start'], train_times['end']):
        lower = samplerate * start 
        upper = samplerate * end   

        #isolate up to next train
        no_train = time_signal[previous_upper:lower]
        #append to final data array
        data.append([no_train, 0])

        #isolate the train signal
        train = time_signal[lower:upper]
        #append to final data array
        data.append([train, 1])

        #set previous upper
        previous_upper = upper

    #save the final no_train
    data.append([time_signal[upper:], 0])

    #save the data array as a panda and then as csv
    data_pandas = pd.DataFrame(data)
    data_pandas.to_pickle(output_file)

#output_to_pickle('hour0001_all_trains.csv', 'hour0001.wav', 'hour0001_trains.pkl')

#returns wavfile and accurate timings of recording
def extract_wavfile_accurate(wavfile, samplerate=1000):
    #load data
    data, sr = librosa.load(wavfile, sr=samplerate)
    #extract hour
    hour = int(wavfile[-8:-6])
    #hours
    hours = ['_060913_220655','_070913_000657','_070913_010731','_070913_020731','_070913_030731','_070913_040731','_070913_050731','_070913_060731','_070913_070731','_070913_080731','_070913_090731','_070913_100730','_070913_110730','_070913_120730','_070913_130730','_070913_140730','_070913_150730','_070913_160730','_070913_170730','_070913_180730','_070913_190730','_070913_200730','_070913_210730','_070913_220730','_070913_230730','_080913_000730','_080913_010730','_080913_020730','_080913_030730','_080913_040730','_080913_050730','_080913_060730','_080913_070730','_080913_080730','_080913_090729','_080913_100729','_080913_110729','_080913_120729','_080913_130729','_080913_140729','_080913_150729','_080913_160729','_080913_170729','_080913_180729','_080913_190729','_080913_200729','_080913_210729','_080913_220729','_080913_230729','_090913_000729','_090913_010729','_090913_020729','_090913_030729','_090913_040729']
    time = datetime.datetime.strptime(hours[hour], '_%d%m%y_%H%M%S')
    for i in range(0,len(data)-1):
        time.append(time[i] + pd.to_timedelta('1 ms'))    
    return data, time

#this extracts wavfile and returns the start time of recording in datetime
def extract_wavfile_and_date(wavfile, samplerate=1000):
    #load data
    data, sr = librosa.load(wavfile, sr=samplerate)
    #extract hour
    hour = int(wavfile[-8:-6])
    #hours
    hours = ['_060913_220655','_070913_000657','_070913_010731','_070913_020731','_070913_030731','_070913_040731','_070913_050731','_070913_060731','_070913_070731','_070913_080731','_070913_090731','_070913_100730','_070913_110730','_070913_120730','_070913_130730','_070913_140730','_070913_150730','_070913_160730','_070913_170730','_070913_180730','_070913_190730','_070913_200730','_070913_210730','_070913_220730','_070913_230730','_080913_000730','_080913_010730','_080913_020730','_080913_030730','_080913_040730','_080913_050730','_080913_060730','_080913_070730','_080913_080730','_080913_090729','_080913_100729','_080913_110729','_080913_120729','_080913_130729','_080913_140729','_080913_150729','_080913_160729','_080913_170729','_080913_180729','_080913_190729','_080913_200729','_080913_210729','_080913_220729','_080913_230729','_090913_000729','_090913_010729','_090913_020729','_090913_030729','_090913_040729']
    time = datetime.datetime.strptime(hours[hour], '_%d%m%y_%H%M%S')

    return data, time

