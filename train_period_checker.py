#intended for use in a notebook, requires the input of judged train timings, will output the judged train timing windows
import pandas as pd
import numpy as np 
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import spectrogram
import datetime
from wavfile_manipulations import extract_wavfile_and_date

def max_pooling(array, pool_size, pool_overlap=(0,0)):
    if pool_overlap:
        if (pool_overlap[0] >= pool_size[0]) and (pool_overlap[1] >= pool_size[1]):
            print('pool_overlap must be less than pool_size')
            return

    output = []
    for i in range(0, array.shape[0]-pool_size[0]+pool_overlap[0], pool_size[0]-pool_overlap[0]):
        row = []
        for j in range(0, array.shape[1]-pool_size[1]+pool_overlap[1], pool_size[1]-pool_overlap[1]):
            maximum = max(np.array(array[i:i+pool_size[0], j:j+pool_size[1]]).flatten())
            row.append(maximum)
        output.append((row))
    return output
 

def train_windows(hours=['00','01','02','03','04','05','06','07','08','09','10','11','12','13'], train_line='cen_eas', wavfile_root='train_recordings/', timetables_root='judged_timetables/', samplerate=1000, nperseg=100, noverlap=0, nfft=10000, pool_size=(5,5), pool_overlap=(1,1), vmin=1e-8, vmax=5e-7, ylim=(50,300)):
    """
    Function requires user input, will be shown surrounding signal to an expected train. Input window start and finish when prompted, input None if incorrect, or input 'min' followed by new min - e.g. 'scale 1e-7' to change scale.
    input   - wavfile - path to wavfile
            - judged_timings - requires path to csv file containing single values of a single train in wavfile
    output  - nothing
    """
    #empty arrays for judgements
    judged_window_start = []
    judged_window_end = []

    for hour in hours:                
        #load data
        single_times = pd.read_csv(timetables_root+'judged_timetable_hour'+hour+'.csv', delimiter=',')[train_line]
        data, date = extract_wavfile_and_date(wavfile_root+'hour'+hour+'01.wav', samplerate=samplerate)

        status = 'okay'

        for time in single_times:
            try:    time=int(time)
            except: continue
            start = time - 15
            end = time + 15

            #plot spectrogram
            fig = plt.figure(figsize=(12,6))
            f0, t0, Sxx0 = spectrogram(data[start*samplerate:end*samplerate], fs=samplerate, window='hamming', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            pooled = max_pooling(Sxx0, pool_size=pool_size, pool_overlap=pool_overlap)
            if ylim:
                lim = [(ylim[0]*len(pooled))//500, (ylim[1]*len(pooled))//500]
                pooled = pooled[lim[0]:lim[1]]
            else:
                ylim = [0,500]

            f = np.linspace(ylim[0],ylim[1],len(pooled))
            t = np.linspace(0,30,len(pooled[0]))
            plt.pcolormesh(t, f, pooled, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            plt.xticks(range(0,30))
            plt.grid(b=True, axis='x', color='white')

            #show plot
            plt.show()
            print('Train time: '+str(time))
            print('Train line: '+train_line)

            #request user input
            print('When does the train start?')
            judged_start = input('Enter start: ')
            print('When does the train end?')
            judged_end = input('Enter end: ')

            try:
                judged_start = int(judged_start)
                judged_end = int(judged_end)
            except:
                pass

            if isinstance(judged_start, int) and isinstance(judged_end, int):
                judged_start = int(judged_start)
                judged_end = int(judged_end)

                #append judged windows to array
                judged_window_start.append(date+datetime.timedelta(seconds=time+judged_start))
                judged_window_end.append(date+datetime.timedelta(seconds=time+judged_end))
            elif (judged_start == 'exit') or (judged_end == 'exit'):
                status = 'exit'
                break

        if status == 'exit':
            break

    #pusblish windows array to csv
    pd.DataFrame(zip(judged_window_start, judged_window_end), columns=['start','end']).to_csv(timetables_root+'judged_windows_'+train_line+'1.csv', index=False)
    print('published to: '+timetables_root+'judged_windows_'+train_line+'1.csv')