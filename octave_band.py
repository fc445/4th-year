import datetime
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import librosa
from expected_train_times import *
from timetables import trains_in_period
from wavfile_manipulations import extract_wavfile_and_date

def moving_average(x, w):
    y_padded = np.pad(x, (w//2, w-1-w//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((w,))/w, mode='valid')
    return y_smooth

def find_octave_band(octave_band, base=2):
    #dictionary of third-octave bands up to 20kHz, with associated band number
    dictionary = {16:1, 20:2, 25:3, 31.5:4, 40:5, 50:6, 63:7, 80:8, 100:9, 125:10, 160:11, 200:12, 250:13, 315:14, 400:15, 500:16, 630:17, 800:18, 1000:19, 1250:20, 1600:21, 2000:22, 2500:23, 3150:24, 4000:25, 5000:26, 6300:27, 8000:28, 10000:29, 12500:30, 16000:31, 20000:32}
    
    #calculation using base 2
    if base==2:
        i = dictionary[octave_band] - 19
        fcentre = 10**3 * 2**(i/3)
        fd = 2**(1/6)
        fupper = fcentre * fd
        flower = fcentre / fd

    #calculation using base 10
    elif base==10:
        i = dictionary[octave_band] + 11
        fcentre = 10**(0.1*i)
        fd = 10**0.05
        fupper = fcentre * fd
        flower = fcentre / fd

    #return boundaries for given octave band
    return flower, fupper

def return_octave_filtered(audio_data, samplerate=1000, octave_band=125):
    flower, fupper = find_octave_band(octave_band)
    b, a = scipy.signal.butter( N=4, Wn=np.array([ flower, fupper])/(samplerate/2), btype='bandpass', analog=False, output='ba')
    filtered = scipy.signal.filtfilt(b, a, audio_data)
    return filtered

def plot_signal_octave_filter(data_bands, samplerate, octave_band):

    flower, fupper = find_octave_band(octave_band)

    b, a = scipy.signal.butter( N=4, Wn=np.array([ flower, fupper])/(samplerate/2), btype='bandpass', analog=False, output='ba')

    for i in data_bands:
        filtered = scipy.signal.filtfilt(b, a, i)
        plt.plot(abs(filtered))
    
    plt.show()

#data1, samplerate1 = librosa.load('train_recordings/hour0101.wav', sr=1000)
#data2, samplerate2 = librosa.load('train37_channel01.wav', sr=1000)
#data3, samplerate3 = librosa.load('train32_channel01.wav', sr=1000)

#plot_signal_octave_filter([data1], samplerate1, 125)

def plot_octave(input_data, bands, samplerate=1000, lines=None, judged_lines=None, start_time=None, ma=None, show_type='test', save_as=None, ymin=None, ymax=None, ticks=False):
    """
    INPUT   - data - bands of data from librosa load or in wavfile string form
            - samplerate - samplerate of loaded datastream
            - bands - bands of octave bands to plot, or single int for one band
            - lines (optional) - train lines to plot
            - start_time (only required if data is bands and lines) - start of period required for plotting trains
            - ma - moving_average (optional) - if included, will apply moving average to the specfied length
            - show_type (optional) - if 'test' will print small for testings, if 'document' will print high quality for documentation

    OUTPUT - returns nothing, will show plot
    """
    #colour bands for plotting lines
    line_colours = {'bak_nor': 'sienna', 'bak_sou': 'peru', 'cen_eas': 'orange', 'cen_wes': 'red', 'vic_nor': 'blue', 'vic_sou': 'dodgerblue'}

    #load data
    if isinstance(input_data, str):
        data, start_time = extract_wavfile_and_date(input_data, samplerate)
    else:
        data = input_data


    ###----------IF LIST OF OCTAVE BANDS--------------###
    if isinstance(bands, list):
        if show_type == 'test':
            fig, ax = plt.subplots(len(bands),1, sharex=True, sharey=True, figsize=[8,4.5])
        elif show_type == 'document':
            fig, ax = plt.subplots(len(bands),1, sharex=True, sharey=True, figsize=[16,9], dpi=300)

        for i,j in enumerate(bands):
            flower, fupper = find_octave_band(j)
            b, a = scipy.signal.butter( N=4, Wn=np.array([flower, fupper])/(samplerate/2), btype='bandpass', analog=False, output='ba')
            filtered = abs(scipy.signal.filtfilt(b, a, data))
            time = np.linspace(0,len(filtered)/samplerate, len(filtered))
            #plot signal 
            if ma: ax[i].plot(time, moving_average(filtered, ma), color='#000000', label=j, linewidth=1)
            else: ax[i].plot(time, filtered, color='#000000', label=j, linewidth=1)
            ax[i].legend(loc='upper right')
            if ymax: ax[i].set_ylim(ymin, ymax)
            #plot train times if lines and wavfile given
            #get duration
            duration = len(data)//samplerate
            if lines: 
                if isinstance(input_data, str):
                    train_timetables = trains_in_period(lines=lines, start_time=start_time, duration='full')
                    for train_line in train_timetables:
                        for train_time in train_timetables[train_line]:
                            ax[i].axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)
            #plot train times if lines and data given 
                else:
                    train_timetables = trains_in_period(lines=lines, start_time=start_time, duration=duration)
                    for train_line in train_timetables:
                        for train_time in train_timetables[train_line]:
                            ax[i].axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)

            if judged_lines: 
                if isinstance(input_data, str):
                    judged_timetables = pd.read_csv('judged_timetables/judged_timetable_hour'+str(input_data[-8:-6])+'.csv', delimiter=',')
                    for train_line in judged_lines:
                        for train_time in judged_timetables[train_line]:
                            ax[i].axvline(x=train_time, color=line_colours[train_line], lw=1.25)
            #plot train times if lines and data given 
                else:
                    #todo: 
                    pass

    ###----------IF SINGLE OCTAVE BAND-----------###
    elif isinstance(bands, int):
        if show_type == 'test':
            fig = plt.figure(figsize=[8,4.5])
        elif show_type == 'document':
            fig = plt.figure(figsize=[16,9], dpi=300)

        flower, fupper = find_octave_band(bands)
        b, a = scipy.signal.butter( N=4, Wn=np.array([flower, fupper])/(samplerate/2), btype='bandpass', analog=False, output='ba')
        filtered = abs(scipy.signal.filtfilt(b, a, data))
        time = np.linspace(0,len(filtered)/samplerate, len(filtered))
        #plot signal 
        if ma: plt.plot(time, moving_average(filtered, ma), color='#000000', label=bands, linewidth=1)
        else: plt.plot(time, filtered, color='#000000', label=bands, linewidth=1)
        plt.legend(loc='upper right')
        if ymax: plt.ylim(ymin, ymax)
        #if ticks
        if ticks:
            ticks = range(0,len(data)//samplerate, 20)
            plt.xticks(ticks)
            plt.grid()
        #plot train times if lines and wavfile given
        #get duration
        duration = len(data)//samplerate
        if lines: 
            if isinstance(input_data, str):
                train_timetables = trains_in_period(lines=lines, start_time=start_time, duration='full')
                for train_line in train_timetables:
                    for train_time in train_timetables[train_line]:
                        plt.axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)
        #plot train times if lines and data given 
            else:
                train_timetables = trains_in_period(lines=lines, start_time=start_time, duration=duration)
                for train_line in train_timetables:
                    for train_time in train_timetables[train_line]:
                        plt.axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)

        if judged_lines: 
            if isinstance(input_data, str):
                judged_timetables = pd.read_csv('judged_timetables/judged_timetable_hour'+str(input_data[-8:-6])+'.csv', delimiter=',')
                for train_line in judged_lines:
                    for train_time in judged_timetables[train_line]:
                        plt.axvline(x=train_time, color=line_colours[train_line], lw=1.25)
        #plot train times if lines and data given 
            else:
                #todo: 
                pass

    if save_as: plt.savefig(save_as)

    plt.show()

def plot_custom_bands(input_data, bands, samplerate=1000, lines=None, judged_lines=None, start_time=None, plotting_type=None, ma=None, show_type='test', save_as=None, ymin=None, ymax=None, vmin=None, vmax=None):
    """
    INPUT   - data - bands of data from librosa load
            - samplerate - samplerate of loaded datastream
            - bands - bands of octave bands to plot
            - lines (optional) - train lines to plot
            - plotting_type (required if lines) - 'judged' or 'predicted'
            - ma - moving_average (optional) - if included, will apply moving average to the specfied length
            - show_type (optional) - if 'test' will print small for testings, if 'document' will print high quality for documentation

    OUTPUT - returns nothing, will show plot
    """
    #colour bands for plotting lines
    line_colours = {'bak_nor': 'sienna', 'bak_sou': 'peru', 'cen_eas': 'orange', 'cen_wes': 'red', 'vic_nor': 'blue', 'vic_sou': 'dodgerblue'}

    #load data
    if isinstance(input_data, str):
        data, start_time = extract_wavfile_and_date(input_data, samplerate)
    else:
        data = input_data

    if show_type == 'test':
        fig, ax = plt.subplots(len(bands),1, sharex=True, sharey=True, figsize=[8,4.5])
    elif show_type == 'document':
        fig, ax = plt.subplots(len(bands),1, sharex=True, sharey=True, figsize=[16,9], dpi=300)
    
    for i,j in enumerate(bands):
        flower, fupper = j
        b, a = scipy.signal.butter( N=4, Wn=np.array([flower, fupper])/(samplerate/2), btype='bandpass', analog=False, output='ba')
        filtered = abs(scipy.signal.filtfilt(b, a, data))
        time = np.linspace(0,len(filtered)/samplerate, len(filtered))
        averaged = moving_average(filtered, ma)

        if vmin: averaged = [0 if x < vmin[i] else x for x in averaged]
        if vmax: averaged = [0 if x > vmax[i] else x for x in averaged]

        #plot signal 
        if ma: ax[i].plot(time, averaged, color='#000000', label=j, linewidth=1)
        else: ax[i].plot(time, filtered, color='#000000', label=j, linewidth=1)
        ax[i].legend(loc='upper right')
        
        if ymax: ax[i].set_ylim(ymin, ymax)
        #plot train times if lines and wavfile given
        #get duration
        duration = len(data)//samplerate
        if lines: 
            if isinstance(input_data, str):
                train_timetables = trains_in_period(lines=lines, start_time=start_time, duration='full')
                for train_line in train_timetables:
                    for train_time in train_timetables[train_line]:
                        ax[i].axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)
        #plot train times if lines and data given 
            else:
                train_timetables = trains_in_period(lines=lines, start_time=start_time, duration=duration)
                for train_line in train_timetables:
                    for train_time in train_timetables[train_line]:
                        ax[i].axvline(x=train_time, color=line_colours[train_line], alpha=0.9, ls='dashed', lw=0.75)

        if judged_lines: 
            if isinstance(input_data, str):
                judged_timetables = pd.read_csv('judged_timetables/judged_timetable_hour'+str(input_data[-8:-6])+'.csv', delimiter=',')
                for train_line in judged_lines:
                    for train_time in judged_timetables[train_line]:
                        ax[i].axvline(x=train_time, color=line_colours[train_line], lw=1.25)
        #plot train times if lines and data given 
            else:
                #todo: 
                pass

    if save_as: plt.savefig(save_as)

    plt.show()

#create a function to extract timings of trains, based on peaks in the 125Hz octave band only
#it must define between two trains, even if no zero in between
#decide on how much time to extract per train
#is there a minimum duration for a passing train
#def extract_train_timings(wav_file_directory, )




