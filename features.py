from librosa.core import audio
import numpy as np
from librosa import feature
from scipy import signal
from octave_band import return_octave_filtered

"""
The below functions should be designed to take the input of a 
*pre-loaded* audio file, with samplerate and output a single feature
"""

"""class features:
    def __init__(self):
        """

###------------ZERO CROSSING RATE-----------###

def zero_crossing_rate(audio_data, samplerate=3200, frame_length=3200, hop_length=1600):
    output = []
    if len(audio_data) > frame_length:
        starts = range(0, len(audio_data)-frame_length, hop_length)
    else:
        starts = [0]
    for start in starts:
        end = start + frame_length
        zero_crosses = np.nonzero(np.diff(audio_data[int(start):int(end)] > 0))[0].size
        output.append(zero_crosses/(frame_length/samplerate))
    
    return output

###-------------SHORT-TIME ENERGY-----------###

def short_time_energy(audio_data, octave_band=None, samplerate=3200, frame_length=3200, hop_length=1600):
    output = []
    if len(audio_data) > frame_length:
        starts = range(0, len(audio_data)-frame_length, hop_length)
    else:
        starts = [0]
    if octave_band: audio_data=return_octave_filtered(audio_data, samplerate, octave_band)
    for start in starts:
        end = start + frame_length
        energy = np.sum([abs(x)**2 for x in audio_data[int(start):int(end)]])
        output.append(energy/frame_length)

    return output

###-----------MOST PROMINENT FREQUENCY-------###

def prominent_frequency(audio_data, frequency_step=20, samplerate=3200, frame_length=3200, hop_length=1600):
    output = []
    frequencies = range(1,(samplerate//2)-frequency_step,frequency_step)
    if len(audio_data) > frame_length:
        starts = range(0, len(audio_data)-frame_length, hop_length)
    else:
        starts = [0]
    for start in starts:
        max0 = 0
        maxf = 0
        end = start + frame_length
        data = audio_data[int(start):int(end)]
        for frequency in frequencies:
            b, a = signal.butter( N=4, Wn=np.array([frequency, frequency+frequency_step])/(samplerate/2), btype='bandpass', analog=False, output='ba')
            data = abs(signal.filtfilt(b, a, data))
            max1 = max(data)
            if max1 > max0:
                max0 = max1
                maxf = frequency + frequency_step//2
        output.append(maxf)

    return output

###------------------RMS------------------###

def RMS(audio_data, frame_length=3200, hop_length=1600):
    output = []
    if frame_length:
        if len(audio_data) > frame_length:
            starts = range(0, len(audio_data)-frame_length, hop_length)
        else:
            starts = [0]

        for start in starts:
            end = start + frame_length
            rms = np.sqrt(np.sum([x**2 for x in audio_data[start:end]]))
            output.append(rms)
            
    else: output = np.sqrt(np.sum([x**2 for x in audio_data]))
    return output

###--------------Pulse Duration-----------###

def pulse_duration(audio_data, frame_length=3200, hop_length=1600):
    output = []
    if frame_length:
        if len(audio_data) > frame_length:
            starts = range(0, len(audio_data)-frame_length, hop_length)
        else:
            starts = [0]
        for start in starts:
            end = start + frame_length
            maxx = max(abs(audio_data[start:end]))
            halfpower = maxx / np.sqrt(2)
            for i,j in enumerate(audio_data[start:end]):
                if j > halfpower:
                    first = i
                    break
            for i,j in enumerate(audio_data[start:end][::-1]):
                if j > halfpower:
                    last = i
                    break
        output.append(last-first)

    else:
        for i,j in enumerate(audio_data):
            if j > halfpower:
                first = i
                break
        for i,j in enumerate(audio_data[::-1]):
            if j > halfpower:
                last = i
                break

    return output

###--------------Peak Value---------------###

def peak_value(audio_data, octave_band=None, samplerate=3200, frame_length=3200, hop_length=1600):
    output = []
    if len(audio_data) > frame_length:
        starts = range(0, len(audio_data)-frame_length, hop_length)
    else:
        starts = [0]
    if octave_band: audio_data=return_octave_filtered(audio_data, samplerate, octave_band)
    for start in starts:
        end = start + frame_length
        output.append(max(audio_data[start:end]))

    return output

###------------Spectral Centroid------------###

def spectral_centroid(audio_data, samplerate=3200, frame_length=3200, hop_length=1600):
    output = []
    if frame_length:
        if len(audio_data) > frame_length:
            starts = range(0, len(audio_data)-frame_length, hop_length)
        else:
            starts = [0]
        for start in starts:
            end = start + frame_length
            centroid = feature.spectral_centroid(audio_data[start:end],sr=samplerate,n_fft=len(audio_data[start:end]),hop_length=len(audio_data[start:end]))
            output.append(centroid[0][0])
    else: output = feature.spectral_centroid(audio_data,sr=samplerate,n_fft=len(audio_data),hop_length=len(audio_data))

    return output

###------------Spectral Roll-off------------###

def spectral_roll_off(audio_data, samplerate=3200, frame_length=3200, hop_length=1600, n_fft=256, fft_hop_length=64):
    output = []
    if frame_length:
        if len(audio_data) > frame_length:
            starts = range(0, len(audio_data)-frame_length, hop_length)
        else:
            starts = [0]
        for start in starts:
            end = start + frame_length
            rolloff = feature.spectral_rolloff(audio_data[start:end],sr=samplerate,n_fft=len(audio_data[start:end]),hop_length=len(audio_data[start:end]))
            output.append(rolloff[0][0])
    else: output = feature.spectral_rolloff(audio_data,sr=samplerate,n_fft=len(audio_data),hop_length=len(audio_data))

    return output
