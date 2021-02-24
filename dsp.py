'''
DSP module for the implementation of PHASEN
Author: Juan S. Martinez
Date: Spring 2021
'''
from scipy.io.wavfile import read as wavread
from scipy.signal import resample
import numpy as np
import sys

# Global variables according to the original paper by Yin et al. (2020)
audio_fs = int(16e3)
hann_win_length = 0.025 # 25 ms
hop_length = 0.01 # 10 ms
fft_size = 512

def read_audio_from(path):
    '''
    Read the audio file from a path
    :param path. The path to the audio file
    :return fs, data. The sampling rate and the data as a floating-point number array
    '''
    # Read the data first
    fs, data = wavread(path)

    # Turn into floating point numbers between -1 and 1
    if data.dtype == 'int16':
        n_bits = 16
    elif data.dtype == 'int32':
        n_bits = 32
    elif data.dtype == 'uint8':
        sys.exit("ERROR: Unexpected type of audio data")
    else: 
        n_bits = -1

    if n_bits > 0:
        
        try:
            # If stereo, we unpack two values
            samples, channels = data.shape
            data = data.astype(float)
            data[:,0] = data[:,0]/(2.0**(n_bits-1)+1)
            data[:,1] = data[:,1]/(2.0**(n_bits-1)+1)
            data = data.sum(axis=1)/2.0
        except ValueError:
            # We have a mono file 
            data = data.astype(float).flatten()/(2.0**(n_bits-1)+1)

    return fs, data


