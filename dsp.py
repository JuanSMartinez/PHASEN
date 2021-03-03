'''
DSP module for the implementation of PHASEN
Author: Juan S. Martinez
Date: Spring 2021
'''
from scipy.io.wavfile import read as wavread
from scipy.signal import resample_poly, stft, check_NOLA
import numpy as np
import math
import sys

# Global variables according to the original paper by Yin et al. (2020)
audio_fs = int(16e3)
hann_win_length = 0.025 # 25 ms
hop_length = 0.01 # 10 ms
fft_size = 512

def read_audio_from(path):
    '''
    Read the audio file from a path
    :param: path. The path to the audio file
    :return: fs, data. The sampling rate and the data as a floating-point number array
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


def resample_signal(data, old_fs, target_fs):
    '''
    Resample a signal in the data arry to a target sampling frequency
    :param : data: np.array. The signal to be resampled
    :param : old_fs: int. The original sampling frequency
    :param : target_fs: int. The target sampling frequency
    :return : resampled. np.array. The resampled signal
    '''

    g = math.gcd(old_fs, target_fs)
    up = target_fs//g
    down = old_fs//g
    # This uses the default FIR low pass filter from scipy, which uses a kaiser window 
    resampled = resample_poly(data, up, down)
    return resampled


def get_stft_spectrogram(data, fs):
    '''
    Compute the spectrogram of the signal in the data array via the STFT
    :param : data. np.array. The signal 
    :param : fs. int. Sample rate of the signal
    :return : spec. Spectogram of the signal
    '''

    # According to the paper, the spectrogram is computed using a Hann window with a length of 25 ms,
    # a hop length of 10 ms and FFt size of 512
    
    # I believe the length of each segment is the hann window length
    n_per_seg = int(hann_win_length*fs)

    # The hop size H = n_per_seg - n_overlap according to scipy 
    n_hop_size = int(hop_length*fs)
    n_overlap = n_per_seg - n_hop_size

    # Compute STFT if the nonzero overlap add constraint is satisfied
    if check_NOLA('hann', n_per_seg, n_overlap):
        f, t, Zxx = stft(data, fs, window='hann', nperseg=n_per_seg, noverlap=n_overlap, nfft=fft_size) 
        # Return the complex spectrogram
        return f, t, Zxx
    else:
        raise Exception("The nonzero overlap constraint was not met while computing a STFT")





