'''
DSP module for the implementation of PHASEN
Author: Juan S. Martinez
Date: Spring 2021
'''
from scipy.io.wavfile import read as wavread
from scipy.signal import resample_poly, stft, check_NOLA, istft
import numpy as np
import torch
import math
import sys

# Global variables according to the original paper by Yin et al. (2020)
audio_fs = int(16e3)
hann_win_length = 0.025 # 25 ms
hop_length = 0.01 # 10 ms
fft_size = 512
K = 10
C = 0.1

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

def compute_cIRM_from(S, Y):
    '''
    Compute the cIRM from tensors of noisy spectrogram Y and clean speech
    spectrogram S. The computation is done as in: Williamson, D. S., 
    Wang, Y., & Wang, D. (2015). Complex ratio masking for monaural speech
    separation. IEEE/ACM transactions on audio, speech, and language processing,
    24(3), 483-492.
    :param: S. A tensor that represents the clean speech.
            The shape must be (*,2,T,F)
    :param: Y. A tensor that represents the noisy speech
            The shape must be (*,2,T,F)
    :return: M. A tensor that represents the cIRM
    '''
    N, Ch, T, F = S.shape
    M = torch.zeros(N, Ch, T, F)
    M[:,0,:,:] = torch.div(Y[:,0,:,:]*S[:,0,:,:] + Y[:,1,:,:]*S[:,1,:,:],
                            Y[:,0,:,:]**2 + Y[:,1,:,:]**2)
    M[:,1,:,:] = torch.div(Y[:,0,:,:]*S[:,1,:,:] - Y[:,1,:,:]*S[:,0,:,:],
                            Y[:,0,:,:]**2 + Y[:,1,:,:]**2)
    return M

def compress_cIRM(M):
    '''
    Compress the cIRM tensor M.
    The computation is done as in: Williamson, D. S., Wang, Y., & Wang, D. (2015).
    Complex ratio masking for monaural speech separation.
    IEEE/ACM transactions on audio, speech, and language processing,
    24(3), 483-492.
    :param: M. A tensor that represents the cIRM.
            The shape must be (*,2,T,F)
    '''
    cM = M.clone()
    cM[:,0,:,:] = K*torch.div(1 - torch.exp(-C*M[:,0,:,:]),
                            1 + torch.exp(-C*M[:,0,:,:]))
    cM[:,1,:,:] = K*torch.div(1 - torch.exp(-C*M[:,1,:,:]),
                            1 + torch.exp(-C*M[:,1,:,:]))
    return cM

def decompress_cIRM(cM):
    '''
    Decompress the cIRM tensor cM.
    The computation is done as in: Williamson, D. S., Wang, Y., & Wang, D. (2015).
    Complex ratio masking for monaural speech separation.
    IEEE/ACM transactions on audio, speech, and language processing,
    24(3), 483-492.
    :param: cM. A tensor that represents the compressed cIRM.
            The shape must be (*,2,T,F)
    '''
    M = cM.clone()
    M[:,0,:,:] = (-1/C)*torch.log(torch.div(K - cM[:,0,:,:],
                                K + cM[:,0,:,:]))
    M[:,1,:,:] = (-1/C)*torch.log(torch.div(K - cM[:,1,:,:],
                                K + cM[:,1,:,:]))
    return M


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


def recover_from_stft_spectrogram(Zxx, fs):
    '''
    Recover the time-domain signal from a spectrogram via the inverse STFT
    :param : Zxx. np.array. The complex spectrogram
    :param : fs. int. Sample rate of the signal
    :return : data. time-domain signal
    '''

    # According to the paper, the spectrogram is computed using a Hann window with a length of 25 ms,
    # a hop length of 10 ms and FFt size of 512

    # I believe the length of each segment is the hann window length
    n_per_seg = int(hann_win_length*fs)

    # The hop size H = n_per_seg - n_overlap according to scipy
    n_hop_size = int(hop_length*fs)
    n_overlap = n_per_seg - n_hop_size

    # Compute inverse STFT if the nonzero overlap add constraint is satisfied
    if check_NOLA('hann', n_per_seg, n_overlap):
        t, data = istft(Zxx, fs,  window='hann', nperseg=n_per_seg, noverlap=n_overlap, nfft=fft_size)
        return t, data
    else:
        raise Exception("The nonzero overlap constraint was not met while computing an inverse STFT")


def clip_audio(fs, audio_data, max_length):
    '''
    Clip audio data to a max_length in seconds
    :param: fs. Sampling frequency of the audio data
    :param: audio_data. Mono audio data as a floating point array
    :max_length: maximum length to clip to. A value in seconds
    :return: clipped signal
    '''

    end_sample = int(max_length*fs)
    # A 'hard-clip' approach of truncating the signal
    return audio_data[0:end_sample]

def padd_signal(signal, extra_zeros):
    '''
    Pad a signal with a number of extra zeros
    :param: signal. Numpy array to be padded
    :param: extra_zeros. Number of additional zeros to add
    :return: padded signal.
    '''
    n_signal = np.zeros(len(signal)+extra_zeros)
    n_signal[0:len(signal)] = signal
    return n_signal
