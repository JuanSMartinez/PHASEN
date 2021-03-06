'''
Custom dataset classes to handle the datasets in PyTorch
Author: Juan S. Martinez
Date: Spring 2021
'''

import os
from torch.utils.data import Dataset
import torch
import dsp


class AVSpeechAudioSet(Dataset):

    def __init__(self, avspeech_path, audioset_path, set_type):
        super(AVSpeechAudioSet, self).__init__()
        if set_type == 'train' or set_type == 'test':
            try:
                avspeech_root = avspeech_path + '/' + set_type + '/'
                audioset_root = audioset_path + '/' + set_type + '/'
                self.avspeech_files = [avspeech_root + p for p in os.listdir(avspeech_root) if not p.startswith('.')]
                self.audioset_files = [audioset_root + p for p in os.listdir(audioset_root) if not p.startswith('.')]
                self.set_type = set_type

                if len(self.avspeech_files) != len(self.audioset_files):
                    raise Exception("The number of files for AVSpeech and AudioSet are inconsistent. "\
                            "Please verify that the scripts under each dataset folder agree in the number of videos to download")
            except:
                raise Exception("No files found in the datasets folder. Make sure to run the scripts in 'datasets' first")
        else:
            raise Exception("Invalid dataset type. Choose 'train' or 'test'")


    def __len__(self):
        return len(self.avspeech_files)

    def __getitem__(self, idx):
        avspeech_file = self.avspeech_files[idx]
        audioset_file = self.audioset_files[idx]
        fs_speech, speech = dsp.read_audio_from(avspeech_file)
        fs_noise, noise = dsp.read_audio_from(audioset_file)

        # Resample both speech and noise to 16 kHz
        re_speech = dsp.resample_signal(speech, fs_speech, dsp.audio_fs)
        re_noise = dsp.resample_signal(noise, fs_noise, dsp.audio_fs)

        # Clip all audio data to 3 seconds
        clipped_speech = dsp.clip_audio(dsp.audio_fs, re_speech, 3)
        clipped_noise = dsp.clip_audio(dsp.audio_fs, re_noise, 3)

        # Combine the signals as in the original paper

        # In case one of the signals is shorter than 3 seconds, padd with zeros
        max_len = int(3*dsp.audio_fs)
        if len(clipped_speech) < max_len:
            clipped_speech = dsp.padd_signal(clipped_speech, max_len - len(clipped_speech))
        if len(clipped_noise) < max_len:
            clipped_noise = dsp.padd_signal(clipped_noise, max_len - len(clipped_noise))

        mix = clipped_speech + 0.3*clipped_noise

        # Compute spectrograms
        f_mix, t_mix, mix_spec = dsp.get_stft_spectrogram(mix, dsp.audio_fs)
        f_truth, t_truth, ground_truth_spec = dsp.get_stft_spectrogram(clipped_speech, dsp.audio_fs)

        # The spectrograms have to be shaped as (2, T, F), but the output of dsp is a complex array of shape (F,T)
        mix_spec_t = torch.tensor(mix_spec).T
        ground_truth_spec_t = torch.tensor(ground_truth_spec).T

        mix_t = torch.cat((mix_spec_t.real.unsqueeze(0), mix_spec_t.imag.unsqueeze(0)),dim=0)
        ground_truth_t = torch.cat((ground_truth_spec_t.real.unsqueeze(0), ground_truth_spec_t.imag.unsqueeze(0)),dim=0)

        return torch.tensor(clipped_speech), torch.tensor(mix), ground_truth_t, mix_t
