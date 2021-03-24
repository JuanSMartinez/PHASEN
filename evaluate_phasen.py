'''
Evaluation of the PHASEN network
Author: Juan S. Martinez
Date: Spring 2021
'''

import argparse
import sys
import torch
import os
import phasen
import numpy as np
import dsp
import custom_datasets
from torch.autograd import Variable
import torch.nn.functional as F
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from scipy.io.wavfile import write as wavwrite

torch.autograd.set_detect_anomaly(True)
#torch.backends.cudnn.enabled=False
# --------------- Global variables --------------------------------------------#

# Available networks to perform ablation studies
networks = ['phasen']

# Argument parser to run the script
parser = argparse.ArgumentParser(description='Evaluate the PHASEN network')
parser.add_argument('operation', type=str, help='Type of operation on a network. Either "train" or "test"')
parser.add_argument('net', type=str, help='Type of network to evaluate. Choices: ' + ','.join(networks))
parser.add_argument('dataset', type=str, help='Dataset to train or test. Choices: "avspeech_audioset"')

# Training configuration
training_config = {
    'epochs': 50,
    'learning_rate': 2e-4,
    'batch_size': 5
}

# Random seed
seed = 930103
torch.manual_seed(seed)
np.random.seed(seed)

# -----------------------------------------------------------------------------#

class ComplexMSELoss(torch.nn.Module):
    def __init__(self, device, p=0.3):
        super(ComplexMSELoss, self).__init__()
        self.p = p
        self.device = device

    def pw_compress_spectrogram(self, spec):
        '''
        Compute the power-law compressed spectrogram on amplitude of a complex
        spectrogram
        '''
        # mag_spec = torch.abs(spec)
        # return mag_spec**self.p * spec
        mag_spec = torch.sqrt(spec[:,0,:,:]**2 + spec[:,1,:,:]**2)
        mag_spec = torch.pow(mag_spec.unsqueeze(1).repeat(1,2,1,1), self.p)
        return mag_spec * spec

    def forward(self, spec_in, spec_out):
        # N, T, Fd = spec_in.shape
        #
        # comp_spec_in = self.pw_compress_spectrogram(spec_in)
        # comp_spec_out = self.pw_compress_spectrogram(spec_out)
        #
        # comp_split_spec_in = torch.zeros(N,2,T,Fd)
        # comp_split_spec_out = torch.zeros(N,2,T,Fd)
        # comp_split_spec_in[:,0,:,:] = comp_spec_in.real
        # comp_split_spec_in[:,1,:,:] = comp_spec_in.imag
        # comp_split_spec_out[:,0,:,:] = comp_spec_out.real
        # comp_split_spec_out[:,1,:,:] = comp_spec_out.imag
        #
        # comp_split_spec_in = Variable(comp_split_spec_in,
        #                             requires_grad=True).to(self.device)
        # comp_split_spec_out = Variable(comp_split_spec_out,
        #                             requires_grad=True).to(self.device)
        #
        # mag_comp_spec_in = Variable(torch.abs(comp_spec_in),
        #                             requires_grad=True).to(self.device)
        # mag_comp_spec_out = Variable(torch.abs(comp_spec_out),
        #                             requires_grad=True).to(self.device)
        #
        # return 0.5*F.mse_loss(mag_comp_spec_in, mag_comp_spec_out) +\
        #         0.5*F.mse_loss(comp_split_spec_in, comp_split_spec_out)

        comp_spec_in = self.pw_compress_spectrogram(spec_in)
        comp_spec_out = self.pw_compress_spectrogram(spec_out)
        mag_spec_in = torch.sqrt(comp_spec_in[:,0,:,:]**2 + comp_spec_in[:,1,:,:]**2)
        mag_spec_out = torch.sqrt(comp_spec_out[:,0,:,:]**2 + comp_spec_out[:,1,:,:]**2)
        return 0.5*F.mse_loss(mag_spec_in, mag_spec_out) +\
                 0.5*F.mse_loss(comp_spec_in, comp_spec_out)




def find_device():
    '''
    Find the best device to perform operations
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_net_of_type(net_type):
    if net_type == 'phasen':
        return phasen.PHASEN()
    else:
        return None

def create_dataset_for(dataset_name, operation):
    if dataset_name == 'avspeech_audioset':
        dataset = custom_datasets.AVSpeechAudioSet('datasets/AVSpeech',
                                                    'datasets/AudioSet',
                                                    operation)
    else:
        dataset = None
    return dataset


def test(device, net_type, model_path, dataset):
    net = create_net_of_type(net_type)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    dataset = create_dataset_for(dataset, 'test')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=2)

    metrics = np.zeros((len(dataset), 3))
    i=0
    for fm, tm, sm, clean_speech, noisy_speech in loader:
        s_out, M, Phi = net(sm.float())
        clean_speech = clean_speech.numpy().flatten()
        noisy_speech = noisy_speech.numpy().flatten()
        # Convert s_out from (1, 2, T, F) to a complex array of shape (F, T)
        s_out = s_out.squeeze(0)
        C, T, F = s_out.shape
        sout_c = torch.zeros(T, F, dtype=torch.cfloat)
        sout_c.real = s_out[0,:,:]
        sout_c.imag = s_out[1,:,:]
        sout_c = sout_c.T.detach().numpy()

        # Recover time domain signal
        t, recovered_speech = dsp.recover_from_stft_spectrogram(sout_c, dsp.audio_fs)
        PESQ = pesq(dsp.audio_fs, clean_speech, recovered_speech, 'wb')
        STOI = stoi(clean_speech, recovered_speech, dsp.audio_fs, extended=False)
        SDR, sir, sar, perm = bss_eval_sources(clean_speech, recovered_speech)
        metrics[i,0] = SDR[0]
        metrics[i,1] = PESQ
        metrics[i,2] = STOI
        i += 1
    np.save(net_type + '_metrics.npy', metrics)
    print("Finished testing of net '{}', metrics saved in {}_metrics.npy".format(net_type, net_type))


def train(device, net_type, save_path, dataset):
    net = create_net_of_type(net_type)
    net = net.to(device)
    dataset = create_dataset_for(dataset, 'train')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=training_config['batch_size'],
                                        shuffle=True,
                                        num_workers=2)
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=training_config['learning_rate'])
    criterion = ComplexMSELoss(device)
    loss_per_epoch = np.zeros((int(training_config['epochs']), 2))
    for epoch in range(training_config['epochs']):
        dataset_idx = 0
        loss_per_pass = np.zeros(len(dataset))
        for fm, tm, sm, ft, tt, st in loader:
            # Put the spectrograms of the mixed signal and ground truth on the
            # training device
            sm = sm.float().to(device)
            st = st.float().to(device)

            # Do an optimization step
            optimizer.zero_grad()
			# s_in, s_out, M, Phi = net(sm)
			# s_in = s_in.to(device)
			# s_out = s_out.to(device)
            s_out, M, Phi = net(sm)
            loss = criterion(sm, s_out)
            loss_per_pass[dataset_idx] = loss.item()
            loss.backward()
            optimizer.step()
            dataset_idx += 1
        loss_per_epoch[epoch, 0] = loss_per_pass.mean()
        loss_per_epoch[epoch, 1] = loss_per_pass.std(ddof=1)
        print('[epoch {}]: loss: {} +/- {}'.format(epoch+1, loss_per_epoch[epoch,0], loss_per_epoch[epoch, 1]))
    torch.save(net.state_dict(), save_path)
    np.save('training_loss.npy', loss_per_epoch)
    print("Finished training network '{}'. Model saved in '{}' and loss saved in 'training_loss.npy'".format(net_type, save_path))

if __name__ == "__main__":
    args = vars(parser.parse_args())
    operation = args['operation']
    net_type = args['net']
    dataset = args['dataset']
    if not operation == 'train' and not operation == 'test':
        print('Invalid choice for the type of operation. Choose "train" or "test"')
        sys.exit(1)
    elif net_type not in networks:
        print('Invalid type of network to evaluate. Available choices: ' + ','.join(networks))
        sys.exit(2)
    elif not dataset == 'avspeech_audioset':
        print('Invalid dataset selection. Available choices: "avspeech_audioset"')
        sys.exit(3)
    else:
        device = find_device()
        if operation == 'train':
            print('Operation "{}" started on device "{}".'.format(operation, device))
            # Check if there is a model already trained
            model_path = net_type + ".pt"
            if os.path.exists(model_path):
                load = input("A model for the network '{}' already exists. "\
                        "Would you like to train and save a new one?[y/n] ".format(net_type))
                if load == 'y':
                    print("Started training on a new model.")
                    train(device, net_type, model_path, dataset)
                elif load == 'n':
                    print('A new model was not trained.')
                    sys.exit(0)
                else:
                    print('Invalid response. Expected "y" or "n"')
                    sys.exit(4)
            else:
                train(device, net_type, model_path, dataset)
        elif operation == 'test':
            print('Operation "{}" started on device "cpu".'.format(operation))
            model_path = net_type + ".pt"
            if os.path.exists(model_path):
                test(device, net_type, model_path, dataset)
            else:
                print('The model "{}" could not be found. Train it first'.format(model_path))
