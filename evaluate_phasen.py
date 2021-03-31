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
torch.autograd.set_detect_anomaly(True)

# --------------- Global variables --------------------------------------------#

# Available networks to perform ablation studies
networks = ['phasen', 'phasen_baseline', 'phasen_1strm', 'phasen_no_ftb', 'phasen_no_a2pp2a', 'phasen_no_p2a']

# Argument parser to run the script
parser = argparse.ArgumentParser(description='Evaluate the PHASEN network')
parser.add_argument('operation', type=str, help='Type of operation on a network. Either "train", "test" or "preprocess_test"')
parser.add_argument('net', type=str, help='Type of network to evaluate. Choices: ' + ','.join(networks))
parser.add_argument('dataset', type=str, help='Dataset to train or test. Choices: "avspeech_audioset"')

# Training configuration
training_config = {
    'epochs': 50,
    'learning_rate': 2e-4,
    'batch_size': 5
}

# Random seed
seed = 210325
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
        mag_spec = torch.sqrt(spec[:,0,:,:]**2 + spec[:,1,:,:]**2 + 1e-8)
        normalized_spec = torch.div(spec, mag_spec.unsqueeze(1).repeat(1,2,1,1))
        mag_spec = torch.pow(mag_spec.unsqueeze(1).repeat(1,2,1,1), self.p)
        return mag_spec * normalized_spec

    def forward(self, spec_in, spec_out):

        comp_spec_in = self.pw_compress_spectrogram(spec_in)
        comp_spec_out = self.pw_compress_spectrogram(spec_out)
        mag_spec_in = torch.sqrt(comp_spec_in[:,0,:,:]**2 + comp_spec_in[:,1,:,:]**2 + 1e-8)
        mag_spec_out = torch.sqrt(comp_spec_out[:,0,:,:]**2 + comp_spec_out[:,1,:,:]**2 + 1e-8)
        return 0.5*F.mse_loss(mag_spec_in, mag_spec_out) +\
                 0.5*F.mse_loss(comp_spec_in, comp_spec_out)


def complex_mse_loss(sgt, sout):
    mag_sgt = torch.sqrt(sgt[:,0,:,:]**2 + sgt[:,1,:,:]**2 + 1e-8)
    mag_out = torch.sqrt(sout[:,0,:,:]**2 + sout[:,1,:,:]**2 + 1e-8)
    pwc_sgt = torch.pow(mag_sgt.unsqueeze(1).repeat(1,2,1,1), 0.3) * torch.div(sgt, mag_sgt.unsqueeze(1).repeat(1,2,1,1))
    pwc_sout = torch.pow(mag_out.unsqueeze(1).repeat(1,2,1,1), 0.3) * torch.div(sout, mag_out.unsqueeze(1).repeat(1,2,1,1))
    mag_pwc_sgt = torch.sqrt(pwc_sgt[:,0,:,:]**2 + pwc_sgt[:,1,:,:]**2 + 1e-8)
    mag_pwc_sout = torch.sqrt(pwc_sout[:,0,:,:]**2 + pwc_sout[:,1,:,:]**2 + 1e-8)
    return 0.5*F.mse_loss(mag_pwc_sgt, mag_pwc_sout) + 0.5*F.mse_loss(pwc_sgt, pwc_sout)

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
    elif net_type == 'phasen_1strm':
        return phasen.PHASEN_one_strm()
    elif net_type == 'phasen_baseline':
        return phasen.PHASEN_baseline()
    elif net_type == 'phasen_no_ftb':
        return phasen.PHASEN_without_ftb()
    elif net_type == 'phasen_no_a2pp2a':
        return phasen.PHASEN_without_A2PP2A()
    elif net_type == 'phasen_no_p2a':
        return phasen.PHASEN_without_P2A()
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


def pre_process_test(device, net_type, model_path, dataset):
    net = create_net_of_type(net_type)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    dataset = create_dataset_for(dataset, 'test')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=1)
    i=0
    for clean, noisy, st, sm in loader:
        clean_speech = clean.numpy().flatten()
        noisy_speech = noisy.numpy().flatten()
        sm = sm.float().to(device)
        st = st.float().to(device)
        if net_type == 'phasen_1strm' or net_type == 'phasen_baseline':
            cIRM_est = net(sm)
            decompressed_cIRM = dsp.decompress_cIRM(cIRM_est)
            decompressed_cIRM = decompressed_cIRM.squeeze(0)
            sm = sm.squeeze(0)
            C, T, F = decompressed_cIRM.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = decompressed_cIRM[0,:,:]*sm[0,:,:] - decompressed_cIRM[1,:,:]*sm[1,:,:]
            sout_c.imag = decompressed_cIRM[0,:,:]*sm[1,:,:] + decompressed_cIRM[1,:,:]*sm[0,:,:]
            sout_c = sout_c.T.detach().cpu().numpy()
        else:
            s_out, M, Phi = net(sm)
            # Convert s_out from (1, 2, T, F) to a complex array of shape (F, T)
            s_out = s_out.squeeze(0)
            C, T, F = s_out.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = s_out[0,:,:]
            sout_c.imag = s_out[1,:,:]
            sout_c = sout_c.T.detach().cpu().numpy()

        # Recover time domain signal
        t, recovered_speech = dsp.recover_from_stft_spectrogram(sout_c, dsp.audio_fs)

        # Save pair of signals
        pair = np.zeros((len(recovered_speech), 2))
        pair[:,0] = clean_speech
        pair[:,1] = recovered_speech
        np.save('preprocessed_test_data_'+net_type+'/pair_' + str(i+1) + '.npy', pair)
        print('[Pair of clean and recovered speech #{} complete]'.format(i+1))
        sys.stdout.flush()
        i += 1
    print("Finished pre-processing test data for net '{}', data saved in preprocessed_test_data_{}".format(net_type, net_type))
    sys.stdout.flush()

def test(device, net_type, model_path, dataset):
    from pystoi import stoi
    from pesq import pesq
    from mir_eval.separation import bss_eval_sources
    net = create_net_of_type(net_type)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    dataset = create_dataset_for(dataset, 'test')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=1)

    metrics = np.zeros((len(dataset), 3))
    i=0
    for clean, noisy, st, sm in loader:
        clean_speech = clean.numpy().flatten()
        noisy_speech = noisy.numpy().flatten()
        sm = sm.float().to(device)
        if net_type == 'phasen_1strm' or net_type == 'phasen_baseline':
            cIRM_est = net(sm)
            decompressed_cIRM = dsp.decompress_cIRM(cIRM_est)
            decompressed_cIRM = decompressed_cIRM.squeeze(0)
            sm = sm.squeeze(0)
            C, T, F = decompressed_cIRM.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = decompressed_cIRM[0,:,:]*sm[0,:,:] - decompressed_cIRM[1,:,:]*sm[1,:,:]
            sout_c.imag = decompressed_cIRM[0,:,:]*sm[1,:,:] + decompressed_cIRM[1,:,:]*sm[0,:,:]
            sout_c = sout_c.T.detach().cpu().numpy()
        else:
            s_out, M, Phi = net(sm)
            # Convert s_out from (1, 2, T, F) to a complex array of shape (F, T)
            s_out = s_out.squeeze(0)
            C, T, F = s_out.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = s_out[0,:,:]
            sout_c.imag = s_out[1,:,:]
            sout_c = sout_c.T.detach().cpu().numpy()

        # Recover time domain signal
        t, recovered_speech = dsp.recover_from_stft_spectrogram(sout_c, dsp.audio_fs)
        PESQ = pesq(dsp.audio_fs, clean_speech, recovered_speech, 'wb')
        STOI = stoi(clean_speech, recovered_speech, dsp.audio_fs, extended=False)
        SDR, sir, sar, perm = bss_eval_sources(clean_speech, recovered_speech)
        metrics[i,0] = SDR[0]
        metrics[i,1] = PESQ
        metrics[i,2] = STOI
        i += 1
        print('[Sample {} Complete]'.format(i))
        sys.stdout.flush()
    np.save(net_type + '_metrics.npy', metrics)
    print("Finished testing of net '{}', metrics saved in {}_metrics.npy".format(net_type, net_type))
    sys.stdout.flush()


def train(device, net_type, save_path, dataset):
    net = create_net_of_type(net_type)
    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("Total params: {}".format(pytorch_total_params))
    dataset = create_dataset_for(dataset, 'train')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=training_config['batch_size'],
                                        shuffle=True,
                                        num_workers=1)
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=training_config['learning_rate'])
    #criterion = ComplexMSELoss(device)
    loss_per_epoch = np.zeros((int(training_config['epochs']), 2))
    for epoch in range(training_config['epochs']):
        batch = 0
        loss_per_pass = np.zeros(len(dataset))
        for clean, noisy, st, sm in loader:
            # Put the spectrograms of the mixed signal and ground truth on the
            # training device
            sm = sm.float().to(device)
            st = st.float().to(device)

            # Do an optimization step
            optimizer.zero_grad()
            if net_type == 'phasen_1strm' or net_type == 'phasen_baseline':
                compressed_cIRM = dsp.compress_cIRM(dsp.compute_cIRM_from(st, sm)).to(device)
                cIRM_est = net(sm)
                #loss = criterion(compressed_cIRM, cIRM_est)
                loss = complex_mse_loss(compressed_cIRM, cIRM_est)
            else:
                s_out, M, Phi = net(sm)
                #loss = criterion(st, s_out)
                loss = complex_mse_loss(st, s_out)
            loss_per_pass[batch] = loss.item()
            loss.backward()
            optimizer.step()
            batch += 1
            #if batch % 100 == 0:
            #    print('\t[epoch {}, batch {}] running_loss: {}'.format(epoch+1, batch, loss_per_pass[batch-1]))
        loss_per_epoch[epoch, 0] = loss_per_pass.mean()
        loss_per_epoch[epoch, 1] = loss_per_pass.std(ddof=1)
        print('[epoch {}] loss: {} +/- {}'.format(epoch+1, loss_per_epoch[epoch,0], loss_per_epoch[epoch, 1]))
        sys.stdout.flush()
    torch.save(net.state_dict(), save_path)
    np.save(net_type + '_training_loss.npy', loss_per_epoch)
    print("Finished training network '{}'. Model saved in '{}' and loss saved in '{}_training_loss.npy'".format(net_type, save_path, net_type))
    sys.stdout.flush()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    operation = args['operation']
    net_type = args['net']
    dataset = args['dataset']
    if not operation == 'train' and not operation == 'test' and not operation == 'preprocess_test':
        print('Invalid choice for the type of operation. Choose "train", "test" or "preprocess_test"')
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
        elif operation == 'preprocess_test':
            print('Operation "{}" started on device "{}".'.format(operation, device))
            model_path = net_type + ".pt"
            if os.path.exists(model_path):
                os.mkdir('preprocessed_test_data_'+net_type)
                pre_process_test(device, net_type, model_path, dataset)
            else:
                print('The model "{}" could not be found. Train it first'.format(model_path))
