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
networks = ['phasen', 'phasen_baseline', 'phasen_1strm', 'phasen_no_ftb', 'phasen_no_a2pp2a', 'phasen_no_p2a', 'altphasen']

# Available operations
operations = ["train", "continue_training", "test", "preprocess_test", "test_preprocessed_data"]

# Argument parser to run the script
parser = argparse.ArgumentParser(description='Evaluate the PHASEN network')
parser.add_argument('operation', type=str, help='Type of operation on a network. Either "train", "test", "preprocess_test", "test_preprocessed_data" ')
parser.add_argument('net', type=str, help='Type of network to evaluate. Choices: ' + ','.join(networks))
parser.add_argument('dataset', type=str, help='Dataset to train or test. Choices: "avspeech_audioset"')
parser.add_argument('--iteration', type=int, help='Iteration number when continuing training')

# Training configuration
training_config = {
    'epochs': 50,
    'learning_rate': 2e-5,
    'batch_size': 5
}

# Random seed
seed = 210325
torch.manual_seed(seed)
np.random.seed(seed)

# -----------------------------------------------------------------------------#

def complex_mse_loss(sgt, sout):
    mag_sgt = torch.sqrt(sgt[:,0,:,:]**2 + sgt[:,1,:,:]**2 + 1e-8)
    mag_out = torch.sqrt(sout[:,0,:,:]**2 + sout[:,1,:,:]**2 + 1e-8)
    pwc_sgt = torch.pow(mag_sgt.unsqueeze(1).repeat(1,2,1,1), 0.3) * torch.div(sgt, mag_sgt.unsqueeze(1).repeat(1,2,1,1))
    pwc_sout = torch.pow(mag_out.unsqueeze(1).repeat(1,2,1,1), 0.3) * torch.div(sout, mag_out.unsqueeze(1).repeat(1,2,1,1))
    mag_pwc_sgt = torch.sqrt(pwc_sgt[:,0,:,:]**2 + pwc_sgt[:,1,:,:]**2 + 1e-8)
    mag_pwc_sout = torch.sqrt(pwc_sout[:,0,:,:]**2 + pwc_sout[:,1,:,:]**2 + 1e-8)
    La = 0.5*F.mse_loss(mag_pwc_sgt, mag_pwc_sout, reduction='sum')
    Lp = 0.5*F.mse_loss(pwc_sgt, pwc_sout, reduction='sum')
    return La + Lp

def find_device():
    '''
    Find the best device to perform operations
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def create_net_of_type(net_type, device, batch_size):
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
    elif net_type == 'altphasen':
        return phasen.AltPHASEN(N=batch_size, device=device)
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
    net = create_net_of_type(net_type, device, 1)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    dataset = create_dataset_for(dataset, 'test')
    total_samples = len(dataset)
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=1)
    i=0
    print("Saving pairs of clean and recovered speech")
    for clean, noisy, st, sm in loader:
        clean_speech = clean.numpy().flatten()
        noisy_speech = noisy.numpy().flatten()
        sm = sm.float().to(device)
        if net_type == 'phasen_1strm' or net_type == 'phasen_baseline':
            cIRM_est = net(sm)
            cIRM_est = cIRM_est.squeeze(0)
            sm = sm.squeeze(0)
            #C, T, F = decompressed_cIRM.shape
            C, T, F = cIRM_est.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = cIRM_est[0,:,:]*sm[0,:,:] - cIRM_est[1,:,:]*sm[1,:,:]
            sout_c.imag = cIRM_est[0,:,:]*sm[1,:,:] + cIRM_est[1,:,:]*sm[0,:,:]
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
        if i < total_samples-1:
            print('[Process {}% complete]'.format(100.0*(i+1)/total_samples), end='\r')
        else:
            print('[Process {}% complete]'.format(100.0*(i+1)/total_samples), end='\n')
        i += 1
    print("Finished pre-processing test data for net '{}', data saved in preprocessed_test_data_{}".format(net_type, net_type))

def test_preprocessed_data(net_type):
    from pystoi import stoi
    import pesq
    from mir_eval.separation import bss_eval_sources
    path = 'preprocessed_test_data_' + net_type + '/'
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        sdr_a = []
        pesq_a = []
        stoi_a = []
        processed = 0
        for i, f in enumerate(files):
            signals = np.load(path + f)
            clean_speech = signals[:,0]
            recovered_speech = signals[:,1]
            if np.any(clean_speech) and np.any(recovered_speech):
                PESQ = pesq.pesq(dsp.audio_fs, clean_speech, recovered_speech, 'wb')
                STOI = stoi(clean_speech, recovered_speech, dsp.audio_fs, extended=False)
                SDR, sir, sar, perm = bss_eval_sources(clean_speech, recovered_speech)
                sdr_a.append(SDR[0])
                pesq_a.append(PESQ)
                stoi_a.append(STOI)
                processed += 1
                if i < len(files)-1:
                    print('[Metric computation: {}% complete]'.format(100.0*(i+1)/len(files)), end='\r')
                else:
                    print('[Metric computation: {}% complete]'.format(100.0*(i+1)/len(files)), end='\n')
        metrics = np.array([sdr_a, pesq_a, stoi_a]).T
        np.save(net_type + '_metrics.npy', metrics)
        print("Finished pre-processed testing of net '{}', {} files out of {} were processed into {}_metrics.npy".format(net_type, processed, len(files), net_type))
    else:
        print("Error: Preprocessed data for the model not found")

def test(device, net_type, model_path, dataset):
    from pystoi import stoi
    from pesq import pesq
    from mir_eval.separation import bss_eval_sources
    net = create_net_of_type(net_type, device, 1)
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
            cIRM_est = cIRM_est.squeeze(0)
            sm = sm.squeeze(0)
            #C, T, F = decompressed_cIRM.shape
            C, T, F = cIRM_est.shape
            sout_c = torch.zeros(T, F, dtype=torch.cfloat)
            sout_c.real = cIRM_est[0,:,:]*sm[0,:,:] - cIRM_est[1,:,:]*sm[1,:,:]
            sout_c.imag = cIRM_est[0,:,:]*sm[1,:,:] + cIRM_est[1,:,:]*sm[0,:,:]
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
        if i < len(dataset)-1:
            print('[Sample {}% Complete]'.format(100*i/len(dataset)), end='\r')
        else:
            print('[Sample {}% Complete]'.format(100*i/len(dataset)), end='\n')
    np.save(net_type + '_metrics.npy', metrics)
    print("Finished testing of net '{}', metrics saved in {}_metrics.npy".format(net_type, net_type))

def train(device, net_type, save_path, dataset):
    net = create_net_of_type(net_type, device, training_config['batch_size'])
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

    loss_per_epoch = np.zeros((int(training_config['epochs']), 2))
    total_batches = len(dataset)/training_config['batch_size']
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
                cIRM = dsp.compute_cIRM_from(st, sm)
                cIRM_est = net(sm)
                loss = complex_mse_loss(cIRM.to(device), cIRM_est)
            else:
                s_out, M, Phi = net(sm)
                loss = complex_mse_loss(st, s_out)
            loss.backward()
            loss_per_pass[batch] = loss.item()
            optimizer.step()
            batch += 1
            if batch < total_batches-1:
                print('\t[epoch {}] mini-batch progress: {}%'.format(epoch+1, 100.0*batch/total_batches), end='\r')
            else:
                print('\t[epoch {}] mini-batch progress: {}%'.format(epoch+1, 100.0*batch/total_batches), end='\n')

        loss_per_epoch[epoch, 0] = loss_per_pass.mean()
        loss_per_epoch[epoch, 1] = loss_per_pass.std(ddof=1)
        print('[epoch {}] loss: {} +/- {}'.format(epoch+1, loss_per_epoch[epoch,0], loss_per_epoch[epoch, 1]))

    torch.save(net.state_dict(), save_path)
    torch.save(optimizer.state_dict(), net_type + '_optim.pt')
    np.save(net_type + '_training_loss_1.npy', loss_per_epoch)
    print("Finished training network '{}'. Model saved in '{}', loss saved in '{}_training_loss_1.npy' and optimizer saved in '{}'_optim.pt".format(net_type, save_path, net_type, net_type))

def continue_training(device, net_type, dataset, iteration):
    net_path = net_type + '.pt'
    optim_path = net_type + '_optim.pt'
    if not os.path.isfile(net_path):
        print("ERROR: No previous model found in current directory")
        sys.exit(5)
    elif not os.path.isfile(optim_path):
        print("ERROR: No preovious optimizer settings found in the current directory")
        sys.exit(6)

    net = create_net_of_type(net_type, device, training_config['batch_size'])
    net.load_state_dict(torch.load(net_path, map_location=device))
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=training_config['learning_rate'])
    optimizer.load_state_dict(torch.load(optim_path, map_location=device))

    dataset = create_dataset_for(dataset, 'train')
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=training_config['batch_size'],
                                        shuffle=True,
                                        num_workers=1)
    loss_per_epoch = np.zeros((int(training_config['epochs']), 2))
    total_batches = len(dataset)/training_config['batch_size']
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
                cIRM = dsp.compute_cIRM_from(st, sm)
                cIRM_est = net(sm)
                loss = complex_mse_loss(cIRM.to(device), cIRM_est)
            else:
                s_out, M, Phi = net(sm)
                loss = complex_mse_loss(st, s_out)
            loss.backward()
            loss_per_pass[batch] = loss.item()
            optimizer.step()
            batch += 1
            if batch < total_batches-1:
                print('\t[epoch {}] mini-batch progress: {}%'.format(epoch+1, 100.0*batch/total_batches), end='\r')
            else:
                print('\t[epoch {}] mini-batch progress: {}%'.format(epoch+1, 100.0*batch/total_batches), end='\n')

        loss_per_epoch[epoch, 0] = loss_per_pass.mean()
        loss_per_epoch[epoch, 1] = loss_per_pass.std(ddof=1)
        print('[epoch {}] loss: {} +/- {}'.format(epoch+1, loss_per_epoch[epoch,0], loss_per_epoch[epoch, 1]))

    torch.save(net.state_dict(), net_path)
    torch.save(optimizer.state_dict(), net_type + '_optim.pt')
    np.save(net_type + '_training_loss_'+str(iteration)+'.npy', loss_per_epoch)
    print("Finished training network '{}' on iteration {}. Model updated in '{}', loss saved in '{}_training_loss_{}.npy' \
     and optimizer updated in '{}'_optim.pt".format(net_type, iteration, net_path, net_type, iteration, net_type))

if __name__ == "__main__":
    args = vars(parser.parse_args())
    operation = args['operation']
    net_type = args['net']
    dataset = args['dataset']
    iteration = args['iteration']
    if operation not in operations:
        print('Invalid choice for the type of operation. Choose one of: ' + ','.join(operations))
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
        elif operation == 'test_preprocessed_data':
            print('Operation "{}" started on device "{}".'.format(operation, device))
            test_preprocessed_data(net_type)
        elif operation == "continue_training":
            if iteration:
                print('Operation "{}" started on device "{}".'.format(operation, device))
                if iteration < 2:
                    print("WARNING: If you are running this option you most likely are doing a second pass of training. Therefore,\
                     'iteration' should be larger or equal to 2.")
                continue_training(device, net_type, dataset, iteration)
            else:
                print("ERROR: Iteration number needed when continuing training")
                sys.exit(7)
