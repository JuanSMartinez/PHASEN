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
import custom_datasets

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
    'epochs': 1e6,
    'learning_rate': 2e-4,
    'batch_size': 8
}

# Random seed
seed = 930103
torch.manual_seed(seed)
np.random.seed(seed)

# -----------------------------------------------------------------------------#

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

def train(device, net_type, save_path, dataset):
    net = create_net_of_type(net_type)
    net = net.to(device)
    dataset = create_dataset_for(dataset, 'train')
    loss_per_epoch = np.zeros(training_config['epochs'])
    loss_per_pass = np.zeros(len(dataset))
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=training_config['batch_size'],
                                        shuffle=True,
                                        num_workers=2)
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=training_config['learning_rate'])
    for epoch in range(training_config['epochs']):
        for fm, tm, sm, ft, tt, st in loader:
            # Put the spectrograms of the mixed signal and ground truth on the
            # training device
            sm = sm.to(device)
            st = st.to(device)

            # Do an optimization step
            optimizer.zero_grad()
            s_out = net(sm)

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
        print('Operation "{}"" started on device "{}".'.format(operation, device))

        if operation == 'train':
            # Check if there is a model already trained
            model_path = net_type + ".pt"
            if os.path.exists(model_path):
                load = input("A model for the network '{}' already exists."\
                        "Would you like to train and save a new one?[y/n]")
                if load == 'y':
                    train(device, net_type, model_path, dataset)
                elif load == 'n':
                    print('A new model was not trained.')
                    sys.exit(0)
                else:
                    print('Invalid response. Expected "y" or "n"')
                    sys.exit(4)
