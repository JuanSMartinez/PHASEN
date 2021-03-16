'''
Evaluation of the PHASEN network
Author: Juan S. Martinez
Date: Spring 2021
'''

import argparse
import sys
import torch

# --------------- Global variables --------------------------------------------#

# Available networks to perform ablation studies
networks = ['phasen']

# Argument parser to run the script
parser = argparse.ArgumentParser(description='Evaluate the PHASEN network')
parser.add_argument('operation', type=str, help='Type of operation on a network. Either "train" or "test"')
parser.add_argument('net', type=str, help='Type of network to evaluate. Choices: ' + ','.join(networks))

# Training configuration
trainin_config = {
    'epochs': 1e6,
    'learning_rate': 2e-4,
    'batch_size': 8
}
# -----------------------------------------------------------------------------#

def find_device():
    '''
    Find the best device to perform operations
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    args = vars(parser.parse_args())
    operation = args['operation']
    net_type = args['net']
    if not operation == 'train' and not operation == 'test':
        print('Invalid choice for the type of operation. Choose "train" or "test"')
        sys.exit(1)
    elif net_type not in networks:
        print('Invalid type of network to evaluate. Available choices: ' + ','.join(networks))
        sys.exit(2)
    else:
        device = find_device()
        print('Operation "{}"" started on device "{}".'.format(operation, device))
