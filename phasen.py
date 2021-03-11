'''
Unofficial implementation of the PHASEN network
Author: Juan S. Martinez
Date: Spring 2021
'''

import torch.nn as nn

class FTB(nn.Module):
    '''
    The FTB module as described in the original paper by Yin et al., (2020)
    '''
    
    def __init__(self, Ca=96, Cr=5, T=301, F=258):
        super(FTB, self).__init__()

        # Sub attention module
        self.conv_att = nn.Sequential(
                nn.Conv2d(Ca, Cr, kernel_size=(1,1), stride=1, dilation=1, padding=0),
                nn.BatchNorm2d(Cr),
                nn.ReLU()
                )
        self.conv_att_1d = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=9, stride=1, dilation=1, padding=4),
                nn.BatchNorm1d(1),
                nn.ReLU()
                )


