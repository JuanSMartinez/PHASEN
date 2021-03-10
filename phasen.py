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
    
    def __init__(self, Ca=96, Cr=5):
        super(FTB, self).__init__()

        # Sub attention module
        self.att_mod = nn.Sequential(
                nn.Conv2d(Ca, Cr, kernel_size=(1,1), stride=1, dilation=1, padding=0),
                nn.BatchNorm2d(Cr),
                nn.ReLU()
                )


