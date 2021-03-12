'''
Unofficial implementation of the PHASEN network
Author: Juan S. Martinez
Date: Spring 2021
'''
import torch
import torch.nn as nn

class FTB(nn.Module):
    '''
    The FTB module as described in the original paper by Yin et al., (2020)
    '''
    
    def __init__(self, Ca=96, Cr=5, T=301, F=258):
        super(FTB, self).__init__()

        # Sub attention module
        self.conv_att = nn.Sequential(
                nn.Conv2d(Ca, Cr, kernel_size=1, stride=1, dilation=1, padding=0),
                nn.BatchNorm2d(Cr),
                nn.ReLU()
                )
        self.conv_att_1d = nn.Sequential(
                nn.Conv1d(F*Cr, F, kernel_size=9, stride=1, dilation=1, padding=4),
                nn.BatchNorm1d(F),
                nn.ReLU()
                )
        # FreqFC as a linear layer with no bias
        self.freq_fc = nn.Linear(F, F, bias=False)

        # Output
        self.conv_out = nn.Sequential(
                nn.Conv2d(2*Ca, Ca, kernel_size=(1,1),stride=1,dilation=1, padding=0),
                nn.BatchNorm2d(Ca),
                nn.ReLU()
                )

    def forward(self, x):
        # NOTE: The input x must be of size (N, Ca, T, F)
        N, Ca, T, F = x.shape
        # Sub-attention module
        x_att = self.conv_att(x)
        N, Cr, T, F = x_att.shape
        x_att = x_att.permute(0,1,3,2)
        x_att = x_att.reshape(N, F*Cr, T)
        x_att = self.conv_att_1d(x_att)
        
        # Point-wise multiply
        x_att = x_att.permute(0,2,1) # Shape is now (N, T, F)
        x_att = x_att.unsqueeze(1)
        x_p = x * x_att.repeat(1, Ca, 1, 1)

        # FreqFC layer
        x_freqfc = self.freq_fc(x_p)

        # Concatenate with input along the channel dimensions
        x_c = torch.cat((x,x_freqfc), dim=1)

        # Output
        x_out = self.conv_out(x_c)
        return x_out

class TSB(nn.Module):
    '''
    The TSB module in the original paper by Yin et. al, (2020)
    '''

    def __init__(self, Ca=96, Cp=48, Cr=5, T=301, F=258):
        super(TSB, self).__init__()

        # Stream A blocks
        self.ftb_1 = FTB(Ca, Cr, T, F)
        self.conv_a_1 = nn.Sequential(
                nn.Conv2d(Ca, Ca, kernel_size=5, stride=1, dilation=1, padding=2),
                nn.BatchNorm2d(Ca),
                nn.ReLU())

        self.conv_a_2 = nn.Sequential(
                nn.Conv2d(Ca, Ca, kernel_size=(25, 1), stride=1, dilation=1, padding=(12,0)),
                nn.BatchNorm2d(Ca),
                nn.ReLU())
        
        self.conv_a_3 = nn.Sequential(
                nn.Conv2d(Ca, Ca, kernel_size=5, stride=1, dilation=1, padding=2),
                nn.BatchNorm2d(Ca),
                nn.ReLU())

        self.ftb_2 = FTB(Ca, Cr, T, F)

        # Stream P blocks
        self.conv_p_1 = nn.Sequential(
                nn.LayerNorm((Cp, T, F)),
                nn.Conv2d(Cp, Cp, kernel_size=(5, 3), stride=1, dilation=1, padding=(2, 1)))

        self.conv_p_2 = nn.Sequential(
                nn.LayerNorm((Cp, T, F)),
                nn.Conv2d(Cp, Cp, kernel_size=(25, 1), stride=1, dilation=1, padding=(12, 0)))
        
        # Convolutional layers for information communication functions
        self.conv_p_to_a = nn.Conv2d(Cp, Ca, kernel_size=1, stride=1, dilation=1, padding=0)
        self.conv_a_to_p = nn.Conv2d(Ca, Cp, kernel_size=1, stride=1, dilation=1, padding=0)

    def forward(self, s_a, s_p):
        # NOTE: The input should be of shape (N, Ca, T, F) for s_a and (N, Cp, T, F) for s_p
        
        # Compute amplitude stream
        x_a = self.ftb_1(s_a)
        x_a = self.conv_a_1(x_a)
        x_a = self.conv_a_2(x_a)
        x_a = self.conv_a_3(x_a)
        x_a = self.ftb_2(x_a)

        # Compute phase stream
        x_p = self.conv_p_1(s_p)
        x_p = self.conv_p_2(x_p)

        # Information communication
        s_a_out = x_a * torch.tanh(self.conv_p_to_a(x_p))
        s_p_out = x_p * torch.tanh(self.conv_a_to_p(x_a))

        return s_a_out, s_p_out


class PHASEN(nn.Module):
    '''
    Unofficial implementation of the PHASEN network by Yin et al., (2020)
    '''

    def __init__(self, Ca=96, Cp=48, Cr_tsb=5, Cr_out=8, T=301, F=258):
        super(PHASEN, self).__init__()

        # Convolutional layers to produce stream A
        self.conv_a = nn.Sequential(
                nn.Conv2d(2, Ca, kernel_size=(1,7), stride=1, dilation=1, padding=(0,3)),
                nn.Conv2d(Ca, Ca, kernel_size=(7,1), stride=1, dilation=1, padding=(3,0))

        # Convolutional layers to produce stream P

