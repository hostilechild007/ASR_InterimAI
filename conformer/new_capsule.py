import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
# from torchvision import datasets, transforms
USE_CUDA = False

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              stride=(1, 1)
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        # each capsule: compute on their inputs -> encapsulate results into a small vec
        # idea: each capsule recognize an obj and its params/feats
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
            for _ in range(num_capsules)])
        
        self.out_channels = out_channels

    def forward(self, x):
        """

        :param x: size = [batch_size, in_channels / prev convolution's out_channel, image_size_x, image_size_y]
        :return:
        """
        print("---------PrimaryCaps--------")
        print("x.size():", x.size())
        # each capsule i's size = [batch_size, out_channel, new_image_x, new_image_y]
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)  # size = [batch_size, num_capsules, out_channel, new_image_x=6, new_image_y=6]
        print("u.size():", u.size())  # w/ mel spec librispeech, size = [50, 8, 32, 56, 645]
        u = u.view(x.size(0), self.out_channels * u.size(3) * u.size(4), -1)  # size = [batch_size, out_channel * new_image_x * new_image_y, num_capsules]
        print()
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 8 * 8, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes  # state size of each capsule/feature
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        """

        :param x: all capsules' outputs from lower level; size = [batch_size, 1 cap/feat size, num_capsules]
        :return: size = [batch_size,
        """
        print("---------DigitCaps------------")
        print("x size:", x.size())
        
        batch_size = x.size(0)
        # size: [batch_size, cap size, curr num caps, lower num caps, 1]
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        # size: [batch_size, num_routes / 1 capsule size, num_capsules, out_channels, in_channels]
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)  # size: [batch_size, num_routes, curr num caps, out_channels, 1]
        print("u_hat.size:", u_hat.size())
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):  # start dynamic routing 3 times (like finding best cluster mean)
            c_ij = F.softmax(b_ij, dim=1)  # calc routing weights for all lower level capsules i
            print("old c_ij:", c_ij.size())
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)  # size: [100, 1152, 10, 1, 1]
            print("new c_ij:", c_ij.size())
            
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            print("s_j:", s_j.size())
            v_j = self.squash(s_j)
            print("v_j:", v_j.size())

            if iteration < num_iterations - 1:  # weight update for each lower level capsule
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        
        print("last v_j:", v_j.size())
        out = v_j.squeeze(1)
        print("out v_j:", out.size())
        print()
        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self, out_channels=256, window_size=28):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels * window_size * window_size),
            nn.Sigmoid()
        )
        
        '''
        # was this:
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        '''
        self.out_channels = out_channels
        self.window_size = window_size

    def forward(self, x):
        print("---------Decoder--------")
        print("x size:", x.size())
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
         
        reconstructions = reconstructions.view(-1, self.out_channels, self.window_size, self.window_size)
        print("reconstructions size:", reconstructions.size())
        print()
        # reconstructions = reconstructions.view(-1, 1, 28, 28)  # reconstruction was THIS and SHOULD output this size: [batch_size, channels=encoder_dim , subsampled_lengths, sumsampled_dim]
        # reconstructions = reconstructions.view(-1, 16, 64)   # b/c 16 * 64 = 1024

        return reconstructions, masked
        
class CapsNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, window_size=28):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.primary_capsules = PrimaryCaps(in_channels=out_channels)
        self.digit_capsules = DigitCaps(num_routes=self.primary_capsules.out_channels * 4 * 4)
        self.decoder = Decoder( out_channels=out_channels, window_size=window_size)

    
    #  def forward(self, data):  # before was THIS
    def forward(self, data: Tensor):
        '''
        want:
        out_channels = out_channels in sub_sampling conv
        input data = specifc frame that's aready padded w/ static length and should have same kernel and stride as sub_sampling Conv
        
        '''
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output)
        return reconstructions

class ProxyCapsNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, window_size=20):
        super(ProxyCapsNet, self).__init__()
        self.caps_net = CapsNet(in_channels=in_channels, out_channels=out_channels, window_size=window_size)
        self.window_size: int = window_size
        self.out_channels = out_channels
        
    def forward(self, data, input_lengths: Tensor):
        '''
        idea: do specific stride, pad, and kernel size and send to CapsNet and stitch it up to be same ish output as ConvSubSampling
        b/c WE ARE NOT subsampling, want the data to match the ORIGINIAL INPUT size --> pad based on KERNEL size
        
        after stiching, should have size = batch_size, channels / out_channels, subsampled_lengths, sumsampled_dim
        
        input data size= (batch, input channel = 1, time, dim)
        
        # !!!! in ConvSubSampling, output size = [batch_size, subsampled_lengths, channels * sumsampled_dim] !!!
        '''
        
        # idea: pad the image data w/ len lower kernel_size // 2 so CapNet doesnt need to pad and input_lengths doesn't change b/c the image didnt change
        padding_len: int = self.window_size // 2
        print("old input size:", data.size())
        data: Tensor = nn.functional.pad(data, (padding_len, padding_len, padding_len, padding_len))  # pads last 2 data's dimensions (time, dim) by padding_len on both sides -> ex: (2 + time + 2, 2 + dim + 2)
        print("new input size:", data.size())
        
        # have static sliding image window size = [28, 28]
        ROWS: int = data.size(2)
        COLS: int = data.size(3)
        
        '''
        out = None
        for r in range(0, 9, 3):
          out_col = window
          for c in range(3, 12, 3):
            # print(torch.cat((out_col, window), dim=3))
            out_col = torch.cat((out_col, window), dim=3)
        
          if out is None:
            print()
            print(out_col.size())
            out = out_col
          else:
            print()
            print(out_col.size())
            print(out.size())
            out = torch.cat((out, out_col), dim=2)
        
        '''
        
        '''
        out = None
        for r in range(0, ROWS, self.window_size):
            if r + self.window_size > ROWS:
                break
            
            if self.window_size > COLS:
                break
            
            window_data: Tensor = data[:, :, r : r + self.window_size, 0 : self.window_size] 
            output_window: Tensor = self.caps_net(window_data)
            out_col = output_window
            
            for c in range(self.window_size, COLS, self.window_size):
                if c + self.window_size > COLS:
                    break
                
                
                window_data: Tensor = data[:, :, r : r + self.window_size, c : c + self.window_size]  # [batchsize, channels, window_size, window_size]
                # before:( batch, input channel = 1, time = window_size, dim = window_size)
                output_window: Tensor = self.caps_net(window_data)
                # # after:( batch, outputchanbel = 256, time = window_size, dim = window_size)
                
                print("!!!!!MADE IT THROUGH!!!!")
                
                out_col = torch.cat((out_col, output_window), dim=3)
                
            
            if out is None:
                print()
                print(out_col.size())
                out = out_col
            else:
                print()
                print(out_col.size())
                print(out.size())
                out = torch.cat((out, out_col), dim=2)
        '''  
        out = torch.rand(data.size(0), self.out_channels, (ROWS // self.window_size) * self.window_size, (COLS // self.window_size) * self.window_size)
        for r in range(0, ROWS, self.window_size):
            if r + self.window_size > ROWS:
                break
            for c in range(0, COLS, self.window_size):
                if c + self.window_size > COLS:
                    break
                
                window_data: Tensor = data[:, :, r : r + self.window_size, c : c + self.window_size]  # [batchsize, channels, window_size, window_size]
                # before:( batch, input channel = 1, time = window_size, dim = window_size)
                output_window: Tensor = self.caps_net(window_data)
                # # after:( batch, outputchanbel = 256, time = window_size, dim = window_size)
                
                print("!!!!!MADE IT THROUGH!!!!")
                out[:, :, r : r + self.window_size, c : c + self.window_size] = output_window
                
        print("out.size:", out.size())
        return out  # ( batch, output channel = 256, time = input time, dim = input dim)
        
