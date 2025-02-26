import torch, torch.nn as nn, torch.nn.functional as F
import math, random
from collections import OrderedDict
import sys
# sys.path.append(".modules")
from .SE_block import *
from .pooling import *

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

'''Base layers'''
class conv1d_padding_same(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv1d = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            dilation=dilation
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        _, _, L = x.size()
        padding_size = max((math.ceil(L / self.stride) - 1) * self.stride + (self.kernel_size - 1) * self.dilation + 1 - L, 0)
        padded_x = F.pad(x, (padding_size//2, padding_size-padding_size//2), "constant", 0)
        return self.activation(self.conv1d(padded_x))

class conv2d_padding_same(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), stride=(1,1), dilation=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv2d = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )

    def forward(self, x):
        _, C, H, W = x.size()
        H_update = (math.ceil(H / self.stride[0]) - 1)
        if H%self.stride[0] != 0: H_update = math.ceil(H / self.stride[0])
        padding_H = max(H_update * self.stride[0] + (self.kernel_size[0] - 1) * self.dilation + 1 - H, 0)
        W_update = (math.ceil(W / self.stride[1]) - 1)
        if W%self.stride[1] != 0: W_update = math.ceil(W / self.stride[1])
        padding_W = max(W_update * self.stride[1] + (self.kernel_size[1] - 1) * self.dilation + 1 - W, 0)
        padded_x = F.pad(x, (padding_H//2, padding_H-padding_H//2, padding_W//2, padding_W-padding_W//2), "constant", 0)
        return self.conv2d(padded_x)

class maxpool2d_padding_same(nn.Module):
    def __init__(self, pool_size, stride) -> None:
        super().__init__()
        self.kernel_size = pool_size
        self.stride = stride
        self.maxpool2d = nn.MaxPool2d(kernel_size=pool_size, stride=stride)

    def forward(self, x):
        _, C, H, W = x.size()
        H_update = (math.ceil(H / self.stride[0]) - 1)
        if H%self.stride[0] != 0: H_update = math.ceil(H / self.stride[0])
        padding_H = max(H_update * self.stride[0] + (self.kernel_size[0] - 1) + 1 - H, 0)
        W_update = (math.ceil(W / self.stride[1]) - 1)
        if W%self.stride[1] != 0: W_update = math.ceil(W / self.stride[1])
        padding_W = max(W_update * self.stride[1] + (self.kernel_size[1] - 1) + 1 - W, 0)
        padded_x = F.pad(x, (padding_H//2, padding_H-padding_H//2, padding_W//2, padding_W-padding_W//2), "constant", 0)
        return self.maxpool2d(padded_x)

class resBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3,3), stride=(1,1), downsampling=False, ConvBlock=conv2d_padding_same) -> None:
        super().__init__()
        self.conv2d_1 = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2d_2 = ConvBlock(out_ch, out_ch, kernel_size=(3,3), stride=(1,1))
        self.downsampling = downsampling
        if self.downsampling:
            self.maxpool = maxpool2d_padding_same(pool_size=stride, stride=stride)
            self.conv2d_3 = ConvBlock(in_ch, out_ch, kernel_size=(1,1), stride=(1,1))

    def forward(self, x): # input x is batch-wise normalized already
        output = self.conv2d_1(x)
        output = self.conv2d_2(F.relu(self.bn1(output)))
        if self.downsampling:
            x = self.maxpool(x)
            x = self.conv2d_3(x)
        output = x + output
        return output


'''Functional layers'''
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print(torch.sum(torch.isnan(x)))
        # print(x.shape)
        return x

class PrintLayer4test(nn.Module):
    def __init__(self):
        super(PrintLayer4test, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print(torch.sum(torch.isnan(x)))
        # print(x.shape)
        return x


'''Interspeech 2025 models'''
class spectrum_branch(nn.Module):
    def __init__(self, input_size: tuple) -> None:
        super().__init__()
        self.convBlock = nn.Sequential(OrderedDict([
            ("conv1d_1", conv1d_padding_same(1, 128, kernel_size=256, stride=64)),
            ("conv1d_2", conv1d_padding_same(128, 128, kernel_size=64, stride=32)),
            ("conv1d_3", conv1d_padding_same(128, 128, kernel_size=32, stride=4)),
        ]))
        self.latent_dim = self.convBlock(torch.ones(1, *input_size)).size()
        _, C, L = self.latent_dim
        self.denseBlock = nn.Sequential(OrderedDict([
            ("dense_layer_1", nn.Linear(C*L, 128)),
            ("batchnorm_1", nn.BatchNorm1d(128)),
            ("activation_1", nn.ReLU()),
            ("dense_layer_2", nn.Linear(128, 128)),
            ("batchnorm_2", nn.BatchNorm1d(128)),
            ("activation_2", nn.ReLU()),
            ("dense_layer_3", nn.Linear(128, 128)),
            ("batchnorm_3", nn.BatchNorm1d(128)),
            ("activation_3", nn.ReLU()),
            ("dense_layer_4", nn.Linear(128, 128)),
            ("batchnorm_4", nn.BatchNorm1d(128)),
            ("activation_4", nn.ReLU()),
        ]))
        self.emb_layer = nn.Linear(128, 128)
    
    def forward(self, x):
        B, C, L = x.shape
        # print("input", x.shape)
        x = self.convBlock(x)
        # print("conv", x.shape)
        x = self.denseBlock(x.view(B, -1))
        # print("dense", x.shape)
        emb = self.emb_layer(x)
        return emb

class spectrogram_branch_msn(nn.Module):
    def __init__(self, input_size: tuple, Tc_scales: list, Fc_scales: list) -> None:
        super().__init__()

        self.Tc_scales=Tc_scales
        self.Fc_scales=Fc_scales

        # self.freq_layernorm = nn.LayerNorm(input_size[-1])# negative effect
        self.within_freq_layer = nn.Sequential(OrderedDict([
            ("print_ori", PrintLayer4test()),
            ("conv2d_1", conv2d_padding_same(256, 32, kernel_size=(3,3), stride=(2,2))),
            ("batchnorm_1", nn.BatchNorm2d(32)),
            ("ReLU", nn.ReLU()),
            ("print1", PrintLayer4test()),
            ("residual_1", resBlock(32, 64, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_3", nn.BatchNorm2d(64)),
            ("print3", PrintLayer4test()),
            ("residual_2", resBlock(64, 128, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_4", nn.BatchNorm2d(128)),
            ("print4", PrintLayer4test()),
            ("residual_3", resBlock(128, 128, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_5", nn.BatchNorm2d(128)),
            ("print5", PrintLayer4test()),
            ("residual_4", resBlock(128, 128)),
            ("print6", PrintLayer4test()),
            # ("maxpooling_2", maxpool2d_padding_same((2, 2), stride=(2, 2))),
            ("pooling", StatsPool()),
        ]))
        self.fc_layer = nn.Sequential(
            nn.Linear(128*len(Tc_scales)*len(Fc_scales)*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

        # self.layernorm = nn.LayerNorm(input_size[-1])
        self.init_selayer = SEModule_combine_v2(freq_ch=513, time_ch=563, fea_ch=1)
        self.down_1 = nn.Sequential(OrderedDict([
            ("conv2d_1", conv2d_padding_same(1, 16, kernel_size=(7,7), stride=(2,2))),
            ("batchnorm_1", nn.BatchNorm2d(16)),
            ("ReLU", nn.ReLU()),
            ("maxpooling", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("batchnorm_2", nn.BatchNorm2d(16)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=128, time_ch=141, fea_ch=16))
        ]))
        self.keep_1 = nn.Sequential(OrderedDict([
            ("residual_1", resBlock(16, 16, kernel_size=(3,3))),
            ("batchnorm_2", nn.BatchNorm2d(16)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=128, time_ch=141, fea_ch=16)),
            ("residual_2", resBlock(16, 16, kernel_size=(3,3))),
        ]))
        self.down_2 = nn.Sequential(OrderedDict([
            ("batchnorm_1", nn.BatchNorm2d(16)),
            ("residual_1", resBlock(16, 32, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_2", nn.BatchNorm2d(32)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=65, time_ch=70, fea_ch=32)),
            ("residual_2", resBlock(32, 32, kernel_size=(3,3))),
        ]))
        self.down_3 = nn.Sequential(OrderedDict([
            ("batchnorm_1", nn.BatchNorm2d(32)),
            ("residual_1", resBlock(32, 64, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_2", nn.BatchNorm2d(64)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=32, time_ch=36, fea_ch=64)),
            ("residual_2", resBlock(64, 64, kernel_size=(3,3))),
        ]))
        self.down_4 = nn.Sequential(OrderedDict([
            ("batchnorm_1", nn.BatchNorm2d(64)),
            ("residual_1", resBlock(64, 128, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_2", nn.BatchNorm2d(128)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=16, time_ch=18, fea_ch=128)),
            ("residual_2", resBlock(128, 128, kernel_size=(3,3)))
        ]))
        self.down_5 = nn.Sequential(OrderedDict([
            ("batchnorm_1", nn.BatchNorm2d(128)),
            ("residual_1", resBlock(128, 256, kernel_size=(3,3), stride=(2,2), downsampling=True)),
            ("batchnorm_2", nn.BatchNorm2d(256)),
            ("print_1", PrintLayer()),
            ("selayer", SEModule_combine_v2(freq_ch=8, time_ch=9, fea_ch=256)),
            ("residual_2", resBlock(256, 256, kernel_size=(3,3)))
        ]))

        self.maxpool = maxpool2d_padding_same((9, 8), stride=(9, 8))
        self.latent_dim = self.maxpool(self.down_5(self.down_4(self.down_3(self.down_2(self.keep_1(self.down_1(torch.ones(1, *input_size).transpose(2,3))))))))
        _, C, H, W = self.latent_dim.size()
        self.emb_layer = nn.Sequential(OrderedDict([
            ("batchnorm", nn.BatchNorm1d(C*H*W+256)),
            ("dense_1", nn.Linear(C*H*W+256, 1024)),
            ("dense_2", nn.Linear(1024, 128)),
        ]))

    def forward(self, x):
        B, C, Freq, Tt = x.shape

        freq_output = []

        # Tc_scales = Tc_scales
        # Fc_scales = Fc_scales

        num_F=8; num_T=32

        for Fc in self.Fc_scales:
            for Tc in self.Tc_scales:
                # Fc_step, Tc_step = Fc//2, Tc//2
                Fc_step = max(1, (Freq - Fc) // (num_F - 1)) if num_F > 1 else Freq - Fc
                Tc_step = max(1, (Tt - Tc) // (num_T - 1)) if num_T > 1 else Tt - Tc

                F_indices = [i * Fc_step for i in range(num_F)]
                T_indices = [j * Tc_step for j in range(num_T)]

                if F_indices[-1] + Fc < Freq:
                    F_indices[-1] = Freq - Fc
                if T_indices[-1] + Tc < Tt:
                    T_indices[-1] = Tt - Tc

                fx = []
                for f_start in F_indices:
                    for t_start in T_indices:
                        chunk = x[:, :, f_start:f_start + Fc, t_start:t_start + Tc]
                        fx.append(chunk)

                fx = torch.stack(fx, dim=0).transpose(0,1).reshape(B, -1, Fc, Tc).contiguous()

                freq_output.append(self.within_freq_layer(fx))

        freq_output = torch.hstack(freq_output)
        # print(freq_output.shape)
        freq_output = self.fc_layer(freq_output)

        # origin spectrogram
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x.transpose(2,3)
        # print("origin:", fx.shape, tx.shape, torch.sum(torch.isnan(fx)), torch.sum(torch.isnan(tx)))
        # print("after:", freq_output.shape, time_output.shape)
        # print("freq and time", torch.sum(torch.isnan(freq_output.cpu())), torch.sum(torch.isnan(time_output.cpu())))
        output = self.init_selayer(x)
        output = self.down_1(output)
        output = self.keep_1(output)
        output = self.down_2(output)
        output = self.down_3(output)
        output = self.down_4(output)
        output = self.down_5(output)
        output = self.maxpool(output)
        output = output.view(B, -1)

        selayer_outputs = {}
        selayer_outputs['init_selayer'] = self.init_selayer.maps
        selayer_outputs['down_1'] = self.down_1[-1].maps
        selayer_outputs['keep_1'] = self.keep_1[-2].maps 
        selayer_outputs['down_2'] = self.down_2[-2].maps
        selayer_outputs['down_3'] = self.down_3[-2].maps
        selayer_outputs['down_4'] = self.down_4[-2].maps
        selayer_outputs['down_5'] = self.down_5[-2].maps

        freq_output = freq_output.view(B, -1)
        emb = self.emb_layer(torch.hstack([output, freq_output]))

        return emb, selayer_outputs

class emb_model_MSN(nn.Module):
    def __init__(self, spectrum_size: tuple, spectrogram_size: tuple, Tc_scales=[16, 32, 64], Fc_scales = [32, 64, 128, 256, 500]) -> None:
        super().__init__()
        self.spectrum_branch = spectrum_branch(spectrum_size)
        self.spectrogram_branch = spectrogram_branch_msn(spectrogram_size, Tc_scales=Tc_scales, Fc_scales=Fc_scales)
    
    def forward(self, x: dict):
        spectrum_emb = self.spectrum_branch(x['spectrum'])
        spectrogram_emb, maps = self.spectrogram_branch(x['spectrogram'])
        return {
            'spectrum_emb':    spectrum_emb,
            'spectrogram_emb': spectrogram_emb,
            'combo_emb':       torch.hstack([spectrum_emb, spectrogram_emb]),
        }


