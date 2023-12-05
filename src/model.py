import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from src.utils import calc_padding, init_weights

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, lrelu_slope=0.1, dilations=(1, 3, 5)):
        super(ResidualBlock, self).__init__()
        self.res_parts = nn.ModuleList()
        for d in dilations:
            self.res_parts.append(nn.Sequential(
                nn.LeakyReLU(lrelu_slope),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, calc_padding(kernel_size, d), dilation=d))
            ))

    def forward(self, x):
        for res_part in self.res_parts:
            x += res_part(x.clone())
        return x

class MRFLayer(nn.Module):
    def __init__(self, channels):
        super(MRFLayer, self).__init__()
        kernel_sizes = [3, 7, 11]
        dilations = (1, 3, 5)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels, k, dilations=dilations) for k in kernel_sizes
        ])


    def forward(self, x):
        return sum(res(x) for res in self.res_blocks) / len(self.res_blocks)

class Generator(nn.Module):
    def __init__(self, in_channels=80):
        super(Generator, self).__init__()
        fl_kernel_size = 7
        kernel_sizes = (16, 16, 4, 4)
        strides = (8, 8, 2, 2)
        channels = 512

        blocks = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels, channels, fl_kernel_size, padding=fl_kernel_size // 2))
        ])

        for kernel_size, stride in zip(kernel_sizes, strides):
            blocks.append(
                nn.Sequential(
                    weight_norm(nn.ConvTranspose1d(channels, channels // 2, kernel_size, stride, padding=(kernel_size - stride) // 2)),
                    MRFLayer(channels // 2)
                )
            )
            channels //= 2

        blocks.append(weight_norm(nn.Conv1d(channels, 1, fl_kernel_size, padding=fl_kernel_size // 2)))
        self.model = nn.Sequential(*blocks)
        self.apply(init_weights)

    def forward(self, x):
        x = self.model(x).squeeze(1)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        disc_periods = (2, 3, 5, 7, 11)
        self.mpd_discriminators = nn.ModuleList([
            PeriodDiscriminator(p, weight_norm) for p in disc_periods
        ])
        self.msd_discriminators = nn.ModuleList([
            ScaleDiscriminator(spectral_norm),
            ScaleDiscriminator(weight_norm),
            ScaleDiscriminator(weight_norm)
        ])
        self.post_pooling = nn.AvgPool1d(4, 2, 2)
        self.apply(init_weights)

    def forward(self, x):
        preds = []
        layer_acts = []

        for model in self.mpd_discriminators:
            pred, hidden = model(x)
            preds.append(pred)
            layer_acts.append(hidden)

        for model in self.msd_discriminators:
            pred, hidden = model(x)
            preds.append(pred)
            layer_acts.append(hidden)
        x = self.post_pooling(x)

        return torch.cat(preds, 1), torch.cat(layer_acts, 1)

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, norm_f):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.activation = nn.LeakyReLU(0.1)
        kernel_size = 5
        stride = 3

        self.layers = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(calc_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(calc_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(calc_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(calc_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))
        ])
        self.post_conv = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        T = x.size(1)
        if T % self.period:
            x = nn.functional.pad(x, (0, self.period - T % self.period), "reflect")
        channels = T // self.period + (1 if T % self.period else 0)
        x = x.view(-1, 1, channels, self.period)

        layer_acts = []
        for layer in self.layers:
            x = self.activation(layer(x))
            layer_acts.append(x.flatten(1))
        x = self.post_conv(x)
        layer_acts.append(x.flatten(1))

        return x.flatten(1, -1), torch.cat(layer_acts, 1)

class ScaleDiscriminator(nn.Module):
    def __init__(self, norm_f):
        super(ScaleDiscriminator, self).__init__()
        self.activation = nn.LeakyReLU(0.1)

        self.layers = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        ])
        self.post_conv = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        x = x.unsqueeze(1)

        layer_acts = []
        for layer in self.layers:
            x = self.activation(layer(x))
            layer_acts.append(x.flatten(1))
        x = self.post_conv(x)
        layer_acts.append(x.flatten(1))

        return x.flatten(1, -1), torch.cat(layer_acts, 1)
        x = self.post_conv(x)
        layer_acts.append(x.flatten(1))

        return x.flatten(1, -1), torch.cat(layer_acts, 1)
