# @title Define Loss function
from itertools import permutations

import numpy as np
import xarray
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.signal import sawtooth, square, detrend

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from torchtools.optim import Ranger
from torchtools.nn import Mish
from torch_audiomentations import Compose, Gain, HighPassFilter, LowPassFilter, PolarityInversion, Shift, PeakNormalization
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

import pytorch_lightning as pl
pl.seed_everything(42)

import os
import random
import glob
import io

import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.patheffects import withStroke

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.filter import envelope
from obspy import UTCDateTime
from obspy.imaging.util import _set_xaxis_obspy_dates


from scipy.signal import spectrogram as _spectrogram
from scipy.ndimage import uniform_filter

from warnings import warn

class FaSNet_base(pl.LightningModule):

    def __init__(self,
                 enc_dim: int = 128,
                 feature_dim: int = 128,
                 hidden_dim: int = 64,
                 layer: int = 1,
                 segment_size: int = 200,
                 nspk: int = 2,
                 win_len: int = 2,
                 lr: float = 1e-3,
                 step_size: int = 1,
                 gamma: float = 0.9,
                 **kwargs):

        super(FaSNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        self.step_size = step_size
        self.gamma = gamma

        self.save_hyperparameters()
        # waveform encoder

        self.encoder = Encoder(win_len, enc_dim)  # [B T]-->[B N L]

        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)  # [B N L]-->[B N L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim,
                                   self.num_spk, self.layer, self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(enc_dim, win_len)
        self.mish = Mish()

        # self.upscale = nn.Conv2d(self.enc_dim-2,self.enc_dim,1)
        # self.gaussian = nn.Sequential(GaussianLayer(2,3),
        #                               Mish())

        self.criterion = SI_SDR_Loss()

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch, T)
        """
        # pass to a DPRNN
        # input = input.to(device)
        B, _ = input.size()

        # mixture, rest = self.pad_input(input, self.window)
        # print('mixture.shape {}'.format(mixture.shape))
        mixture_w = self.encoder(input)  # B, E, L
        score_ = self.enc_LN(mixture_w)  # B, E, L
        score_ = self.separator(score_)  # B, nspk, T, N
        score_ = score_.view(B * self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nspk, N, T
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        score = score.view(B, self.num_spk, self.enc_dim, -1)  # [B*nspk, E, L] -> [B, nspk, E, L]

        # est_mask = self.gaussian(score)
        # est_mask = self.upscale(est_mask.permute(0,2,1,3)).permute(0,2,1,3)

        # est_mask = nn.Softmax(dim=1)(est_mask)

        # est_mask = nn.Softmax(dim=1)(score)
        est_mask = F.softmax(score / score.flatten().std(), dim=1)

        # print(est_mask_old.size(), est_mask.size())
        est_source = self.decoder(mixture_w, est_mask)  # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        # est_source = self.decoder(mixture_w, score) # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]
        return est_source

    def training_step(self, batch, batch_idx):
        signal, n_sources = batch
        mix = torch.sum(signal, dim=1, keepdim=False)
        estimates = model(mix)

        loss, reorder_estimate_source = self.criterion(signal, estimates)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.global_step % 100 == 0:
            self.prediction = [signal, reorder_estimate_source]
            self.logger.experiment.add_image('Results', plot_tb_figure(self.prediction[0], self.prediction[1]),
                                             self.global_step, dataformats='HWC')
        self.logger.experiment.flush()

        return loss

    def validation_step(self, batch, batch_idx):
        signal, n_sources = batch
        mix = torch.sum(signal, dim=1, keepdim=False)
        estimates = model(mix)

        loss, reorder_estimate_source = self.criterion(signal, estimates)

        self.prediction = [signal, reorder_estimate_source]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.flush()
        return loss

    def test_step(self, batch, batch_idx):
        signal, n_sources = batch
        mix = torch.sum(signal, dim=1, keepdim=False)
        estimates = model(mix)

        loss, reorder_estimate_source = self.criterion(signal, estimates)

        self.prediction = [signal, reorder_estimate_source]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        step_size = self.hparams.step_size
        gamma = self.hparams.gamma

        optimizer = Ranger(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                     'interval': 'epoch'}

        return [optimizer], [scheduler]

    def on_epoch_end(self):

        self.logger.experiment.add_image('Results', plot_tb_figure(self.prediction[0], self.prediction[1]),
                                         self.current_epoch, dataformats='HWC')
        self.logger.experiment.flush()