from torchtools.optim import Ranger
from torchtools.nn import Mish

import pytorch_lightning as pl

from sedenoss.utils import *
from sedenoss.loss import SI_SDR_Loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPS = 1e-8

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """Custom Conv2d module. Refer to nn.Conv2d documentation
        """
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        print(weight.size())
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        """Custom Conv1d module. Refer to nn.Conv1d documentation
        """
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



# Based on https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation implementation
class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
        Args:
            mixture: [B, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, N, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
    """

    def __init__(self, W=2, N=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.N = W, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Sequential(nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False),
                                      Mish(),
                                      )
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=W, stride=W // 2, bias=False)
        self.mish = Mish()

    def forward(self, mixture):

        mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
        mixture_w = self.mish(self.conv1d_U(mixture))  # [B, N, L]
        return mixture_w


class Decoder(nn.Module):
    """
    Decoder module.
    Args:
        mixture_w: [B, E, L]
        est_mask: [B, C, E, L]
    Returns:
        est_source: [B, C, T]
    """
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask):
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3)  # [B, C, L, E]

        est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W // 2)  # B x C x T
        return est_source


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0.2, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, hidden = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        self.output = nn.Sequential(Mish(),
                                    nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        batch_size, _, dim1, dim2 = input.shape
        output = input

        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N

            #             row_input = self.attn(row_input.permute(0,2,1)).view(batch_size * dim2, dim1, -1)

            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H

            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            #             col_input = self.attn(col_input.permute(1,2,0)).view(batch_size * dim1, dim2, -1)

            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H

            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                             2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output)  # B, output_size, dim1, dim2

        return output


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2,
                 layer=4, segment_size=100, bidirectional=True, rnn_type='LSTM'):
        """DPRNN base module.
        Args:
            input_dim: int
            feature_dim: int
            hidden_dim: int
            num_spk: int, refers to number of speakers
            layer: int, refers to number of layers,
            segment_size: int,
            bidirectional: bool,
            rnn_type: str, e.g. 'LSTM'
            """
        super(DPRNN_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPRNN model
        self.DPRNN = DPRNN(rnn_type, self.feature_dim, self.hidden_dim,
                           self.feature_dim * self.num_spk,
                           num_layers=layer, bidirectional=bidirectional)

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass


# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    """Beamforming module
    """
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, input):
        # input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input)  # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B, N, L, K: L is the segment_size
        # print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPRNN

        output = self.DPRNN(enc_segments).view(batch_size * self.num_spk, self.feature_dim, self.segment_size,
                                               -1)  # B*nspk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*nspk, N, T

        #         output = self.attn(output)

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, T

        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1,
                                                                self.feature_dim)  # B, nspk, T, N

        return bf_filter

class FaSNet_base(pl.LightningModule):
    """Model module used for the study
    Args:
        enc_dim: int, Encoder dimensions
        feature_dim: int, Feature dimensions
        hidden_dim: int, Hidden dimensions
        layer: int, number of layers to use
        segment_size: int, segment size to use
        nspk: int, number of speakers (sources)
        win_len: int, window length to use
        lr: float, learning rate
        step_size: int, step size for scheduling the optimization
        gamma: float, decay factor for scheduling the optimization
    """
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
                 gamma: float = 0.9):
        super().__init__()

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
        self.lr = lr

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
        B, _ = input.size()

        mixture_w = self.encoder(input)  # B, E, L
        score_ = self.enc_LN(mixture_w)  # B, E, L
        score_ = self.separator(score_)  # B, nspk, T, N
        score_ = score_.view(B * self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()  # B*nspk, N, T
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        score = score.view(B, self.num_spk, self.enc_dim, -1)  # [B*nspk, E, L] -> [B, nspk, E, L]

        est_mask = F.softmax(score / score.flatten().std(), dim=1)
        est_source = self.decoder(mixture_w, est_mask)  # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]
        return est_source

    def training_step(self, batch, batch_idx):
        signal, n_sources = batch
        mix = torch.sum(signal, dim=1, keepdim=False)
        estimates = self(mix)

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
        estimates = self(mix)

        loss, reorder_estimate_source = self.criterion(signal, estimates)

        self.prediction = [signal, reorder_estimate_source]
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.logger.experiment.flush()
        return loss

    def test_step(self, batch, batch_idx):
        signal, n_sources = batch
        mix = torch.sum(signal, dim=1, keepdim=False)
        estimates = self(mix)

        loss, reorder_estimate_source = self.criterion(signal, estimates)

        self.prediction = [signal, reorder_estimate_source]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        step_size = self.step_size
        gamma = self.gamma

        optimizer = Ranger(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                     'interval': 'epoch'}

        return [optimizer], [scheduler]

    def on_epoch_end(self):
        self.logger.experiment.add_image('Results', plot_tb_figure(self.prediction[0], self.prediction[1]),
                                         self.current_epoch, dataformats='HWC')
        self.logger.experiment.flush()