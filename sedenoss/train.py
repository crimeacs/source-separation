# # @title Define Loss function
# from itertools import permutations
#
# import numpy as np
# import xarray
# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from scipy.signal import sawtooth, square, detrend
#
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as data_utils
# from torch.utils.data import Dataset, DataLoader
# from torchtools.optim import Ranger
# from torchtools.nn import Mish
# from torch_audiomentations import Compose, Gain, HighPassFilter, LowPassFilter, PolarityInversion, Shift, PeakNormalization
# from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
#
# import pytorch_lightning as pl
# pl.seed_everything(42)
#
# import os
# import random
# import glob
# import io
#
# import matplotlib.pyplot as plt
# from matplotlib.colorbar import make_axes
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import LogLocator, NullFormatter
# from matplotlib.patheffects import withStroke
#
# import obspy
# from obspy.clients.fdsn import Client
# from obspy import UTCDateTime
# from obspy.signal.filter import envelope
# from obspy import UTCDateTime
# from obspy.imaging.util import _set_xaxis_obspy_dates
#
#
# from scipy.signal import spectrogram as _spectrogram
# from scipy.ndimage import uniform_filter
#
# from warnings import warn
#
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
#
# # @title Prepare the data and set augmentations
# if denoising_mode:
#     results = []
#     for line in train_attributes.snr_db:
#         results.append(process_snr(line))
#
#     snr = pd.DataFrame(results, columns=['Z', 'N', 'E'])
#     ind = snr[snr.Z > 20].index
#
#     print('Using %.0f per cent of the data' % (len(ind) / len(train_attributes) * 100))
#     print('Using %i samples' % (len(ind)))
#
#     train_val_data = full_data[ind][:-1000]
#     test_data = full_data[ind][-1000:]
#
# else:
#     train_val_data = full_data[:-1000]
#     test_data = full_data[-1000:]
#
# train_val_noise = full_noise[:-1000]
# test_noise = full_noise[-1000:]
#
# # Define augmentations
# # Initialize augmentation callable
# if denoising_mode:
#     augmentation_signal = Compose(
#         transforms=[
#             HighPassFilter(
#                 min_cutoff_freq=0.5,
#                 max_cutoff_freq=1.5,
#                 mode="per_example",
#                 p=1,
#             ),
#
#             LowPassFilter(
#                 min_cutoff_freq=10,
#                 max_cutoff_freq=14,
#                 mode="per_example",
#                 p=0.5,
#             ),
#
#             PolarityInversion(p=0.5)
#         ]
#     )
#
#     augmentation_noise = Compose(
#         transforms=[
#             Gain(
#                 min_gain_in_db=-10.0,
#                 max_gain_in_db=0,
#                 p=1,
#             ),
#
#             HighPassFilter(
#                 min_cutoff_freq=0.1,
#                 max_cutoff_freq=1,
#                 mode="per_example",
#                 p=0.5,
#             ),
#
#             LowPassFilter(
#                 min_cutoff_freq=10,
#                 max_cutoff_freq=14,
#                 mode="per_example",
#                 p=0.5,
#             ),
#
#             PolarityInversion(p=0.5)
#         ]
#     )
# else:
#     augmentation_noise = Compose(
#         transforms=[
#
#             HighPassFilter(
#                 min_cutoff_freq=0.5,
#                 max_cutoff_freq=1.5,
#                 mode="per_channel",
#                 p=0.5,
#             ),
#
#             LowPassFilter(
#                 min_cutoff_freq=5,
#                 max_cutoff_freq=14,
#                 mode="per_channel",
#                 p=0.5,
#             ),
#
#             PolarityInversion(p=0.5),
#
#             PeakNormalization(
#                 mode="per_example",
#                 p=1
#             ),
#
#         ]
#     )
# augmentation_signal = augmentation_noise
#
# if 'data_train' not in globals():
#     data_train, data_val  = train_test_split(train_val_data, train_size=0.95)
#     noise_train, noise_val  = train_test_split(train_val_noise, train_size=0.95)
#
# train_dataset = TrainSignals(data_train, noise_train, transform_signal=augmentation_signal, transform_noise=augmentation_noise, denoising_mode=denoising_mode)
# val_dataset   = TrainSignals(data_val, noise_val, transform_signal=False, transform_noise=False, denoising_mode=denoising_mode)
# test_dataset  = TrainSignals(test_data, test_noise, transform_signal=augmentation_signal, transform_noise=augmentation_noise, denoising_mode=denoising_mode, test=True)
#
# generate_data_samples(test_dataset)
#
# mc = ModelCheckpoint(monitor='val_loss', save_top_k=3)
#
# dm = DataModule()
# model = FaSNet_base()#.load_from_checkpoint(checkpoint_path="/gdrive/MyDrive/TraML/lightning_logs/version_29/checkpoints/epoch=4-step=12094.ckpt")
#
# trainer = pl.Trainer(gpus=-1,
#                      progress_bar_refresh_rate=50,
#                      default_root_dir='/gdrive/MyDrive/TraML/',
#                      benchmark=True,
#
#                      max_epochs=200,
#                      terminate_on_nan=True,
#
#                      callbacks=[mc],
#                      num_sanity_val_steps=1,
#                      precision=16,
#
#                     #  fast_dev_run = True,
#                      )
#
# trainer.fit(model, dm)