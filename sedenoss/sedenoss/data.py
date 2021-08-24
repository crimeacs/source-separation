import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import random
from sklearn.model_selection import train_test_split
from torch_audiomentations import Compose, Gain, HighPassFilter, LowPassFilter, PolarityInversion, PeakNormalization
import xarray

from sedenoss.utils import *

# @title Define dataset class
class TrainSignals(Dataset):
    def __init__(self, data, noise, transform_signal, transform_noise, denoising_mode=False, test=False):
        self.data = data
        self.noise = noise
        self.transform_signal = transform_signal
        self.transform_noise = transform_noise
        self.denoising_mode = denoising_mode

        self.n_sources = 2
        self.sampling_rate = 30
        self.test = test

    def __getitem__(self, idx):

        # Loading
        signal_1 = self.data[idx]
        np.random.seed()
        signals = []

        signal_1 = signal_1 - signal_1.mean()
        signal_1 = torch.from_numpy(signal_1).unsqueeze(1).float()

        if self.transform_signal != False:
            signal_1 = self.transform_signal(signal_1.T.unsqueeze(0), sample_rate=self.sampling_rate)
            signal_1 = taper(signal_1, func=torch.hann_window, max_percentage=0.05).view(1800, 1)
        signals.append(signal_1)

        for i in range(1, self.n_sources):
            if self.denoising_mode:
                if self.test:
                    signal = self.noise[idx]
                else:
                    signal = self.noise[np.random.randint(0, len(self.noise))]
            else:
                choice = random.choice([0, 1])
                if self.test:
                    signal = self.data[np.random.randint(0, len(self.data))]
                else:
                    if choice == 0:
                        signal = self.noise[np.random.randint(0, len(self.noise))]
                    else:
                        signal = self.data[np.random.randint(0, len(self.data))]

            signal = signal - signal.mean()
            signal = torch.from_numpy(signal).unsqueeze(1).float()

            if self.transform_noise != False:
                signal = self.transform_noise(signal.T.unsqueeze(0), sample_rate=self.sampling_rate)
                signal = taper(signal, func=torch.hann_window, max_percentage=0.05).view(1800, 1)

            signals.append(signal)

        signal = torch.cat(signals, dim=-1)
        signal = signal / signal.abs().max(dim=0, keepdim=True).values.max()
        return signal.T, self.n_sources

    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 192, num_workers: int = 4, denoising_mode: bool = False,
                 data_path: str = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_signal.nc',
                 noise_path: str = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_noise.nc',
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.denoising_mode = denoising_mode
        self.data_path = data_path
        self.noise_path = noise_path

        assert self.data_path[-2:] == 'nc', "Please provide data in .nc Xarray format"
        assert self.noise_path[-2:] == 'nc', "Please provide data in .nc Xarray format"

        da_data = xarray.open_dataset(self.data_path, engine='netcdf4')
        da_noise = xarray.open_dataset(self.noise_path, engine='netcdf4')
        train_val_data = da_data.sel(channel=0).to_array().values[0]
        train_val_noise = da_noise.sel(channel=0).to_array().values[0]

        data_train, data_val = train_test_split(train_val_data, train_size=0.95)
        noise_train, noise_val = train_test_split(train_val_noise, train_size=0.95)

        augmentation_signal, augmentation_noise = self.choose_agmentations()

        self.train_dataset = TrainSignals(data_train, noise_train, transform_signal=augmentation_signal,
                                     transform_noise=augmentation_noise, denoising_mode=self.denoising_mode)
        self.val_dataset = TrainSignals(data_val, noise_val, transform_signal=False, transform_noise=False,
                                   denoising_mode=self.denoising_mode)

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def choose_agmentations(self):
        # Define augmentations
        # Initialize augmentation callable
        if self.denoising_mode:
            augmentation_signal = Compose(
                transforms=[
                    HighPassFilter(
                        min_cutoff_freq=0.5,
                        max_cutoff_freq=1.5,
                        mode="per_example",
                        p=1,
                    ),

                    LowPassFilter(
                        min_cutoff_freq=10,
                        max_cutoff_freq=14,
                        mode="per_example",
                        p=0.5,
                    ),

                    PolarityInversion(p=0.5)
                ]
            )

            augmentation_noise = Compose(
                transforms=[
                    Gain(
                        min_gain_in_db=-10.0,
                        max_gain_in_db=0,
                        p=1,
                    ),

                    HighPassFilter(
                        min_cutoff_freq=0.1,
                        max_cutoff_freq=1,
                        mode="per_example",
                        p=0.5,
                    ),

                    LowPassFilter(
                        min_cutoff_freq=10,
                        max_cutoff_freq=14,
                        mode="per_example",
                        p=0.5,
                    ),

                    PolarityInversion(p=0.5)
                ]
            )
        else:
            augmentation_noise = Compose(
                transforms=[

                    HighPassFilter(
                        min_cutoff_freq=0.5,
                        max_cutoff_freq=1.5,
                        mode="per_channel",
                        p=0.5,
                    ),

                    LowPassFilter(
                        min_cutoff_freq=5,
                        max_cutoff_freq=14,
                        mode="per_channel",
                        p=0.5,
                    ),

                    PolarityInversion(p=0.5),

                    PeakNormalization(
                        mode="per_example",
                        p=1
                    ),

                ]
            )
            augmentation_signal = augmentation_noise

        return augmentation_signal, augmentation_noise

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=True, worker_init_fn=self.worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, worker_init_fn=self.worker_init_fn)
