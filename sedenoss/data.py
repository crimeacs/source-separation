import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import random


# @title Define dataset class
class TrainSignals(Dataset):
    def __init__(self, data, noise, transform_signal=False, transform_noise=False, denoising_mode=False, test=False):
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

    def __init__(self, batch_size: int = 192, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

