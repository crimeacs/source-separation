import xarray

from sklearn.model_selection import train_test_split
from torch_audiomentations import Compose, Gain, HighPassFilter, LowPassFilter, PolarityInversion, PeakNormalization

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def choose_agmentations(denoising_mode):
    # Define augmentations
    # Initialize augmentation callable
    if denoising_mode:
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

def produce_datasets(data_path='/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_signal.nc',
                     noise_path='/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_noise.nc',
                     denoising_mode=True):

    da_data = xarray.open_dataset(data_path,
                                  engine='netcdf4')
    da_noise = xarray.open_dataset(noise_path,
                                   engine='netcdf4')
    train_val_data = da_data.sel(channel=0).to_array().values[0]
    train_val_noise = da_noise.sel(channel=0).to_array().values[0]

    if 'data_train' not in globals():
        data_train, data_val = train_test_split(train_val_data, train_size=0.95)
        noise_train, noise_val = train_test_split(train_val_noise, train_size=0.95)

    augmentation_signal, augmentation_noise = choose_agmentations(denoising_mode)

    train_dataset = TrainSignals(data_train, noise_train, transform_signal=augmentation_signal, transform_noise=augmentation_noise, denoising_mode=denoising_mode)
    val_dataset   = TrainSignals(data_val, noise_val, transform_signal=False, transform_noise=False, denoising_mode=denoising_mode)

    return train_dataset, val_dataset

# generate_data_samples(test_dataset)

mc = ModelCheckpoint(monitor='val_loss', save_top_k=3)

dm = DataModule()
model = FaSNet_base()#.load_from_checkpoint(checkpoint_path="/gdrive/MyDrive/TraML/lightning_logs/version_29/checkpoints/epoch=4-step=12094.ckpt")

trainer = pl.Trainer(gpus=-1,
                     progress_bar_refresh_rate=50,
                     default_root_dir='/gdrive/MyDrive/TraML/',
                     benchmark=True,

                     max_epochs=200,
                     terminate_on_nan=True,

                     callbacks=[mc],
                     num_sanity_val_steps=1,
                     precision=16,

                    #  fast_dev_run = True,
                     )

trainer.fit(model, dm)