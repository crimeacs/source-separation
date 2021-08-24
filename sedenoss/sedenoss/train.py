
import argparse, yaml


from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sedenoss.data import DataModule
from sedenoss.models import FaSNet_base
from pytorch_lightning.utilities.cli import LightningCLI

# generate_data_samples(test_dataset)

if __name__ == "__main__":

    #
    # print('Loading data into a dataset module, may take sometime depending on the size of the data')
    #
    # mc = ModelCheckpoint(monitor='val_loss', save_top_k=3)
    #
    # dm = DataModule(data_path=data_path, noise_path=noise_path)
    # model = FaSNet_base()  # .load_from_checkpoint(checkpoint_path="/gdrive/MyDrive/TraML/lightning_logs/version_29/checkpoints/epoch=4-step=12094.ckpt")
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
    #                      #  fast_dev_run = True,
    #                      )
    data_path = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/sample_data.nc'
    noise_path = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/sample_noise.nc'

    # dm = DataModule(data_path=data_path, noise_path=noise_path)
    # model = FaSNet_base# .load_from_checkpoint(checkpoint_path="/gdrive/MyDrive/TraML/lightning_logs/version_29/checkpoints/epoch=4-step=12094.ckpt")
    # 
    cli = LightningCLI(FaSNet_base, DataModule)