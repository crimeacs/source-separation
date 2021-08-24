
import argparse, yaml


from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from sedenoss.data import DataModule

# generate_data_samples(test_dataset)

if __name__ == "__main__":
    data_path = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_signal.nc'
    noise_path = '/gdrive/MyDrive/Seismic GAN/STEAD_data_JUL_2021/waveforms_noise.nc'

    train_dataset, val_dataset = DataModule.produce_datasets(data_path, noise_path)

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