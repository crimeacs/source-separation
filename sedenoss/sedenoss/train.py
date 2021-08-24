from sedenoss.data import DataModule
from sedenoss.models import FaSNet_base
from pytorch_lightning.utilities.cli import LightningCLI

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cli = LightningCLI(FaSNet_base, DataModule)