from sedenoss import data, loss, models, train, utils

#Fix the random seed for reporducability
import pytorch_lightning as pl
pl.seed_everything(42)