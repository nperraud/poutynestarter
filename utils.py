import os
import shutil

from conf import PATH_SUMMARY, PATH_CHECKPOINTS
from pathlib import Path
from poutyne import TensorBoardLogger, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter


def saferm(path):
    """Remove the folder."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + str(path))
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + srt(path))

def metric_flatten(func):
    def inner(y1, y2, *args, **kwargs):
        return func(y1.flatten(), y2.flatten(), *args, **kwargs)
    inner.__name__ = "f_"+func.__name__
    return inner

def get_poutyne_callbacks(experiment_name):

    summary_dir = PATH_SUMMARY / Path(experiment_name)
    checkpoint_dir = PATH_CHECKPOINTS / Path(experiment_name)
    summary_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(str(checkpoint_dir / Path('last_epoch.ckpt'))),

        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(str(checkpoint_dir / Path('best_epoch_{epoch}_mse_{val_loss:.2}.ckpt')), monitor='val_loss', mode='min', 
                        save_best_only=True, restore_best=True, verbose=True),

        TensorBoardLogger(SummaryWriter(summary_dir))
    ]

    return callbacks, summary_dir, checkpoint_dir