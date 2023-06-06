"""
Callbacks
"""

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


# class ActivationRecordingCallback(pl.Callback):
#     """
#     Callback to record the activations of a network.
#     """
#     def __init__(self):
#         super(ActivationRecordingCallback, self).__init__()
#         self.activations = []
#
#     def on_forward(self, trainer, pl_module, batch, batch_idx, output, **kwargs):
#         self.activations.append(output.detach().clone())


class LogPredictionsCallback(Callback):

    def __init__(self, wandb_logger, save_n=20, log_every=1):
        """
        Callback to log predictions to wandb.
        Args:
            wandb_logger:  wandb logger object
            save_n: number of predictions to log
            log_every: log predictions every XXX batches
        """
        super(LogPredictionsCallback, self).__init__()
        self.wandb_logger = wandb_logger
        self.save_n = save_n
        self.log_every = log_every

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx % self.log_every == 0:
            n = min(len(batch), self.save_n)
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]

            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            self.wandb_logger.log_table(key='sample_table', columns=columns, data=data)


class ActivationRecordingHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activations = []

    def hook_fn(self, module, input, output):
        self.activations.append(output.detach().clone())

    def close(self):
        self.hook.remove()
        del self

    def __del__(self):
        self.close()
