"""
Callbacks
"""

import functools

import h5py

import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


def record_weights(module):
    weights = {}
    for name, param in module.named_parameters():
        if param.requires_grad and 'weight' in name:
            weights[name] = param.detach().clone()
    return weights


from pytorch_lightning.loggers import CSVLogger


# class BatchCSVLogger(CSVLogger):
#     def on_batch_end(self, trainer, pl_module):
#         # Access the current batch information from the trainer
#         batch_idx = trainer.batch_idx
#         global_step = trainer.global_step
#
#         # Log the batch-level information
#         metrics = trainer.callback_metrics
#         self.log_metrics(metrics, step=global_step, epoch=trainer.current_epoch, batch=batch_idx)
#
#         # Call the original on_batch_end method from CSVLogger
#         super().on_batch_end(trainer, pl_module)
#
#     def log_metrics(self, metrics, step=None, epoch=None, batch=None):
#         for key, value in metrics.items():
#             self.experiment.log_metric(key, value, step=step, epoch=epoch, batch=batch)


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

    def __init__(self, wandb_logger, save_n=20, log_every=256):
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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
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
        self.activations = {}
        self.hook_fns = []

        for name, module in module.named_modules():
            self.activations[name] = []
            self.hook_fns.append(module.register_forward_hook(
                functools.partial(self.hook_fn, name=name)
            ))

    def hook_fn(self, module, input, output, name=None):
        if name is not None:
            self.activations[name].append(output.detach().clone())
        else:
            print("No name for module, not recording activations")

    def close(self):
        for hook in self.hook_fns:
            hook.remove()
        self.activations = None


class GradientRecordingHook:
    def __init__(self, module):
        self.gradients = {}
        self.hook_fns = []

        for name, module in module.named_modules():
            self.gradients[name] = []
            self.hook_fns.append(module.register_backward_hook(
                functools.partial(self.hook_fn, name=name)
            ))

    def hook_fn(self, module, grad_input, grad_output, name=None):
        if name is not None:
            self.gradients[name].append(grad_output[0].detach().clone())
        else:
            print("No name for module, not recording gradients")

    def close(self):
        for hook in self.hook_fns:
            hook.remove()
        self.gradients = None


class WeightGradientRecordingHook:
    def __init__(self, module):
        self.weights = {}
        self.hook_fns = []

        for name, module in module.named_modules():
            if hasattr(module, 'weight'):
                self.weights[name] = []
                self.hook_fns.append(module.weight.register_hook(
                    functools.partial(self.hook_fn, name=name)
                ))

    def hook_fn(self, grad, name=None):
        if name is not None:
            self.weights[name].append(grad.detach().clone())
        else:
            print("No name for module, not recording weights")

    def close(self):
        for hook in self.hook_fns:
            hook.remove()
        self.weights = None


class LogActivationsCallback(Callback):
    """
    Callback that records activations of the model during each validation batch.
    """

    def __init__(self, h5_filepath):
        """
        Args:
            h5_filepath: path to save activations to
        """
        super(LogActivationsCallback, self).__init__()
        self.h5_filepath = h5_filepath
        self.current_epoch = None
        self.current_hook_train = None
        self.current_hook_validation = None

    # def _hook_fn(self, module, input, output, name=None, activations=None):
    #     if name is not None:
    #         activations[name].append(output.detach().clone())
    #     else:
    #         print("No name for module, not recording activations")

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # delete previous activations by deleting the file. otherwise extra activations from the sanity check being run on the validation epoch will be saved
        if self.h5_filepath.exists():
            self.h5_filepath.unlink()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = trainer.current_epoch
        self.current_hook_train = ActivationRecordingHook(pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"Saving training activations for epoch {self.current_epoch} to {self.h5_filepath}")

        if self.current_epoch is not None:
            # write to h5 file
            with h5py.File(self.h5_filepath, 'a') as f:
                for layer_name, layer_activations in self.current_hook_train.activations.items():
                    if layer_name and len(layer_activations) > 0:
                        processed_activations = torch.cat(layer_activations, dim=0).detach().cpu().numpy()
                        # indexing name under epoch_XXX/layer_name (with leading zeros)
                        epoch_name = f'epoch_{self.current_epoch:03d}'
                        layer_name = f'{layer_name}'

                        # if epoch_name/train not in f:
                        f.create_dataset(f'{epoch_name}/train/{layer_name}', data=processed_activations)

        self.current_hook_train.close()
        self.current_hook_train = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = trainer.current_epoch
        self.current_hook_validation = ActivationRecordingHook(pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"Saving validation activations for epoch {self.current_epoch} to {self.h5_filepath}")

        if self.current_epoch is not None:
            # write to h5 file
            with h5py.File(self.h5_filepath, 'a') as f:
                for layer_name, layer_activations in self.current_hook_validation.activations.items():
                    if layer_name and len(layer_activations) > 0:
                        processed_activations = torch.cat(layer_activations, dim=0).detach().cpu().numpy()
                        # indexing name under epoch_XXX/layer_name (with leading zeros)
                        epoch_name = f'epoch_{self.current_epoch:03d}'
                        layer_name = f'{layer_name}'

                        # if epoch_name/validation not in f:
                        f.create_dataset(f'{epoch_name}/validation/{layer_name}', data=processed_activations)

        self.current_hook_validation.close()

        # delete the hook from memory
        self.current_hook_validation = None


class LogGradientsCallback(Callback):
    """
    Callback that records gradients of the model during each validation batch.
    """

    def __init__(self, h5_filepath):
        """
        Args:
            h5_filepath: path to save gradients to
        """
        super(LogGradientsCallback, self).__init__()
        self.h5_filepath = h5_filepath
        self.current_epoch = None
        self.current_hook_train = None

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # delete previous gradients by deleting the file
        if self.h5_filepath.exists():
            self.h5_filepath.unlink()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = trainer.current_epoch
        self.current_hook_train = GradientRecordingHook(pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"Saving training gradients for epoch {self.current_epoch} to {self.h5_filepath}")

        if self.current_epoch is not None:
            # write to h5 file
            with h5py.File(self.h5_filepath, 'a') as f:
                for layer_name, layer_gradients in self.current_hook_train.gradients.items():
                    if layer_name and len(layer_gradients) > 0:
                        processed_gradients = torch.cat(layer_gradients, dim=0).detach().cpu().numpy()
                        # indexing name under epoch_XXX/layer_name (with leading zeros)
                        epoch_name = f'epoch_{self.current_epoch:03d}'
                        layer_name = f'{layer_name}'

                        f.create_dataset(f'{epoch_name}/train/{layer_name}', data=processed_gradients)

        self.current_hook_train.close()
        self.current_hook_train = None


class LogWeightGradientsCallback(Callback):
    """
    Callback that records weight gradients of the model during each validation batch.
    """

    def __init__(self, h5_filepath):
        """
        Args:
            h5_filepath: path to save weights to
        """
        super(LogWeightGradientsCallback, self).__init__()
        self.h5_filepath = h5_filepath
        self.current_epoch = None
        self.current_hook_train = None

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # delete previous weights by deleting the file
        if self.h5_filepath.exists():
            self.h5_filepath.unlink()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = trainer.current_epoch
        self.current_hook_train = WeightGradientRecordingHook(pl_module)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"Saving training weight gradients for epoch {self.current_epoch} to {self.h5_filepath}")

        if self.current_epoch is not None:
            # write to h5 file
            with h5py.File(self.h5_filepath, 'a') as f:
                for layer_name, layer_weights in self.current_hook_train.weights.items():
                    if layer_name and len(layer_weights) > 0:
                        processed_weights = torch.stack(layer_weights, dim=0).detach().cpu().numpy()
                        # indexing name under epoch_XXX/layer_name (with leading zeros)
                        epoch_name = f'epoch_{self.current_epoch:03d}'
                        layer_name = f'{layer_name}'

                        f.create_dataset(f'{epoch_name}/train/{layer_name}', data=processed_weights)

        self.current_hook_train.close()
        self.current_hook_train = None


class LogWeightCallback(Callback):
    """
    Callback that records weights of the model during each validation batch.
    """

    def __init__(self, h5_filepath):
        """
        Args:
            h5_filepath: path to save weights to
        """
        super(LogWeightCallback, self).__init__()
        self.h5_filepath = h5_filepath
        self.current_epoch = None

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # delete previous weights by deleting the file
        if self.h5_filepath.exists():
            self.h5_filepath.unlink()

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.current_epoch = trainer.current_epoch
        weights = record_weights(pl_module)
        # write to h5 file
        print(f"Saving training weights for epoch {self.current_epoch} to {self.h5_filepath}")
        with h5py.File(self.h5_filepath, 'a') as f:
            epoch_name = f'epoch_{self.current_epoch:03d}'
            for layer_name, layer_weights in weights.items():
                if layer_name:
                    f.create_dataset(f'{epoch_name}/train/{layer_name}', data=layer_weights)
