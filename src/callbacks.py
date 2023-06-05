"""
Callbacks
"""

import torch

import pytorch_lightning as pl

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



# make a pytorch hook to record activations

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
