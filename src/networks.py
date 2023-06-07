"""
Networks
"""

# imports
import functools
from typing import Sequence, Tuple, Callable, Text
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer


class MLPNet(pl.LightningModule):
    """
    Multilayer perceptron (MLP) network.

    Args:
        input_size: input size
        hidden_sizes: hidden layer sizes
        output_size: output size
        loss_fn: loss function
        activation: activation function
        lr: learning rate
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            loss_fn: Text,
            # activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            # final_activation: Callable[[torch.Tensor], torch.Tensor] = F.softmax,
            activation: Text = "relu",
            final_activation: Text = "identity",
            lr: float = 0.001,
            is_image: bool = True,
    ):
        super(MLPNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.is_image = is_image

        # loss function
        if loss_fn == "cross_entropy":
            self.loss_fn = F.cross_entropy
        elif loss_fn == "mse":
            self.loss_fn = F.mse_loss
        else:
            raise NotImplementedError(f"Loss function {loss_fn} not implemented.")

        # final activation
        if final_activation == "softmax":
            self.final_activation = functools.partial(F.softmax, dim=-1)
        elif final_activation == "sigmoid":
            self.final_activation = F.sigmoid
        elif final_activation == "relu":
            self.final_activation = F.relu
        elif final_activation == "tanh":
            self.final_activation = F.tanh
        elif final_activation == "identity":
            self.final_activation = lambda x: x
        else:
            raise NotImplementedError(f"Final activation {final_activation} not implemented.")

        self.lr = lr

        # activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")

        # Define network
        self.fc_layers = self._create_fc_layers()

    def _create_fc_layers(self) -> nn.ModuleList:
        layers = []
        sizes = [self.input_size] + list(self.hidden_sizes) + [self.output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image
        if self.is_image:
            x = x.view(x.size(0), -1)

        # should call each layer with activation except for last layer
        for layer in self.fc_layers[:-1]:
            x = self.activation(layer(x))
        x = self.fc_layers[-1](x)
        x = self.final_activation(x)
        return x

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs, preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs, preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return preds

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs, preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return preds

    def _get_preds_loss_accuracy(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, targets = batch
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)
        loss = self.loss_fn(outputs, targets)
        acc = accuracy(outputs, targets, task="multiclass", num_classes=self.output_size)

        return outputs, preds, loss, acc
