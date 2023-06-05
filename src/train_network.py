"""
Trains the network.

Usage:
python train_network --config experiments/train_mnist.yaml
"""

import argparse
import yaml
import datetime

from networks import MLPNet
from datasets import load_dataset
# from callbacks import ActivationRecordingHook

import pytorch_lightning as pl

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train the network.")
    parser.add_argument("--config", type=str, default="experiments/train_mnist.yaml",
                        help="Path to the config file.")
    args = parser.parse_args()

    # Read in config file with pyyaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # load data

    # run name plus date with time
    run_name = config["run_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = config["save_dir"]
    model_params = config["model"]
    dataset_params = config["data"]
    training_params = config["training"]
    logging_params = config["logging"]

    # load dataset
    train_loader, test_loader = load_dataset(
        dataset_name=dataset_params["dataset_name"],
        data_dir=dataset_params["data_dir"],
        batch_size=training_params["batch_size"],
        shuffle_train=dataset_params["shuffle_train"],
        shuffle_test=dataset_params["shuffle_test"],
        num_workers=dataset_params["num_workers"],
    )

    # load model
    if model_params["model_type"] == "mlp":
        model = MLPNet(
            input_size=model_params["input_size"],
            hidden_sizes=model_params["hidden_sizes"],
            output_size=model_params["output_size"],
            loss_fn=model_params["loss_fn"],
            activation=model_params["activation"],
            final_activation=model_params["final_activation"],
            lr=training_params["lr"],
        )
    else:
        raise NotImplementedError(f"Model {model_params['model_type']} not implemented.")

    # load callbacks
    callbacks = []
    # if logging_params["record_activations"]:
    #     callbacks.append(ActivationRecordingCallback())

    # train model
    trainer = pl.Trainer(
        max_epochs=training_params["epochs"],
        default_root_dir=model_params["model_save_dir"],
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, test_loader)

    # save model
    trainer.save_checkpoint(model_params["model_save_dir"] + "/" + run_name + ".ckpt")


