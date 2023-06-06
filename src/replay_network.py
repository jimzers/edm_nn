"""
Runs the network on the test dataset, and records activations of the network.

Example usage:
python replay_network.py --config experiments/train_mnist.yaml --run_name mnist_mlp_20230531-194230
"""

import argparse
import yaml
import datetime
import h5py
import pathlib

import torch

from networks import MLPNet
from datasets import load_dataset
from callbacks import ActivationRecordingHook

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test the network.")
    parser.add_argument("--config", type=str, default="experiments/train_mnist.yaml",
                        help="Path to the config file.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run to load.")
    args = parser.parse_args()

    # Read in config file with pyyaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # run name plus date with time

    # run_name = config["run_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name
    save_dir = config["save_dir"]
    model_params = config["model"]
    dataset_params = config["data"]
    training_params = config["training"]
    logging_params = config["logging"]

    # set up saving directories
    save_dir = pathlib.Path(save_dir)
    run_dir = save_dir / run_name

    model_save_dir = run_dir / model_params["model_save_dir"]
    model_save_path = model_save_dir / (run_name + ".ckpt")

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

    # load the model from a checkpoint file
    model = model.load_from_checkpoint(
        model_save_path,
        **model_params,
    )

    # setup the activation recording hook
    activation_recording_hook = ActivationRecordingHook(model)

    model.eval()
    model.freeze()

    # run the model on the test dataset, and gather the loss and accuracy
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += model.loss_fn(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.2f}%")

    # save the activations
    activations_save_dir = run_dir / model_params["activations_save_dir"]
    activations_save_dir.mkdir(parents=True, exist_ok=True)
    activations_save_path = activations_save_dir / ("replay_activations" + ".h5")

    activations = activation_recording_hook.activations
    with h5py.File(activations_save_path, "w") as f:
        # import ipdb; ipdb.set_trace()
        for layer_name, layer_activations in activations.items():
            if layer_name and len(layer_activations) > 0:
                processed_activations = torch.cat(layer_activations, dim=0).cpu().numpy()
                f.create_dataset(layer_name, data=processed_activations)

    print(f"Saved activations to {activations_save_path}")
