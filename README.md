# Studying Neural Network Dynamics through Behavioral Motifs

In this work, we apply techniques from the field of behavioral analysis and ethology
to understand the learning dynamics of neural networks.

## Installation

From **conda**:

First, make your conda environment:

```bash
conda env create -f environment.yml
```

Then, activate the environment:

```bash
conda activate edm_nn
```

## Usage

Train:

```bash
# make sure you are in the root directory of the project
cd src
python train_network.py --config experiments/train_mnist.yaml
```

Replay:

```bash
cd src
python replay_network.py --config experiments/train_mnist.yaml --run_name mnist_mlp_20230531-201430
```

Analysis: Check the notebooks in `src/notebooks`
