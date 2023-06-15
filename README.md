# Empirical Dynamical Modeling of Neural Networks

TODO:

1. initial ML model (DONE)
2. train mnist model (DONE)
3. record activations (DONE)
4. record weights at every step (DONE)
5. record loss at every step (DONE)
6. time series analysis (TODO)
7. record more weights

Behavioral / "kinematics" analysis:
- PCA
- PSD
- tSNE + watershed clustering

MotionMapper:
1. decompose to PCA (dimensionality reduction, could use a different method but PCA is easy)
(let's try PCA of dimension k=8)
2. Morlet wavelet transform
3. Normalization
4. Map to tSNE
5. Gaussian smoothed density
5. Watershed transformation

Correlation between variables:
- ARIMA
- CCM

Needed for preliminary submission:

1. Title
2. Abstract

Outline

1. Our goal: Understand the training dynamics of a neural network
2. The problem / current methods: Try to use spectral analysis, NTKs / GPs, MFs, try to understand the network from a
   linear POV.
3. The tool we are using: Empirical Dynamical Modeling. (baseline model: ARIMA / autoregressive model)
4. Why: Handles nonlinear connections, state dependence --> can we find distinctive states of learning in a neural
   network?
5. Ultimately, EDM is good at finding causal relationships between variables.

Some examples of X causes Y:

Examples of X:

1. Layer size
2. Manipulating / resetting the weights of a certain layer in the network

Can X change over time? Here are some ideas:

1. Epoch (lol)
2. Prediction error
3. Hyperparameters
4. Adam values
5. Set of values fed into the model
6. Batch size
7. learning rate
8. Weight density (how many weights are non-zero)

Candidates of interest for "Y":

1. Loss (train and test)
2.

Scope of experiments:

1. Other manifold reduction techniques
2. Try to learn underlying dynamics of a known model
3. Analyse network trained on learned dynamics

Desired figures:

1. Learning curves (training, test losses)
2.

Technical stack:

1. PyEDM
2. Pytorch

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