# Empirical Dynamical Modeling of Neural Networks

Needed for preliminary submission:

1. Title
2. Abstract

Outline

1. Our goal: Understand the training dynamics of a neural network
2. The problem / current methods: Try to use spectral analysis, NTKs / GPs, MFs, try to understand the network from a linear POV.
3. The tool we are using: Empirical Dynamical Modeling.
4. Why: Handles nonlinear connections, state dependence --> can we find distinctive states of learning in a neural network?


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