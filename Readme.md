# Adversarial Perturbations straight on JPEG Coefficients

This repository contains the Tensorflow implementation for the GCPR 2023 paper "Adversarial Perturbations Straight on
JPEG
coefficients" by Kolja Sielmann and Peer Stelldinger.

It only contains some example models (for Cifar10 and our LPIPS model).
The full set of models can be
downloaded [here](https://drive.google.com/drive/folders/1JmQq7_LVbHHBZp3jWavPvcDtwiuGnPWb?usp=sharing).

## Package Installation

Some functions require the package torchjpeg that is only available on linux.
It is recommended to use the requirements.txt to install the packages needed.

## Code

The code can be found in the adversarial_attacks folder.
Its structure is the following:

- **tutorial.ipynb**: contains examples for all of our code's main functions
- attacks: This folder contains implementation of our attacks and the attacks we compare our method to.
- config: Setting data and model folders, batch sizes etc.
- datasets: Implementation of all datasets. Our datasets (original, adversarial) always wrap a tf.data.Dataset.
- models: Models are defined here. Training function is also written in the models.py. The transformation modules for
  JPEG to RGB and inverse are defined here as well.
- utils: Some utilities for plotting, for our experiments e.g. computing the norms, for transformation between
  JPEG, RGB, YCbCR and to deal with coefficients, e.g. idct, dequantization etc.
- main.py: Initialization of Tensorflow, Logging and creating necessary directories.
