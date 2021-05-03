# CIFAR10 ResNet in JAX

This repo provides a simple ResNet implementation for CIFAR-10 using JAX.
I built upon [Haiku](https://github.com/deepmind/dm-haiku) and [Optax](https://github.com/deepmind/optax) for high-level neural net API.
I used PyTorch's DataLoader for data loading pipeline.

## Setup

- JAX
- Haiku
- Optax
- dm-tree
- PyTorch
- Torchvision

## Run

``` bash
python train.py
```