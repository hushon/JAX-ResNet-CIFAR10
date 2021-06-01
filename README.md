# Simple CIFAR10 ResNet in JAX

This repo provides ResNet example for CIFAR-10 using [Google's JAX](https://github.com/google/jax). I aim to provide a simple baseline code for deep learning researchers who want to quickly get started with JAX. For those who are not famlilar with JAX, it is [Autograd](https://github.com/HIPS/autograd) + [XLA](https://www.tensorflow.org/xla).

I built upon Deepmind's [Haiku](https://github.com/deepmind/dm-haiku) and [Optax](https://github.com/deepmind/optax) for high-level neural net API.
I used PyTorch and Torchvision for data loading pipeline.
My ResNet implementation is based on [this repo](https://github.com/akamaster/pytorch_resnet_cifar10).

Updates:
- Support for mixed precision training using [JMP](https://github.com/deepmind/jmp).
- Support for multi-GPU training: `train_multigpu.py`

## Requirements

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

### Mixed precision training

``` bash
python train_mp.py
```

## Benchmarks

| Model | Size | Test Acc |
| --- | --- | --- |
| ResNet20 | 0.27 M | 91.5 % |
| ResNet32 | 0.46 M | 92.5 % |
| ResNet44 | 0.66 M | 93.1 % |
| ResNet56 | 0.85 M | 93.2 % |
