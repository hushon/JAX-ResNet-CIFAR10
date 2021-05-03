import jax
import jax.lax
# from jax import random, grad, jit, vmap, value_and_grad
import numpy as np
import jax.numpy as jnp

import torch
from torchvision import datasets, transforms
import tqdm
import os
import PIL.Image

import haiku as hk
import optax

from typing import Any, Iterable, Mapping, NamedTuple, Tuple
import atexit
import resnet_cifar
import tree

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

KEY = jax.random.PRNGKey(1)
BATCH_SIZE = 128
DATA_ROOT = '/workspace/data/'
LOG_ROOT = '/workspace/runs/'
MAX_EPOCH = 200
INIT_LR = 1e-1
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.485, 0.456, 0.406)
CIFAR100_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    # loss_scale: jmp.LossScale


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ArrayNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'Input should be ndarray. Got {type(arr)}.')

        if arr.ndim < 3:
            raise ValueError(
                f'Expected array to be a image of size (..., H, W, C). Got {arr.shape}.')

        dtype = arr.dtype
        mean = np.asarray(self.mean, dtype=dtype)
        std = np.asarray(self.std, dtype=dtype)
        if (std == 0).any():
            raise ValueError(
                f'std evaluated to zero after conversion to {dtype}, leading to division by zero.')
        if mean.ndim == 1:
            mean = mean.reshape(1, 1, -1)
        if std.ndim == 1:
            std = std.reshape(1, 1, -1)
        arr -= mean
        arr /= std
        return arr


class ToArray(torch.nn.Module):
    '''convert image to float and 0-1 range'''
    dtype = np.float32

    def __call__(self, x):
        assert isinstance(x, PIL.Image.Image)
        x = np.asarray(x, dtype=self.dtype)
        x /= 255.0
        return x


def softmax_cross_entropy(logits, labels, reduce=False):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(logp, labels[:, None], axis=-1)
    if reduce:
        loss = loss.mean()
    return loss


def l2_loss(params):
    # l2_params = jax.tree_util.tree_leaves(params)
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(
        params) if 'batchnorm' not in mod_name]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def onecycle_schedule(step, total_steps, init_lr):
    def true_fun(step, init_lr, milestone):
        return _annealing_linear(
            start_lr=0.,
            end_lr=init_lr,
            pct=step / milestone
        )

    def false_fun(step, init_lr, milestone):
        return _annealing_cos(
            start_lr=init_lr,
            end_lr=0.,
            pct=(step - milestone) / milestone
        )
    milestone = 0.25 * total_steps
    lr = jax.lax.cond(step < milestone, true_fun,
                      false_fun, (step, init_lr, milestone))
    return lr


def _annealing_cos(start_lr, end_lr, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = jnp.cos(jnp.pi * pct) + 1.
    return end_lr + (start_lr - end_lr) / 2.0 * cos_out


def _annealing_linear(start_lr, end_lr, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return (end_lr - start_lr) * pct + start_lr


def cosine_schedule(init_lr, total_steps):
    def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
        ratio = jnp.maximum(0., step / total_steps)
        mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
        learning_rate = mult * init_lr
        return learning_rate
    return lr_schedule


def forward(images, is_training: bool):
    net = resnet_cifar.ResNet32(num_classes=10, bn_config={'decay_rate': 1e-5})
    return net(images, is_training=is_training)


@jax.jit
def ema_update(params, avg_params):
    '''polyak averaging'''
    avg_params = optax.incremental_update(params, avg_params, step_size=0.001)
    return avg_params


def main():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ToArray(),
        ArrayNormalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        ToArray(),
        ArrayNormalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_dataset = datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        DATA_ROOT, train=False, transform=transform_test)
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        collate_fn=numpy_collate,
    )
    test_loader = MultiEpochsDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE*2,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=numpy_collate,
    )

    model = hk.transform_with_state(forward)
    # model = hk.without_apply_rng(model)

    sample_input = jnp.ones((1, 32, 32, 3))
    params, state = model.init(KEY, sample_input, is_training=True)
    print(
        sum([p.size for p in jax.tree_util.tree_leaves(params)])
    )

    # learning_rate_fn = cosine_schedule(INIT_LR, len(train_loader) * MAX_EPOCH)
    # learning_rate_fn = optax.cosine_onecycle_schedule(
    #     transition_steps=len(train_loader) * MAX_EPOCH,
    #     peak_value=INIT_LR,
    #     pct_start=0.3,
    #     div_factor=25.0,
    #     final_div_factor=1e4
    # )
    # learning_rate_fn = optax.cosine_decay_schedule(
    #     init_value=INIT_LR,
    #     decay_steps=len(train_loader) * MAX_EPOCH,
    #     alpha=0.0
    #     )
    learning_rate_fn = optax.piecewise_constant_schedule(
        init_value=INIT_LR,
        boundaries_and_scales={
            len(train_loader) * 100: 0.1,
            len(train_loader) * 150: 0.1,
        }
    )
    optimizer = optax.sgd(learning_rate_fn, momentum=0.9, nesterov=False)
    opt_state = optimizer.init(params)
    train_state = TrainState(params, state, opt_state)

    @jax.jit
    def train_step(train_state, batch):
        params, state, opt_state = train_state
        input = batch['image']
        target = batch['label']
        def loss_fn(p):
            logits, state_new = model.apply(
                p, state, KEY, input, is_training=True)
            ce_loss = softmax_cross_entropy(logits, target).mean()
            loss = ce_loss + 1e-4 * l2_loss(p)
            return loss, state_new
        (val, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        deltas, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, deltas)
        # params = jax.tree_multimap(lambda p, u: p + u, params, deltas)
        train_state = TrainState(params, state, opt_state)
        return train_state, val

    @jax.jit
    def eval_step(train_state, batch):
        params, state, _ = train_state
        input, target = batch['image'], batch['label']
        logits, _ = model.apply(params, state, KEY, input, is_training=False)
        correct = jnp.argmax(logits, -1) == target
        loss = softmax_cross_entropy(logits, target)
        return correct, loss

    def evaluate(dataloader):
        corrects = []
        losses = []
        for input, target in dataloader:
            batch = {
                'image': input,
                'label': target,
            }
            correct, loss = eval_step(train_state, batch)
            corrects.append(correct)
            losses.append(loss)
        avg_acc = jnp.concatenate(corrects).mean()
        avg_loss = jnp.concatenate(losses).mean()
        return avg_acc, avg_loss

    def save_pickle():
        pickle = train_state
        pickle_path = os.path.join(LOG_ROOT, 'model.pickle')
        torch.save(pickle, pickle_path)
        print(f'[SAVE] {pickle_path}')

    atexit.register(save_pickle)

    for epoch in tqdm.trange(MAX_EPOCH, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1.):
        for input, target in train_loader:
            batch = {
                'image': input,
                'label': target,
            }
            train_state, train_loss = train_step(train_state, batch)
        acc, loss = evaluate(test_loader)
        print(f'[{epoch}/{MAX_EPOCH}] LR: {INIT_LR:.3f} | Train Loss: {train_loss:.3f} | Test Loss: {loss:.3f} | Acc: {acc:.3f}')


if __name__ == '__main__':
    main()
