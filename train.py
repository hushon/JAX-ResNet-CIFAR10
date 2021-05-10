import jax
import jax.lax
# from jax import random, grad, jit, vmap, value_and_grad
import numpy as np
import jax.numpy as jnp

import torch
from torchvision import datasets, transforms
from tqdm import trange, tqdm
import os
import PIL.Image

import haiku as hk
import optax

from typing import Any, Iterable, Mapping, NamedTuple, Tuple
import atexit
import resnet_cifar
import tree
import shutil


class FLAGS(NamedTuple):
    KEY = jax.random.PRNGKey(1)
    BATCH_SIZE = 128
    DATA_ROOT = '/workspace/data/'
    LOG_ROOT = '/workspace/runs/'
    MAX_EPOCH = 200
    INIT_LR = 1e-1
    N_WORKERS = 4
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


shutil.copytree('./', FLAGS.LOG_ROOT, dirs_exist_ok=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['NVIDIA_TF32_OVERRIDE'] = '1'


def tprint(obj):
    tqdm.write(obj.__str__())


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


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
        assert isinstance(arr, np.ndarray), f'Input should be ndarray. Got {type(arr)}.'
        assert arr.ndim >= 3, f'Expected array to be a image of size (..., H, W, C). Got {arr.shape}.'

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


def softmax_cross_entropy(logits, labels):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(logp, labels[:, None], axis=-1)
    return loss


def l2_loss(params):
    # l2_params = jax.tree_util.tree_leaves(params)
    l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(
        params) if 'batchnorm' not in mod_name]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def forward(images, is_training: bool):
    net = resnet_cifar.ResNet32(num_classes=10, bn_config={'decay_rate': 0.9})
    return net(images, is_training=is_training)


@jax.jit
def ema_update(params, params_avg):
    '''polyak averaging'''
    params_avg = optax.incremental_update(params, params_avg, step_size=0.001)
    return params_avg


def main():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ToArray(),
        ArrayNormalize(FLAGS.CIFAR10_MEAN, FLAGS.CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        ToArray(),
        ArrayNormalize(FLAGS.CIFAR10_MEAN, FLAGS.CIFAR10_STD),
    ])
    train_dataset = datasets.CIFAR10(
        FLAGS.DATA_ROOT, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        FLAGS.DATA_ROOT, train=False, transform=transform_test)
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=FLAGS.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=FLAGS.N_WORKERS,
        collate_fn=numpy_collate,
    )
    test_loader = MultiEpochsDataLoader(
        test_dataset,
        batch_size=FLAGS.BATCH_SIZE*2,
        shuffle=False,
        drop_last=False,
        num_workers=FLAGS.N_WORKERS,
        collate_fn=numpy_collate,
    )


    ## INITIALIZE MODEL ##
    model = hk.transform_with_state(forward)
    # model = hk.without_apply_rng(model)

    sample_input = jnp.ones((1, 32, 32, 3))
    params, state = model.init(FLAGS.KEY, sample_input, is_training=True)
    tprint(
        sum([p.size for p in jax.tree_util.tree_leaves(params)])
    )

    ## INITIALIZE OPTIMIZER ##
    # learning_rate_fn = optax.cosine_onecycle_schedule(
    #     transition_steps=len(train_loader) * FLAGS.MAX_EPOCH,
    #     peak_value=FLAGS.INIT_LR,
    #     pct_start=0.3,
    #     div_factor=25.0,
    #     final_div_factor=1e4
    # )
    learning_rate_fn = optax.cosine_decay_schedule(
        init_value=FLAGS.INIT_LR,
        decay_steps=len(train_loader) * FLAGS.MAX_EPOCH,
        alpha=0.0
        )
    # learning_rate_fn = optax.piecewise_constant_schedule(
    #     init_value=FLAGS.INIT_LR,
    #     boundaries_and_scales={
    #         len(train_loader) * 100: 0.1,
    #         len(train_loader) * 150: 0.1,
    #     }
    # )
    optimizer = optax.sgd(learning_rate_fn, momentum=0.9, nesterov=False)
    opt_state = optimizer.init(params)

    ## INITIALIZE TRAIN STATE ##
    train_state = TrainState(params, state, opt_state)


    @jax.jit
    def train_step(train_state: TrainState, batch: dict):
        params, state, opt_state = train_state
        input, target = batch['image'], batch['label']
        def loss_fn(p):
            logits, state_new = model.apply(
                p, state, FLAGS.KEY, input, is_training=True)
            ce_loss = softmax_cross_entropy(logits, target).mean()
            loss = ce_loss + 1e-4 * l2_loss(p)
            return loss, state_new
        (val, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        deltas, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, deltas)
        train_state = TrainState(params, state, opt_state)
        return train_state, val


    @jax.jit
    def eval_step(train_state: TrainState, batch: dict):
        params, state, _ = train_state
        input, target = batch['image'], batch['label']
        logits, _ = model.apply(params, state, FLAGS.KEY, input, is_training=False)
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
        state_dict = {
            'params': train_state.params,
            'state' : train_state.state,
            'opt_state': train_state.opt_state,
        }
        pickle_path = os.path.join(FLAGS.LOG_ROOT, 'model.pickle')
        torch.save(state_dict, pickle_path)
        tprint(f'[SAVE] {pickle_path}')


    atexit.register(save_pickle)

    for epoch in trange(FLAGS.MAX_EPOCH, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1.):
        for input, target in train_loader:
            batch = {
                'image': input,
                'label': target,
            }
            train_state, train_loss = train_step(train_state, batch)
        acc, loss = evaluate(test_loader)
        last_lr = learning_rate_fn(train_state.opt_state[-1].count.item())
        tprint(f'[{epoch}/{FLAGS.MAX_EPOCH}] LR: {last_lr:.3f} | Train Loss {train_loss:.3f} | Test Loss {loss:.3f} Acc: {acc:.3f}')

        if epoch % 20 == 0:
            save_pickle()


if __name__ == '__main__':
    main()
