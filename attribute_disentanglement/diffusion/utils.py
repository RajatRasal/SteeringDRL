from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import shutil
import urllib
import warnings
from typing import Tuple, Union

import safetensors
import torch
from torch import nn, optim


def normalise(x):
    return x * 2 - 1


def unnormalise(x):
    return (x + 1) * 0.5


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """Downloads a file if it does not exist, optionally checking its SHA-256 hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, 'wb') as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if digest != file_digest:
            raise OSError(f'hash of {path} (url: {url}) failed to validate')
    return path


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


class InverseLR(optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch) #, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """Implements an exponential learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr. Decays the learning rate
    continuously by decay (default 0.5) every num_steps steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (float): The number of steps to decay the learning rate by decay in.
        decay (float): The factor by which to decay the learning rate every num_steps
            steps. Default: 0.5.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, num_steps, decay=0.5, warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.num_steps = num_steps
        self.decay = decay
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch) #, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ConstantLRWithWarmup(optim.lr_scheduler._LRScheduler):
    """Implements a constant learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup=0., last_epoch=-1, verbose=False):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        super().__init__(optimizer, last_epoch) #, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        return [warmup * base_lr for base_lr in self.base_lrs]


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """CosineLR with Warmup, restart after decay.

    Code adopted from:
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification. Default: 1.
        max_lr (float | Tuple): First cycle's max learning rate. Default: 0.1.
        min_lr (float | Tuple): Min learning rate. Default: 0.001.
        warmup_steps (int): Linear warmup step size. Default: 0.
        gamma (float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.,
        max_lr: Union[float, Tuple] = 0.1,
        min_lr: Union[float, Tuple] = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        num_opt = len(optimizer.param_groups)
        if isinstance(max_lr, float):
            max_lr = [max_lr] * num_opt
        else:
            assert isinstance(max_lr, tuple)
            assert len(max_lr) == num_opt
        if isinstance(min_lr, float):
            min_lr = [min_lr] * num_opt
        else:
            assert isinstance(min_lr, tuple)
            assert len(min_lr) == num_opt

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = self.min_lr
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i]

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (max_lr - base_lr) * self.step_in_cycle / self.warmup_steps +
                base_lr for max_lr, base_lr in zip(self.max_lr, self.base_lrs)
            ]
        else:
            return [
                base_lr + (max_lr - base_lr) *
                (1 + math.cos(math.pi *
                              (self.step_in_cycle - self.warmup_steps) /
                              (self.cur_cycle_steps - self.warmup_steps))) / 2
                for max_lr, base_lr in zip(self.max_lr, self.base_lrs)
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) *
                    self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log((epoch / self.first_cycle_steps *
                                  (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps *
                                                     (self.cycle_mult**n - 1) /
                                                     (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * \
                        (self.cycle_mult**n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = [
            lr * (self.gamma**self.cycle) for lr in self.base_max_lr
        ]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = Path(filename)
        self.columns = columns
        if self.filename.exists():
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, 'w')
            self.write(*self.columns)

    def write(self, *args):
        print(*args, sep=',', file=self.file, flush=True)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """A context manager that sets whether TF32 is allowed on cuDNN or matmul."""
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old


def get_safetensors_metadata(path):
    """Retrieves the metadata from a safetensors file."""
    return safetensors.safe_open(path, "pt").metadata()


def ema_update_dict(values, updates, decay):
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values
