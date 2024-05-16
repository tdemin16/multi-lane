# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI-LANE
# Thomas De Min thomas.demin@unitn.it
# ------------------------------------------
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
import numpy as np
from collections import defaultdict, deque
import datetime
import matplotlib.pyplot as plt
import seaborn as sb
from typing import List, Tuple, Callable

from timm.optim import create_optimizer

import torch
import torch.distributed as dist
import torchmetrics as tm
from torchmetrics import ConfusionMatrix
    

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Adapted from: Token Merging original source code https://github.com/facebookresearch/ToMe

    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    def do_nothing(x: torch.Tensor) -> torch.Tensor:
        return x

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1] # num_patches
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # Split the metric tensor into two parts and compute dot product
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        # Find best matching node 
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from: Token Merging original source code https://github.com/facebookresearch/ToMe

    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size    


def mean_average_precision(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: [bsz, num_classes]
    targets: [bsz, num_classes]
    """
    logits = torch.nn.functional.sigmoid(logits)
    return tm.functional.average_precision(logits, targets, task='multilabel', num_labels=logits.size(1)) * 100


def f1_score_overall(predicts, targets, zero_division=0):
    predicts = torch.nn.functional.sigmoid(predicts).gt(0.8)
    _op = precision_score_overall(targets, predicts, zero_division)
    _or = recall_score_overall(targets, predicts, zero_division)
    return ((2 * _op * _or) / (_op + _or)) * 100 if (_op + _or) > 0 else torch.tensor([zero_division], dtype=torch.float32)

def precision_score_overall(targets, predicts, zero_division=0):
    _nc = (targets * predicts).sum().float()
    _np = predicts.sum().float()
    return (_nc / _np) if _np > 0 else torch.tensor([zero_division], dtype=torch.float32, device=predicts.device)

def recall_score_overall(targets, predicts, zero_division=0):
    _nc = (targets * predicts).sum().float()
    _ng = targets.sum().float()
    return (_nc / _ng) if _ng > 0 else torch.tensor([zero_division], dtype=torch.float32, device=predicts.device)


def f1_score_per_class(predicts, targets, zero_division=0):
    predicts = torch.nn.functional.sigmoid(predicts).gt(0.8)
    _cp = precision_score_per_class(targets, predicts, zero_division)
    _cr = recall_score_per_class(targets, predicts, zero_division)

    nu = (2 * _cp * _cr)
    de = (_cp + _cr)

    nu[de == 0] = zero_division
    de[de == 0] = 1.
    return (nu / de).mean() * 100, nu / de

def precision_score_per_class(targets, predicts, zero_division=0):
    _nc = (targets * predicts).sum(axis=0).type(torch.float32)
    _np = predicts.sum(axis=0).type(torch.float32)

    _nc[_np == 0] = zero_division
    _np[_np == 0] = 1.
    return _nc / _np

def recall_score_per_class(targets, predicts, zero_division=0):
    _nc = (targets * predicts).sum(axis=0).type(torch.float32)
    _ng = targets.sum(axis=0).type(torch.float32)

    _nc[_ng == 0] = zero_division
    _ng[_ng == 0] = 1.
    return _nc / _ng


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def is_trainable(args, name: str) -> bool:
    if 'prompts' in name or 'head' in name or 'keys' in name or 'selectors' in name \
    or 'identifier' in name or 'task_cls_token' in name or 'token_scale' in name:
        return True
    return False


def get_optimizer(args, model):
    backbone_params = [param for name, param in model.named_parameters() if 'head' not in name and param.requires_grad]
    head_params = [param for name, param in model.named_parameters() if 'head' in name and param.requires_grad]
    params = [
        {'params': backbone_params, 'weight_decay': args.weight_decay},
        {'params': head_params, 'weight_decay': 0},
    ]
    return create_optimizer(args, params)
    

def mask_logits(logits: torch.Tensor, mask: torch.Tensor, nb_classes: int, task_id: List[int],
                fill_value: float=float('-inf')):
    device = logits.device
    mask = [c for t in task_id for c in mask[t]]
    not_mask = np.setdiff1d(np.arange(nb_classes), mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
    logits = logits.index_fill(dim=1, index=not_mask, value=fill_value)
    return logits


def remove_logits(logits: torch.Tensor, mask: torch.Tensor, task_id: List[int]):
    mask = sorted([c for t in task_id for c in mask[t]])
    mask = torch.tensor(mask, dtype=torch.int64).to(logits.device)
    return logits[:, mask]


def save_confusion_matrix(cm: tuple, path: str):
    pred, target = cm
    n_cls = torch.max(target).item() + 1
    confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=n_cls, normalize='true')
    cm = confusion_matrix(pred, target)

    plt.figure(dpi=250)
    sb.set()
    ax = sb.heatmap(cm, fmt='.2f', annot=True, xticklabels='auto', yticklabels='auto', cbar=False)
    ax.set(xlabel="Pred Task", ylabel="Target Task")

    parent = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(parent):
        os.makedirs(parent)
    plt.savefig(path)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    args.gpu_name = torch.cuda.get_device_name(0)
