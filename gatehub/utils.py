import datetime
import os
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist


def frame_level_map_n_cap(results):
    """Per-frame mAP and calibrated mAP over the C action classes."""
    all_probs = results['probs']
    all_labels = results['labels']

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        all_cls_ap.append(psum / np.sum(this_cls_gt))
        all_cls_acp.append(cpsum / np.sum(this_cls_gt))

    mean_ap = sum(all_cls_ap) / len(all_cls_ap)
    mean_cap = sum(all_cls_acp) / len(all_cls_acp)
    return mean_ap, all_cls_ap, mean_cap, all_cls_acp


def setup_for_distributed(is_master):
    """Silence printing on non-master processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


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
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class SmoothedValue(object):
    """Track a series of values and report a windowed median and global average."""

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
        """Note: does not synchronize the deque, only count and total."""
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
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

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
        return self.fmt.format(median=self.median, avg=self.avg,
                               global_avg=self.global_avg, max=self.max, value=self.value)


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
        return self.delimiter.join(
            "{}: {}".format(name, str(meter)) for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        parts = [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                 'time: {time}', 'data: {data}']
        if torch.cuda.is_available():
            parts.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(parts)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                fields = dict(eta=eta_string, meters=str(self),
                              time=str(iter_time), data=str(data_time))
                if torch.cuda.is_available():
                    fields['memory'] = torch.cuda.max_memory_allocated() / MB
                print(log_msg.format(i, len(iterable), **fields))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, str(datetime.timedelta(seconds=int(total_time))),
            total_time / len(iterable)))


def reduce_dict(input_dict, average=True):
    """Average the values of a dict across processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = torch.stack([input_dict[k].clone().detach().float() for k in names], dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if average:
            values /= world_size
        return {k: v for k, v in zip(names, values)}
