import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from gatehub import utils
from gatehub.config import get_args_parser
from gatehub.criterion import GateHUBCriterion
from gatehub.dataset import THUMOSDataLayer
from gatehub.engine import evaluate, train_one_epoch
from gatehub.logger import setup_logger
from gatehub.models import GateHUB


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_datasets(args):
    if args.dataset == 'THUMOS':
        return (THUMOSDataLayer(phase='train', args=args),
                THUMOSDataLayer(phase='test', args=args))
    raise NotImplementedError(
        '{} is not wired up yet; THUMOS is supported today.'.format(args.dataset))


def main(args):
    utils.init_distributed_mode(args)
    command = 'python ' + ' '.join(sys.argv)
    output_dir = Path(args.output_dir)

    if args.removelog and utils.is_main_process():
        for name in ('log_dist.txt', 'log_train_test.txt'):
            path = output_dir / name
            if path.exists():
                os.remove(path)

    logger = setup_logger(str(output_dir / 'log_dist.txt'), command=command)
    for arg in vars(args):
        logger.output_print("{}:{}".format(arg, getattr(args, arg)))

    if args.distributed:
        torch.cuda.set_device(args.gpu)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)

    model = GateHUB(args=args).to(device)
    criterion = GateHUBCriterion(
        num_classes=args.numclass, losses=['bg_suppression'], args=args).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    elif args.dataparallel:
        model = nn.DataParallel(model)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.output_print('number of params: {}'.format(n_parameters))

    dataset_train, dataset_val = build_datasets(args)
    num_classes = args.numclass

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, pin_memory=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=torch_rng)
    data_loader_val = DataLoader(
        dataset_val, args.val_batch_size, sampler=sampler_val, drop_last=False,
        pin_memory=True, num_workers=args.num_workers,
        worker_init_fn=seed_worker, generator=torch_rng)

    if args.lr_scheduler == "CyclicLR":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == "OneCycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, steps_per_epoch=len(data_loader_train),
            pct_start=args.pct_start, epochs=args.epochs)
    elif args.lr_scheduler == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.base_lr, max_lr=args.max_lr)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        print('loading weights from', args.frozen_weights)
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        state_dict = GateHUB.remap_legacy_state_dict(checkpoint['model'])
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        model_without_ddp.load_state_dict(state_dict, strict=False)

    if args.resume:
        print('checkpoint: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(
            GateHUB.remap_legacy_state_dict(checkpoint['model']))
        if not args.eval and all(k in checkpoint for k in ('optimizer', 'lr_scheduler', 'epoch')):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    dense_loss = not args.no_dense_loss

    if args.eval:
        evaluate(model, criterion, data_loader_val, device, logger, args, epoch=0,
                 nprocs=utils.get_world_size(), dense_loss=dense_loss, num_class=num_classes)
        return

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, lr_scheduler, args, device,
            epoch, args.clip_max_norm, args.lr_scheduler, dense_loss, num_classes)

        if args.lr_scheduler == "StepLR":
            lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0:
                checkpoint_paths.append(output_dir / 'checkpoint{:04}.pth'.format(epoch))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(
            model, criterion, data_loader_val, device, logger, args, epoch,
            nprocs=utils.get_world_size(), dense_loss=dense_loss, num_class=num_classes)

        log_stats = {**{'train_{}'.format(k): v for k, v in train_stats.items()},
                     **{'test_{}'.format(k): v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log_train_test.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser('GateHUB training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.numclass = len(args.class_index)

    if args.rgb_only and args.flow_only:
        parser.error('--rgb_only and --flow_only are mutually exclusive')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
