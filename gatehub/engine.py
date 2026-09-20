"""Train and evaluation loops."""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from . import utils


def _build_targets(logits, target, gamma, num_class, dense_loss):
    """Flatten the present-window predictions and targets for the criterion."""
    if dense_loss:
        steps = logits.shape[1]
        outputs = {'bg_suppression': logits.reshape(-1, num_class)}
        targets = {'bg_suppression': (target[:, -steps:].reshape(-1, num_class),
                                      gamma[:, -steps:].reshape(-1))}
    else:
        outputs = {'bg_suppression': logits[:, -1].reshape(-1, num_class)}
        targets = {'bg_suppression': (target[:, -1].reshape(-1, num_class),
                                      gamma[:, -1].reshape(-1))}
    return outputs, targets


def train_one_epoch(model, criterion, data_loader, optimizer, lr_scheduler, args, device,
                    epoch, max_norm=0, optim_type="OneCycle", dense_loss=True, num_class=22):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for (_, present_rgb, present_flow, history_rgb, history_flow,
         present_target, _, present_gamma) in metric_logger.log_every(data_loader, 500, header):

        present_rgb = present_rgb.to(device)
        present_flow = present_flow.to(device)
        history_rgb = history_rgb.to(device)
        history_flow = history_flow.to(device)
        present_target = present_target.to(device)
        present_gamma = present_gamma.to(device)

        logits = model(present_rgb, present_flow, history_rgb, history_flow)
        outputs, targets = _build_targets(
            logits, present_target, present_gamma, num_class, dense_loss)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {'{}_unscaled'.format(k): v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_reduced_scaled.values()).item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if optim_type in ("OneCycle", "CyclicLR"):
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, logger, args, epoch, nprocs=4,
             dense_loss=True, num_class=22):
    action_names = args.class_index[1:num_class - 1]

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    all_probs, all_classes = [], []

    for (_, present_rgb, present_flow, history_rgb, history_flow,
         present_target, _, present_gamma) in metric_logger.log_every(data_loader, 500, 'Test:'):

        present_rgb = present_rgb.to(device)
        present_flow = present_flow.to(device)
        history_rgb = history_rgb.to(device)
        history_flow = history_flow.to(device)
        present_target = present_target.to(device)
        present_gamma = present_gamma.to(device)

        logits = model(present_rgb, present_flow, history_rgb, history_flow)
        outputs, targets = _build_targets(
            logits, present_target, present_gamma, num_class, dense_loss)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {'{}_unscaled'.format(k): v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        if args.distributed:
            gathered_logits = [torch.zeros_like(logits) for _ in range(nprocs)]
            torch.distributed.all_gather(gathered_logits, logits)
            logits = torch.cat(gathered_logits, dim=0)

            gathered_targets = [torch.zeros_like(present_target) for _ in range(nprocs)]
            torch.distributed.all_gather(gathered_targets, present_target)
            present_target = torch.cat(gathered_targets, dim=0)

        # Score only the current frame over background + the C action classes.
        scored = slice(0, num_class - 1)
        all_probs += list(F.softmax(logits[:, -1, scored], dim=-1).cpu().numpy())
        all_classes += list(present_target[:, -1, scored].cpu().numpy())

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if args.distributed and not utils.is_main_process():
        return stats

    results = {'probs': np.asarray(all_probs).T, 'labels': np.asarray(all_classes).T}
    mean_ap, aps, _, _ = utils.frame_level_map_n_cap(results)
    logger.output_print('[Epoch-{}] mAP: {:.4f}\n'.format(epoch, mean_ap))
    for name, ap in zip(action_names, aps):
        logger.output_print('{}: {:.4f}'.format(name, ap))

    stats['mAP'] = mean_ap
    return stats
