import torch
from torch import nn


class GateHUBCriterion(nn.Module):
    """Background suppression objective (Eqn. 6).

    A focal-style cross entropy that applies a different exponent to background
    and action frames: gamma_b on frames labelled background, gamma_a otherwise.
    The per-frame exponent is precomputed by the dataset and passed in as
    ``gamma``, so this module just applies it.
    """

    def __init__(self, num_classes, losses, args):
        super().__init__()
        self.num_classes = num_classes
        self.losses = losses
        self.args = args
        self.weight_dict = {
            'cross_entropy': args.classification_h_loss_coef,
            'bg_suppression': args.classification_h_loss_coef,
        }
        # THUMOS reserves the last class for 'Ambiguous', which is excluded from the loss.
        self.ignore_index = num_classes - 1 if args.dataset == 'THUMOS' else -1
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _reduce(self, per_sample, target, logits):
        if self.ignore_index < 0:
            return torch.mean(per_sample)
        if per_sample.sum() == 0:
            return torch.zeros((), device=logits.device, dtype=per_sample.dtype)
        return torch.mean(per_sample[target[:, self.ignore_index] != 1])

    def _scored_classes(self, num_columns):
        if self.ignore_index < 0:
            return list(range(num_columns))
        return [i for i in range(num_columns) if i != self.ignore_index]

    def loss_bg_suppression(self, logits, targets, name):
        target, gamma = targets
        target = target.float()
        gamma = gamma.float()

        columns = self._scored_classes(target.shape[-1])
        log_prob = self.logsoftmax(logits[:, columns])
        prob = torch.exp(log_prob)

        per_sample = torch.sum(
            -target[:, columns] * ((1 - prob) ** gamma.unsqueeze(1)) * log_prob
            * self.args.focal_alpha, dim=1)

        return {name: self._reduce(per_sample, target, logits)}

    def loss_cross_entropy(self, logits, targets, name):
        target, _ = targets
        target = target.float()

        columns = self._scored_classes(target.shape[-1])
        log_prob = self.logsoftmax(logits[:, columns])

        per_sample = torch.sum(-target[:, columns] * log_prob, dim=1)

        return {name: self._reduce(per_sample, target, logits)}

    def get_loss(self, loss, logits, targets):
        loss_map = {
            'bg_suppression': self.loss_bg_suppression,
            'cross_entropy': self.loss_cross_entropy,
        }
        assert loss in loss_map, 'unknown loss {}'.format(loss)
        return loss_map[loss](logits, targets, name=loss)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs[loss], targets[loss]))
        return losses
