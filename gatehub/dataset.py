import os.path as osp
import pickle

import numpy as np
import torch
import torch.utils.data as data


def _load_anno(anno_root, dataset, split):
    path = osp.join(anno_root, '{}_{}_anno.npz'.format(dataset.lower(), split))
    with np.load(path) as handle:
        return {session: handle[session] for session in handle.files}


def _session_key(store, session):
    """Feature stores are keyed either by session name or by '<session>.pkl'."""
    if session in store:
        return session
    return session + '.pkl'


class THUMOSDataLayer(data.Dataset):
    """Per-frame online action detection samples for THUMOS'14.

    Each item yields the present window (t_pr frames) and the history window
    (T frames) ending at the same current frame, plus the per-frame targets and
    the per-frame gamma used by the background suppression objective.

    ``feature_stride`` is the number of stored feature frames per model
    timestep: 1 for features stored at 4 FPS, 6 for the 24 FPS dense variant.
    """

    def __init__(self, args, phase='train'):
        self.args = args
        self.training = phase == 'train'
        self.split = 'val' if self.training else 'test'
        self.sessions = getattr(args, phase + '_session_set')

        self.stride = args.feature_stride
        self.history_stride = self.stride * args.ltm_interval
        self.present_frames = args.present_frames
        self.history_frames = args.history_frames

        present_span = self.present_frames * self.stride
        history_span = self.history_frames * self.history_stride
        sample_stride = args.train_stride * self.stride if self.training else self.stride

        anno_all = _load_anno(args.anno_root, args.dataset, self.split)

        self.inputs = []
        for session in self.sessions:
            target = anno_all[session]
            gamma = self.background_suppression_gamma(target)
            for idx in range(1, target.shape[0], sample_stride):
                end = idx + 1
                start = max(0, end - present_span)
                history_start = max(0, end - history_span)

                present_target = target[start:end][::self.stride]
                present_gamma = gamma[start:end][::self.stride]
                history_target = target[history_start:end][::self.history_stride]

                if present_target.argmax() != args.numclass - 1:
                    self.inputs.append([session, history_start, start, end,
                                        present_target, history_target, present_gamma])

        self.features = pickle.load(
            open(osp.join(args.data_root, args.feature_file.format(self.split)), 'rb'))

        self.rgb_features = None
        self.future_features = None
        if args.tfk:
            self.rgb_features = pickle.load(
                open(osp.join(args.data_root, args.rgb_feature_file), 'rb'))
            if args.future_file:
                self.future_features = pickle.load(
                    open(osp.join(args.data_root, args.future_file), 'rb'))

    def background_suppression_gamma(self, anno):
        """Per-frame gamma of Eqn. 6: gamma_a on action frames, gamma_b on background."""
        is_action = np.float_(anno[:, 1:].sum(axis=1) > 0)
        is_background = np.float_(anno[:, 1:].sum(axis=1) == 0)

        is_action *= self.args.focal_ac_gamma * self.args.focal_gamma
        is_background *= self.args.focal_bg_gamma * self.args.focal_gamma

        return is_action + is_background

    def _rgb(self, session, start, end, stride):
        """RGB features, using Future-augmented History where it is available."""
        if self.rgb_features is None:
            return self.features[session]['rgb'][start:end][::stride]

        key = _session_key(self.rgb_features, session)
        past = self.rgb_features[key]
        if self.future_features is None:
            return past[start:end][::stride]

        # Frames older than the present window have observed futures, so they are
        # encoded from [t, t+t_f] instead of [t-t_ps, t] (Eqn. 5).
        future = self.future_features[_session_key(self.future_features, session)]
        boundary = end - self.present_frames * self.stride
        if boundary > start:
            feats = np.vstack((future[start:boundary], past[boundary:end]))
        else:
            feats = past[start:end]
        return feats[::stride]

    def _flow(self, session, start, end, stride):
        return self.features[session]['flow'][start:end][::stride]

    @staticmethod
    def _pad_front(tensor, length):
        if tensor.shape[0] >= length:
            return tensor
        pad = torch.zeros((length - tensor.shape[0],) + tuple(tensor.shape[1:]),
                          dtype=torch.float32)
        return torch.cat([pad, tensor], dim=0)

    @staticmethod
    def _pad_target_front(target, length):
        if target.shape[0] >= length:
            return target
        pad = torch.zeros((length - target.shape[0], target.shape[1]), dtype=torch.float32)
        pad[:, -1] = 1  # padded steps carry the ignored class
        return torch.cat([pad, target], dim=0)

    def __getitem__(self, index):
        session, history_start, start, end, present_target, history_target, present_gamma = \
            self.inputs[index]

        present_rgb = torch.as_tensor(
            np.ascontiguousarray(self._rgb(session, start, end, self.stride)), dtype=torch.float32)
        present_flow = torch.as_tensor(
            np.ascontiguousarray(self._flow(session, start, end, self.stride)), dtype=torch.float32)
        history_rgb = torch.as_tensor(
            np.ascontiguousarray(self._rgb(session, history_start, end, self.history_stride)),
            dtype=torch.float32)
        history_flow = torch.as_tensor(
            np.ascontiguousarray(self._flow(session, history_start, end, self.history_stride)),
            dtype=torch.float32)

        present_target = torch.as_tensor(present_target, dtype=torch.float32)
        history_target = torch.as_tensor(history_target, dtype=torch.float32)
        present_gamma = torch.as_tensor(present_gamma, dtype=torch.float32)

        present_rgb = self._pad_front(present_rgb, self.present_frames)
        present_flow = self._pad_front(present_flow, self.present_frames)
        history_rgb = self._pad_front(history_rgb, self.history_frames)
        history_flow = self._pad_front(history_flow, self.history_frames)
        present_gamma = self._pad_front(present_gamma, self.present_frames)
        present_target = self._pad_target_front(present_target, self.present_frames)
        history_target = self._pad_target_front(history_target, self.history_frames)

        return (session, present_rgb, present_flow, history_rgb, history_flow,
                present_target, history_target, present_gamma)

    def __len__(self):
        return len(self.inputs)
