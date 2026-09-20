import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'GateHUB: Gated History Unit with Background Suppression', add_help=False)

    # Training
    parser.add_argument('--output_dir', default='./logs/outputs/result',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true', help='run evaluation only')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='checkpoint to initialise from, excluding the classifier')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--dataparallel', action='store_true')
    parser.add_argument('--removelog', action='store_true')

    # Background suppression objective (Eqn. 6)
    parser.add_argument('--classification_h_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.75, type=float,
                        help='global scale on the loss')
    parser.add_argument('--focal_gamma', default=0.1, type=float,
                        help='base gamma; multiplied by the per-class factors below')
    parser.add_argument('--focal_ac_gamma', default=0.5, type=float,
                        help='gamma_a = focal_ac_gamma * focal_gamma (action frames)')
    parser.add_argument('--focal_bg_gamma', default=1.0, type=float,
                        help='gamma_b = focal_bg_gamma * focal_gamma (background frames)')
    parser.add_argument('--no_dense_loss', action='store_true',
                        help='supervise only the current frame instead of all t_pr present frames')

    # Optimisation
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_lr', default=1.2e-4, type=float)
    parser.add_argument('--base_lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--pct_start', default=0.25, type=float)
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=1., type=float)
    parser.add_argument('--lr_scheduler', default='OneCycle', type=str,
                        choices=['OneCycle', 'CyclicLR', 'StepLR'])

    # Data
    parser.add_argument('--dataset', type=str, default='THUMOS',
                        choices=['THUMOS', 'TVSERIES'])
    parser.add_argument('--dataset_file', type=str, default='data/data_info_new.json')
    parser.add_argument('--anno_root', type=str, default='data',
                        help='directory holding <dataset>_<split>_anno.npz')
    parser.add_argument('--data_root', type=str, required=False, default='',
                        help='directory holding the extracted feature pickles')
    parser.add_argument('--feature_file', type=str,
                        default='thumos_all_feature_{}_tsn_v2.pickle',
                        help="TSN feature pickle with 'rgb'/'flow' keys; '{}' is the split")
    parser.add_argument('--rgb_feature_file', type=str,
                        default='thumos_timesformer_k600_96_1s_temp_end.pkl',
                        help='RGB feature pickle used when --tfk is set')
    parser.add_argument('--feature_stride', default=1, type=int,
                        help='stored feature frames per model timestep '
                             '(1 for 4 FPS features, 6 for the 24 FPS dense variant)')
    parser.add_argument('--train_stride', default=1, type=int,
                        help='training sample stride in model timesteps')
    parser.add_argument('--ltm_interval', default=1, type=int,
                        help='additional subsampling of the history; 1 keeps the paper '
                             'setting of 1024 frames at 4 FPS (256 s)')

    # Model
    parser.add_argument('--rgb_only', action='store_true', help='drop the optical flow stream')
    parser.add_argument('--flow_only', action='store_true', help='drop the RGB stream')
    parser.add_argument('--embedding_dim', default=1024, type=int, help='D')
    parser.add_argument('--dim_feature', default=4096, type=int,
                        help='input feature dim (M), e.g. 2048 RGB + 2048 flow')
    parser.add_argument('--numclass', default=22, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--positional_encoding_type', default='learned', type=str,
                        choices=['learned', 'fixed'])
    parser.add_argument('--dec_init', default='zero', type=str, choices=['zero', 'random'])
    parser.add_argument('--tfk', action='store_true',
                        help='use TimeSformer features with Future-augmented History')
    parser.add_argument('--future_file', default='', type=str,
                        help='pickle of FaH features extracted from t_f future frames')

    parser.add_argument('--present_frames', default=8, type=int,
                        help='t_pr, number of present frames (8 = 2 s at 4 FPS)')
    parser.add_argument('--history_frames', default=1024, type=int,
                        help='T, number of history frames (1024 = 256 s at 4 FPS)')
    parser.add_argument('--latent_size', default=16, type=int, help='L, latent encoding size')
    parser.add_argument('--history_layers', default=2, type=int,
                        help='N, self-attention layers after the GHU')
    parser.add_argument('--decoder_layers', default=2, type=int,
                        help='present decoder cross-attention repeats')
    parser.add_argument('--decoder_embedding_dim_out', default=1024, type=int)
    parser.add_argument('--decoder_attn_dropout_rate', default=0.0, type=float)
    parser.add_argument('--decoder_num_heads', default=16, type=int, help='N_heads')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:12342')

    return parser
