import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=10, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='adp', type=str, choices=['adp'],
                    help='Use ADP as reward (default: ADP)')

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=4, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--allow_partial_load', action='store_true',
                    help='Allow resuming when some position checkpoint architecture does not match '
                         'the current model (those positions restart from scratch). Default: hard error.')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')

parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='Exploration epsilon for play positions (end value if decay)')
parser.add_argument('--exp_epsilon_start', default=0.05, type=float,
                    help='Exploration epsilon start value (decays to exp_epsilon)')
parser.add_argument('--bid_exp_epsilon', default=0.3, type=float,
                    help='Exploration epsilon for bidding (end value if decay)')
parser.add_argument('--bid_exp_epsilon_start', default=0.5, type=float,
                    help='Bidding exploration epsilon start value')
parser.add_argument('--epsilon_decay_games', default=20000, type=int,
                    help='Per-actor games over which epsilon decays from start→end (0=disabled)')
parser.add_argument('--bid_curriculum_games', default=0, type=int,
                    help='Number of games to decay P0 bid-0 bias from 1.0→0.0 (0=disabled)')
parser.add_argument('--decay', default=1., type=float,
                    help='MC 回报的逐步折扣系数 (1.0=整局共用同一回报, 不做折扣)')

parser.add_argument('--batch_size', default=30, type=int,
                    help='Learner batch size (play positions)')
parser.add_argument('--bid_batch_size', default=8, type=int,
                    help='Learner batch size for bidding positions (default: 8)')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                    help='Number of batches to accumulate gradients over before one optimizer step (default: 1, no accumulation)')

# Oracle auxiliary-head hyperparameters (see oracle_heads_design.md §2.3)
parser.add_argument('--oracle_decay_ratio', default=0.2, type=float,
                    help='Fraction of total_frames over which oracle λ decays (default: 0.2, i.e. first 20%% of training)')
parser.add_argument('--oracle_lambda_landlord', default=1.0, type=float,
                    help='Initial oracle λ for n_active (default: 1.0)')
parser.add_argument('--oracle_lambda_peasant', default=0.6, type=float,
                    help='Initial oracle λ for n_passive/delta_n (default: 0.6)')
parser.add_argument('--oracle_lambda_min', default=0.03, type=float,
                    help='Minimum oracle λ floor (default: 0.03)')

# Model architecture
parser.add_argument('--use_v2', action='store_true',
                    help='Use V2 Cross-Attention architecture (GeneralModelTransformerV2)')
parser.add_argument('--d_model', default=256, type=int,
                    help='Transformer embedding dimension')
parser.add_argument('--nhead', default=8, type=int,
                    help='Number of attention heads')
parser.add_argument('--num_encoder_layers', default=6, type=int,
                    help='Number of Transformer encoder layers')
parser.add_argument('--dim_feedforward', default=1024, type=int,
                    help='Transformer FFN hidden dimension')

# Logging
parser.add_argument('--log_game_steps', action='store_true',
                    help='Log per-game step count and throughput (can be noisy with many actors)')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--epsilon', default=1e-8, type=float,
                    help='Optimizer epsilon (RAdam)')
