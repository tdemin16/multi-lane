import argparse
import os
from .custom_types import *

def get_args_parser(subparsers: argparse.ArgumentParser):
    subparsers.add_argument('--name', default=None, type=str, help='Name of the run')
    subparsers.add_argument('--notes', default='', type=str, help='Description of the run')
    subparsers.add_argument('--batch_size', default=256, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=30, type=int)
    
    # Method parameters
    subparsers.add_argument('--method', type=str, 
                            choices=['prompts'], 
                            default='prompts',
    )
    subparsers.add_argument('--num_prompts', type=int, default=10)
    subparsers.add_argument('--detach', action='store_true', help='Detach the cls token befor computing similarity')
    subparsers.add_argument('--num_selectors', type=int, default=10)
    subparsers.add_argument('--tome', type=int, default=0, help='Use Token Merging')
    subparsers.add_argument('--disable_dandr', action='store_true', help='Disable drop and replace')
    subparsers.add_argument('--num_prompt_layers', type=int, default=5)
    subparsers.add_argument('--prompt_init', default='orthogonal', type=str, choices=['orthogonal', 'uniform'])
    subparsers.add_argument('--normalize', type=str, choices=['none', 'pre-head'], default='pre-head', help='Normalize the prompt vectors')
    subparsers.add_argument('--temperature', type=float, default=1.0)
    subparsers.add_argument('--head_mode', type=str, choices=['concat', 'task'], default='concat')
    subparsers.add_argument('--accumulate_grad_batches', type=int, default=1)
    subparsers.add_argument('--store_model', action='store_true', help='Store the model')
    subparsers.add_argument('--wandb', action='store_true', help='Enable Weights and Biases')
    subparsers.add_argument('--project', default='multi-lane')

    # Model parameters
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    subparsers.add_argument('--input_size', default=224, type=int, help='images input size')
    subparsers.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    subparsers.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    subparsers.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    subparsers.add_argument('--opt_betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    subparsers.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    subparsers.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    subparsers.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    subparsers.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.03)')
    subparsers.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on_off epoch percentages')
    subparsers.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    subparsers.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std_dev (default: 1.0)')
    subparsers.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    subparsers.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    subparsers.add_argument('--decay_epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    subparsers.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    subparsers.add_argument('--cooldown_epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    subparsers.add_argument('--patience_epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    subparsers.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    subparsers.add_argument('--unscale_lr', type=bool, default=False, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    subparsers.add_argument('--min_scale', type=float, default=0.05)
    subparsers.add_argument('--normalize_input', action='store_true', help='add normalization layer to the input')
    subparsers.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand_m9-mstd0.5-inc1)'),
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    subparsers.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    subparsers.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    subparsers.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    subparsers.add_argument('--data_path', default='datasets', type=str, help='dataset path')
    subparsers.add_argument('--dataset', default='Split-VOC', type=str, help='dataset name')
    subparsers.add_argument('--shuffle', type=str, default='none', choices=['none', 'yes'], help='shuffle the data order')
    subparsers.add_argument('--drop_last', action='store_true', help='drop last batch size')
    subparsers.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    subparsers.add_argument('--device', default='cuda', help='device to use for training / testing')
    subparsers.add_argument('--seed', default=0, type=int)
    subparsers.add_argument('--eval', action='store_true', help='Perform evaluation only')
    subparsers.add_argument('--eval_dir', type=str, help='Directory on which perform evaluation')
    subparsers.add_argument('--num_workers', default=os.cpu_count()//4, type=int)
    subparsers.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    subparsers.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    subparsers.set_defaults(pin_mem=True)

    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    subparsers.add_argument('--num_tasks', default=5, type=int, help='number of sequential tasks')
    subparsers.add_argument('--base_classes', default=4, type=int, help='number of base classes')
    subparsers.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='if doing task incremental')

    # ViT parameters
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, help='type of global pooling for final sequence')
    subparsers.add_argument('--head_type', default='token', choices=['token', 'gap', 'prompt', 'token_prompt'], type=str, help='input type of classification head')

    # Misc parameters
    subparsers.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')