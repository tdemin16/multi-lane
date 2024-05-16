# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI_LANE
# Author: Thomas De Min thomas.demin@unitn.it
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model

from multi_lane.configs import CONFIGS
from multi_lane.datasets import build_continual_dataloader
from multi_lane.engine import *
import multi_lane.models # used to register custom vit architecture
from multi_lane import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # train_dataloader, val_dataloader = tmp_dl(args)
    data_loader, class_mask = build_continual_dataloader(args)

    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        args=args,
    )
    model.init(args)
    model.to(device)
    model.class_mask = class_mask #! TMP
        
    # freeze everything except head and layernorm
    learnable_params = []
    for n, p in model.named_parameters():
        if utils.is_trainable(args, n):
            p.requires_grad = True
            learnable_params.append((n, p))
        else:
            p.requires_grad = False
        
    n_parameters = sum(p.numel() for _, p in learnable_params)
    print(f'Name: {args.name}')
    print(f'Description: {args.notes}')
    print(f"Learnable Parameters {[n for n, _ in learnable_params]}")
    print('Number of params:', n_parameters)
    print(args)

    if args.eval:
        raise NotImplementedError("Use the evaluation script.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0 * args.accumulate_grad_batches

    if args.opt == 'sgd':
        args.opt_betas = None

    if 'COCO' in args.dataset or 'VOC' in args.dataset:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    train_and_evaluate(model=model, 
                       model_without_ddp=model_without_ddp, 
                       criterion=criterion, 
                       data_loader=data_loader, 
                       device=device,
                       class_mask=class_mask, 
                       args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    if args.store_model:
        output_path = os.path.join(args.output_dir, 'checkpoints.pth')
        print(f"Saving trained model to {output_path}")
        state_dict = {
            'args': args,
            'model': model_without_ddp.state_dict(),
        }

        torch.save(state_dict, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    
    subparser = parser.add_subparsers(dest='subparser_name')
    config_parser = subparser.add_parser(config)
    
    get_args_parser = CONFIGS[config]
    get_args_parser(config_parser)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)