# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI-LANE
# Thomas De Min thomas.demin@unitn.it
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
from typing import Iterable, List
import wandb
import datetime

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.utils import accuracy

import multi_lane.utils as utils
import multi_lane.datasets as datasets


def train_one_epoch(model: nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, set_training_mode=True, task_id=-1, 
                    class_mask=None, run=None, args=None,):

    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for i, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input)

        # here is the trick to mask out classes of non-current tasks
        fill_value = float('-inf') if type(criterion) == torch.nn.CrossEntropyLoss else 0
        logits = utils.mask_logits(logits, class_mask, args.num_classes, [task_id], fill_value=fill_value)

        if type(criterion) == torch.nn.BCEWithLogitsLoss:
            target = utils.mask_logits(target, class_mask, args.num_classes, [task_id], fill_value=0)

        if type(criterion) == torch.nn.CrossEntropyLoss:
            loss = criterion(logits / args.temperature, target)
        else:
            loss = criterion(logits / args.temperature, target.float())
        
        if args.head_mode  == 'task':
            loss -= sim

        if type(criterion) == torch.nn.CrossEntropyLoss:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            mAP = utils.mean_average_precision(logits, target)
            clean_logits = utils.remove_logits(logits, class_mask, [task_id])
            clean_target = utils.remove_logits(target, class_mask, [task_id])
            of1 = utils.f1_score_overall(clean_logits, clean_target)
            cf1, _ = utils.f1_score_per_class(clean_logits, clean_target)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        
        loss.backward()
        if i % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        # metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Loss'].update(loss.item(), n=input.shape[0])
        if type(criterion) == torch.nn.CrossEntropyLoss:
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        else:
            metric_logger.meters['mAP'].update(mAP.item(), n=input.shape[0])
            metric_logger.meters['oF1'].update(of1.item(), n=input.shape[0])
            metric_logger.meters['cF1'].update(cf1.item(), n=input.shape[0])

        if args.head_mode  == 'task':
            metric_logger.meters['Sim'].update(sim.item(), n=input.shape[0])

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    wandb_dict = {}
    if run is not None:
        if 'Acc@1' in metric_logger.meters:
            wandb_dict['train/acc1'] = metric_logger.meters['Acc@1'].global_avg
        if 'Acc@5' in metric_logger.meters:
            wandb_dict['train/acc5'] = metric_logger.meters['Acc@5'].global_avg
        if 'mAP' in metric_logger.meters:
            wandb_dict['train/mAP'] = metric_logger.meters['mAP'].global_avg
        if 'oF1' in metric_logger.meters:
            wandb_dict['train/oF1'] = metric_logger.meters['oF1'].global_avg
        if 'cF1' in metric_logger.meters:
            wandb_dict['train/cF1'] = metric_logger.meters['cF1'].global_avg
        if 'Loss' in metric_logger.meters:
            wandb_dict['train/loss'] = metric_logger.meters['Loss'].global_avg
        if 'Lr' in metric_logger.meters:
            wandb_dict['train/lr'] = metric_logger.meters['Lr'].global_avg
        if 'Sim' in metric_logger.meters:
            wandb_dict['train/sim'] = metric_logger.meters['Sim'].global_avg
    return wandb_dict


@torch.no_grad()
def evaluate(model: nn.Module, criterion, data_loader: Iterable, device, task_id=-1, tasks_so_far=-1,
             class_mask=None, args=None,):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    predictions = []
    targets = []
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input, eval=True)

        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.num_classes, list(range(tasks_so_far+1)))
        loss = criterion(logits / args.temperature, target)
        
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc_task = (tasks == task_id).float().mean()

        metric_logger.meters['Loss'].update(loss.item())
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        if args.head_mode == 'task':
            metric_logger.meters['Sim'].update(sim.item(), n=input.shape[0])
            metric_logger.meters['Acc_Task'].update(acc_task.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    stats = '* '
    stats += f"Acc@1 {metric_logger.meters['Acc@1'].global_avg:.3f} Acc@5 {metric_logger.meters['Acc@5'].global_avg:.3f} "
    stats += f"Loss {metric_logger.meters['Loss'].global_avg:.3f}"
    if args.head_mode  == 'task':
        stats += f" Acc_Task {metric_logger.meters['Acc_Task'].global_avg:.3f}"
    
    print(stats)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: nn.Module, criterion, data_loader, device, task_id=-1, acc_matrix=None,
                      class_mask=None, run=None, args=None,):
    stat_matrix = np.zeros((4, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader[i]['val'], 
                              device=device, task_id=i, tasks_so_far=task_id, class_mask=class_mask, 
                              args=args)

        # save stats
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc_Task'] if args.head_mode  == 'task' else 0

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    diagonal = np.diag(acc_matrix)

    result_str = f"[Average accuracy till task{task_id+1}]"
    result_str += f"\tAcc@1: {avg_stat[0]:.4f}\tAcc@5: {avg_stat[1]:.4f}"
    
    if args.head_mode  == 'task':
        result_str += f"\tAcc Task: {avg_stat[3]:.4f}"
    
    result_str += f"\tLoss: {avg_stat[2]:.4f}"

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    wandb_dict = {}
    if run is not None:
        for task in range(task_id+1):
            wandb_dict[f'test/task{task}_acc'] = stat_matrix[0][task]
            if args.head_mode  == 'task':
                wandb_dict[f'test/task{task}_acc_task'] = stat_matrix[3][task]
        wandb_dict.update({
            f'test/avg_acc': avg_stat[0],
        })
        if task_id > 0:
            wandb_dict.update({
                'test/forgetting': forgetting,
            })

    return wandb_dict


@torch.no_grad()
def evaluate_till_now_multi(model: nn.Module, criterion, data_loader, device: torch.device, 
                            task_id=-1, mAP_vector=None, class_mask=None, run=None, args=None,):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    #! ------- Extract predictions ------
    predictions = []
    targets = []

    header = f'Till task {task_id+1}'
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input, eval=True)

        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.num_classes, task_id=list(range(task_id+1)), 
                                fill_value=0)
        target = utils.mask_logits(target, class_mask, args.num_classes, task_id=list(range(task_id+1)),
                                fill_value=0)
        
        loss = criterion(logits / args.temperature, target.float())
        metric_logger.meters['Loss'].update(loss.item(), n=input.shape[0])

        predictions.append(logits)
        targets.append(target)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    #! ----------------------------------
    
    #$ ------- Calculate metrics ----------
    mAP = utils.mean_average_precision(predictions, targets)
    mAP_vector[task_id] = mAP.item()
    clean_predictions = utils.remove_logits(predictions, class_mask, list(range(task_id+1)))
    clean_targets = utils.remove_logits(targets, class_mask, list(range(task_id+1)))
    of1 = utils.f1_score_overall(clean_predictions, clean_targets)
    cf1, class_wise_cf1 = utils.f1_score_per_class(clean_predictions, clean_targets)
    #$ ------------------------------------

    #? ------- Print statistics for histogram -------
    target_frequencies = clean_targets.sum(dim=0)
    sorted_classes, indices = torch.sort(target_frequencies, descending=True)
    class_wise_cf1 = class_wise_cf1[indices]
    class_names = [data_loader.dataset.dataset.category2name[i.item()] for i in indices]
    with open('class_wise_cf1.txt', 'w') as f:
        for i, cf1_ in enumerate(class_wise_cf1):
            f.write(f'{class_names[i]}-{indices[i].item()}-{sorted_classes[i].item()}: {cf1_.item():.4f}\n')
    #? ----------------------------------------------

    result_str = f"[Average performances till task {task_id+1}]"
    result_str += f"\tmAP: {mAP.item():.4f}\tamAP: {mAP_vector[:task_id+1].mean():.4f}\toF1: {of1.item():.4f}\tcF1: {cf1.item():.4f}"
    result_str += f"\tLoss: {metric_logger.meters['Loss'].global_avg:.4f}"
    print(result_str)

    wandb_dict = {}
    if run is not None:
        wandb_dict['test/mAP'] = mAP.item()
        wandb_dict['test/oF1'] = of1.item()
        wandb_dict['test/cF1'] = cf1.item()
        wandb_dict['test/loss'] = metric_logger.meters['Loss'].global_avg

    return wandb_dict


def train_and_evaluate(model: nn.Module, model_without_ddp: nn.Module, 
                       criterion, data_loader: Iterable, device: torch.device, class_mask=None, 
                       args=None):
    
    if args.wandb and args.rank == 0:
        wandb_name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-{args.name}'
        run = wandb.init(
            project=args.project,
            entity='user',
            name=wandb_name,
            notes=args.notes,
            config=args,
        )
    else:
        run = None

    # create matrix to save end-of-task accuracies
    if type(criterion) == torch.nn.CrossEntropyLoss:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    else:
        mAP_vector = np.zeros((args.num_tasks, 1))

    for task_id in range(args.num_tasks):
        # transfer parameters to new task
        if task_id > 0:
            if args.distributed:
                model.module.next_task()
            else:
                model.next_task()

        # Create new optimizer for each task to clear optimizer status
        optimizer = utils.get_optimizer(args, model_without_ddp)
        lr_scheduler = None
        if args.sched == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        train_dataloader = data_loader[task_id]['train']
        
        wandb_dict = {}
        for epoch in range(args.epochs):
            stats = train_one_epoch(model=model,criterion=criterion, data_loader=train_dataloader,
                                    optimizer=optimizer, device=device, epoch=epoch,
                                    set_training_mode=True, task_id=task_id, class_mask=class_mask, 
                                    run=run, args=args,)                
            if lr_scheduler:
                lr_scheduler.step()
        wandb_dict.update(stats)

        if type(criterion) == torch.nn.CrossEntropyLoss:
            stats = evaluate_till_now(model=model, criterion=criterion,data_loader=data_loader,
                                    device=device, task_id=task_id, acc_matrix=acc_matrix, 
                                    class_mask=class_mask, run=run, args=args)

        else:
            _, val_seen_dataset = datasets.get_dataset(args.dataset.replace('Split-', ''), 
                                                     datasets.build_transform(is_train=True, args=args),
                                                     datasets.build_transform(is_train=False, args=args),
                                                     args=args)
            seen_classes = [c for m in class_mask[:task_id+1] for c in m]
            val_seen_indices = []
            for k in range(len(val_seen_dataset.targets)):
                if set(val_seen_dataset.targets[k]).intersection(set(seen_classes)) != set():
                    val_seen_indices.append(k)
                    
            val_seen_dataset = torch.utils.data.Subset(val_seen_dataset, val_seen_indices)
            val_seen_dataloader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=24, 
                                                              shuffle=False, num_workers=args.num_workers, 
                                                              pin_memory=True)
            stats = evaluate_till_now_multi(model=model, criterion=criterion, data_loader=val_seen_dataloader,
                                            device=device, task_id=task_id, mAP_vector=mAP_vector, 
                                            class_mask=class_mask, run=run, args=args)
            
    
        wandb_dict.update(stats)
        if run is not None:
            run.log(wandb_dict)
    