# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # used to collect all predictions and true labels
    all_preds = []
    all_targets = []
    
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # get prediction results
        preds = output.argmax(dim=1)
        
        # collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        acc1, acc5 = accuracy(output, target, topk=(1, 3))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    # calculate the accuracy of each class
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # calculate the average accuracy of all classes
    mean_class_acc = per_class_acc.mean()

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # # print the accuracy of each class
    # class_names = ['crack', 'lack_of_fusion', 'lack_of_penetration', 'porosity', 'slag']
    # class_accuracies = {class_name: float(acc) for class_name, acc in zip(class_names, per_class_acc)}
    # for i, acc in enumerate(per_class_acc):
    #     print(f'* {class_names[i]} Accuracy: {acc:.3f}')
    
    # print the average accuracy of all classes
    print(f'*=================== Mean Class Accuracy: {mean_class_acc:.3f}===================')

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['confusion_matrix'] = conf_matrix
    results['mean_acc'] = mean_class_acc * 100
    # results['class_accuracies'] = class_accuracies  # add class accuracy dictionary to results
    return results



def plot_confusion_matrix(conf_matrix, save_path, class_names=['crack', 'lack_of_fusion', 'lack_of_penetration', 'porosity', 'slag'], figsize=(10, 8), dpi=300):
    """
    Plot and save the confusion matrix image
    
    Parameters:
        conf_matrix (np.array): confusion matrix
        save_path (str): save path
        class_names (list): class name list, default=None
        figsize (tuple): image size, default=(10, 8)
        dpi (int): image resolution, default=100
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    # # calculate percentage
    # conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # conf_matrix_percent = np.round(conf_matrix_percent * 100, 2)
    
    # plot heatmap
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',    # .2f 
                cmap='Blues',
                xticklabels=class_names if class_names else True,
                yticklabels=class_names if class_names else True)
    
    # set title and labels
    plt.title('Confusion Matrix (%)', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # adjust layout
    plt.tight_layout()
    
    # save image
    plt.savefig(save_path + 'confusion_matrix.png')
    plt.close()    
