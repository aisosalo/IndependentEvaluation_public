import os
import operator

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from qhoptim.pyt import QHAdam

from termcolor import colored

from src.modeling.models import SplitBreastModel

from pipeline.kvs import GlobalKVS


def init_model(device):
    """Initialize model

    @param device: torch.device
    @return model: torch.nn.Module with parameters on the torch.device object
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    if kvs['args'].model_mode == 'view_split':
        model_input_channels = 3 if kvs['args'].use_heatmaps else 1
        model = SplitBreastModel(model_input_channels)
    else:
        raise NotImplementedError

    # Load a pre-trained model state dictionary
    if kvs['args'].use_pretrained:
        model.load_state_dict(torch.load(kvs['args'].model_path)["model"], strict=False)

    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model)

    model = model.to(device)

    return model


def init_optimizer(params):
    """Initialize optimizer

    @param params: model parameters
    @return model: torch.optim or qhoptim instance
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    # Define optimizer
    if kvs['args'].optimizer == 'Adam':  # https://pytorch.org/docs/master/optim.html#torch.optim.Adam
        return Adam(params=params, lr=kvs['args'].lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=kvs['args'].wd, amsgrad=False)
    elif kvs['args'].optimizer == 'QHAdam':  # https://github.com/facebookresearch/qhoptim/blob/e81dea3f2765780cf4fbb90b87b22ba7604b8625/qhoptim/pyt/qhadam.py#L12
        return QHAdam(params=params, lr=kvs['args'].lr, betas=(0.99, 0.999), nus=(0.7, 1.0), eps=1e-8, weight_decay=kvs['args'].wd, decouple_weight_decay=False)
    else:
        raise NotImplementedError


def init_scheduler(optimizer):
    """Initialize scheduler

    @param optimizer:
    @return: torch.optim.lr_scheduler.MultiStepLR instance
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    if kvs['args'].scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=kvs['args'].lr_drop, gamma=kvs['args'].learning_rate_decay)
    else:
        raise NotImplementedError


def init_loss(device):
    """Initialize loss

    @param device: torch.device
    @return loss: torch.nn.NLLLoss instance
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    if kvs['args'].loss == 'nlll':  # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        loss = nn.NLLLoss().to(device)
        return loss
    else:
        raise NotImplementedError


def log_metrics(board_logger, train_loss, val_loss, val_metrics=None):
    """Log metrics

    @param board_logger: SummaryWriter object
    @param train_loss:
    @param val_loss:
    @param val_metrics: (optional)
    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    print(colored(f'Training loss: {train_loss:.5f}', 'green'))
    print(colored(f'Validation loss: {val_loss:.5f}', 'green'))

    losses = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': kvs['curr_epoch']
    }
    kvs.update(f'losses_fold_[{kvs["curr_fold"]}]', losses)

    if val_metrics is not None:
        raise NotImplementedError
    
    board_logger.add_scalars('Losses', {'train': train_loss, 'val': val_loss}, kvs['curr_epoch'])

    # Save session
    kvs.save_pkl(kvs['args'].snapshots_root / kvs['snapshot_name'] / 'session.pkl')


def save_checkpoint(model, optimizer, val_parameter_name, comparator='lt'):  # lt, less than
    """Save checkpoint

    @param model:
    @param optimizer:
    @param val_parameter_name:
    @param comparator:
    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    fold_id = kvs['curr_fold']
    epoch = kvs['curr_epoch']

    if 'loss' in val_parameter_name:
        val_metric = kvs[f'losses_fold_[{fold_id}]'][epoch][0][val_parameter_name]  # see init_folds() for details
    else:
        raise NotImplementedError

    comparator = getattr(operator, comparator)

    cur_snapshot_name = kvs['args'].snapshots_root / kvs['snapshot_name'] / f'fold_{fold_id}_epoch_{epoch + 1}.pth'

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if kvs['prev_model'] is None:
        print(colored(f'Snapshot was saved to {cur_snapshot_name}', 'red'))
        torch.save(state, cur_snapshot_name)
        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)
    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored(f'Snapshot was saved to {cur_snapshot_name}', 'red'))
            os.remove(kvs['prev_model'])
            torch.save(state, cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)

    # Save session
    kvs.save_pkl(kvs['args'].snapshots_root / kvs['snapshot_name'] / 'session.pkl')
