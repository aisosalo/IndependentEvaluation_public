import gc

import torch

from tqdm import tqdm

from termcolor import colored

from src.constants import VIEWS

from pipeline.kvs import GlobalKVS
from pipeline.session import init_loaders
from pipeline.training import init_model, init_optimizer, init_scheduler, init_loss
from pipeline.training import log_metrics, save_checkpoint


def init_epoch_pass(model, optimizer, loader):
    """Initialize epoch

    @param model: torch.nn.Module
    @param optimizer:
    @param loader: torch.utils.data.DataLoader instance
    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    model.train(optimizer is not None)

    running_loss = 0.0

    n_batches = len(loader)
    pbar = tqdm(total=n_batches)

    curr_epoch = kvs['curr_epoch']
    max_epoch = kvs['args'].n_epoch

    device = next(model.parameters()).device

    return running_loss, n_batches, pbar, curr_epoch, max_epoch, device


def epoch_pass(model, optimizer, criterion, loader):
    """Training / Validation

    @param model: torch.nn.Module
    @param optimizer:
    @param criterion:
    @param loader: torch.utils.data.DataLoader instance
    @return:
    """

    # Initialize training / validation variables
    running_loss, n_batches, pbar, curr_epoch, max_epoch, device = init_epoch_pass(model, optimizer, loader)

    with torch.set_grad_enabled(optimizer is not None):
        for i, batch in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()  # sets the gradients to zero before starting backpropagation

            input_batch = batch['inputs']
            target_batch = batch['targets']

            inputs = {view: torch.stack(input_batch[view]).squeeze(0).permute(0, 3, 1, 2).to(device) for view in VIEWS.LIST}

            # Pass inputs to the SplitBreastModel
            output = model(inputs)

            num_samples, label_count, num_classes = output['CC'].shape  # output['MLO'].shape is identical

            targets_cc = torch.stack(target_batch['CC']).type(torch.int64).to(device)
            targets_mlo = torch.stack(target_batch['MLO']).type(torch.int64).to(device)

            loss_cc = criterion(output['CC'].view(num_samples * label_count, num_classes), targets_cc.view(num_samples * label_count))
            loss_mlo = criterion(output['MLO'].view(num_samples * label_count, num_classes), targets_mlo.view(num_samples * label_count))
            loss = loss_cc + loss_mlo

            if optimizer is not None:
                loss.backward()  # accumulates the gradient (by addition) for each parameter
                optimizer.step()  # performs a parameter update based on the current gradient
            else:
                pass

            running_loss += loss.item()

            if optimizer is not None:
                pbar.set_description(f'Training [{curr_epoch} / {max_epoch}]: {running_loss / (i + 1):.3f}')
            else:
                pbar.set_description(f'Validating [{curr_epoch} / {max_epoch}]: {running_loss / (i + 1):.3f}')
            pbar.update()

            gc.collect()

    gc.collect()
    pbar.close()

    if optimizer is not None:
        return running_loss / n_batches
    else:
        return running_loss / n_batches


def train_folds(device, summary_writers):
    """Train folds

    @param device:
    @param summary_writers:
    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    scheduler = None

    for fold_id in kvs['cv_split_train_val']:
        kvs.update('curr_fold', fold_id)
        kvs.update('prev_model', None)

        print(colored(f'Training fold: {fold_id}', 'blue'))

        train_index, val_index = kvs['cv_split_train_val'][fold_id]

        # Initialize loaders
        train_loader, val_loader = init_loaders(train_split=kvs['metadata'].iloc[train_index],
                                                val_split=kvs['metadata'].iloc[val_index])

        # Initialize model
        model = init_model(device)

        output_layer_params = list(map(id, model.output_layer_cc.parameters()))  # See https://discuss.pytorch.org/t/loss-problem-in-net-finetuning/18311/44
        output_layer_params.append(map(id, model.output_layer_mlo.parameters()))

        base_params = filter(lambda p: id(p) not in output_layer_params, model.parameters())
        output_params = filter(lambda p: id(p) in output_layer_params, model.parameters())

        # Initialize optimizer
        optimizer = init_optimizer(params=[{'params': output_params, 'lr': kvs['args'].lr}])

        # Initialize criterion
        criterion = init_loss(device)

        # Initialize scheduler
        if kvs['args'].use_scheduler:  # this is optional
            scheduler = init_scheduler(optimizer)

        for curr_epoch in range(kvs['args'].n_epoch):
            kvs.update('curr_epoch', curr_epoch)

            print(colored(f'Epoch: {curr_epoch}', 'blue'))

            if kvs['args'].use_pretrained and curr_epoch == kvs['args'].unfreeze_epoch:
                optimizer.add_param_group({'params': base_params, 'lr': kvs['args'].lr_base})

                if kvs['args'].use_scheduler:  # this is optional
                    scheduler = init_scheduler(optimizer)

            train_loss = epoch_pass(model, optimizer, criterion, train_loader)
            val_loss = epoch_pass(model, None, criterion, val_loader)

            # Log metrics
            log_metrics(summary_writers[fold_id], train_loss, val_loss, None)

            # Save checkpoint
            save_checkpoint(model, optimizer, val_parameter_name='val_loss', comparator='lt')  # lt, less than

            if kvs['args'].use_scheduler:  # this is optional
                scheduler.step(curr_epoch)

            gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()
