import sys
import random
import time

import numpy as np
from numpy.random import RandomState

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold

from tensorboardX import SummaryWriter

from pipeline.kvs import GlobalKVS
from pipeline.utils import concatenate_column_values
from pipeline.data import MammogramDataset

DEBUG = sys.gettrace() is not None


def init_session(args):
    """Initialize session

    @param args:
    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    if DEBUG:
        args.n_threads = 0

    # Set the seeds for generating random numbers
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set model path
    if args.use_pretrained:
        model_path = args.model_root_dir / args.model_dict_file
        args.model_path = str(model_path)  # path for weights

    # Set directory for snapshots
    snapshots_dir = args.root_dir / 'snapshots' / args.dataset_name
    snapshot_name = time.strftime(f'%Y_%m_%d_%H_%M_%S')
    kvs.update('snapshot_name', snapshot_name)
    (snapshots_dir / snapshot_name).mkdir(exist_ok=True, parents=True)
    args.snapshots_root = snapshots_dir

    # Set directory for logs
    logs_dir = args.root_dir / 'logs' / args.dataset_name
    logs_dir.mkdir(exist_ok=True, parents=True)
    args.logs_dir = logs_dir

    # Set directory where the experiment data is located
    dataset_dir = args.dataset_root_dir / args.dataset_name
    args.dataset_loc = dataset_dir

    # Set training metadata location
    training_metadata_loc = args.dataset_root_dir / args.dataset_name / 'metadata' / args.training_metadata
    args.training_metadata_loc = training_metadata_loc

    # Update KVS with args
    kvs.update('args', args)

    # Update KVS with CUDA information
    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('device_count', torch.cuda.device_count())
        print(f'CUDA version: {torch.version.cuda}')
        print(f'CUDA device count: {torch.cuda.device_count()}')
    else:
        kvs.update('cuda', None)
        kvs.update('device_count', None)

    # Store information on how the script was executed
    kvs.save_pkl(snapshots_dir / snapshot_name / 'session.pkl')


def init_device():
    """Initialize device

    @return device: torch.device
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    device = torch.device(kvs['args'].device if torch.cuda.is_available() else 'cpu')

    return device


def init_splits():
    """Initialize splits

    :return:
    """
    kvs = GlobalKVS()

    metadata = kvs['metadata']

    # Create K-fold iterator with non-overlapping groups
    gkf = GroupKFold(n_splits=kvs['args'].n_folds)  # FIXME: Replace with Stratified Group KFold

    y = concatenate_column_values(dframe=metadata, cols=['PatientAgeGroup',
                                                         'ScreeningScoreLeft',
                                                         'ScreeningScoreRight', ]).astype(str)

    gfk_split = gkf.split(metadata,
                          y=y,
                          groups=metadata.PatientID.astype(str))

    cv_split = [x for x in gfk_split]

    # Update KVS with split information
    kvs.update('cv_split_all_folds', cv_split)

    # Save session
    kvs.save_pkl(kvs['args'].snapshots_root / kvs['snapshot_name'] / 'session.pkl')


def init_folds():
    """Initialize folds

    @return writers: SummaryWriter object
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    writers = {}
    cv_split_train_val = {}
    for fold_id, split in enumerate(kvs['cv_split_all_folds']):
        if fold_id != kvs['args'].train_fold and kvs['args'].train_fold > -1:
            continue

        kvs.update(f'losses_fold_[{fold_id}]', None, list)

        cv_split_train_val[fold_id] = split

        log_dir = kvs['args'].logs_dir / 'fold_{}'.format(fold_id) / kvs['snapshot_name']
        writers[fold_id] = SummaryWriter(comment='IndependentEvaluation',
                                         log_dir=str(log_dir),
                                         flush_secs=15,
                                         max_queue=1)

    kvs.update('cv_split_train_val', cv_split_train_val)

    # Save session
    kvs.save_pkl(kvs['args'].snapshots_root / kvs['snapshot_name'] / 'session.pkl')

    return writers


def init_loaders(train_split, val_split):
    """Initialize loaders

    @param train_split: training subset of the dataset
    @param val_split: validation subset of the dataset
    @return: torch.utils.data.DataLoader instances for training and validation
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    train_ds = MammogramDataset(split=train_split,
                                exam_list_file=kvs['args'].exam_list_filename,
                                rng=RandomState(kvs['args'].seed),
                                dataset_loc=kvs['args'].dataset_loc,
                                use_heatmaps=kvs['args'].use_heatmaps,
                                augmentation=kvs['args'].use_augmentations,)

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=kvs['args'].train_bs,
                              num_workers=kvs['args'].n_threads,
                              shuffle=kvs['args'].resuffle_train,
                              drop_last=True,  # drops the last incomplete batch, if the dataset size is not divisible by the batch size
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    val_ds = MammogramDataset(split=val_split,
                              exam_list_file=kvs['args'].exam_list_filename,
                              rng=RandomState(kvs['args'].seed),
                              dataset_loc=kvs['args'].dataset_loc,
                              use_heatmaps=kvs['args'].use_heatmaps,
                              augmentation=False,)

    val_loader = DataLoader(dataset=val_ds,
                            batch_size=kvs['args'].val_bs,
                            num_workers=kvs['args'].n_threads,
                            shuffle=False)

    return train_loader, val_loader
