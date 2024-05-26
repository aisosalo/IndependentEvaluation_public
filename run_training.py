import sys
import argparse

from pathlib import Path

import cv2

from src.constants import MODELMODES

from pipeline.session import init_session, init_device, init_splits, init_folds
from pipeline.data import init_metadata
from pipeline.training import train_folds
from pipeline.kvs import GlobalKVS

print(sys.version, sys.platform, sys.executable)

DEBUG = sys.gettrace() is not None
print('Debug: ', DEBUG)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, default='')

    parser.add_argument('--dataset_root_dir', type=Path, default='')
    parser.add_argument('--dataset_name', choices=['',], default='')
    parser.add_argument('--training_metadata', default='training_metadata.csv')
    parser.add_argument('--exam_list_filename', type=Path, default='cropped_exam_list_with_optimal_centers.pkl')

    parser.add_argument('--use_pretrained', type=bool, choices=[False, True], default=True)
    parser.add_argument('--model_root_dir', type=Path, default='/breast_cancer_classifier/models/')
    parser.add_argument('--model_dict_file', choices=['sample_imageheatmaps_model.p',], default='sample_imageheatmaps_model.p')
    parser.add_argument('--model_mode', type=str, default=MODELMODES.VIEW_SPLIT)

    parser.add_argument('--use_heatmaps', type=bool, choices=[False, True], default=True)

    parser.add_argument('--use_augmentations', type=bool, choices=[False, True], default=True)

    parser.add_argument('--optimizer', type=str, choices=['QHAdam', 'Adam'], default='QHAdam')
    parser.add_argument('--lr', type=float, choices=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], default=1e-4)  # 1e-4 = 0.0001
    parser.add_argument('--lr_base', type=float, choices=[1e-4, 1e-5, 1e-6], default=1e-4)  # 1e-4 = 0.0001
    parser.add_argument('--wd', type=float, choices=[1e-4, 1e-5], default=1e-5)  # 1e-5 = 0.00001

    parser.add_argument('--loss', type=str, choices=['nlll',], default='nlll')

    parser.add_argument('--scheduler', type=str, choices=['MultiStepLR',], default='MultiStepLR')
    parser.add_argument('--learning_rate_decay', type=float, choices=[0.1,], default=0.1)  # MultiStepLR
    parser.add_argument('--lr_drop', type=list, choices=[[20,]], default=[20,])  # MultiStepLR
    parser.add_argument('--use_scheduler', type=bool, choices=[False, True], default=False)

    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--val_bs', type=int, default=16)
    parser.add_argument('--resuffle_train', type=bool, choices=[False, True], default=True)

    parser.add_argument('--n_threads', type=int, choices=[0, 12, 24], default=0)

    parser.add_argument('--train_fold', type=int, default=-1)
    parser.add_argument('--n_folds', type=int, default=5)

    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--unfreeze_epoch', type=int, default=2)

    parser.add_argument('--device', type=str, choices=['cuda'], default='cuda')

    parser.add_argument('--seed', type=int, default=444444)
    args = parser.parse_args()

    kvs = GlobalKVS()  # Global Key-Value Storage

    # Initialize experiment
    init_session(args)

    # Initialize device
    device = init_device()

    # Initialize metadata
    init_metadata()

    # Initialize splits
    init_splits()

    # Initialize folds and get summary writers
    writers = init_folds()

    # Run training
    train_folds(device=device, summary_writers=writers)
