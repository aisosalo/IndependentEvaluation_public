import torch
from torch.utils.data import Dataset

import pandas as pd

import numpy as np

import cv2

import src.utilities.pickling as pickling
import src.data_loading.loading as loading
from src.constants import VIEWS
from src.constants import VIEWANGLES
from src.constants import LABELS

from pipeline.kvs import GlobalKVS

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

IMAGE_INDEX = 0  # See https://github.com/nyukat/breast_cancer_classifier for details


class MammogramDataset(Dataset):
    def __init__(self, split, exam_list_file, rng, dataset_loc=None, use_heatmaps=False, augmentation=False):
        self.split = split  # train_split/val_split metadata
        self.exam_list_file = exam_list_file
        self.rng = rng
        self.dataset_root = dataset_loc
        self.use_heatmaps = use_heatmaps
        self.augmentation = augmentation

    def __getitem__(self, ind):
        if isinstance(ind, torch.Tensor):
            ind = ind.item()

        curr_exam = self.split.iloc[ind]

        exam_id = curr_exam['StudyInstanceUID']

        image_path = self.dataset_root / str(curr_exam['StudyInstanceUID']) / 'cropped_images/'
        heatmaps_path = self.dataset_root / str(curr_exam['StudyInstanceUID']) / 'heatmaps/'

        exam_list_path = self.dataset_root / str(curr_exam['StudyInstanceUID']) / 'metadata/' / str(self.exam_list_file)
        exam = pickling.unpickle_from_file(str(exam_list_path))[0]

        loaded_image_dict = {view: [] for view in VIEWS.LIST}
        loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
        for view in VIEWS.LIST:  # See https://github.com/nyukat/breast_cancer_classifier for details
            for file_path in exam[view]:
                loaded_image = loading.load_image(
                    image_path=str(image_path / f"{file_path}.png"),
                    view=view,
                    horizontal_flip=exam["horizontal_flip"],
                )

                if self.use_heatmaps:
                    loaded_heatmaps = loading.load_heatmaps(
                        benign_heatmap_path=str(heatmaps_path / "heatmap_benign/" / f"{file_path}.hdf5"),
                        malignant_heatmap_path=str(heatmaps_path / "heatmap_malignant/" / f"{file_path}.hdf5"),
                        view=view,
                        horizontal_flip=exam["horizontal_flip"],
                    )
                else:
                    loaded_heatmaps = None

                loaded_image_dict[view].append(loaded_image)
                loaded_heatmaps_dict[view].append(loaded_heatmaps)

        exam_dict = {view: [] for view in VIEWS.LIST}
        for view in VIEWS.LIST:  # See https://github.com/nyukat/breast_cancer_classifier for details
            cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                image=loaded_image_dict[view][IMAGE_INDEX],
                auxiliary_image=loaded_heatmaps_dict[view][IMAGE_INDEX],
                view=view,
                best_center=exam["best_center"][view][IMAGE_INDEX],
                random_number_generator=self.rng,
                augmentation=self.augmentation,
                max_crop_noise=(100, 100),  # See https://github.com/nyukat/breast_cancer_classifier for details
                max_crop_size_noise=100,  # See https://github.com/nyukat/breast_cancer_classifier for details
            )

            if loaded_heatmaps_dict[view][IMAGE_INDEX] is None:
                exam_dict[view].append(cropped_image[:, :, np.newaxis])
            else:
                exam_dict[view].append(np.concatenate([cropped_image[:, :, np.newaxis], cropped_heatmaps,], axis=2))

        labels = exam['image_labels']

        targets_dict = {view: [] for view in VIEWANGLES.LIST}
        for view in VIEWANGLES.LIST:
            tg = [labels[(LABELS.LEFT_BENIGN, view)], labels[(LABELS.RIGHT_BENIGN, view)],
                  labels[(LABELS.LEFT_MALIGNANT, view)], labels[(LABELS.RIGHT_MALIGNANT, view)]]
            tg = np.asarray(tg, dtype=np.long)
            targets_dict[view].append(tg[:, np.newaxis])

        return {'inputs': exam_dict, 'targets': targets_dict, 'exam': exam_id}

    def __len__(self):
        return self.split.shape[0]


def init_metadata():
    """Initialize exam metadata

    @return:
    """
    kvs = GlobalKVS()  # Global Key-Value Storage

    # Read metadata
    metadata = pd.read_csv(kvs['args'].training_metadata_loc, sep=',')

    # Update metadata
    kvs.update('metadata', metadata)
