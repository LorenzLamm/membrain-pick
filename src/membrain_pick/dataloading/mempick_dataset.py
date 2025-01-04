import os
from typing import Dict

import imageio as io
import numpy as np
from torch.utils.data import Dataset

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram, get_csv_data
from membrain_seg.segmentation.dataloading.memseg_augmentation import (
    get_training_transforms,
    get_validation_transforms,
)


class CryoETMemPickDataset(Dataset):
    """
    A custom Dataset for Cryo-ET membrane protein localization.

    This Dataset is designed to work with Mask-RCNN
    """

    def __init__(
        self,
        img_folder: str,
        train: bool = False,
        use_fourier_aug: bool = False,
        use_mw_aug: bool = False,
        aug_prob_to_one: bool = False,
        train_pct: float = 0.8,
        bounding_box_size: int = 8,
    ) -> None:
        """
        Constructs all the necessary attributes for the CryoETMemSegDataset object.

        Parameters
        ----------
        img_folder : str
            The path to the directory containing the image files.
        train : bool, default False
            A flag indicating whether the dataset is used for training or validation.
        use_fourier_aug : bool, default False
            A flag indicating whether the Fourier augmentation should be used or not.
        use_mw_aug : bool, default False
            A flag indicating whether the MW augmentation should be used or not.
        aug_prob_to_one : bool, default False
            A flag indicating whether the probability of augmentation should be set
            to one or not.
        """
        self.train = train
        self.img_folder = img_folder
        self.train_pct = train_pct
        self.bounding_box_size = bounding_box_size

        self.initialize_imgs_paths()
        self.load_data()
        self.transforms = (
            get_training_transforms(
                prob_to_one=aug_prob_to_one,
                use_vectors=False,
                use_fourier_aug=use_fourier_aug,
                use_mw_aug=use_mw_aug,
            )
            if self.train
            else get_validation_transforms(use_vectors=False)
        )

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing an image-label pair for the provided index.

        Data augmentations are applied before returning the dictionary.

        Parameters
        ----------
        idx : int
            Index of the sample to be fetched.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.
        """
        idx_dict = {
            "image": np.expand_dims(self.imgs[idx], 0),
            "label": self.labels[idx],
        }
        # idx_dict = self.transforms(idx_dict)
        return idx_dict

    def __len__(self) -> int:
        """
        Returns the number of image-label pairs in the dataset.

        Returns
        -------
        int
            The number of image-label pairs in the dataset.
        """
        return len(self.imgs)

    def load_data(self) -> None:
        """
        Loads image-label pairs into memory from the specified directories.

        In addition to the image-label pairs, the normals are also loaded
        if the return_normals flag is set to True.

        Notes
        -----
        This function assumes the image and label files are in NIFTI format.
        """
        print("Loading images into dataset.")
        self.imgs = []
        self.labels = []

        for entry in self.data_paths:
            tomogram = load_tomogram(entry[0])
            points_positions = get_csv_data(entry[1])
            points_positions = np.array(points_positions, dtype=int)
            bounding_boxes = []
            for point in points_positions:
                bounding_boxes.append(
                    (
                        point[0] - self.bounding_box_size,
                        point[1] - self.bounding_box_size,
                        point[0] + self.bounding_box_size,
                        point[1] + self.bounding_box_size,
                    )
                )
            bounding_boxes = np.array(bounding_boxes)
            self.imgs.append(tomogram)
            self.labels.append(bounding_boxes)

    def initialize_imgs_paths(self) -> None:
        """
        Initializes the list of paths to image-label pairs.

        Notes
        -----
        This function assumes the image and label files are in parallel directories
        and have the same file base names.
        """


        self.data_paths = []
        for filename in os.listdir(self.img_folder):
            if filename.endswith(".mrc"):
                self.data_paths.append(filename)

        self.data_paths.sort()
        self.data_paths = [
            (os.path.join(self.img_folder, filename), os.path.join(self.img_folder, filename[:-4] + "_sampled_points.csv"),)
            for filename in self.data_paths
        ]
        if self.train:
            self.data_paths = self.data_paths[:int(len(self.data_paths) * self.train_pct)]
        else:
            self.data_paths = self.data_paths[int(len(self.data_paths) * self.train_pct):]


    def test(self, test_folder: str, num_files: int = 20) -> None:
        """
        Tests the data loading and augmentation process.

        The 2D images and corresponding labels are generated and then
            saved in the specified directory for inspection.

        Parameters
        ----------
        test_folder : str
            The path to the directory where the generated images and labels
            will be saved.
        num_files : int, default 20
            The number of image-label pairs to be generated and saved.
        """
        os.makedirs(test_folder, exist_ok=True)

        for i in range(num_files):
            test_sample = self.__getitem__(i % self.__len__())
            for num_img in range(0, test_sample["image"].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_img{i}_group{num_img}.png"),
                    test_sample["image"][0, :, :, num_img],
                )

            for num_mask in range(0, test_sample["label"][0].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_mask{i}_group{num_mask}.png"),
                    test_sample["label"][0][0, :, :, num_mask],
                )

            for num_mask in range(0, test_sample["label"][1].shape[0], 15):
                io.imsave(
                    os.path.join(test_folder, f"test_mask_ds2_{i}_group{num_mask}.png"),
                    test_sample["label"][1][0, :, :, num_mask],
                )

    def test_with_normals(self, test_folder, num_files=20):
        """
        Testing of dataloading and augmentations.

        To test data loading and augmentations, 2D images are
        stored for inspection.
        """
        os.makedirs(test_folder, exist_ok=True)

        for i in range(num_files):
            print("sample", i + 1)
            idx = np.random.randint(0, len(self))
            test_sample = self.__getitem__(idx)

            test_sample["image"] = np.array(test_sample["image"])
            test_sample["label"][0] = np.array(test_sample["label"][0])
            test_sample["vectors"][0] = np.array(test_sample["vectors"][0])

            for num_img in range(0, test_sample["image"].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_img{i}_group{num_img}.png"),
                    (test_sample["image"] * 10 + 128).astype(np.uint8)[
                        0, :, :, num_img
                    ],
                )

            for num_mask in range(0, test_sample["label"][0].shape[-1], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_mask{i}_group{num_mask}.png"),
                    (test_sample["label"][0] * 64).astype(np.uint8)[0, :, :, num_mask],
                )

            for num_mask in range(0, test_sample["vectors"][0].shape[-2], 30):
                io.imsave(
                    os.path.join(test_folder, f"test_vecs{i}_group{num_mask}.png"),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp1.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 0
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp2.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 1
                    ],
                )
                io.imsave(
                    os.path.join(
                        test_folder, f"test_vecs{i}_group{num_mask}_comp3.png"
                    ),
                    (test_sample["vectors"][0] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask, 2
                    ],
                )

            for num_mask in range(0, test_sample["label"][1].shape[0], 15):
                io.imsave(
                    os.path.join(test_folder, f"test_mask_ds2_{i}_group{num_mask}.png"),
                    (test_sample["label"][1] * 64 + 64).astype(np.uint8)[
                        0, :, :, num_mask
                    ],
                )


def get_dataset_token(patch_name):
    """
    Get the dataset token from the patch name.

    Parameters
    ----------
    patch_name : str
        The name of the patch.

    Returns
    -------
    str
        The dataset token.

    """
    basename = os.path.basename(patch_name)
    dataset_token = basename.split("_")[0]
    return dataset_token
