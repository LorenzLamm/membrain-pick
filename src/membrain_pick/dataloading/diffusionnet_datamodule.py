import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from membrain_pick.dataloading.diffusionnet_dataset import MemSegDiffusionNetDataset


class MemSegDiffusionNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_folder_train: str,
        csv_folder_val: str,
        load_n_sampled_points: int = 2000,
        overfit: bool = False,
        force_recompute: bool = False,
        overfit_mb: bool = False,
        cache_dir: Optional[str] = None,
        augment_all: bool = True,
        pixel_size: float = 1.0,
        max_tomo_shape: int = 928,
        k_eig: int = 128,
        batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.csv_folder_train = csv_folder_train
        self.csv_folder_val = csv_folder_val
        self.load_n_sampled_points = load_n_sampled_points
        self.overfit = overfit
        self.force_recompute = force_recompute
        self.overfit_mb = overfit_mb
        self.cache_dir = cache_dir
        self.augment_all = augment_all
        self.pixel_size = pixel_size
        self.max_tomo_shape = max_tomo_shape

        self.k_eig = k_eig
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Placeholder for the datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MemSegDiffusionNetDataset(
                csv_folder=self.csv_folder_train,
                train=True,
                train_pct=1.0,
                load_only_sampled_points=self.load_n_sampled_points,
                max_tomo_shape=self.max_tomo_shape,
                overfit=self.overfit,
                force_recompute=self.force_recompute,
                overfit_mb=self.overfit_mb,
                cache_dir=self.cache_dir,
                augment_all=self.augment_all,
                pixel_size=self.pixel_size,
                k_eig=self.k_eig,
            )
            self.val_dataset = MemSegDiffusionNetDataset(
                csv_folder=self.csv_folder_val,
                train=False,
                load_only_sampled_points=self.load_n_sampled_points,
                max_tomo_shape=self.max_tomo_shape,
                overfit=self.overfit,
                force_recompute=self.force_recompute,
                overfit_mb=self.overfit_mb,
                cache_dir=self.cache_dir,
                augment_all=self.augment_all,
                pixel_size=self.pixel_size,
                k_eig=self.k_eig,
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
