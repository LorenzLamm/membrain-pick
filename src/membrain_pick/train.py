import os 

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from membrain_pick.dataloading.diffusionnet_datamodule import MemSegDiffusionNetDataModule
from membrain_pick.optimization.diffusion_training_pylit import DiffusionNetModule


def train(
    data_dir: str,
    training_dir: str = "./training_output",
    project_name: str = "test_diffusion",
    sub_name: str = "0",
    allpos: bool=False,
    use_psii: bool=True,
    use_b6f: bool=False,
    use_uk: bool=False,

    # Dataset parameters
    overfit: bool = False,
    overfit_mb: bool = False,
    partition_size: int = 2000,
    force_recompute_partitioning: bool = False,
    augment_all: bool = True,
    aug_prob_to_one: bool = False,
    input_pixel_size: float = 10.0,
    process_pixel_size: float = 15.0,
    k_eig: int = 128,

    # Model parameters
    N_block: int = 6,
    C_width: int = 16,
    dropout: bool = False,
    with_gradient_features: bool = True,
    with_gradient_rotations: bool = True,
    device: str = "cuda:0",
    one_D_conv_first: bool = False,

    # Mean shift parameters
    mean_shift_output: bool = False,
    mean_shift_bandwidth: float = 7.,
    mean_shift_max_iter: int = 10,
    mean_shift_margin: float = 2.,

    # Training parameters
    max_epochs: int = 1000,

    ):

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    cache_dir_mb = os.path.join(training_dir, "mesh_cache")
    log_dir = os.path.join(training_dir, "logs")

    # Create the data module
    data_module = MemSegDiffusionNetDataModule(
        csv_folder_train=train_path,
        csv_folder_val=val_path,
        load_n_sampled_points=partition_size,
        overfit=overfit,
        force_recompute=force_recompute_partitioning,
        overfit_mb=overfit_mb,
        cache_dir=cache_dir_mb,
        augment_all=augment_all,
        aug_prob_to_one=aug_prob_to_one,
        input_pixel_size=input_pixel_size,
        process_pixel_size=process_pixel_size,
        allpos=allpos,
        use_psii=use_psii,
        use_b6f=use_b6f,
        use_uk=use_uk,
        k_eig=k_eig,
        batch_size=1,
        num_workers=0,
        pin_memory=False,

    )
    data_module.setup()
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=0, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=1, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=2, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=3, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=4, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=5, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=15, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=25, times=10)
    # data_module.train_dataset.test_loading(out_dir="./test_loading/", idx=50, times=10)
    # exit()

    model = DiffusionNetModule(
        C_in=data_module.parameter_len,
        C_out=1,
        C_width=C_width,
        N_block=N_block,
        mean_shift_output=mean_shift_output,
        mean_shift_bandwidth=mean_shift_bandwidth,
        mean_shift_max_iter=mean_shift_max_iter,
        mean_shift_margin=mean_shift_margin,
        dropout=dropout,
        with_gradient_features=with_gradient_features,
        with_gradient_rotations=with_gradient_rotations,
        device=device,
        one_D_conv_first=one_D_conv_first,
        max_epochs=max_epochs,
    )

    checkpointing_name = project_name + "_" + sub_name
    # Set up logging
    csv_logger = pl_loggers.CSVLogger(log_dir)
    wandb_logger = pl_loggers.WandbLogger(name=checkpointing_name, project=project_name)

    # Set up model checkpointing
    checkpoint_callback_val_loss = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=100,
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch}-{val_loss:.2f}",
        verbose=True,  # Print a message when a checkpoint is saved
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    class PrintLearningRate(Callback):
        def on_epoch_start(self, trainer, pl_module):
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"Epoch {trainer.current_epoch}: Learning Rate = {current_lr}")

    print_lr_cb = PrintLearningRate()
    # Set up the trainer
    trainer = pl.Trainer(
        precision="32",
        logger=[csv_logger, wandb_logger],
        callbacks=[
            checkpoint_callback_val_loss,
            checkpoint_callback_regular,
            lr_monitor,
            print_lr_cb,
        ],
        max_epochs=max_epochs,
    )

    # Start the training process
    trainer.fit(model, data_module)
