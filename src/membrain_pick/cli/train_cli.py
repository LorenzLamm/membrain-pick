import os
from typing import List

from typer import Option
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli



@cli.command(name="train", no_args_is_help=True)
def train(
    data_dir: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the training data.", **PKWARGS
    ),
    training_dir: str = Option(  # noqa: B008
        "./training_output",
        help="Path to the folder where the training output should be stored.",
    ),
    project_name: str = Option(  # noqa: B008
        "membrain_pick", help="Name of the project."
    ),
    sub_name: str = Option("0", help="Subname of the project."),  # noqa: B008
    force_recompute_partitioning: bool = Option(  # noqa: B008
        False, help="Should the partitioning be recomputed?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        10.0, help="Pixel size of the tomogram."
    ),
    device: str = Option("cuda:0", help="Device to use."),  # noqa: B008
    max_epochs: int = Option(200, help="Maximum number of epochs."),  # noqa: B008
):
    """Train a diffusion net model.

    Example
    -------
    membrain-pick train --data-dir <path-to-your-folder> --training-dir <path-to-your-folder>
    """
    from membrain_pick.train import train as _train

    _train(
        data_dir=data_dir,
        training_dir=training_dir,
        project_name=project_name,
        sub_name=sub_name,
        overfit=False,
        overfit_mb=False,
        partition_size=2000,
        force_recompute_partitioning=force_recompute_partitioning,
        augment_all=True,
        aug_prob_to_one=True,
        input_pixel_size=input_pixel_size,
        k_eig=128,
        N_block=4,
        C_width=16,
        conv_width=16,
        dropout=False,
        with_gradient_features=True,
        with_gradient_rotations=True,
        device=device,
        one_D_conv_first=True,
        mean_shift_output=False,
        max_epochs=max_epochs,
    )


@cli.command(name="train_advanced", no_args_is_help=True)
def train_advanced(
    data_dir: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the training data.", **PKWARGS
    ),
    training_dir: str = Option(  # noqa: B008
        "./training_output",
        help="Path to the folder where the training output should be stored.",
    ),
    project_name: str = Option(  # noqa: B008
        "membrain_pick", help="Name of the project."
    ),
    sub_name: str = Option("0", help="Subname of the project."),  # noqa: B008
    position_tokens: List[str] = Option(  # noqa: B008
        None,
        help="Tokens for the positions, as they are also specified in the _rlnClassNumber column of the GT star file. If columns are not present, the tokens are ignored and all positions used.",
    ),
    force_recompute_partitioning: bool = Option(  # noqa: B008
        False, help="Should the partitioning be recomputed?"
    ),
    augment_all: bool = Option(  # noqa: B008
        True, help="Should all data be augmented?"
    ),
    aug_prob_to_one: bool = Option(  # noqa: B008
        True, help="Should the probability be set to one?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        10.0, help="Pixel size of the tomogram."
    ),
    k_eig: int = Option(128, help="Number of eigenvectors."),  # noqa: B008
    N_block: int = Option(4, help="Number of blocks."),  # noqa: B008
    C_width: int = Option(16, help="Width of the convolution."),  # noqa: B008
    conv_width: int = Option(16, help="Width of the convolution."),  # noqa: B008
    dropout: bool = Option(False, help="Should dropout be used?"),  # noqa: B008
    with_gradient_features: bool = Option(  # noqa: B008
        True, help="Should the gradient features be used?"
    ),
    with_gradient_rotations: bool = Option(  # noqa: B008
        True, help="Should the gradient rotations be used?"
    ),
    device: str = Option("cuda:0", help="Device to use."),  # noqa: B008
    one_D_conv_first: bool = Option(  # noqa: B008
        True, help="Should 1D convolution be used first?"
    ),
    mean_shift_output: bool = Option(  # noqa: B008
        False, help="Should the output be mean shifted?"
    ),
    mean_shift_bandwidth: float = Option(  # noqa: B008
        7.0, help="Bandwidth for the mean shift."
    ),
    mean_shift_max_iter: int = Option(  # noqa: B008
        10, help="Maximum number of iterations for the mean shift."
    ),
    mean_shift_margin: float = Option(  # noqa: B008
        2.0, help="Margin for the mean shift."
    ),
    max_epochs: int = Option(200, help="Maximum number of epochs."),  # noqa: B008
):
    """Train a diffusion net model.

    Example
    -------
    membrain-pick train --data-dir <path-to-your-folder> --training-dir <path-to-your-folder>
    """
    from membrain_pick.train import train as _train

    _train(
        data_dir=data_dir,
        training_dir=training_dir,
        project_name=project_name,
        sub_name=sub_name,
        overfit=False,
        overfit_mb=False,
        partition_size=2000,
        position_tokens=position_tokens,
        force_recompute_partitioning=force_recompute_partitioning,
        augment_all=augment_all,
        aug_prob_to_one=aug_prob_to_one,
        input_pixel_size=input_pixel_size,
        k_eig=k_eig,
        N_block=N_block,
        C_width=C_width,
        conv_width=conv_width,
        dropout=dropout,
        with_gradient_features=with_gradient_features,
        with_gradient_rotations=with_gradient_rotations,
        device=device,
        one_D_conv_first=one_D_conv_first,
        mean_shift_output=mean_shift_output,
        mean_shift_bandwidth=mean_shift_bandwidth,
        mean_shift_max_iter=mean_shift_max_iter,
        mean_shift_margin=mean_shift_margin,
        max_epochs=max_epochs,
    )
