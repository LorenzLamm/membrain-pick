import os
from typing import List

from typer import Option
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="predict", no_args_is_help=True)
def predict(
    data_dir: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the data to predict.", **PKWARGS
    ),
    ckpt_path: str = Option(  # noqa: B008
        ..., help="Path to the checkpoint.", **PKWARGS
    ),
    out_dir: str = Option(  # noqa: B008
        "./predict_output", help="Path to the folder where the output should be stored."
    ),
    partition_size: int = Option(2000, help="Size of the partition."),  # noqa: B008
    input_pixel_size: float = Option(  # noqa: B008
        10.0, help="Pixel size of the tomogram."
    ),
    force_recompute_partitioning: bool = Option(  # noqa: B008
        False, help="Should the partitioning be recomputed?"
    ),
    N_block: int = Option(4, help="Number of blocks."),  # noqa: B008
    C_width: int = Option(16, help="Width of the convolution."),  # noqa: B008
    conv_width: int = Option(16, help="Width of the convolution."),  # noqa: B008
    k_eig: int = Option(128, help="Number of eigenvectors."),  # noqa: B008
    mean_shift_output: bool = Option(  # noqa: B008
        True, help="Should the output be mean shifted?"
    ),
    mean_shift_bandwidth: float = Option(  # noqa: B008
        7.0, help="Bandwidth for the mean shift."
    ),
    mean_shift_max_iter: int = Option(  # noqa: B008
        150, help="Maximum number of iterations for the mean shift."
    ),
    mean_shift_margin: float = Option(  # noqa: B008
        0.0, help="Margin for the mean shift."
    ),
    mean_shift_score_threshold: float = Option(  # noqa: B008
        9.0, help="Score threshold for the mean shift."
    ),
    mean_shift_device: str = Option(  # noqa: B008
        # "cuda:0", help="Device to use for the mean shift."
        "cpu",
        help="Device to use for the mean shift.",
    ),
):
    """Predict the output of the trained model on the given data.

    Example
    -------
    membrain-pick predict --data-dir <path-to-your-folder> --ckpt-path <path-to-your-checkpoint> --out-dir <path-to-store-output>
    """
    # predict CLI
    from membrain_pick.predict import predict as _predict

    _predict(
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        is_single_mb=False,
        partition_size=partition_size,
        input_pixel_size=input_pixel_size,
        force_recompute_partitioning=force_recompute_partitioning,
        k_eig=k_eig,
        N_block=N_block,
        C_width=C_width,
        conv_width=conv_width,
        mean_shift_output=mean_shift_output,
        mean_shift_bandwidth=mean_shift_bandwidth,
        mean_shift_max_iter=mean_shift_max_iter,
        mean_shift_margin=mean_shift_margin,
        mean_shift_score_threshold=mean_shift_score_threshold,
        mean_shift_device=mean_shift_device,
    )


# @cli.command(name="mean_shift", no_args_is_help=True)
# def mean_shift_for_csv(
#     csv_path: str = Option(..., help="Path to the CSV file.", **PKWARGS),  # noqa: B008
#     out_dir: str = Option(  # noqa: B008
#         "./mean_shift_output",
#         help="Path to the folder where the output should be stored.",
#     ),
#     bandwidth: float = Option(7.0, help="Bandwidth for the mean shift."),  # noqa: B008
#     max_iter: int = Option(  # noqa: B008
#         150, help="Maximum number of iterations for the mean shift."
#     ),
#     margin: float = Option(0.0, help="Margin for the mean shift."),  # noqa: B008
#     score_threshold: float = Option(  # noqa: B008
#         9.0, help="Score threshold for the mean shift."
#     ),
#     device: str = Option(  # noqa: B008
#         "cuda:0", help="Device to use for the mean shift."
#     ),
# ):
#     """Perform mean shift on the given CSV file.

#     Example
#     -------
#     membrain-pick mean_shift --csv-path <path-to-your-csv> --out-dir <path-to-store-output>
#     """
#     from membrain_pick.clustering.mean_shift_inference import (
#         mean_shift_for_csv as _mean_shift_for_csv,
#     )

#     _mean_shift_for_csv(
#         csv_file=csv_path,
#         out_dir=out_dir,
#         bandwidth=bandwidth,
#         max_iter=max_iter,
#         margin=margin,
#         score_threshold=score_threshold,
#         device=device,
#     )


@cli.command(name="mean_shift", no_args_is_help=True)
def mean_shift_for_h5(
    h5_path: str = Option(..., help="Path to the .h5 file.", **PKWARGS),  # noqa: B008
    out_dir: str = Option(  # noqa: B008
        "./mean_shift_output",
        help="Path to the folder where the output should be stored.",
    ),
    bandwidth: float = Option(7.0, help="Bandwidth for the mean shift."),  # noqa: B008
    max_iter: int = Option(  # noqa: B008
        150, help="Maximum number of iterations for the mean shift."
    ),
    margin: float = Option(0.0, help="Margin for the mean shift."),  # noqa: B008
    score_threshold: float = Option(  # noqa: B008
        9.0, help="Score threshold for the mean shift."
    ),
    method: str = Option(  # noqa: B008
        "membrain_pick",
        help="Method to use for the mean shift. Choose from 'membrain_pick' or 'membrainv1'.",
    ),
    device: str = Option(  # noqa: B008
        "cuda:0", help="Device to use for the mean shift."
    ),
):
    """Perform mean shift on the given CSV file.

    Example
    -------
    membrain-pick mean_shift --csv-path <path-to-your-csv> --out-dir <path-to-store-output>
    """
    from membrain_pick.clustering.mean_shift_inference import (
        mean_shift_for_h5 as _mean_shift_for_h5,
    )

    _mean_shift_for_h5(
        h5_file=h5_path,
        out_dir=out_dir,
        bandwidth=bandwidth,
        max_iter=max_iter,
        margin=margin,
        score_threshold=score_threshold,
        method=method,
        device=device,
    )


@cli.command(name="assign_angles", no_args_is_help=True)
def assign_angles(
    position_file: str = Option(  # noqa: B008
        ...,
        help="Path to the positions file. Can be a CSV file (first 3 columns are x, y, z) or a star file (relion or stopgap).",
        **PKWARGS,
    ),
    obj_file: str = Option(  # noqa: B008
        default="",
        help="Path to an obj file with the membrane mesh. Provide either this or the segmentation file.",
    ),
    segmentation_file: str = Option(  # noqa: B008
        default="",
        help="Path to a membrane segmentation file. Provide either this or the obj file.",
    ),
    out_dir: str = Option(  # noqa: B008
        "./angle_output",
        help="Path to the folder where the output should be stored.",
    ),
    position_scale_factor: float = Option(
        1.0, help="Rescale points to match mesh dimensions"
    ),  # noqa: B008
    out_format: str = Option(  # noqa: B008
        "RELION",
        help="Output format for the angles. Choose from RELION or STOPGAP.",
    ),
):
    """Assign initial membrane alignment angles to the given positions.

    Example
    -------
    membrain_pick assign_angles --position-file <path-to-your-positions> --obj-file <path-to-your-obj-file> --out-dir <path-to-store-output>
    """
    from membrain_pick.orientation import orientation_from_files

    orientation_from_files(
        positions_file=position_file,
        out_dir=out_dir,
        mesh_file=obj_file if len(obj_file) > 0 else None,
        segmentation_file=segmentation_file if len(segmentation_file) > 0 else None,
        positions_scale_factor=position_scale_factor,
        out_format=out_format,
    )
