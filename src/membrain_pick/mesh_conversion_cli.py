"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup

from membrain_pick.train import train as _train


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    MemBrain-pick's data conversion / mesh projection module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain-pick --help

    -------

    Example:
    -------
    membrain-pick process-folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> 
        --output-folder <path-to-store-meshes>

    -------
    """


from typing import List

from typer import Option

from membrain_pick.mesh_projection import (
    mesh_for_single_mb_file,
    mesh_for_tomo_mb_folder,
    meshes_for_folder_structure
)


@cli.command(name="convert_file", no_args_is_help=True)
def convert_single_file(
    tomogram_path: str = Option(  # noqa: B008
        ..., help="Path to the tomogram to be projected", **PKWARGS
    ),
    mb_path: str = Option(  # noqa: B008
        ...,
        help="Path to the segmentation to convert.",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./mesh_data", help="Path to the folder where mesh projections should be stored."
    ),
    only_obj: bool = Option(  # noqa: B008
        False, help="Should only .obj files be computed? --> compatible with Surforama"
    ),
    match_size_flag: bool = Option(  # noqa: B008
        False, help="Should tomograms and membranes be converted to a specific pixel size?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    temp_folder: str = Option(  # noqa: B008
        "./temp_mesh_data",
        help="Path to the folder where temporary data should be stored.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        0.25, help="Step size for the normal vectors. Default: 0.25"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        1.0, help="Barycentric area for the mesh. Default: 1.0"
    ),
    crop_box_flag: bool = Option(  # noqa: B008
        False, help="Should the mesh be cropped to the bounding box of the segmentation?"
    ),
    only_largest_component: bool = Option(  # noqa: B008
        True, help="Should only the largest connected component be used?"
    ),
    min_connected_size: int = Option(  # noqa: B008
        1e4,
        help="Minimum size of the connected component. Only used if only_largest_component is True.",
    ),
):
    """Convert a single membrane segmentation to a mesh.

    Example
    -------
    membrain-pick convert_file --tomogram-path <path-to-your-tomo> --mb-path <path-to-your-membrane-segmentation> --out-folder <path-to-store-meshes>
    """
    
    mesh_for_single_mb_file(
        mb_file=mb_path,
        tomo_file=tomogram_path,
        out_folder=out_folder,
        tomo=None,
        only_obj=only_obj,
        match_size_flag=match_size_flag,
        temp_folder=temp_folder,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        output_pixel_size=output_pixel_size,
        crop_box_flag=crop_box_flag,
        only_largest_component=only_largest_component,
        min_connected_size=min_connected_size,
    )


@cli.command(name="convert_mb_folder", no_args_is_help=True)
def convert_mb_folder(
    mb_folder: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the membrane segmentations.", **PKWARGS
    ),
    tomo_path: str = Option(  # noqa: B008
        ..., help="Path to the tomogram to be projected.", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        "./mesh_data", help="Path to the folder where mesh projections should be stored."
    ),
    only_obj: bool = Option(  # noqa: B008
        False, help="Should only .obj files be computed? --> compatible with Surforama"
    ),
    match_size_flag: bool = Option(  # noqa: B008
        False, help="Should tomograms and membranes be converted to a specific pixel size?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    temp_folder: str = Option(  # noqa: B008
        "./temp_mesh_data",
        help="Path to the folder where temporary data should be stored.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        0.25, help="Step size for the normal vectors. Default: 0.25"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        1.0, help="Barycentric area for the mesh. Default: 1.0"
    ),
    crop_box_flag: bool = Option(  # noqa: B008
        False, help="Should the mesh be cropped to the bounding box of the segmentation?"
    ),
    only_largest_component: bool = Option(  # noqa: B008
        True, help="Should only the largest connected component be used?"
    ),
    min_connected_size: int = Option(  # noqa: B008
        1e4,
        help="Minimum size of the connected component. Only used if only_largest_component is True.",
    ),
):
    """Convert a folder of membrane segmentations to meshes.

    Example
    -------
    membrain-pick convert_mb_folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> --out-folder <path-to-store-meshes>
    """
    mesh_for_tomo_mb_folder(
        mb_folder=mb_folder,
        tomo_path=tomo_path,
        out_folder=out_folder,
        only_obj=only_obj,
        match_size_flag=match_size_flag,
        temp_folder=temp_folder,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        output_pixel_size=output_pixel_size,
        crop_box_flag=crop_box_flag,
        only_largest_component=only_largest_component,
        min_connected_size=min_connected_size,
    )

@cli.command(name="convert_folder_structure", no_args_is_help=True)
def convert_folder_structure(
    mb_folder: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the membrane segmentations.", **PKWARGS
    ),
    tomo_folder: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the tomograms to be projected.", **PKWARGS
    ),
    out_folder: str = Option(  # noqa: B008
        "./mesh_data", help="Path to the folder where mesh projections should be stored."
    ),
    only_obj: bool = Option(  # noqa: B008
        False, help="Should only .obj files be computed? --> compatible with Surforama"
    ),
    match_size_flag: bool = Option(  # noqa: B008
        False, help="Should tomograms and membranes be converted to a specific pixel size?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    temp_folder: str = Option(  # noqa: B008
        "./temp_mesh_data",
        help="Path to the folder where temporary data should be stored.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        0.25, help="Step size for the normal vectors. Default: 0.25"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        1.0, help="Barycentric area for the mesh. Default: 1.0"
    ),
    crop_box_flag: bool = Option(  # noqa: B
        False, help="Should the mesh be cropped to the bounding box of the segmentation?"
    ),
    only_largest_component: bool = Option(  # noqa: B
        True, help="Should only the largest connected component be used?"
    ),
    min_connected_size: int = Option(  # noqa: B
        1e4,
        help="Minimum size of the connected component. Only used if only_largest_component is True.",
    ),
):
    """Convert a folder structure of membrane segmentations to meshes.

    Example
    -------
    membrain-pick convert_folder_structure --mb-folder <path-to-your-folder> --tomo-folder <path-to-your-tomo-folder> --out-folder <path-to-store-meshes>
    """
    meshes_for_folder_structure(
        mb_folder=mb_folder,
        tomo_folder=tomo_folder,
        out_folder=out_folder,
        only_obj=only_obj,
        match_size_flag=match_size_flag,
        temp_folder=temp_folder,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        output_pixel_size=output_pixel_size,
        crop_box_flag=crop_box_flag,
        only_largest_component=only_largest_component,
        min_connected_size=min_connected_size,
    )



@cli.command(name="train", no_args_is_help=True)
def train(
    data_dir: str = Option(  # noqa: B008
        ..., help="Path to the folder containing the training data.", **PKWARGS
    ),
    training_dir: str = Option(  # noqa: B008
        "./training_output", help="Path to the folder where the training output should be stored."
    ),
    project_name: str = Option(  # noqa: B008
        "test_diffusion", help="Name of the project."
    ),
    sub_name: str = Option(  # noqa: B008
        "0", help="Subname of the project."
    ),
    overfit: bool = Option(  # noqa: B008
        False, help="Should the model be overfitted?"
    ),
    overfit_mb: bool = Option(  # noqa: B008
        False, help="Should the model be overfitted to the membrane?"
    ),
    partition_size: int = Option(  # noqa: B008
        2000, help="Size of the partition."
    ),
    force_recompute_partitioning: bool = Option(  # noqa: B008
        False, help="Should the partitioning be recomputed?"
    ),
    augment_all: bool = Option(  # noqa: B008
        True, help="Should all data be augmented?"
    ),
    pixel_size: float = Option(  # noqa: B008
        1.0, help="Pixel size of the tomogram."
    ),
    max_tomo_shape: int = Option(  # noqa: B008
        928, help="Maximum shape of the tomogram."
    ),
    k_eig: int = Option(  # noqa: B008
        128, help="Number of eigenvectors."
    ),
    N_block: int = Option(  # noqa: B008
        6, help="Number of blocks."
    ),
    C_width: int = Option(  # noqa: B008
        16, help="Width of the convolution."
    ),
    dropout: bool = Option(  # noqa: B008
        False, help="Should dropout be used?"
    ),
    with_gradient_features: bool = Option(  # noqa: B008
        True, help="Should the gradient features be used?"
    ),
    with_gradient_rotations: bool = Option(  # noqa: B008
        True, help="Should the gradient rotations be used?"
    ),
    device: str = Option(  # noqa: B008
        "cuda:0", help="Device to use."
    ),
    one_D_conv_first: bool = Option(  # noqa: B008
        False, help="Should 1D convolution be used first?"
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
    max_epochs: int = Option(  # noqa: B008
        1000, help="Maximum number of epochs."
    ),
):
    """Train a diffusion net model.

    Example
    -------
    membrain-pick train --data-dir <path-to-your-folder> --training-dir <path-to-your-folder>
    """
    _train(
        data_dir=data_dir,
        training_dir=training_dir,
        project_name=project_name,
        sub_name=sub_name,
        overfit=overfit,
        overfit_mb=overfit_mb,
        partition_size=partition_size,
        force_recompute_partitioning=force_recompute_partitioning,
        augment_all=augment_all,
        pixel_size=pixel_size,
        max_tomo_shape=max_tomo_shape,
        k_eig=k_eig,
        N_block=N_block,
        C_width=C_width,
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



# predict CLI
from membrain_pick.predict import predict as _predict

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
    is_single_mb: bool = Option(  # noqa: B008
        False, help="Should the prediction be done for a single membrane?"
    ),
    partition_size: int = Option(  # noqa: B008
        2000, help="Size of the partition."
    ),
    pixel_size: float = Option(  # noqa: B008
        1.0, help="Pixel size of the tomogram."
    ),
    max_tomo_shape: int = Option(  # noqa: B008
        928, help="Maximum shape of the tomogram."
    ),
    k_eig: int = Option(  # noqa: B008
        128, help="Number of eigenvectors."
    ),
    mean_shift_output: bool = Option(  # noqa: B008
        False, help="Should the output be mean shifted?"
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
        "cuda:0", help="Device to use for the mean shift."
    ),
):
    """Predict the output of the trained model on the given data.

    Example
    -------
    membrain-pick predict --data-dir <path-to-your-folder> --ckpt-path <path-to-your-checkpoint> --out-dir <path-to-store-output>
    """
    _predict(
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        is_single_mb=is_single_mb,
        partition_size=partition_size,
        pixel_size=pixel_size,
        max_tomo_shape=max_tomo_shape,
        k_eig=k_eig,
        mean_shift_output=mean_shift_output,
        mean_shift_bandwidth=mean_shift_bandwidth,
        mean_shift_max_iter=mean_shift_max_iter,
        mean_shift_margin=mean_shift_margin,
        mean_shift_score_threshold=mean_shift_score_threshold,
        mean_shift_device=mean_shift_device,
    )



from membrain_pick.mean_shift_inference import mean_shift_for_csv as _mean_shift_for_csv

@cli.command(name="mean_shift", no_args_is_help=True)
def mean_shift_for_csv(
    csv_path: str = Option(  # noqa: B008
        ..., help="Path to the CSV file.", **PKWARGS
    ),
    out_dir: str = Option(  # noqa: B008
        "./mean_shift_output", help="Path to the folder where the output should be stored."
    ),
    bandwidth: float = Option(  # noqa: B008
        7.0, help="Bandwidth for the mean shift."
    ),
    max_iter: int = Option(  # noqa: B008
        150, help="Maximum number of iterations for the mean shift."
    ),
    margin: float = Option(  # noqa: B008
        0.0, help="Margin for the mean shift."
    ),
    score_threshold: float = Option(  # noqa: B008
        9.0, help="Score threshold for the mean shift."
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
    _mean_shift_for_csv(
        csv_file=csv_path,
        out_dir=out_dir,
        bandwidth=bandwidth,
        max_iter=max_iter,
        margin=margin,
        score_threshold=score_threshold,
        device=device,
    )