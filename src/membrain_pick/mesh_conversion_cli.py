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
    add_completion=False,
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

from membrain_pick.mesh_conversion_wrappers import (
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
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        2.5, help="Step size for the normal vectors. Default: 2.5"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        400.0, help="Barycentric area for the mesh. Default: 1.0"
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
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        2.5, help="Step size for the normal vectors. Default: 2.5"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        400.0, help="Barycentric area for the mesh. Default: 1.0"
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
        tomo_file=tomo_path,
        out_folder=out_folder,
        only_obj=only_obj,
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
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    output_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the output tomogram. Only used if match_size_flag is True.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-6, 7),
        help="Step numbers for the normal vectors. Default: (-6, 7)",
    ),
    step_size: float = Option(  # noqa: B008
        2.5, help="Step size for the normal vectors. Default: 2.5"
    ),
    mesh_smoothing: int = Option(  # noqa: B008
        1000, help="Smoothing factor for the mesh. Default: 1000"
    ),
    barycentric_area: float = Option(  # noqa: B008
        400.0, help="Barycentric area for the mesh. Default: 1.0"
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
    aug_prob_to_one: bool = Option(  # noqa: B008
        False, help="Should the probability be set to one?"
    ),
    input_pixel_size: float = Option(  # noqa: B008
        10.0, help="Pixel size of the tomogram."
    ),
    process_pixel_size: float = Option(  # noqa: B008
        15.0, help="Pixel size of the processed tomogram."
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
    conv_width: int = Option(  # noqa: B008
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
    allpos: bool = Option(  # noqa: B008
        False, help="Should all positive samples be used?"
    ),
    use_psii: bool = Option(  # noqa: B008
        True, help="Should PSII be used?"
    ),
    use_uk: bool = Option(  # noqa: B008
        False, help="Should UK be used?"
    ),
    use_b6f: bool = Option(  # noqa: B008
        False, help="Should b6f be used?"
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
        aug_prob_to_one=aug_prob_to_one,
        input_pixel_size=input_pixel_size,
        process_pixel_size=process_pixel_size,
        k_eig=k_eig,
        allpos=allpos,
        use_psii=use_psii,
        use_uk=use_uk,
        use_b6f=use_b6f,
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
    input_pixel_size: float = Option(  # noqa: B008
        10.0, help="Pixel size of the tomogram."
    ),
    process_pixel_size: float = Option(  # noqa: B008
        15.0, help="Pixel size of the tomogram."
    ),
    force_recompute_partitioning: bool = Option(  # noqa: B008
        False, help="Should the partitioning be recomputed?"
    ),
    N_block: int = Option(  # noqa: B008
        4, help="Number of blocks."
    ),
    C_width: int = Option(  # noqa: B008
        64, help="Width of the convolution."
    ),
    conv_width: int = Option(  # noqa: B008
        16, help="Width of the convolution."
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
        # "cuda:0", help="Device to use for the mean shift."
        "cpu", help="Device to use for the mean shift."
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
        input_pixel_size=input_pixel_size,
        process_pixel_size=process_pixel_size,
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






@cli.command(name="surforama", no_args_is_help=True)
def surforama(
    h5_path: str = Option(  # noqa: B008
        ...,
        help="Path to the h5 container.",
        **PKWARGS,
    ),
    tomogram_path: str = Option(  # noqa: B008
        default="", help="Path to the tomogram to be projected", 
    ),
):
    import os
    import napari
    from surforama.app import QtSurforama
    import numpy as np
    from matplotlib.pyplot import get_cmap
    from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5, get_csv_data
    from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
    from membrain_pick.scalar_selection import ScalarSelectionWidget

    from membrain_pick.surforama_cli_utils import (
            display_tomo,
            get_pixel_size,
            get_points_and_faces,
            display_scores,
            display_cluster_centers,
            initialize_surforama_widget,
            display_input_normal_values
        )

    viewer = napari.Viewer(ndisplay=3)

    mesh_files = None
    if os.path.isdir(h5_path):
        mesh_files = [os.path.join(h5_path, f) for f in os.listdir(h5_path) if f.endswith(".h5")]
    else:
        mesh_files = [h5_path]

    for h5_nr, h5_path in enumerate(mesh_files):
        mesh_data = load_mesh_from_hdf5(h5_path)

        if h5_nr == 0:
            volume_layer = display_tomo(viewer, mesh_data, tomogram_path)
            pixel_size = get_pixel_size(mesh_data, None)

        points, faces = get_points_and_faces(mesh_data, pixel_size)
        scores_layer = display_scores(viewer, mesh_data, points, faces)

        if h5_nr == 0:
            surforama_widget = initialize_surforama_widget(points, faces, volume_layer, viewer)
        display_cluster_centers(viewer, mesh_data, pixel_size, surforama_widget)

        if h5_nr == 0:
            display_input_normal_values(viewer, mesh_data, points, faces)
    napari.run()


@cli.command(name="tomotwin_extract", no_args_is_help=True)
def tomotwin_extract(
    h5_path: str = Option(  # noqa: B008
        ...,
        help="Path to the h5 container with predicted positions.",
        **PKWARGS,
    ),
    output_dir: str = Option(  # noqa: B008
        "./subvolumes", help="Path to the folder where the subvolumes should be stored."
    ),
    tomogram_path: str = Option(  # noqa: B008
        default="", help="Path to the tomogram to extract subvolumes from", 
    ),
):
    """
    Extract subvolumes from the tomogram using the predicted positions.
    
    The subvolumes are saved as single .mrc files in size (37, 37, 37). This is the
    format required by TomoTwin. 
    The extracted subvolumes can therefore be used to generate TomoTwin embeddings
    for the predicted positions.
    """

    from membrain_pick.tomotwin_extract import tomotwin_extract_subvolumes

    tomotwin_extract_subvolumes(
        h5_path=h5_path,
        output_dir=output_dir,
        tomogram_path=tomogram_path,
    )

@cli.command(name="tomotwin_embeddings", no_args_is_help=True)
def tomotwin_embeddings(
        subvolume_folder: str = Option(  # noqa: B008
            ..., help="Path to the folder containing the subvolumes.", **PKWARGS
        ),
        output_folder: str = Option(  # noqa: B008
            "./embeddings", help="Path to the folder where the embeddings should be stored."
        ),
        model_path: str = Option(  # noqa: B008
            ..., help="Path to the model checkpoint.", **PKWARGS
        ),
        batch_size: int = Option(  # noqa: B008
            12, help="Batch size."
        ),):
        """
        This command generates TomoTwin embeddings for the subvolumes using the given model checkpoint.

        IMPORTANT: The TomoTwin embeddings command must be executed in a Python environment with the TomoTwin package installed.
        TomoTwin can be installed via
        pip install tomotwin-cryoet

        More information at:
        https://tomotwin-cryoet.readthedocs.io/en/stable/installation.html#installation

        WARNING: Having MemBrain and TomoTwin installed in the same environment can lead to conflicts. 
        Therefore, the safest way to generate TomoTwin embeddings is to create a separate environment for TomoTwin.
        If you encounter any issues, please report on GitHub.
        """
        import shutil
        import subprocess

        if shutil.which('tomotwin_embed.py') is None:
            print("tomotwin_embed.py command is not available. Please use a Python environment with TomoTwin installed.")
            print("In this environment, run the following code to generate TomoTwin embeddings:")

            print(f"tomotwin_embed.py subvolumes -m {model_path} -v {subvolume_folder}/*.mrc -b {batch_size} -o {output_folder}")
            return

        # Construct the embedding command
        command = [
            "tomotwin_embed.py",
            "subvolumes",
            "-m", model_path,
            "-v", f"{subvolume_folder}/*.mrc",
            "-b", "12",
            "-o", output_folder
        ]
        
        # Execute the embedding command
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        
        if result.returncode != 0:
            print(f"Error executing the embedding command: {result.stderr}")
        else:
            print(f"Embedding command executed successfully: {result.stdout}")

