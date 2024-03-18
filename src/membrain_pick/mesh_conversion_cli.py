"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup


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
