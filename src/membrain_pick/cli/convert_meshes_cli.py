import os
from typing import List

from typer import Option
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli



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
        "./mesh_data",
        help="Path to the folder where mesh projections should be stored.",
    ),
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-10, 10),
        help="Step numbers for the normal vectors. Default: (-10, 10)",
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
    imod_meshing: bool = Option(  # noqa: B008
        False,
        help="Should the mesh be generated using IMOD? WARNING: This is highly experimental.",
    ),
):
    """Convert a single membrane segmentation to a mesh.

    Example
    -------
    membrain-pick convert_file --tomogram-path <path-to-your-tomo> --mb-path <path-to-your-membrane-segmentation> --out-folder <path-to-store-meshes>
    """

    from membrain_pick.mesh_projections.mesh_conversion_wrappers import mesh_for_single_mb_file

    mesh_for_single_mb_file(
        mb_file=mb_path,
        tomo_file=tomogram_path,
        out_folder=out_folder,
        tomo=None,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        imod_meshing=imod_meshing,
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
        "./mesh_data",
        help="Path to the folder where mesh projections should be stored.",
    ),
    input_pixel_size: float = Option(  # noqa: B008
        None,
        help="Pixel size of the input tomogram. Only used if match_size_flag is True. If not provided, the pixel size will be read from the tomogram.",
    ),
    step_numbers: List[int] = Option(  # noqa: B008
        (-10, 10),
        help="Step numbers for the normal vectors. Default: (-10, 10)",
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
    imod_meshing: bool = Option(  # noqa: B008
        False,
        help="Should the mesh be generated using IMOD? WARNING: This is highly experimental.",
    ),
):
    """Convert a folder of membrane segmentations to meshes.

    Example
    -------
    membrain-pick convert_mb_folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> --out-folder <path-to-store-meshes>
    """
    from membrain_pick.mesh_projections.mesh_conversion_wrappers import mesh_for_tomo_mb_folder

    mesh_for_tomo_mb_folder(
        mb_folder=mb_folder,
        tomo_file=tomo_path,
        out_folder=out_folder,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        imod_meshing=imod_meshing,
    )
