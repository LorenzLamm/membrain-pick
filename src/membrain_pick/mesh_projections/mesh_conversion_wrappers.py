import os
import numpy as np

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
from membrain_pick.mesh_projections.mesh_projection import convert_to_mesh


def meshes_for_folder_structure(
    mb_folder: str,
    tomo_folder,
    out_folder,
    only_obj=False,
    step_numbers=(-6, 7),
    step_size=2.5,  # in Angstrom
    mesh_smoothing=1000,
    barycentric_area=1.0,
    input_pixel_size=None,
    crop_box_flag=False,
    only_largest_component=True,
    min_connected_size=1e4,
    imod_meshing=False,
):
    """
    This assumes the following folder structure:

    tomo_folder
    ├── tomo1.mrc
    ├── tomo2.mrc
    └── ...

    mb_folder
    ├── tomo1
    │   ├── seg1.mrc
    │   ├── seg2.mrc
    │   └── ...
    ├── tomo2
    │   ├── seg1.mrc
    │   ├── seg2.mrc
    │   └── ...
    └── ...

    and will create the following folder structure:

    out_folder
    ├── tomo1
    │   ├── seg1_mesh_data.csv
    │   ├── seg1_mesh_faces.csv
    │   ├── seg1_mesh_normals.csv
    │   ├── seg1_mesh_normals.vtp
    │   ├── seg1_psii_pos.csv
    │   ├── seg1_seg.mrc
    │   ├── seg1.mrc
    │   ├── seg2_mesh_data.csv
    │   ├── seg2_mesh_faces.csv
    │   ├── seg2_mesh_normals.csv
    │   ├── seg2_mesh_normals.vtp
    │   ├── seg2_psii_pos.csv
    │   ├── seg2_seg.mrc
    │   ├── seg2.mrc
    │   └── ...
    ├── tomo2
    │   ├── seg1_mesh_data.csv
    │   ├── ...

    """
    os.makedirs(out_folder, exist_ok=True)

    tomo_files = [
        os.path.join(tomo_folder, f)
        for f in os.listdir(tomo_folder)
        if f.endswith(".mrc") or f.endswith(".rec")
    ]
    mb_subfolders = [
        os.path.join(mb_folder, f)
        for f in os.listdir(mb_folder)
        if os.path.isdir(os.path.join(mb_folder, f))
    ]
    mb_subfolders = [
        os.path.join(mb_folder, os.path.basename(f).split(".")[0])
        for f in tomo_files
        if os.path.isdir(os.path.join(mb_folder, os.path.basename(f).split(".")[0]))
    ]

    assert len(tomo_files) == len(mb_subfolders)

    for tomo_file, mb_folder in zip(tomo_files, mb_subfolders):
        out_tomo_folder = os.path.join(
            out_folder, os.path.basename(tomo_file).split(".")[0]
        )
        os.makedirs(out_tomo_folder, exist_ok=True)

        tomo = load_tomogram(tomo_file).data
        tomo_token = os.path.basename(mb_folder)

        mesh_for_tomo_mb_folder(
            tomo_file=tomo_file,
            mb_folder=mb_folder,
            out_folder=out_tomo_folder,
            tomo=tomo,
            tomo_token=tomo_token,
            only_obj=only_obj,
            step_numbers=step_numbers,
            step_size=step_size,
            mesh_smoothing=mesh_smoothing,
            barycentric_area=barycentric_area,
            input_pixel_size=input_pixel_size,
            crop_box_flag=crop_box_flag,
            only_largest_component=only_largest_component,
            min_connected_size=min_connected_size,
            imod_meshing=imod_meshing,
        )


def mesh_for_tomo_mb_folder(
    tomo_file: str,
    mb_folder: str,
    out_folder: str,
    tomo: np.ndarray = None,
    tomo_token=None,
    only_obj=False,
    step_numbers=(-6, 7),
    step_size=2.5,  # in Angstrom
    mesh_smoothing=1000,
    barycentric_area=1.0,
    input_pixel_size=None,
    crop_box_flag=False,
    only_largest_component=True,
    min_connected_size=1e4,
    imod_meshing=False,
):
    """
    This function assumes the following folder structure:

    mb_folder
    ├── seg1.mrc
    ├── seg2.mrc
    └── ...

    and will create the following folder structure:

    out_folder
    ├── seg1_mesh_data.csv
    ├── seg1_mesh_faces.csv
    ├── seg1_mesh_normals.csv
    ├── seg1_mesh_normals.vtp
    ├── seg1_psii_pos.csv
    ├── seg1_seg.mrc
    ├── seg1.mrc
    ├── ...
    """

    os.makedirs(out_folder, exist_ok=True)

    mb_files = [
        os.path.join(mb_folder, f) for f in os.listdir(mb_folder) if f.endswith(".mrc")
    ]
    print(mb_files)

    if tomo is None:
        tomo = load_tomogram(tomo_file)
        input_pixel_size = (
            tomo.voxel_size if input_pixel_size is None else input_pixel_size
        )
        tomo = tomo.data
        if tomo_token is None:
            tomo_token = os.path.basename(tomo_file).split(".")[0]

    if tomo_token is None:
        tomo_token = "Tomo"

    for mb_file in mb_files:
        mesh_for_single_mb_file(
            mb_file=mb_file,
            tomo_file=tomo_file,
            out_folder=out_folder,
            tomo=tomo,
            tomo_token=tomo_token,
            only_obj=only_obj,
            step_numbers=step_numbers,
            step_size=step_size,
            mesh_smoothing=mesh_smoothing,
            barycentric_area=barycentric_area,
            input_pixel_size=input_pixel_size,
            crop_box_flag=crop_box_flag,
            only_largest_component=only_largest_component,
            min_connected_size=min_connected_size,
            imod_meshing=imod_meshing,
        )


def mesh_for_single_mb_file(
    mb_file: str,
    tomo_file: str,
    out_folder: str,
    tomo: np.ndarray = None,
    tomo_token: str = None,
    only_obj=False,
    step_numbers=(-6, 7),
    step_size=2.5,  # in Angstrom
    mesh_smoothing=1000,
    barycentric_area=1.0,
    input_pixel_size=None,
    crop_box_flag=False,
    only_largest_component=True,
    min_connected_size=1e4,
    imod_meshing=False,
):
    """ """
    os.makedirs(out_folder, exist_ok=True)

    if tomo is None:
        tomo = load_tomogram(tomo_file)
        input_pixel_size = (
            tomo.voxel_size if input_pixel_size is None else input_pixel_size
        )
        tomo = tomo.data
        if tomo_token is None:
            tomo_token = os.path.basename(tomo_file).split(".")[0]

    if tomo_token is None:
        tomo_token = "Tomo"

    convert_to_mesh(
        mb_file,
        tomo_file,
        out_folder,
        tomo=tomo,
        token=tomo_token,
        only_obj=only_obj,
        step_numbers=step_numbers,
        step_size=step_size,
        mesh_smoothing=mesh_smoothing,
        barycentric_area=barycentric_area,
        input_pixel_size=input_pixel_size,
        crop_box_flag=crop_box_flag,
        only_largest_component=only_largest_component,
        min_connected_size=min_connected_size,
        imod_meshing=imod_meshing,
    )
