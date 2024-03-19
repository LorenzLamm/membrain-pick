from time import time


import os
import numpy as np

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram, store_tomogram, store_array_in_csv, store_point_and_vectors_in_vtp
from membrain_seg.tomo_preprocessing.pixel_size_matching.match_pixel_size import match_pixel_size
from membrain_seg.tomo_preprocessing.pixel_size_matching.match_pixel_size_seg import match_segmentation_pixel_size_to_tomo
from membrain_pick.compute_mesh_projection import convert_seg_to_evenly_spaced_mesh, compute_values_along_normals
from membrain_pick.mesh_class import Mesh

from membrain_pick.mesh_projection_utils import get_connected_components, get_cropped_arrays, get_normals_from_face_order, get_connected_components


def match_tomo_pixel_size(tomo_file, tomo_file_out, pixel_size_out=14.08, pixel_size_in=None, match_file=None, tmp_folder="./temp_mesh_data"):
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder, exist_ok=True)

    if match_file is not None:
        match_segmentation_pixel_size_to_tomo(
            seg_path=tomo_file,
            orig_tomo_path=match_file,
            output_path=tomo_file_out
        )
    else:
        match_pixel_size(
            input_tomogram=tomo_file,
            output_path=tomo_file_out,
            pixel_size_out=pixel_size_out,
            pixel_size_in=pixel_size_in,
            disable_smooth=False
        )

def meshes_for_folder_structure(mb_folder: str, 
                                tomo_folder, 
                                out_folder,
                                only_obj=False,
                                match_size_flag=False,
                                temp_folder="./temp_mesh_data",
                                step_numbers=(-6, 7),
                                step_size=0.25,
                                mesh_smoothing=1000,
                                barycentric_area=1.0,
                                input_pixel_size=None,
                                output_pixel_size=None,
                                crop_box_flag=False,
                                only_largest_component=True,
                                min_connected_size=1e4):
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

    tomo_files = [os.path.join(tomo_folder, f) for f in os.listdir(tomo_folder) if f.endswith(".mrc") or f.endswith(".rec")]
    mb_subfolders = [os.path.join(mb_folder, f) for f in os.listdir(mb_folder) if os.path.isdir(os.path.join(mb_folder, f))]
    
    for tomo_file, mb_folder in zip(tomo_files, mb_subfolders):
        out_tomo_folder = os.path.join(out_folder, os.path.basename(tomo_file).split(".")[0])
        os.makedirs(out_tomo_folder, exist_ok=True)

        if match_size_flag:
            tomo_file_tmp = os.path.join(temp_folder, os.path.basename(tomo_file))
            match_tomo_pixel_size(tomo_file, tomo_file_tmp, pixel_size_out=output_pixel_size, pixel_size_in=input_pixel_size)
            tomo_file = tomo_file_tmp
        
        tomo = load_tomogram(tomo_file).data

        mesh_for_tomo_mb_folder(tomo_file=tomo_file,
                                mb_folder=mb_folder,
                                out_folder=out_tomo_folder,
                                tomo=tomo,
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
                                min_connected_size=min_connected_size)
        
                
def mesh_for_tomo_mb_folder(tomo_file: str,
                            mb_folder: str,
                            out_folder: str,
                            tomo: np.ndarray = None,
                            only_obj=False,
                            match_size_flag=False,
                            temp_folder="./temp_mesh_data",
                            step_numbers=(-6, 7),
                            step_size=0.25,
                            mesh_smoothing=1000,
                            barycentric_area=1.0,
                            input_pixel_size=None,
                            output_pixel_size=None,
                            crop_box_flag=False,
                            only_largest_component=True,
                            min_connected_size=1e4):
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

    mb_files = [os.path.join(mb_folder, f) for f in os.listdir(mb_folder) if f.endswith(".mrc")]

    if match_size_flag:
        if tomo is None:
            tomo_file_tmp = os.path.join(temp_folder, os.path.basename(tomo_file))
            match_tomo_pixel_size(tomo_file, tomo_file_tmp, pixel_size_out=output_pixel_size, pixel_size_in=input_pixel_size)
            tomo_file = tomo_file_tmp
    
    if tomo is None:
        tomo = load_tomogram(tomo_file).data

    for mb_file in mb_files:
        mesh_for_single_mb_file(mb_file=mb_file,
                                tomo_file=tomo_file,
                                out_folder=out_folder,
                                tomo=tomo,
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
                                min_connected_size=min_connected_size)


def mesh_for_single_mb_file(mb_file: str,
                            tomo_file: str,
                            out_folder: str,
                            tomo: np.ndarray = None,
                            only_obj=False,
                            match_size_flag=False,
                            temp_folder="./temp_mesh_data",
                            step_numbers=(-6, 7),
                            step_size=0.25,
                            mesh_smoothing=1000,
                            barycentric_area=1.0,
                            input_pixel_size=None,
                            output_pixel_size=None,
                            crop_box_flag=False,
                            only_largest_component=True,
                            min_connected_size=1e4):
    """
    """
    os.makedirs(out_folder, exist_ok=True)

    if match_size_flag:
        if tomo is None:
            tomo_file_tmp = os.path.join(temp_folder, os.path.basename(tomo_file))
            match_tomo_pixel_size(tomo_file, tomo_file_tmp, pixel_size_out=output_pixel_size, pixel_size_in=input_pixel_size)
            tomo_file = tomo_file_tmp

        mb_file_tmp = os.path.join(temp_folder, os.path.basename(mb_file))
        match_tomo_pixel_size(mb_file, mb_file_tmp, pixel_size_out=output_pixel_size, pixel_size_in=input_pixel_size, match_file=tomo_file)
        mb_file = mb_file_tmp

    if tomo is None:
        tomo = load_tomogram(tomo_file).data

    convert_to_mesh(mb_file, 
                    tomo_file, 
                    out_folder, 
                    tomo=tomo, 
                    only_obj=only_obj,
                    step_numbers=step_numbers,
                    step_size=step_size,
                    mesh_smoothing=mesh_smoothing,
                    barycentric_area=barycentric_area,
                    crop_box_flag=crop_box_flag,
                    only_largest_component=only_largest_component, 
                    min_connected_size=min_connected_size)


def convert_to_mesh(mb_file, 
                    tomo_file, 
                    out_folder, 
                    tomo=None, 
                    only_obj=False,
                    step_numbers=(-6, 7),
                    step_size=0.25,
                    mesh_smoothing=1000,
                    barycentric_area=1.0,
                    crop_box_flag=False,
                    only_largest_component=True, 
                    min_connected_size=1e4):
    
    
    print(f"Processing {mb_file}")
    mb_key = os.path.basename(mb_file).split(".")[0]
    seg = load_tomogram(mb_file).data

    seg = get_connected_components(seg, only_largest=only_largest_component)

    if tomo is None:
        print(f"Loading tomo {tomo_file}")
        tomo = load_tomogram(tomo_file).data

    sub_seg_count = 0
    for k in range(1, seg.max() + 1):
        if np.sum(seg == k) < min_connected_size:
            continue
        print(f"Processing sub-seg {k} of {mb_key}. Total seg count {float(seg.max())}")
        sub_seg_count += 1
        cur_mb_key = mb_key + f"_{sub_seg_count}"

        if crop_box_flag:
            cur_seg, cur_tomo = get_cropped_arrays(seg, tomo)
        else:
            cur_seg = seg == k
            cur_tomo = tomo

        mesh = convert_seg_to_evenly_spaced_mesh(seg=cur_seg,
                                                 smoothing=mesh_smoothing,
                                                 barycentric_area=barycentric_area)
        
        points, faces, point_normals = get_normals_from_face_order(mesh)

        if not only_obj:
            normal_values = compute_values_along_normals(mesh=mesh, 
                                                        tomo=cur_tomo, 
                                                        steps=step_numbers, 
                                                        step_size=step_size, 
                                                        verts=points,
                                                        normals=point_normals)
            print("Computed values along normals")

        out_data = np.concatenate([mesh.points, normal_values], axis=1)
        out_file = os.path.join(out_folder, cur_mb_key + "_mesh_data.csv")
        out_file_faces = os.path.join(out_folder, cur_mb_key + "_mesh_faces.csv")
        out_file_normals = os.path.join(out_folder, cur_mb_key + "_mesh_normals.csv")
        out_file_normals_vtp = os.path.join(out_folder, mb_key + "_mesh_normals.vtp")

        if not only_obj:
            store_array_in_csv(out_file, out_data)
            store_array_in_csv(out_file_faces, faces)
            store_array_in_csv(out_file_normals, mesh.point_normals*(-1))

            store_point_and_vectors_in_vtp(out_file_normals_vtp, mesh.points, mesh.point_normals, in_scalars=[normal_values[:, k] for k in range(normal_values.shape[1])])

        mesh = Mesh(vertices=mesh.points, triangle_combos=faces+1)
        mesh.store_in_file(out_file.replace(".csv", ".obj"))
        store_tomogram(out_file.replace(".csv", ".mrc"), cur_tomo)
        store_tomogram(out_file.replace(".csv", "_seg.mrc"), cur_seg)



def main():
    mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/4Lorenz/Tomo_1/membranes/"        
    tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/4Lorenz/Tomo_1/Tomo1L1_bin4.rec"
    out_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data/Chlamy_old"
    temp_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data/Chlamy_old/temp"

    match_size_flag = False
    recompute_matching = False
    crop = True

    mesh_for_tomo_mb_folder(
        tomo_file=tomo_file,
        mb_folder=mb_folder,
        out_folder=out_folder,
        match_size_flag=match_size_flag,
        temp_folder=temp_folder,
        crop_box_flag=crop
    )


if __name__ == "__main__":
    main()
