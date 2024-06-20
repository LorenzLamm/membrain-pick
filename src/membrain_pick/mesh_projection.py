import os
import numpy as np
from scipy.ndimage import zoom

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram, store_tomogram
from membrain_pick.dataloading.data_utils import store_array_in_csv, store_point_and_vectors_in_vtp, store_mesh_in_hdf5
from membrain_pick.compute_mesh_projection import convert_seg_to_evenly_spaced_mesh, compute_values_along_normals
from membrain_pick.mesh_class import Mesh

from membrain_pick.mesh_projection_utils import get_connected_components, get_cropped_arrays, get_normals_from_face_order, get_connected_components, remove_unused_vertices


def load_data(
    mb_file: str, only_largest_component: bool, tomo_file: str, tomogram: np.ndarray
) -> np.ndarray:
    """
    Load and process the segmentation data.

    Parameters
    ----------
    mb_file : str
        Path to the membrane file.
    only_largest_component : bool
        Flag to process only the largest connected component.
    tomo_file : str
        Path to the tomogram file.
    tomogram : np.ndarray
        Tomogram data -- if provided, this will be used instead of loading the data from the file.

    Returns
    -------
    np.ndarray
        The tomogram and the segmentation data.
    str
        The membrane key.
    """
    mb_key = os.path.basename(mb_file).split(".")[0]
    seg = load_tomogram(mb_file).data
    seg = zoom(seg, (0.5, 0.5, 0.5), order=0)
    seg = get_connected_components(seg, only_largest=only_largest_component)

    if tomogram is None:
        tomogram = load_tomogram(tomo_file).data

    return tomogram, seg, mb_key


def get_sub_segment(
    seg: np.ndarray, k: int, tomo: np.ndarray, crop_box_flag: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a sub-segment of the segmentation data.

    Parameters
    ----------
    seg : np.ndarray
        Segmentation data.
    k : int
        Sub-segment index.
    tomo : np.ndarray
        Tomogram data.
    crop_box_flag : bool
        Flag to crop the arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Processed sub-segment and tomogram data.
    """
    if crop_box_flag:
        cur_seg, cur_tomo = get_cropped_arrays(seg, tomo)
    else:
        cur_seg = seg == k
        cur_tomo = tomo
    return cur_seg, cur_tomo

def get_cur_mb_key(mb_key, only_largest_component, k, sub_seg_count, seg):
    if not only_largest_component:
        print(f"Processing sub-seg {k} of {mb_key}. Total seg count {float(seg.max())}")
        sub_seg_count += 1
        cur_mb_key = mb_key + f"_{sub_seg_count}"
    else:
        cur_mb_key = mb_key
    return cur_mb_key, sub_seg_count


def save_mesh_data(
    out_file_base: str, mesh: Mesh, points: np.ndarray, faces: np.ndarray,
    point_normals: np.ndarray, normal_values: np.ndarray, only_obj: bool, tomo_file: str = None,
    pixel_size: float = None
) -> None:
    """
    Save the mesh data to files.

    Parameters
    ----------
    out_file_base : str
        Base name for the output files.
    mesh : Mesh
        Mesh object.
    points : np.ndarray
        Points data.
    faces : np.ndarray
        Faces data.
    point_normals : np.ndarray
        Point normals data.
    normal_values : np.ndarray
        Normal values data.
    only_obj : bool
        Flag to store only the .obj file.
    tomo_file : str, optional
        Path to the tomogram file.

    Returns
    -------
    None
    """
    out_file = f"{out_file_base}_mesh_data.csv"
    out_file_faces = f"{out_file_base}_mesh_faces.csv"
    out_file_normals = f"{out_file_base}_mesh_normals.csv"
    out_file_normals_vtp = f"{out_file_base}_mesh_normals.vtp"

    if not only_obj:
        out_data = np.concatenate([points, normal_values], axis=1)
        store_array_in_csv(out_file, out_data)
        store_array_in_csv(out_file_faces, faces)
        store_array_in_csv(out_file_normals, point_normals * -1)
        store_mesh_in_hdf5(
            out_file=out_file_base + "_mesh.h5",
            points=points,
            faces=faces,
            normals=point_normals,
            normal_values=normal_values,
            tomo_file=tomo_file,
            pixel_size=pixel_size
        )
        store_point_and_vectors_in_vtp(
            out_file_normals_vtp, points, point_normals,
            in_scalars=[normal_values[:, k] for k in range(normal_values.shape[1])]
        )

    mesh = Mesh(points, faces + 1)
    mesh.store_in_file(out_file.replace(".csv", ".obj"))
    return out_file



def convert_to_mesh(
    mb_file: str, 
    tomo_file: str, 
    out_folder: str, 
    tomo: np.ndarray = None, 
    only_obj: bool = False,
    token: str = None,
    step_numbers: tuple[int, int] = (-6, 7),
    step_size: float = 2.5,
    mesh_smoothing: int = 1000,
    barycentric_area: float = 1.0,
    input_pixel_size: float = None,
    output_pixel_size: float = None,
    crop_box_flag: bool = False,
    only_largest_component: bool = True, 
    min_connected_size: float = 1e4
) -> None:
    """
    Converts segmentation data into a mesh format and stores it.

    Parameters
    ----------
    mb_file : str
        Path to the membrane file.
    tomo_file : str
        Path to the tomogram file.
    out_folder : str
        Path to the output folder.
    tomo : np.ndarray, optional
        Preloaded tomogram data.
    only_obj : bool, optional
        Flag to store only the .obj file.
    token : str, optional
        Unique identifier for output files.
    step_numbers : tuple[int, int], optional
        Range of steps for computing values along normals.
    step_size : float, optional
        Step size for computing values along normals. Units are in Angstrom.
    mesh_smoothing : int, optional
        Smoothing factor for mesh generation.
    barycentric_area : float, optional
        Barycentric area for mesh generation.
    input_pixel_size : float, optional
        Input pixel size for scaling.
    output_pixel_size : float, optional
        Output pixel size for scaling.
    crop_box_flag : bool, optional
        Flag to crop the arrays.
    only_largest_component : bool, optional
        Flag to process only the largest connected component.
    min_connected_size : float, optional
        Minimum size for connected components to be processed.

    Returns
    -------
    None
    """
    
    print(f"Processing {mb_file}")
    tomo, seg, mb_key = load_data(mb_file, only_largest_component, tomo_file, tomo)

    sub_seg_count = 0
    for k in range(1, seg.max() + 1):
        if np.sum(seg == k) < min_connected_size:
            continue
        cur_mb_key, sub_seg_count = get_cur_mb_key(mb_key, only_largest_component, k, sub_seg_count, seg)
        cur_seg, cur_tomo = get_sub_segment(seg, k, tomo, crop_box_flag)
 
        # This returns vertices in the new pixel size -- be careful!!
        mesh = convert_seg_to_evenly_spaced_mesh(seg=cur_seg,
                                                 smoothing=mesh_smoothing,
                                                 was_rescaled=True, #TODO: make adjustable
                                                 input_pixel_size=input_pixel_size,
                                                 output_pixel_size=output_pixel_size,
                                                 barycentric_area=barycentric_area)
        
        points, faces, point_normals = get_normals_from_face_order(mesh)
        points, faces, point_normals = remove_unused_vertices(points, faces, point_normals)

        if not only_obj:
            normal_values = compute_values_along_normals(mesh=mesh, 
                                                        tomo=cur_tomo, 
                                                        steps=step_numbers, 
                                                        step_size=step_size, 
                                                        input_pixel_size=input_pixel_size,
                                                        output_pixel_size=output_pixel_size,
                                                        verts=points,
                                                        normals=point_normals)

        out_file = save_mesh_data(
            out_file_base=os.path.join(out_folder, token + "_" + cur_mb_key),
             mesh=mesh, 
             points=points, 
             faces=faces, 
             point_normals=point_normals, 
             normal_values=normal_values, 
             only_obj=only_obj,
             tomo_file=tomo_file,
             pixel_size=1. # points dimension corresponds already to tomogram dimensions
             )

        if crop_box_flag:
            store_tomogram(out_file.replace(".csv", ".mrc"), cur_tomo)
            store_tomogram(out_file.replace(".csv", "_seg.mrc"), cur_seg)
