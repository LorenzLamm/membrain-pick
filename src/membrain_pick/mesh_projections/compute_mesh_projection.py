import tempfile
import numpy as np
import os
from subprocess import run
from membrain_seg.segmentation.dataloading.data_utils import (
    store_tomogram,
)
import pyvista as pv
import pyacvd
from skimage import measure
from scipy.ndimage import map_coordinates
import trimesh


def imod_mesh_conversion(seg, temp_folder="./tmp_data"):
    """
    Convert a segmentation to a mesh using IMOD.

    Parameters
    ----------
    seg : np.ndarray
        The segmentation array.
    temp_folder : str, optional
        The temporary folder to store the mesh. Default is "./".

    Returns
    -------
    trimesh.Trimesh
        The resulting mesh.
    """
    os.makedirs(temp_folder, exist_ok=True)

    # define temporary files
    temp_out = os.path.join(temp_folder, "tmp_seg.mrc")
    temp_mod_file = os.path.join(temp_folder, "tomo_segmentation.mod")
    mesh_path = os.path.join(temp_folder, "tmp_mesh.obj")

    # Save the segmentation to a temporary file
    store_tomogram(temp_out, seg)

    # Run IMOD to convert the segmentation to a mesh
    run(f"imodauto -h 0.1 {temp_out} {temp_mod_file}", shell=True)
    run(f"imodmesh -s -f -C {temp_mod_file}", shell=True)
    run(f"imod2obj {temp_mod_file} {mesh_path}", shell=True)

    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)

    # delete the temporary files
    os.remove(temp_out)
    os.remove(temp_mod_file)
    os.remove(mesh_path)

    return mesh.vertices, mesh.faces


def convert_seg_to_mesh(
    seg: np.ndarray, smoothing: int, voxel_size: float = 1.0, imod_meshing=False
) -> pv.PolyData:
    """
    Convert a segmentation array to a mesh using marching cubes.

    Parameters
    ----------
    seg : np.ndarray
        The segmentation array.
    smoothing : int
        The number of smoothing iterations to apply to the mesh.
    voxel_size : float, optional
        The voxel size of the segmentation array. Default is 1.0.
    imod_meshing : bool, optional
        Whether to use IMOD for meshing instead of marching cubes. Default is False.

    Returns
    -------
    pv.PolyData
        The resulting mesh.
    """
    if imod_meshing:
        verts, faces = imod_mesh_conversion(seg)
    else:
        verts, faces, _, _ = measure.marching_cubes(
            seg, 0.5, step_size=1.5, method="lewiner"
        )
    verts = verts * voxel_size
    all_col = np.ones((faces.shape[0], 1), dtype=int) * 3  # Prepend 3 for vtk format
    faces = np.concatenate((all_col, faces), axis=1)
    surf = pv.PolyData(verts, faces)
    surf = surf.smooth_taubin(n_iter=smoothing)
    surf = surf.decimate(0.95)
    return surf.smooth_taubin(n_iter=smoothing)


def compute_positions_along_normals(
    mesh,  
    steps=(-6, 7),
    step_size=2.5,  # in Angstrom
    verts=None,
    normals=None,
    input_pixel_size=14.08,
):
    """
    Compute 3D positions along normal directions from mesh vertices.

    Parameters
    ----------
    mesh : object
        Mesh object with points and point_normals attributes.
    steps : tuple of int, optional
        Range of steps along the normal (start, end). Default is (-6, 7).
    step_size : float, optional
        Step size in Angstrom for sampling along normals. Default is 2.5.
    verts : np.ndarray, optional
        Mesh vertices. If None, uses mesh.points. Default is None.
    normals : np.ndarray, optional
        Vertex normals. If None, uses mesh.point_normals. Default is None.
    input_pixel_size : float, optional
        Pixel size in Angstrom for unit conversion. Default is 14.08.

    Returns
    -------
    np.ndarray
        Array of 3D positions along normal directions with shape (n_vertices, n_steps, 3).
    """
    # Get vertices and triangle combinations
    if verts is None:
        verts = mesh.points
    if normals is None:
        normals = mesh.point_normals

    tomo_step_size = step_size / input_pixel_size

    positions = (
        verts[:, None, :]
        + normals[:, None, :]
        * np.arange(steps[0], steps[1])[None, :, None]
        * tomo_step_size
    )

    return positions


def compute_values_along_normals(
    mesh,  # TODO: remove
    tomo,
    steps=(-6, 7),
    input_pixel_size=10.0,
    step_size=2.5,  # in Angstrom
    verts=None,
    normals=None,
):
    """
    Compute tomogram values along normal directions from mesh vertices.

    Parameters
    ----------
    mesh : object
        Mesh object with points and point_normals attributes.
    tomo : np.ndarray
        3D tomogram volume array.
    steps : tuple of int, optional
        Range of steps along the normal (start, end). Default is (-6, 7).
    input_pixel_size : float, optional
        Pixel size in Angstrom for unit conversion. Default is 10.0.
    step_size : float, optional
        Step size in Angstrom for sampling along normals. Default is 2.5.
    verts : np.ndarray, optional
        Mesh vertices. If None, uses mesh.points. Default is None.
    normals : np.ndarray, optional
        Vertex normals. If None, uses mesh.point_normals. Default is None.

    Returns
    -------
    np.ndarray
        Array of tomogram values along normal directions with shape (n_vertices, n_steps).
    """
    # Get vertices and triangle combinations
    positions = compute_positions_along_normals(
        mesh,
        steps=steps,
        step_size=step_size,
        verts=verts,
        normals=normals,
        input_pixel_size=input_pixel_size,
    )

    # Get values along normals
    positions_shape = positions.shape
    positions = positions.reshape(-1, 3)
    positions_transposed = positions.T
    normal_values = map_coordinates(tomo, positions_transposed, order=3)
    normal_values = normal_values.reshape(
        positions_shape[0], positions_shape[1], order="C"
    )
    return normal_values


def convert_seg_to_mesh_pymeshlab(seg, input_pixel_size=14.08, barycentric_area=10):
    """
    Convert segmentation to mesh using PyMeshLab for advanced processing.

    Parameters
    ----------
    seg : np.ndarray
        The segmentation array.
    input_pixel_size : float, optional
        Pixel size in Angstrom for unit conversion. Default is 14.08.
    barycentric_area : float, optional
        Target barycentric area for remeshing. Default is 10.

    Returns
    -------
    pv.PolyData
        Processed mesh with smoothing and remeshing applied.
    """
    import pymeshlab as ml

    # Generate vertices and faces from marching cubes
    vertices, faces, _, _ = measure.marching_cubes(
        seg, 0.5, step_size=1.5, method="lewiner"
    )
    all_col = np.ones((faces.shape[0], 1), dtype=int) * 3  # Prepend 3 for vtk format
    faces = np.concatenate((all_col, faces), axis=1)
    faces = faces.flatten()

    # Create a PolyData mesh with PyVista
    mesh = pv.PolyData(vertices, faces)

    # Use a temporary file for storing intermediate mesh
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        temp_mesh_path = temp_file.name
        mesh.save(temp_mesh_path)

    # Load Mesh into PyMeshLab for processing
    ms = ml.MeshSet()
    ms.load_new_mesh(temp_mesh_path)

    # Apply Laplacian Smoothing
    ms.apply_filter(
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=10,
        boundary=False,
        cotangentweight=True,
        selected=False,
    )

    # Calculate target side length for remeshing
    calc_barycentric_area = barycentric_area / (input_pixel_size**2)
    estimated_face_area = calc_barycentric_area * 0.5
    side_len = np.sqrt(4 * estimated_face_area / np.sqrt(3))

    # Check if AbsoluteValue or PureValue is available
    # (PureValue is available in newer versions of PyMeshLab)
    try:
        target_len = ml.AbsoluteValue(side_len)  # For older versions
    except AttributeError:
        target_len = ml.PureValue(side_len)  # For newer versions

    # Apply Isotropic Explicit Remeshing
    ms.apply_filter(
        "meshing_isotropic_explicit_remeshing", iterations=10, targetlen=target_len
    )

    # Save the final simplified mesh back to a new temporary file
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as final_temp_file:
        simplified_mesh_path = final_temp_file.name
        ms.save_current_mesh(simplified_mesh_path)

    # Load the final mesh with PyVista
    simplified_mesh = pv.read(simplified_mesh_path)

    simplified_mesh = simplified_mesh.smooth_taubin(n_iter=500)

    return simplified_mesh


def convert_seg_to_evenly_spaced_mesh(
    seg,
    smoothing=2000,
    barycentric_area=10,
    was_rescaled=False,  # TODO: remove
    input_pixel_size=14.08,
    imod_meshing=False,
):
    """
    Convert segmentation to evenly spaced mesh with clustering-based remeshing.

    Parameters
    ----------
    seg : np.ndarray
        The segmentation array.
    smoothing : int, optional
        Number of smoothing iterations to apply. Default is 2000.
    barycentric_area : float, optional
        Target barycentric area for mesh clustering. Default is 10.
    was_rescaled : bool, optional
        Whether the segmentation was previously rescaled. Default is False.
    input_pixel_size : float, optional
        Pixel size in Angstrom for unit conversion. Default is 14.08.
    imod_meshing : bool, optional
        Whether to use IMOD for meshing instead of marching cubes. Default is False.

    Returns
    -------
    pv.PolyData
        Evenly spaced remeshed surface with applied smoothing.
    """
    # Convert segmentation to mesh using marching cubes
    mesh = convert_seg_to_mesh(
        seg=seg,
        smoothing=smoothing,
        imod_meshing=imod_meshing,
    )
    if was_rescaled:
        mesh.points = mesh.points * 2  # rescale to original size

    mesh.points *= (
        input_pixel_size  # rescale to phyiscal size to compute barycentric area
    )

    cluster_points = int(mesh.area / barycentric_area)
    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(cluster_points)
    remesh = clus.create_mesh()

    remesh.points /= input_pixel_size  # rescale back to original size
    remesh = remesh.smooth_taubin(n_iter=smoothing // 100)

    return remesh
