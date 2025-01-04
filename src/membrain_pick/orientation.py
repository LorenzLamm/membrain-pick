import os
import numpy as np
import scipy.spatial as spatial
import trimesh
from surforama.utils.geometry import rotate_around_vector
from scipy.spatial.transform import Rotation as R
from membrain_pick.dataloading.data_utils import (
    get_csv_data,
    read_star_file,
    store_array_in_csv,
    store_array_in_star,
)
from membrain_pick.mesh_projections.compute_mesh_projection import convert_seg_to_mesh
from membrain_seg.segmentation.dataloading.data_utils import load_tomogram


def orientation_from_mesh(coordinates, mesh):
    """
    Get the orientation of a point cloud from a mesh.

    This is taken in large parts from the Surforama codebase.

    Parameters
    ----------
    coordinates : np.ndarray
        The coordinates of the point cloud.
    mesh : trimesh.Trimesh
        The mesh.
    maximum_distance : float, optional
        The maximum distance to consider. Default is None.

    Returns
    -------
    np.ndarray
        The orientations of the point cloud.
    """
    # get the closest vertices
    tree = spatial.cKDTree(mesh.vertices)
    distances, vertex_indices = tree.query(coordinates)

    if np.any(distances > 200):
        print(
            "Warning: Some points are more than 200 units away from the mesh. This might be an error. Check rescaling factors."
        )

    mesh.vertices = mesh.vertices[:, ::-1]

    # get the normals of the closest vertices
    normals = mesh.vertex_normals[vertex_indices]
    # normalize the normals
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    # get perpendicular vectors
    up_vectors = np.cross(normals, np.array([0, 0, 1]))
    up_vectors /= np.linalg.norm(up_vectors, axis=1)[:, None]

    # draw a random angle between 0 and 360
    random_angle = np.random.uniform(0, 10)
    random_angle = np.zeros_like(random_angle)

    # rotate the up vectors around the normals
    up_vectors = rotate_around_vector(
        rotate_around=normals,
        to_rotate=up_vectors,
        angle=random_angle,
    )

    third_basis = np.cross(normals, up_vectors)
    n_points = len(coordinates)
    particle_orientations = np.zeros((n_points, 3, 3))
    particle_orientations[:, :, 0] = third_basis[:, ::-1]
    particle_orientations[:, :, 1] = up_vectors[:, ::-1]
    particle_orientations[:, :, 2] = normals[:, ::-1]

    euler_angles = (
        R.from_matrix(particle_orientations).inv().as_euler(seq="ZYZ", degrees=True)
    )  # This gives Euler angles in Relion format: Rot, Tilt, Psi
    return euler_angles


def convert_relion_to_stopgap(relion_angles):
    """
    Convert Euler angles from Relion format to Stopgap format.

    Parameters
    ----------
    relion_angles : np.ndarray
        The Euler angles in Relion format.

    Returns
    -------
    np.ndarray
        The Euler angles in Stopgap format.
    """

    # relion angles are ZYZ: Rot, Tilt, Psi
    # stopgap angles are ZXZ: Phi, Theta, Psi

    rot = relion_angles[:, 0]
    tilt = relion_angles[:, 1]
    psi = relion_angles[:, 2]

    phi = -rot - 90
    theta = -tilt
    psi = -psi + 90

    return np.stack([phi, theta, psi], axis=1)


def read_positions(positions_file):
    """
    Read positions from a file.

    Parameters
    ----------
    positions_file : str
        The path to the positions file.

    Returns
    -------
    np.ndarray
        The positions.
    """
    if positions_file.endswith(".csv"):
        return get_csv_data(positions_file)[:, :3]
    elif positions_file.endswith(".star"):
        star_data = read_star_file(positions_file)
        if "rlnCoordinateX" in star_data:
            return np.stack(
                [
                    star_data["rlnCoordinateX"],
                    star_data["rlnCoordinateY"],
                    star_data["rlnCoordinateZ"],
                ],
                axis=1,
            )
        elif "orig_x" in star_data:
            return np.stack(
                [
                    star_data["orig_x"],
                    star_data["orig_y"],
                    star_data["orig_z"],
                ],
                axis=1,
            )


def _store_array(in_file: str, out_dir: str, data: np.ndarray, out_format: str):
    """
    Store an array in a file.

    Parameters
    ----------
    in_file : str
        The input file.
    out_file : str
        The output file.
    data : np.ndarray
        The data to store.
    out_format : str
        The output format.
    """
    os.makedirs(out_dir, exist_ok=True)
    if in_file.endswith(".csv"):
        out_file = os.path.join(
            out_dir, os.path.basename(in_file).replace(".csv", "_withOrientation.star")
        )
        store_array_in_csv(out_file, data)
    elif in_file.endswith(".star"):
        out_file = os.path.join(
            out_dir, os.path.basename(in_file).replace(".star", "_withOrientation.star")
        )
        if out_format == "RELION":
            header = [
                "rlnCoordinateX",
                "rlnCoordinateY",
                "rlnCoordinateZ",
                "rlnAngleRot",
                "rlnAngleTilt",
                "rlnAnglePsi",
            ]
        elif out_format == "STOPGAP":
            header = [
                "orig_x",
                "orig_y",
                "orig_z",
                "phi",
                "theta",
                "psi",
            ]
        store_array_in_star(out_file, data, header=header)


def orientation_from_files(
    positions_file: str,
    out_dir: str,
    mesh_file: str = None,
    segmentation_file: str = None,
    positions_scale_factor: float = 1.0,
    out_format: str = "RELION",
):
    """
    Get the orientation of a point cloud from a mesh.

    Parameters
    ----------
    positions_file : str
        The path to the positions file.
    out_dir : str
        The output directory.
    mesh_file : str, optional
        The path to the mesh file. Default is None.
    segmentation_file : str, optional
        The path to the segmentation file. Default is None.
    maximum_distance : float, optional
        The maximum distance to consider. Default is None.
    positions_scale_factor : float, optional
        The scale factor for the positions. Default is 1.0.
    out_format : str, optional
        The output format. Default is "RELION".
    """
    positions = read_positions(positions_file)
    if positions_scale_factor != 1.0:
        positions *= positions_scale_factor

    if mesh_file is not None:
        mesh = trimesh.load_mesh(mesh_file)
    elif segmentation_file is not None:
        segmentation = load_tomogram(segmentation_file).data
        mesh = convert_seg_to_mesh(segmentation, smoothing=1000)
        # convert from pyvista to trimesh
        verts = mesh.points
        faces = mesh.faces
        faces = faces.reshape(-1, 4)[:, 1:]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    orientations = orientation_from_mesh(positions, mesh)
    if out_format == "RELION":
        orientations = convert_relion_to_stopgap(orientations)

    positions = np.concatenate([positions, orientations], axis=1)

    _store_array(
        in_file=positions_file,
        out_dir=out_dir,
        data=positions,
        out_format=out_format,
    )
