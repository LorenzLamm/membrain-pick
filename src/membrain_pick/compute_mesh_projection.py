from time import time
import numpy as np

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
from membrain_pick.dataloading.data_utils import store_array_in_csv

import pyvista as pv
import pyacvd

from skimage import measure

from scipy.ndimage import map_coordinates


def convert_seg_to_mesh(
    seg: np.ndarray, smoothing: int, voxel_size: float = 1.0
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

    Returns
    -------
    pv.PolyData
        The resulting mesh.
    """
    verts, faces, _, _ = measure.marching_cubes(
        seg, 0.5, step_size=1.5, method="lewiner"
    )
    verts = verts * voxel_size
    all_col = np.ones((faces.shape[0], 1), dtype=int) * 3  # Prepend 3 for vtk format
    faces = np.concatenate((all_col, faces), axis=1)
    surf = pv.PolyData(verts, faces)
    return surf.smooth(n_iter=smoothing)


def vectorized_cubicTex3DSimple(volTexture, pos, texSize):
    coord_grid = pos - 0.5

    index = np.floor(coord_grid).astype(int)
    fraction = coord_grid - index
    index = index + 0.5  # Correcting index to match original logic

    xyz_range = np.arange(-1, 3)
    x, y, z = np.meshgrid(xyz_range, xyz_range, xyz_range, indexing="ij")

    # Calculate bspline values for all combinations of x, y, z
    bspline_values = bspline(x - fraction[0], y - fraction[1], z - fraction[2])

    # Calculate u, v, w for all combinations
    u = (index[0] + x).flatten()
    v = (index[1] + y).flatten()
    w = (index[2] + z).flatten()

    # Fetch voxel values using advanced indexing
    voxel_values = map_coordinates(
        volTexture, np.array([u, v, w]), order=1, mode="nearest"
    ).reshape(x.shape)

    # Compute result
    result = np.sum(bspline_values * voxel_values) * texSize

    return result


def bspline(x, y, z):
    # Vectorized bspline computation
    t = np.sqrt(x**2 + y**2 + z**2)
    a = 2.0 - t

    result = np.zeros_like(t)
    mask1 = t < 1.0
    mask2 = t < 2.0

    result[mask1] = 2.0 / 3.0 - 0.5 * t[mask1] * t[mask1] * a[mask1]
    result[mask2 & ~mask1] = a[mask2 & ~mask1] ** 3 / 6.0

    return result


def get_tomo_values_along_normal(
    point, normal, tomo, b_splines=True, steps=(-6, 7), step_size=0.25
):
    values = []
    for add in range(steps[0], steps[1]):
        if not b_splines:
            idx = point + add * normal
            idx = np.round(idx).astype(int)
            tomo_val = tomo[idx[0], idx[1], idx[2]]
        else:
            tomo_val = vectorized_cubicTex3DSimple(
                tomo, point + step_size * add * normal, texSize=0.1
            )
            # tomo_val_compare = cubicTex3DSimple(tomo, point + 0.4*add*normal, texSize=0.1)
        values.append(tomo_val)
    values = np.stack(values, axis=0)
    return values


def compute_positions_along_normals(
    mesh,  # TODO: clean up
    steps=(-6, 7),
    step_size=2.5,  # in Angstrom
    verts=None,
    normals=None,
    input_pixel_size=14.08,
    output_pixel_size=10.0,
):  # TODO: remove
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

    # # Go back to original pixel size to match tomogram #TODO: remove
    # pixel_factor = input_pixel_size / output_pixel_size #TODO: remove
    # positions = positions / pixel_factor #TODO: remove
    return positions


def compute_values_along_normals(
    mesh,  # TODO: remove
    tomo,
    steps=(-6, 7),
    input_pixel_size=10.0,
    output_pixel_size=10.0,
    step_size=2.5,  # in Angstrom
    verts=None,
    normals=None,
):
    # Get vertices and triangle combinations
    positions = compute_positions_along_normals(
        mesh,
        steps=steps,
        step_size=step_size,
        verts=verts,
        normals=normals,
        input_pixel_size=input_pixel_size,
        output_pixel_size=output_pixel_size,
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


def convert_seg_to_evenly_spaced_mesh(
    seg,
    smoothing=2000,
    barycentric_area=10,
    was_rescaled=False,
    input_pixel_size=14.08,  # TODO: remove
):
    # Convert segmentation to mesh using marching cubes
    mesh = convert_seg_to_mesh(
        seg=seg,
        smoothing=smoothing,
    )
    mesh = mesh.decimate(0.5)

    # pixel_factor = input_pixel_size / output_pixel_size #TODO:remove
    # mesh.points = mesh.points * pixel_factor * 2 #TODO: remove
    if was_rescaled:
        mesh.points = mesh.points * 2  # rescale to original size

    mesh.points *= (
        input_pixel_size  # rescale to phyiscal size to compute barycentric area
    )

    # Remesh the mesh to have evenly spaced triangles
    cluster_points = int(mesh.area / barycentric_area)
    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(cluster_points)
    remesh = clus.create_mesh()

    remesh.points /= input_pixel_size  # rescale back to original size

    return remesh
