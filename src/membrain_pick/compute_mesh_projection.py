from time import time 
import os
import numpy as np

from membrain_seg.normal_processing.mesh_utils import convert_seg_to_mesh
from membrain_seg.segmentation.dataloading.data_utils import load_tomogram, store_tomogram, store_array_in_csv

import pyvista as pv
import pyacvd

from scipy.ndimage import map_coordinates

import trimesh


# def bspline_orig(t):
#     t = abs(t)
#     a = 2.0 - t

#     if t < 1.0: 
#         return 2.0 / 3.0 - 0.5 * t * t * a
#     elif t < 2.0: 
#         return a * a * a / 6.0
#     else: 
#         return 0.0


# def cubicTex3DSimple(volTexture, pos, texSize):
#     # transform the coordinate from [0,extent] to [-0.5, extent-0.5]
#     coord_grid = pos - 0.5

#     index = np.floor(coord_grid)
#     fraction = coord_grid - index
#     index = index + 0.5  #move from [-0.5, extent-0.5] to [0, extent]

#     result = 0.0
#     for z in np.arange(-1, 2.5, 1):  #range [-1, 2]
#         bsplineZ = bspline_orig(z - fraction[2])
#         w = index[2] + z
#         for y in np.arange(-1, 2.5, 1):
#             bsplineYZ = bspline_orig(y - fraction[1]) * bsplineZ
#             v = index[1] + y
#             for x in np.arange(-1, 2.5, 1):
#                 bsplineXYZ = bspline_orig(x - fraction[0]) * bsplineYZ
#                 u = index[0] + x
#                 voxel_value = map_coordinates(volTexture, np.array([[u], [v], [w]]), order=1)
#                 result += bsplineXYZ * voxel_value * texSize
#     return result[0]



def vectorized_cubicTex3DSimple(volTexture, pos, texSize):
    coord_grid = pos - 0.5

    index = np.floor(coord_grid).astype(int)
    fraction = coord_grid - index
    index = index + 0.5  # Correcting index to match original logic

    xyz_range = np.arange(-1, 3)
    x, y, z = np.meshgrid(xyz_range, xyz_range, xyz_range, indexing='ij')
    
    # Calculate bspline values for all combinations of x, y, z
    bspline_values = bspline(x - fraction[0], y - fraction[1], z - fraction[2])

    # Calculate u, v, w for all combinations
    u = (index[0] + x).flatten()
    v = (index[1] + y).flatten()
    w = (index[2] + z).flatten()

    # Fetch voxel values using advanced indexing
    voxel_values = map_coordinates(volTexture, np.array([u, v, w]), order=1, mode='nearest').reshape(x.shape)

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


def get_tomo_values_along_normal(point, normal, tomo, b_splines=True, steps=(-6, 7), step_size=0.25):
    values = []
    for add in range(steps[0], steps[1]):
        if not b_splines:
            idx = point + add * normal
            idx = np.round(idx).astype(int)
            tomo_val = tomo[idx[0], idx[1], idx[2]]
        else:
            tomo_val = vectorized_cubicTex3DSimple(tomo, point + step_size*add*normal, texSize=0.1)
            # tomo_val_compare = cubicTex3DSimple(tomo, point + 0.4*add*normal, texSize=0.1)
        values.append(tomo_val)
    values = np.stack(values, axis=0)
    return values


def compute_values_along_normals(mesh, tomo, b_splines=True, steps=(-6, 7), step_size=0.25, verts=None, normals=None):
    # Get vertices and triangle combinations
    if verts is None:
        verts = mesh.points
    if normals is None:
        normals = mesh.point_normals

    normal_values = []
    time_zero = time()
    for i in range(len(verts)):
        if i % 5000 == 0:
            print(i, "/", len(verts), time() - time_zero)
        norm_vals = get_tomo_values_along_normal(verts[i], normals[i], tomo, b_splines=b_splines, steps=steps, step_size=step_size)
        normal_values.append(norm_vals)
    normal_values = np.stack(normal_values, axis=0)
    return normal_values


def correct_normals(mesh):
    vertices = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # Reshape faces and remove the leading count

    # Create a trimesh mesh using the vertices and faces from PyVista
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    broken = trimesh.repair.broken_faces(mesh_trimesh, color=None)

    # Repair the mesh
    trimesh.repair.fill_holes(mesh_trimesh)

    # Repair the mesh
    trimesh.repair.fix_inversion(mesh_trimesh, multibody=True)

    # Now you can use trimesh's functionalities
    trimesh.repair.fix_normals(mesh_trimesh, multibody=True)

    # If you want to convert it back to PyVista mesh after manipulation
    mesh = pv.PolyData(mesh_trimesh.vertices, np.hstack([np.full((mesh_trimesh.faces.shape[0], 1), 3, dtype=int), mesh_trimesh.faces]))

    return mesh

def convert_seg_to_evenly_spaced_mesh(seg, smoothing=2000, barycentric_area=10, normals_correction=False):
    mesh = convert_seg_to_mesh(seg=seg,
                        smoothing=smoothing,
                        voxel_size=1.0)
    mesh = mesh.decimate(0.9)
    
    cluster_points = int(mesh.area / barycentric_area)
    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(cluster_points)
    remesh = clus.create_mesh()
    if normals_correction:
        remesh = correct_normals(remesh)
    return remesh


def main():
    path_to_seg = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/230425_tomo3_cc_cropped_mb.mrc"
    path_to_tomo = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/230425_tomo3_cc_cropped.rec"
    out_mesh = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/230425_tomo3_cc_cropped_mb_mesh.vtp"
    out_mesh_vox = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/230425_tomo3_cc_cropped_mb_mesh_voxelized.vtp"
    out_csv = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/230425_tomo3_cc_cropped_mb_mesh.csv"
    tomo = load_tomogram(path_to_tomo).data
    # seg = load_tomogram(path_to_seg).data
    # mesh = convert_seg_to_mesh(seg=seg,
    #                     smoothing=1000,
    #                     voxel_size=1.0)
    # mesh = mesh.decimate(0.9)
    # mesh.save(out_mesh)
    # exit()
    # read mesh in again
    mesh = pv.read(out_mesh)
    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)
    clus.cluster(200000)

    remesh = clus.create_mesh()
    remesh.save(out_mesh_vox)
    
    mesh = remesh
    # Get vertices and triangle combinations
    verts = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    normals = mesh.point_normals

    print(verts.shape, faces.shape, normals.shape)

    normal_values = []
    time_zero = time()
    for i in range(len(verts)):
        if i % 500 == 0:
            print(i, "/", len(verts), time() - time_zero)
        norm_vals = get_tomo_values_along_normal(verts[i], normals[i], tomo, b_splines=True)
        normal_values.append(norm_vals)
    print("Computing values took", time()-time_zero, "Bsplines were computed?", True)
    normal_values = np.stack(normal_values, axis=0)
    point_data = np.concatenate((verts, normals, normal_values), axis=1)
    store_array_in_csv(out_csv, point_data)

if __name__ == "__main__":
    main()