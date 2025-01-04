"""
This module contains the implementation of the mesh partitioning functionalities.
"""

import os
import numpy as np
import hashlib
from typing import List, Tuple, Optional, Dict
from scipy.spatial import cKDTree
import potpourri3d as pp3d
from tqdm import tqdm


def compute_nearest_distances(point_data, PSII_pos):
    kd_tree = cKDTree(PSII_pos)
    distances, nn_idcs = kd_tree.query(point_data, k=1)
    return distances, nn_idcs


def get_array_hash(array):
    array_bytes = array.tobytes()
    hasher = hashlib.sha256()
    hasher.update(array_bytes)
    return hasher.hexdigest()


def exclude_faces_from_candidates(
    face_list: np.ndarray,
    candidates: np.ndarray,
    faces_weights: np.ndarray,
    min_weight: float,
):
    """
    Excludes faces from a list of candidates.

    Parameters
    ----------
    mb_idx : int
        Index of the membrane to be sampled.
    face_list : np.ndarray
        List of faces to be excluded.
    candidates : np.ndarray
        List of candidates to be filtered.
    faces_weights : np.ndarray
        List of weights for the faces.

    Returns
    -------
    np.ndarray
        Filtered list of candidates.
    """
    faces_weights = np.mean(faces_weights, axis=1)
    mask = np.isin(candidates, np.array(face_list)[faces_weights > min_weight])
    return candidates[~mask]


def get_adjacent_triangles(faces, edge_to_face_map, triangle_index):
    triangle = faces[triangle_index]
    adjacent_triangles = set()

    for i in range(3):
        edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
        for face_index in edge_to_face_map.get(edge, []):
            if face_index != triangle_index:
                adjacent_triangles.add(face_index)

    return list(adjacent_triangles)


def build_edge_to_face_map(faces):
    edge_to_face = {}
    for i, face in enumerate(faces):
        edges = [(face[j], face[(j + 1) % 3]) for j in range(3)]
        for edge in edges:
            edge = tuple(sorted(edge))  # Ensure the edge tuple is in a consistent order
            if edge not in edge_to_face:
                edge_to_face[edge] = []
            edge_to_face[edge].append(i)
    return edge_to_face


def gaussian_weights(dists, sigma):
    return np.exp(-(dists**2) / (2 * sigma**2))


def find_adjacent_faces(
    faces, verts: np.ndarray, start_face: int, max_sampled_points: int, solver=None
):
    """
    Finds all faces adjacent to the specified face.

    Parameters
    ----------
    faces : np.ndarray
        List of faces.
    verts : np.ndarray
        List of vertices.
    start_face : int
        Index of the face to start the search from.
    max_sampled_points : int
        Maximum number of sampled vertices.

    Returns
    -------
    np.ndarray
        Indices of the adjacent faces.
    """

    if solver is None:
        solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)

    # Choose first vertex of the face as the starting point
    start_vert = faces[start_face][0]

    # Compute geometric distances to all vertices
    dists = solver.compute_distance(start_vert)

    # Sort vertices by distance and select the closest ones
    sorted_idcs = np.argsort(dists)
    sorted_idcs = sorted_idcs[:max_sampled_points]

    # Find faces that contain the selected vertices
    # TODO: Can this lead to degenerate cases?
    mask = np.isin(faces, sorted_idcs)
    mask = np.all(mask, axis=1)
    cur_faces = np.where(mask)[0]

    # Compute weights for the selected faces
    face_verts = faces[cur_faces]
    face_dists = dists[face_verts]
    weights = gaussian_weights(face_dists, 0.5 * np.max(face_dists))

    return cur_faces, weights


def find_closest_gt_positions(gt_pos, mb_verts, max_tomo_shape, dist_threshold=3.0):
    """
    Finds the GT positions closest to the vertices of the current membrane patch.

    Parameters
    ----------
    mb_idx : int
        Index of the membrane to be sampled.
    max_tomo_shape : int
        Maximum shape of the tomogram.

    Returns
    -------
    np.ndarray
        Indices of the patches.
    """
    nn_dists, _ = compute_nearest_distances(gt_pos, mb_verts[:, :3])
    gt_mask = nn_dists < dist_threshold / max_tomo_shape
    return gt_pos[gt_mask]


def renumber_faces(face_verts, face_list):
    """
    Renumber the faces to refer to the new vertex indices.

    Parameters
    ----------
    face_verts : np.ndarray
        List of vertices.
    face_list : np.ndarray
        List of faces.

    Returns
    -------
    np.ndarray
        Indices of the patches.
    """
    unique_vertex_indices = np.unique(face_verts)
    index_mapping = {
        old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)
    }
    face_list_updated = np.vectorize(index_mapping.get)(face_list)
    return face_list_updated


def get_partition_from_face_list(
    mb: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    vert_normals: np.ndarray,
    gt_pos: np.ndarray,
    face_list: np.ndarray,
    face_weight_list: np.ndarray,
    max_tomo_shape: int,
):
    """
    Returns the partitioning of the membrane into patches based on a list of faces.

    Parameters
    ----------
    mb_idx : int
        Index of the membrane to be sampled.
    face_list : np.ndarray
        List of faces to be used for partitioning.
    face_weight_list : np.ndarray
        List of weights for the faces.

    Returns
    -------
    np.ndarray
        Indices of the patches.
    """
    face_verts = faces[face_list]
    face_verts = face_verts.reshape(-1)
    # vert_weights = np.repeat(face_weight_list, 3)
    vert_weights = face_weight_list.reshape(-1)

    unique_weights = np.unique(vert_weights)
    vert_weight_dict = {}
    for weight in unique_weights:
        for vert in face_verts[vert_weights == weight]:
            vert_weight_dict[vert] = weight

    face_verts = np.unique(face_verts)

    mb_verts = mb[face_verts]
    mb_faces = faces[face_list]
    mb_labels = labels[face_verts]
    mb_normals = vert_normals[face_verts]
    mb_vert_weights = np.array([vert_weight_dict[vert] for vert in face_verts])

    # Find close GT positions
    part_gts = find_closest_gt_positions(
        gt_pos, mb_verts, max_tomo_shape, dist_threshold=3.0
    )

    # Account for the renumbering of the vertices
    mb_faces_updated = renumber_faces(face_verts, mb_faces)

    return mb_verts, mb_labels, mb_faces_updated, mb_normals, part_gts, mb_vert_weights


def load_cached_partitioning(
    cache_dir: str,
    cur_cache_path: str,
    cache_path: str,
    force_recompute: bool,
    part_lists: Tuple[List[np.ndarray], ...],
    mb_idx: int,
    overfit: bool,
):
    cur_cache_count = 0
    cache_found = False
    while (
        cache_dir is not None and os.path.isfile(cur_cache_path) and not force_recompute
    ):
        cache = load_from_cache(cur_cache_path)

        # Appending data from cache
        append_partitioning_data(cache, part_lists, mb_idx)

        # Update cache path for next iteration
        cur_cache_count += 1
        cur_cache_path = cache_path[:-4] + f"_partnr{cur_cache_count}.npz"
        cache_found = True

        # Break if overfitting and cache count exceeds 3
        if overfit and cur_cache_count > 3:
            break
    return cache_found


def compute_and_cache_partitioning(
    mb: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    vert_normals: np.ndarray,
    gt_pos: np.ndarray,
    max_sampled_points: int,
    overfit: bool,
    cache_path: str,
    list_of_partitioning_data: Tuple[List[np.ndarray], ...],
    mb_idx: int,
    min_gaussian_weight: float,
):
    face_candidates = np.arange(faces[mb_idx].shape[0])
    print(
        "Precomputing partitioning for membrane",
        mb_idx,
        "with",
        face_candidates.shape[0],
        "faces.",
    )
    part_counter = 0

    # Initialize tqdm progress bar with the total equal to the initial number of face candidates
    total_len = face_candidates.shape[0]
    prev_len = 0
    no_improv_count = 0
    pbar = tqdm(total=total_len)

    solver = pp3d.MeshHeatMethodDistanceSolver(mb[:, :3], faces[mb_idx])
    while face_candidates.shape[0] > 0:
        if cache_path is not None:
            cur_cache_path = cache_path[:-4] + "_partnr" + str(part_counter) + ".npz"
        # random_idx = np.random.randint(face_candidates.shape[0])
        face_start = face_candidates[
            no_improv_count
        ]  # better fixed starting point for reproducibility
        adj_faces, adj_faces_weights = find_adjacent_faces(
            faces[mb_idx], mb[:, :3], face_start, max_sampled_points, solver=solver
        )
        (
            cur_part_verts,
            cur_part_labels,
            cur_part_faces,
            cur_part_normals,
            cur_part_gts,
            cur_part_vert_weights,
        ) = get_partition_from_face_list(
            mb=mb,
            faces=faces[mb_idx],
            labels=labels[mb_idx],
            vert_normals=vert_normals[mb_idx],
            gt_pos=gt_pos[mb_idx],
            face_list=adj_faces,
            face_weight_list=adj_faces_weights,
            max_tomo_shape=928,
        )
        # part_verts, part_labels, part_faces, part_normals, part_gts, part_vert_weights = get_partition_from_face_list(mb_idx, adj_faces, adj_faces_weights)
        cache = {
            "part_verts": cur_part_verts,
            "part_labels": cur_part_labels,
            "part_faces": cur_part_faces,
            "part_normals": cur_part_normals,
            "part_gt_pos": cur_part_gts,
            "part_vert_weights": cur_part_vert_weights,
        }
        append_partitioning_data(cache, list_of_partitioning_data, mb_idx)
        face_candidates = exclude_faces_from_candidates(
            adj_faces, face_candidates, adj_faces_weights, min_gaussian_weight
        )
        if overfit and part_counter > 3:
            break
        if cache_path is not None:
            np.savez(cur_cache_path, **cache)
        part_counter += 1
        pbar.update(total_len - face_candidates.shape[0] - prev_len)
        if total_len - face_candidates.shape[0] - prev_len == 0:
            no_improv_count += 1
            if face_candidates.shape[0] <= no_improv_count:
                no_improv_count = 0
                break
            if no_improv_count > 10:
                break
        else:
            no_improv_count = 0
        prev_len = total_len - face_candidates.shape[0]
    pbar.close()


def precompute_partitioning(
    membranes: list,
    faces: list,
    labels: list,
    vert_normals: list,
    gt_pos: list,
    max_sampled_points: int,
    overfit: bool,
    overfit_mb: bool,
    cache_dir: str = None,
    force_recompute: bool = False,
    min_gaussian_weight: float = 0.35,
):
    """
    Precomputes the partitioning of the membranes into patches.

    If already done in another run, load it from the cache. Otherwise, compute it,
    store it in the cache and load it.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Initialize lists to hold partitioning results
    part_verts, part_labels, part_faces, part_normals = [], [], [], []
    part_mb_idx, part_gt_pos, part_vert_weights = [], [], []

    for mb_idx, mb in enumerate(membranes):
        # encode both membrane data and self.load_only_sampled_points
        cache_out = get_cache_path(cache_dir, mb, max_sampled_points, overfit)
        if cache_out is None:
            cache_path = None
            cur_cache_path = None
            cache_found = False
        else:
            cache_path, cur_cache_path = cache_out
            cache_found = False

            cache_found = load_cached_partitioning(
                cache_dir,
                cur_cache_path,
                cache_path,
                force_recompute,
                (
                    part_verts,
                    part_labels,
                    part_faces,
                    part_normals,
                    part_mb_idx,
                    part_gt_pos,
                    part_vert_weights,
                ),
                mb_idx,
                overfit,
            )
        if cache_found:
            print(f"Loaded partitioning for membrane {mb_idx}.")
        else:
            print(f"Computing partitioning for membrane {mb_idx}.")
            compute_and_cache_partitioning(
                mb=mb,
                faces=faces,
                labels=labels,
                vert_normals=vert_normals,
                gt_pos=gt_pos,
                max_sampled_points=max_sampled_points,
                overfit=overfit,
                cache_path=cache_path,
                list_of_partitioning_data=(
                    part_verts,
                    part_labels,
                    part_faces,
                    part_normals,
                    part_mb_idx,
                    part_gt_pos,
                    part_vert_weights,
                ),
                mb_idx=mb_idx,
                min_gaussian_weight=min_gaussian_weight,
            )

        if overfit_mb:
            break
    return (
        part_verts,
        part_labels,
        part_faces,
        part_normals,
        part_mb_idx,
        part_gt_pos,
        part_vert_weights,
    )


def get_cache_path(
    cache_dir: str,
    mb: np.ndarray,
    max_sampled_points: int,
    overfit: bool,
) -> str:
    """
    Generates a cache file path for a given set of parameters.

    Parameters
    ----------
    cache_dir : str
        The directory where cache files are stored.
    mb_cache_hash : str
        Hash of the membrane data.
    sampled_points_cache_hash : str
        Hash of the max sampled points setting.
    overfit_cache_hash : str
        Hash of the overfit setting.
    part_counter : int
        The partition counter to differentiate between different partitions of the same membrane.

    Returns
    -------
    str
        The generated file path for the cache file.
    """
    if cache_dir is None:
        return None
    mb_cache_hash = get_array_hash(mb)
    sampled_points_cache_hash = str(hash(max_sampled_points))
    overfit_cache_hash = str(hash(overfit))
    part_counter = 0  # Initialize partition counter
    filename = f"{mb_cache_hash}_{sampled_points_cache_hash}_{overfit_cache_hash}_partnr{part_counter}.npz"
    cache_path = os.path.join(
        cache_dir,
        mb_cache_hash
        + "_"
        + sampled_points_cache_hash
        + "_"
        + overfit_cache_hash
        + ".npz",
    )
    return cache_path, os.path.join(cache_dir, filename)


def load_from_cache(cur_cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempts to load partitioning data from a cache file.

    Parameters
    ----------
    cur_cache_path : str
        Path to the cache file.

    Returns
    -------
    Optional[Dict[str, np.ndarray]]
        The loaded partitioning data if successful, None otherwise.
    """
    if os.path.isfile(cur_cache_path):
        print(f"Loading partitioning data from {cur_cache_path}")
        return np.load(cur_cache_path)
    else:
        return None


def append_partitioning_data(
    partitioning_data: Dict[str, np.ndarray],
    part_lists: Tuple[List[np.ndarray], ...],
    mb_idx: int,
):
    """
    Appends loaded or computed partitioning data to the respective lists.

    Parameters
    ----------
    partitioning_data : Dict[str, np.ndarray]
        The partitioning data to append.
    part_lists : Tuple[List[np.ndarray], ...]
        The tuple of lists to which the data will be appended.
    mb_idx : int
        The index of the current membrane being processed.
    """
    (
        part_verts,
        part_labels,
        part_faces,
        part_normals,
        part_mb_idx,
        part_gt_pos,
        part_vert_weights,
    ) = part_lists
    part_verts.append(partitioning_data["part_verts"])
    part_labels.append(partitioning_data["part_labels"])
    part_faces.append(partitioning_data["part_faces"])
    part_normals.append(partitioning_data["part_normals"])
    part_mb_idx.append(mb_idx)
    part_gt_pos.append(partitioning_data["part_gt_pos"])
    part_vert_weights.append(partitioning_data["part_vert_weights"])
