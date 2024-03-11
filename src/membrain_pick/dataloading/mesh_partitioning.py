"""
This module contains the implementation of the mesh partitioning functionalities.
"""
import os
import numpy as np
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree


def compute_nearest_distances(point_data, PSII_pos):
    kd_tree = cKDTree(PSII_pos)
    distances, nn_idcs = kd_tree.query(point_data, k=1)
    return distances, nn_idcs


def get_array_hash(array):
    array_bytes = array.tobytes()
    hasher = hashlib.sha256()
    hasher.update(array_bytes)
    return hasher.hexdigest()


def exclude_faces_from_candidates(face_list: np.ndarray, candidates: np.ndarray, faces_weights: np.ndarray):
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

        mask = np.isin(candidates, np.array(face_list)[faces_weights == 1.0])
        return candidates[~mask]


def get_adjacent_triangles(faces, edge_to_face_map, triangle_index):
        triangle = faces[triangle_index]
        adjacent_triangles = set()

        for i in range(3):
            edge = tuple(sorted((triangle[i], triangle[(i+1) % 3])))
            for face_index in edge_to_face_map.get(edge, []):
                if face_index != triangle_index:
                    adjacent_triangles.add(face_index)

        return list(adjacent_triangles)

def build_edge_to_face_map(faces):
        edge_to_face = {}
        for i, face in enumerate(faces):
            edges = [(face[j], face[(j+1) % 3]) for j in range(3)]
            for edge in edges:
                edge = tuple(sorted(edge))  # Ensure the edge tuple is in a consistent order
                if edge not in edge_to_face:
                    edge_to_face[edge] = []
                edge_to_face[edge].append(i)
        return edge_to_face

def find_adjacent_faces(faces, mb_idx: int, start_face: int, edge_to_face_map: dict, max_sampled_points: int):
        """
        Finds all faces adjacent to the specified face.

        Parameters
        ----------
        mb_idx : int
            Index of the membrane to be sampled.
        start_face : int
            Index of the face to start the search from.

        Returns
        -------
        np.ndarray
            Indices of the adjacent faces.
        """
        back_1_weight = 0.0
        back_2_weight = 0.125
        back_3_weight = 0.250
        back_4_weight = 0.375
        back_5_weight = 0.500

        faces = faces[mb_idx]
        
        cur_faces = [start_face]
        cur_faces_weights = {start_face: back_1_weight}

        faces_back_1 = []
        faces_back_2 = []
        faces_back_3 = []
        faces_back_4 = []
        faces_back_5 = []

        while len(cur_faces) < max_sampled_points:
            prev_len = len(cur_faces)
            for face in cur_faces:
                
                # add_faces = find_faces_sharing_vertices(faces, faces[face])
                add_faces = get_adjacent_triangles(faces, edge_to_face_map, face)
                cur_faces.extend(add_faces)
                cur_faces = list(np.unique(cur_faces))
                
                if len(cur_faces) >= max_sampled_points:
                    break

            back_1_mask = np.isin(cur_faces, faces_back_1)
            back_2_mask = np.isin(cur_faces, faces_back_2)
            back_3_mask = np.isin(cur_faces, faces_back_3)
            back_4_mask = np.isin(cur_faces, faces_back_4)
            back_5_mask = np.isin(cur_faces, faces_back_5)

            cur_faces = np.array(cur_faces)
            cur_faces_weights = {face: back_1_weight for face in cur_faces}
            cur_faces_weights.update({face: back_2_weight for face in cur_faces[back_1_mask]})
            cur_faces_weights.update({face: back_3_weight for face in cur_faces[back_2_mask]})
            cur_faces_weights.update({face: back_4_weight for face in cur_faces[back_3_mask]})
            cur_faces_weights.update({face: back_5_weight for face in cur_faces[back_4_mask]})
            cur_faces_weights.update({face: 1.0 for face in cur_faces[back_5_mask]})

            cur_faces = list(cur_faces)


            faces_back_5 = faces_back_5 + faces_back_4
            faces_back_4 = faces_back_3
            faces_back_3 = faces_back_2
            faces_back_2 = faces_back_1
            faces_back_1 = cur_faces


            new_len = len(cur_faces)
            if new_len == prev_len:
                print("No new faces added, breaking.")
                break
        cur_faces = np.unique(cur_faces)
        cur_faces_weights[start_face] = 1.0
        weights = np.array([cur_faces_weights[face] for face in cur_faces])
        return cur_faces, weights

def get_partition_from_face_list(mb: np.ndarray,
                                 faces: np.ndarray, 
                                 labels: np.ndarray,
                                 vert_normals: np.ndarray,
                                 gt_pos: np.ndarray,
                                 face_list: np.ndarray, 
                                 face_weight_list: np.ndarray, 
                                 max_tomo_shape: int):
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
        vert_weights = np.repeat(face_weight_list, 3)
        vert_weights = vert_weights.reshape(-1)
        
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
        nn_dists, _ = compute_nearest_distances(gt_pos, mb_verts[:, :3])
        gt_mask = nn_dists < 3. / max_tomo_shape
        part_gts = gt_pos[gt_mask]

        # Create a mapping from old vertex indices to new indices
        unique_vertex_indices = np.unique(face_verts)
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Update the faces to refer to the new vertex indices
        mb_faces_updated = np.vectorize(index_mapping.get)(mb_faces)
        # return mb, mb_labels, mb_faces

        # return mb, labels, faces
        return mb_verts, mb_labels, mb_faces_updated, mb_normals, part_gts, mb_vert_weights


def load_cached_partitioning(cache_dir: str,
                             cur_cache_path: str,
                             cache_path: str,
                             force_recompute: bool,
                             part_lists: Tuple[List[np.ndarray], ...],
                             mb_idx: int,
                             overfit: bool

):
     cur_cache_count = 0
     cache_found = False
     while cache_dir is not None and os.path.isfile(cur_cache_path) and not force_recompute:
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
            mb_idx: int
):
    edge_to_face_map = build_edge_to_face_map(faces[mb_idx])
    face_candidates = np.arange(faces[mb_idx].shape[0])
    print("Precomputing partitioning for membrane", mb_idx, "with", face_candidates.shape[0], "faces.")
    part_counter = 0
    while face_candidates.shape[0] > 0:
        cur_cache_path = cache_path[:-4] + "_partnr" + str(part_counter) + ".npz"
        face_start = face_candidates[0]
        print("Starting from face", face_start, "with", face_candidates.shape[0], "faces left.")
        adj_faces, adj_faces_weights = find_adjacent_faces(faces, mb_idx, face_start, edge_to_face_map, max_sampled_points)
        cur_part_verts, cur_part_labels, cur_part_faces, cur_part_normals, cur_part_gts, cur_part_vert_weights = get_partition_from_face_list(
                mb=mb,
                faces=faces[mb_idx],
                labels=labels[mb_idx],
                vert_normals=vert_normals[mb_idx],
                gt_pos=gt_pos[mb_idx],
                face_list=adj_faces,
                face_weight_list=adj_faces_weights,
                max_tomo_shape=928
        )
        # part_verts, part_labels, part_faces, part_normals, part_gts, part_vert_weights = get_partition_from_face_list(mb_idx, adj_faces, adj_faces_weights)
        cache = {
            "part_verts": cur_part_verts,
            "part_labels": cur_part_labels,
            "part_faces": cur_part_faces,
            "part_normals": cur_part_normals,
            "part_gt_pos": cur_part_gts,
            "part_vert_weights": cur_part_vert_weights
        }
        append_partitioning_data(cache, list_of_partitioning_data, mb_idx)
        face_candidates = exclude_faces_from_candidates(adj_faces, face_candidates, adj_faces_weights)
        if overfit and part_counter > 2:
            break
        print("Saving partitioning for membrane", mb_idx, "to cache.", "with", part_counter, "patches.")
        print("Cache file:", cur_cache_path)
        np.savez(cur_cache_path, **cache)
        part_counter += 1


def precompute_partitioning(
        membranes: list,
        faces: list,
        labels: list,
        vert_normals: list,
        gt_pos: list,
        max_sampled_points: int,
        overfit: bool,
        overfit_mb: bool,
        cache_dir: str=None,
        force_recompute: bool=False
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
            cache_path, cur_cache_path = get_cache_path(cache_dir, mb, max_sampled_points, overfit)
            cache_found = False
            print(f"Loading partitioning for membrane {mb_idx} from cache.")

            # Load from cache if available
            cache_found = load_cached_partitioning(cache_dir, 
                                                   cur_cache_path, 
                                                   cache_path, 
                                                   force_recompute, 
                                                   (part_verts, part_labels, part_faces, part_normals, part_mb_idx, part_gt_pos, part_vert_weights), 
                                                   mb_idx, 
                                                   overfit)


            if not cache_found:
                compute_and_cache_partitioning(
                    mb=mb,
                    faces=faces,
                    labels=labels,
                    vert_normals=vert_normals,
                    gt_pos=gt_pos,
                    max_sampled_points=max_sampled_points,
                    overfit=overfit,
                    cache_path=cache_path,
                    list_of_partitioning_data=(part_verts, part_labels, part_faces, part_normals, part_mb_idx, part_gt_pos, part_vert_weights),
                    mb_idx=mb_idx
                )
            
            if overfit_mb:
                break
        return part_verts, part_labels, part_faces, part_normals, part_mb_idx, part_gt_pos, part_vert_weights


def get_cache_path(cache_dir: str, 
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
    mb_cache_hash = get_array_hash(mb)
    sampled_points_cache_hash = str(hash(max_sampled_points))
    overfit_cache_hash = str(hash(overfit))
    part_counter = 0  # Initialize partition counter
    filename = f"{mb_cache_hash}_{sampled_points_cache_hash}_{overfit_cache_hash}_partnr{part_counter}.npz"
    cache_path = os.path.join(cache_dir, mb_cache_hash + "_" + sampled_points_cache_hash + "_" + overfit_cache_hash + ".npz")
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
    

def append_partitioning_data(partitioning_data: Dict[str, np.ndarray], part_lists: Tuple[List[np.ndarray], ...], mb_idx: int):
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
    (part_verts, part_labels, part_faces, part_normals, part_mb_idx, part_gt_pos, part_vert_weights) = part_lists
    part_verts.append(partitioning_data["part_verts"])
    part_labels.append(partitioning_data["part_labels"])
    part_faces.append(partitioning_data["part_faces"])
    part_normals.append(partitioning_data["part_normals"])
    part_mb_idx.append(mb_idx)
    part_gt_pos.append(partitioning_data["part_gt_pos"])
    part_vert_weights.append(partitioning_data["part_vert_weights"])

