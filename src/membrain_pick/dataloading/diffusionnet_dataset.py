import cProfile
import pstats

import os
from typing import Dict
from time import time

import numpy as np
from torch.utils.data import Dataset
import pyvista as pv

from membrain_seg.segmentation.dataloading.data_utils import get_csv_data, store_point_and_vectors_in_vtp

from scipy.spatial import cKDTree
from collections import defaultdict
import hashlib

def compute_nearest_distances(point_data, PSII_pos):
    kd_tree = cKDTree(PSII_pos)
    distances, nn_idcs = kd_tree.query(point_data, k=1)
    return distances, nn_idcs


def get_array_hash(array):
    array_bytes = array.tobytes()
    hasher = hashlib.sha256()
    hasher.update(array_bytes)
    return hasher.hexdigest()


def find_faces_sharing_vertices(mesh_faces, initial_face):
    """
    Find all faces in the mesh that share at least two vertices with the initial face.

    Parameters:
        mesh_faces (np.ndarray): Array of faces in the mesh, shape (N, 3).
        initial_face (np.ndarray): The initial face, shape (3,).

    Returns:
        np.ndarray: Array of indices of faces in mesh_faces that share at least two vertices with initial_face.
    """
    # Count the number of shared vertices between each face in the mesh and the initial face

    shared_vertices_count = np.sum(np.isin(mesh_faces, initial_face), axis=1)

    # Find the indices of faces that share at least two vertices with the initial face
    faces_indices = np.where(shared_vertices_count == 2)[0]

    return faces_indices


# def precompute_vertex_to_face_mapping(mesh_faces):
#     """
#     Precompute a mapping from each vertex to the faces that contain it.

#     Parameters:
#         mesh_faces (np.ndarray): Array of faces in the mesh, shape (N, 3).

#     Returns:
#         dict: A dictionary where keys are vertex indices and values are lists of face indices.
#     """
#     vertex_to_faces = defaultdict(set)
#     for i, face in enumerate(mesh_faces):
#         for vertex in face:
#             vertex_to_faces[vertex].add(i)
#     return vertex_to_faces


# def find_faces_sharing_vertices(mesh_faces, initial_face, vertex_to_faces):
#     """
#     Find all faces in the mesh that share at least two vertices with the initial face.

#     Parameters:
#         mesh_faces (np.ndarray): Array of faces in the mesh, shape (N, 3).
#         initial_face (np.ndarray): The initial face, shape (3,).
#         vertex_to_faces (dict): Mapping from each vertex to faces that contain it.

#     Returns:
#         set: Set of indices of faces in mesh_faces that share at least two vertices with initial_face.
#     """
#     # Get sets of candidate faces for each vertex in the initial face
#     candidate_faces = [vertex_to_faces[vertex] for vertex in initial_face]

#     # Find the intersection of candidate faces - faces that contain at least two of the initial face's vertices
#     shared_faces = set.intersection(*candidate_faces) - {np.where((mesh_faces == initial_face).all(axis=1))[0][0]}

#     return shared_faces

class MemSegDiffusionNetDataset(Dataset):
    """
    A custom Dataset for Cryo-ET membrane protein localization.

    This Dataset is designed to work with Mask-RCNN
    """

    def __init__(
        self,
        csv_folder: str,
        pointcloud: bool = False,
        mesh_data: bool = False,
        train: bool = False,
        train_pct: float = 0.8,
        max_tomo_shape: int = 928,
        load_only_sampled_points: int = None,
        overfit: bool = False,
        cache_dir: str = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/diffusion_net_training/mesh_cache",
        force_recompute: bool = False,
        overfit_mb: bool = False,
    
    ) -> None:
        """
        Constructs all the necessary attributes for the CryoETMemSegDataset object.
        """
        assert pointcloud ^ mesh_data, "Either pointcloud or mesh_data must be True"
        self.train = train
        self.pointcloud = pointcloud
        self.mesh_data = mesh_data
        self.train_pct = train_pct
        self.csv_folder = csv_folder
        self.max_tomo_shape = max_tomo_shape
        self.load_only_sampled_points = load_only_sampled_points
        self.overfit = overfit
        self.overfit_mb = overfit_mb
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute


        self.initialize_csv_paths()
        self.load_data()

        if self.load_only_sampled_points is not None:
            self._precompute_partitioning()

        self.patch_dicts = [None] * len(self)
        

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing an data-label pair for the provided index.

        Parameters
        ----------
        idx : int
            Index of the sample to be fetched.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing a membrane and its corresponding label.
        """
        if self.load_only_sampled_points is not None:
            idx_dict = {
                "membrane": np.expand_dims(self.part_verts[idx], 0),
                "label": np.expand_dims(self.part_labels[idx], 0),
                "faces": np.expand_dims(self.part_faces[idx], 0),
                "normals": np.expand_dims(self.part_normals[idx], 0),
                "mb_idx": self.part_mb_idx[idx],
                "gt_pos": self.part_gt_pos[idx]
            }
            
            # mb_idx = idx // (self.membranes[0].shape[0] // self.load_only_sampled_points)
            # idx_dict = self.select_random_mb_area(mb_idx, idx, self.load_only_sampled_points)
        else:
            idx_dict = {
                "membrane": np.expand_dims(self.membranes[idx], 0),
                "label": np.expand_dims(self.labels[idx], 0),
                "faces": np.expand_dims(self.faces[idx], 0),
                "normals": np.expand_dims(self.vert_normals[idx], 0),
                "mb_idx": idx,
                "gt_pos": self.gt_pos[idx]
            }
        # idx_dict = self.transforms(idx_dict)
        return idx_dict
    
    def _precompute_partitioning(self):
        """
        Precomputes the partitioning of the membranes into patches.

        If already done in another run, load it from the cache. Otherwise, compute it,
        store it in the cache and load it.
        """
        self.part_verts = []
        self.part_labels = []
        self.part_faces = []
        self.part_normals = []
        self.part_mb_idx = []
        self.part_gt_pos = []

        
        for mb_idx, mb in enumerate(self.membranes):
            # encode both membrane data and self.load_only_sampled_points
            mb_cache_hash = get_array_hash(mb)
            sampled_points_cache_hash = str(hash(self.load_only_sampled_points))
            overfit_cache_hash = str(hash(self.overfit))
            cache_path = os.path.join(self.cache_dir, mb_cache_hash + "_" + sampled_points_cache_hash + "_" + overfit_cache_hash + ".npz")

            cache_found = False
            print(f"Loading partitioning for membrane {mb_idx} from cache.")
            cur_cache_count = 0
            cur_cache_path = cache_path[:-4] + "_partnr0.npz"

            while os.path.isfile(cur_cache_path) and not self.force_recompute:
                # Append loaded data to the lists
                cache = np.load(cur_cache_path)
                self.part_verts.append(cache[f"part_verts"])
                self.part_labels.append(cache[f"part_labels"])
                self.part_faces.append(cache[f"part_faces"])
                self.part_normals.append(cache[f"part_normals"])
                self.part_mb_idx.append(mb_idx)
                self.part_gt_pos.append(cache[f"part_gt_pos"])
                print("Appending data from", cur_cache_path, "to partitioning.")
                cur_cache_path = cache_path[:-4] + "_partnr" + str(cur_cache_count) + ".npz"
                cur_cache_count += 1
                cache_found = True

            if not cache_found:
                
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

                from time import time
                time_zero = time()
                self.edge_to_face_map = build_edge_to_face_map(self.faces[mb_idx])
                face_candidates = np.arange(self.faces[mb_idx].shape[0])
                print("Precomputing partitioning for membrane", mb_idx, "with", face_candidates.shape[0], "faces.")
                part_counter = 0
                while face_candidates.shape[0] > 0:
                    cur_cache_path = cache_path[:-4] + "_partnr" + str(part_counter) + ".npz"
                    face_start = face_candidates[0]
                    print("Starting from face", face_start, "with", face_candidates.shape[0], "faces left.")
                    adj_faces = self.find_adjacent_faces(mb_idx, face_start)
                    part_verts, part_labels, part_faces, part_normals, part_gts = self.get_partition_from_face_list(mb_idx, adj_faces)
                    self.part_verts.append(part_verts)
                    self.part_labels.append(part_labels)
                    self.part_faces.append(part_faces)
                    self.part_normals.append(part_normals)
                    self.part_mb_idx.append(mb_idx)
                    self.part_gt_pos.append(part_gts) 
                    face_candidates = self.exclude_faces_from_candidates(adj_faces, face_candidates)
                    if self.overfit and part_counter > 0:
                        break
                    print("Saving partitioning for membrane", mb_idx, "to cache.", "with", len(self.part_verts), "patches.")
                    print("Cache file:", cur_cache_path)
                    np.savez(cur_cache_path,
                    **{f"part_verts": part_verts,
                        f"part_labels": part_labels,
                        f"part_faces": part_faces,
                        f"part_normals": part_normals,
                        f"part_gt_pos": part_gts})
                    part_counter += 1
            if self.overfit_mb:
                break



    def exclude_faces_from_candidates(self, face_list: np.ndarray, candidates: np.ndarray):
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

        Returns
        -------
        np.ndarray
            Filtered list of candidates.
        """

        mask = np.isin(candidates, face_list)
        return candidates[~mask]


    def get_partition_from_face_list(self, mb_idx: int, face_list: np.ndarray):
        """
        Returns the partitioning of the membrane into patches based on a list of faces.

        Parameters
        ----------
        mb_idx : int
            Index of the membrane to be sampled.
        face_list : np.ndarray
            List of faces to be used for partitioning.

        Returns
        -------
        np.ndarray
            Indices of the patches.
        """
        mb = self.membranes[mb_idx]
        faces = self.faces[mb_idx]
        labels = self.labels[mb_idx]
        vert_normals = self.vert_normals[mb_idx]

        face_verts = faces[face_list]
        face_verts = face_verts.reshape(-1)
        face_verts = np.unique(face_verts)

        mb_verts = mb[face_verts]
        mb_faces = faces[face_list]
        mb_labels = labels[face_verts]
        mb_normals = vert_normals[face_verts]

        # Find close GT positions
        gt_pos = self.gt_pos[mb_idx]
        nn_dists, _ = compute_nearest_distances(gt_pos, mb_verts[:, :3])
        gt_mask = nn_dists < 5. / self.max_tomo_shape
        part_gts = gt_pos[gt_mask]


        # Create a mapping from old vertex indices to new indices
        unique_vertex_indices = np.unique(face_verts)
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Update the faces to refer to the new vertex indices
        mb_faces_updated = np.vectorize(index_mapping.get)(mb_faces)
        # return mb, mb_labels, mb_faces

        # return mb, labels, faces
        return mb_verts, mb_labels, mb_faces_updated, mb_normals, part_gts


    def find_adjacent_faces_v2(self, mb_idx: int, start_face: int):
        faces = self.faces[mb_idx]

        cur_faces = set([start_face])
        while len(cur_faces) < self.load_only_sampled_points:
            new_faces = set()
            for face in cur_faces:
                add_faces = set(find_faces_sharing_vertices(faces, faces[face]))
                new_faces.update(add_faces)
            cur_faces.update(new_faces)
            if len(cur_faces) >= self.load_only_sampled_points:
                break

        return np.array(list(cur_faces))
    

    def get_adjacent_triangles(self, faces, edge_to_face_map, triangle_index):
        triangle = faces[triangle_index]
        adjacent_triangles = set()

        for i in range(3):
            edge = tuple(sorted((triangle[i], triangle[(i+1) % 3])))
            for face_index in edge_to_face_map.get(edge, []):
                if face_index != triangle_index:
                    adjacent_triangles.add(face_index)

        return list(adjacent_triangles)

    def find_adjacent_faces(self, mb_idx: int, start_face: int):
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
        faces = self.faces[mb_idx]
        
        cur_faces = [start_face]
        while len(cur_faces) < self.load_only_sampled_points:
            prev_len = len(cur_faces)
            for face in cur_faces:
                
                # add_faces = find_faces_sharing_vertices(faces, faces[face])
                add_faces = self.get_adjacent_triangles(faces, self.edge_to_face_map, face)
                cur_faces.extend(add_faces)
                cur_faces = list(np.unique(cur_faces))
                
                if len(cur_faces) >= self.load_only_sampled_points:
                    break
            new_len = len(cur_faces)
            if new_len == prev_len:
                print("No new faces added, breaking.")
                print("No new faces added, breaking.")
                print("No new faces added, breaking.")
                print("No new faces added, breaking.")
                break
        cur_faces = np.unique(cur_faces)
        return cur_faces


    def __len__(self) -> int:
        """
        Returns the number of image-label pairs in the dataset.

        Returns
        -------
        int
            The number of image-label pairs in the dataset.
        """
        if self.load_only_sampled_points is not None:
            return len(self.part_verts)
            # return len(self.membranes) * (self.membranes[0].shape[0] // self.load_only_sampled_points)
        return len(self.membranes)

    def load_data(self) -> None:
        """
        Loads data-label pairs into memory from the specified directories.
        """
        print("Loading  membranes into dataset.")
        self.membranes = []
        self.labels = []
        self.faces = []
        self.vert_normals = []
        self.gt_pos = []

        for entry in self.data_paths:
            points = np.array(get_csv_data(entry[0]), dtype=float)
            if os.path.isfile(entry[1]):
                gt_pos = np.array(get_csv_data(entry[1]), dtype=float)
            else:
                gt_pos = np.zeros((1, 3))
            faces = np.array(get_csv_data(entry[2]), dtype=int)
            vert_normals = np.array(get_csv_data(entry[3]), dtype=float)

            distances, nn_idcs = compute_nearest_distances(points[:, :3], gt_pos)

            # Move GT along nearest normal
            _, nn_idcs_psii = compute_nearest_distances(gt_pos, points[:, :3])
            psii_normals = vert_normals[nn_idcs_psii]

            nearest_PSII_pos = gt_pos[nn_idcs] + psii_normals[nn_idcs]*20
            connection_vectors = nearest_PSII_pos - (points[:, :3])

            angle_to_normal = np.einsum("ij,ij->i", connection_vectors, vert_normals)
            mask = angle_to_normal < 0
            distances[mask] = 10

            distances[distances > 10] = 10
            points[:, :3] /= self.max_tomo_shape
            gt_pos /= self.max_tomo_shape

            self.membranes.append(points)
            self.labels.append(distances)
            self.faces.append(faces)
            self.vert_normals.append(vert_normals)
            self.gt_pos.append(gt_pos)
            if self.overfit:
                break
        
    def initialize_csv_paths(self) -> None:
        """
        Initializes the list of paths to data-label pairs.
        """


        self.data_paths = []
        for filename in os.listdir(self.csv_folder):
            # if not filename.startswith("T17S1M13") and not filename.startswith("T17S1M11"):# and not filename.startswith("T17S2M5") and not filename.startswith("T17S1M7") and not filename.startswith("T17S1M10"):
            #     continue
            if filename.endswith("data.csv"):
                self.data_paths.append(filename)

        self.data_paths.sort()
        self.data_paths = [
            (os.path.join(self.csv_folder, filename), 
             os.path.join(self.csv_folder, filename[:-14] + "_psii_pos.csv"),
             os.path.join(self.csv_folder, filename[:-14] + "_mesh_faces.csv"),
             os.path.join(self.csv_folder, filename[:-14] + "_mesh_normals.csv"),)
            for filename in self.data_paths
        ]
        if self.train:
            self.data_paths = self.data_paths[:int(len(self.data_paths) * self.train_pct)]
        else:
            self.data_paths = self.data_paths[int(len(self.data_paths) * self.train_pct):]

    def test_loading(self, out_dir, idx: int) -> None:
        """
        Tests the loading of a data-label pair.

        Parameters
        ----------
        idx : int
            Index of the sample to be loaded.
        """
        idx_dict = self.__getitem__(idx)
        points = idx_dict["membrane"][0]
        labels = idx_dict["label"][0]
        faces = idx_dict["faces"][0]

        print(points)
        print("Points shape:", points.shape)
        print(faces)
        print(faces.shape)
        # store_point_and_vectors_in_vtp(
        #     out_path=os.path.join(out_dir, "test%d.vtp" % idx),
        #     in_points=points[:, :3],
        #     in_scalars=[points[:, 4], labels]
        # )
        create_and_store_mesh(faces, points[:, :3], os.path.join(out_dir, "test%d_mesh.vtp" % idx))


def create_and_store_mesh(mesh_faces, mesh_verts, out_path):
    """
    Creates a mesh from the specified faces and vertices and stores it in the specified path.

    Parameters
    ----------
    mesh_faces : np.ndarray
        Array of faces in the mesh, shape (N, 3).
    mesh_verts : np.ndarray
        Array of vertices in the mesh, shape (M, 3).
    out_path : str
        Path to store the mesh in.
    """
    # Use pyvista to create a mesh from the faces and vertices
    print("HI")

    # from membrain_pick.mesh_class import Mesh

    # mesh = Mesh(mesh_verts, mesh_faces)
    # mesh.store_in_file(out_path)
    mesh_faces = np.concatenate((np.ones((mesh_faces.shape[0], 1), dtype=int)*3, mesh_faces), axis=1)
    mesh = pv.PolyData(mesh_verts, mesh_faces)
    # print("s")
    mesh.save(out_path)