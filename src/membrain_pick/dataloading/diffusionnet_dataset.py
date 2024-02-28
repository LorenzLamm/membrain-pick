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
from scipy.spatial import KDTree
from collections import defaultdict
import hashlib
from membrain_pick.dataloading.pointcloud_augmentations import get_test_transforms, get_training_transforms






################# Precomputation section ########################

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












######################################################
















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


def project_points_to_nearest_hyperplane(points, candiate_points):
    print("Projecting points to nearest hyperplane.")
    # Find nearest three points for each point
    tree = KDTree(candiate_points)
    _, nn_idcs = tree.query(points, k=3)
    P1 = candiate_points[nn_idcs[:, 0]]
    P2 = candiate_points[nn_idcs[:, 1]]
    P3 = candiate_points[nn_idcs[:, 2]]

    for i, point in enumerate(points):
        P1 = candiate_points[nn_idcs[i, 0]]
        P2 = candiate_points[nn_idcs[i, 1]]
        P3 = candiate_points[nn_idcs[i, 2]]
        projection = project_point_to_hyperplane(point, P1, P2, P3)
        points[i] = projection
    return points


def project_point_to_hyperplane(P, P1, P2, P3):
    # Calculate vectors P1P2 and P1P3
    P1P2 = P2 - P1
    P1P3 = P3 - P1

    # Calculate the normal vector (A, B, C) by taking the cross product of P1P2 and P1P3
    normal_vector = np.cross(P1P2, P1P3)

    # Calculate D using one of the points, say P1
    D = -np.dot(normal_vector, P1)

    # Calculate the projection of P onto the hyperplane
    # Using the formula: P_proj = P - ((A*Px + B*Py + C*Pz + D) / (A^2 + B^2 + C^2)) * normal_vector
    numerator = np.dot(normal_vector, P) + D
    denominator = np.dot(normal_vector, normal_vector)
    projection = P - (numerator / denominator) * normal_vector

    normal_vector, D, projection
    return projection



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
        cache_dir: str = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/diffusion_net_training/mesh_cache2",
        force_recompute: bool = False,
        overfit_mb: bool = False,
        augment_noise: bool = False,
        gaussian_smoothing: bool = False,
        random_erase: bool = False,
        median_filter: bool = False,
        brightness_transform: bool = False,
        brightness_gradient: bool = False,
        local_brightness_gamma: bool = False,
        normalize_features: bool = False,
        contrast: bool = False,
        contrast_with_stats_inversion: bool = False,

    
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
        self.augment_noise = augment_noise
        self.gaussian_smoothing = gaussian_smoothing
        self.median_filter = median_filter
        self.random_erase = random_erase
        self.brighness_transform = brightness_transform
        self.brightness_gradient = brightness_gradient
        self.local_brightness_gamma = local_brightness_gamma
        self.normalize_features = normalize_features
        self.contrast = contrast
        self.contrast_with_stats_inversion = contrast_with_stats_inversion

        self.initialize_csv_paths()
        self.load_data()

        if self.load_only_sampled_points is not None:
            self._precompute_partitioning()

        self.transforms = (
            get_training_transforms(self.max_tomo_shape) if self.train else get_test_transforms()
        )
        if self.train:
            self.kdtrees = [KDTree(mb[:, :3]) for mb in (self.membranes if self.load_only_sampled_points is None else self.part_verts)]
        else:
            self.kdtrees = [None] * len(self)
        

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
                "membrane": self.part_verts[idx].copy(),
                "label": self.part_labels[idx],
                "faces": self.part_faces[idx],
                "normals": self.part_normals[idx],
                "mb_idx": self.part_mb_idx[idx],
                "gt_pos": self.part_gt_pos[idx],
                "vert_weights": self.part_vert_weights[idx]
            }
            
            # mb_idx = idx // (self.membranes[0].shape[0] // self.load_only_sampled_points)
            # idx_dict = self.select_random_mb_area(mb_idx, idx, self.load_only_sampled_points)
        else:
            idx_dict = {
                "membrane": self.membranes[idx].copy(),
                "label": self.labels[idx],
                "faces": self.faces[idx],
                "normals": self.vert_normals[idx],
                "mb_idx": idx,
                "gt_pos": self.gt_pos[idx],
                "vert_weights": np.ones(self.membranes[idx].shape[0])
            }
        idx_dict = self.transforms(idx_dict, keys=["membrane"], mb_tree=self.kdtrees[idx])

        for key in idx_dict:
            if key in ["membrane", "label", "faces", "normals", "vert_weights"]:
                idx_dict[key] = np.expand_dims(idx_dict[key], 0)
        # idx_dict = self._augment_sample(idx_dict, idx)

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
        self.part_vert_weights = []

        
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
                self.part_vert_weights.append(cache[f"part_vert_weights"])
                print("Appending data from", cur_cache_path, "to partitioning.")
                cur_cache_path = cache_path[:-4] + "_partnr" + str(cur_cache_count) + ".npz"
                cur_cache_count += 1
                cache_found = True
                if self.overfit and cur_cache_count > 3:
                    break

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
                    adj_faces, adj_faces_weights = self.find_adjacent_faces(mb_idx, face_start)
                    part_verts, part_labels, part_faces, part_normals, part_gts, part_vert_weights = self.get_partition_from_face_list(mb_idx, adj_faces, adj_faces_weights)
                    self.part_verts.append(part_verts)
                    self.part_labels.append(part_labels)
                    self.part_faces.append(part_faces)
                    self.part_normals.append(part_normals)
                    self.part_mb_idx.append(mb_idx)
                    self.part_gt_pos.append(part_gts) 
                    self.part_vert_weights.append(part_vert_weights)
                    face_candidates = exclude_faces_from_candidates(adj_faces, face_candidates, adj_faces_weights)
                    if self.overfit and part_counter > 2:
                        break
                    print("Saving partitioning for membrane", mb_idx, "to cache.", "with", len(self.part_verts), "patches.")
                    print("Cache file:", cur_cache_path)
                    np.savez(cur_cache_path,
                    **{f"part_verts": part_verts,
                        f"part_labels": part_labels,
                        f"part_faces": part_faces,
                        f"part_normals": part_normals,
                        f"part_vert_weights": part_vert_weights,
                        f"part_gt_pos": part_gts})
                    part_counter += 1
            if self.overfit_mb:
                break


    # def exclude_faces_from_candidates(self, face_list: np.ndarray, candidates: np.ndarray, faces_weights: np.ndarray):
    #     """
    #     Excludes faces from a list of candidates.

    #     Parameters
    #     ----------
    #     mb_idx : int
    #         Index of the membrane to be sampled.
    #     face_list : np.ndarray
    #         List of faces to be excluded.
    #     candidates : np.ndarray
    #         List of candidates to be filtered.
    #     faces_weights : np.ndarray
    #         List of weights for the faces.

    #     Returns
    #     -------
    #     np.ndarray
    #         Filtered list of candidates.
    #     """

    #     mask = np.isin(candidates, np.array(face_list)[faces_weights == 1.0])
    #     return candidates[~mask]


    def get_partition_from_face_list(self, mb_idx: int, face_list: np.ndarray, face_weight_list: np.ndarray):
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
        mb = self.membranes[mb_idx]
        faces = self.faces[mb_idx]
        labels = self.labels[mb_idx]
        vert_normals = self.vert_normals[mb_idx]

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
        gt_pos = self.gt_pos[mb_idx]
        nn_dists, _ = compute_nearest_distances(gt_pos, mb_verts[:, :3])
        gt_mask = nn_dists < 3. / self.max_tomo_shape
        part_gts = gt_pos[gt_mask]


        # Create a mapping from old vertex indices to new indices
        unique_vertex_indices = np.unique(face_verts)
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Update the faces to refer to the new vertex indices
        mb_faces_updated = np.vectorize(index_mapping.get)(mb_faces)
        # return mb, mb_labels, mb_faces

        # return mb, labels, faces
        return mb_verts, mb_labels, mb_faces_updated, mb_normals, part_gts, mb_vert_weights
    
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
        back_1_weight = 0.0
        back_2_weight = 0.125
        back_3_weight = 0.250

        faces = self.faces[mb_idx]
        
        cur_faces = [start_face]
        cur_faces_weights = {start_face: back_1_weight}

        faces_back_1 = []
        faces_back_2 = []
        faces_back_3 = []
        while len(cur_faces) < self.load_only_sampled_points:
            prev_len = len(cur_faces)
            for face in cur_faces:
                
                # add_faces = find_faces_sharing_vertices(faces, faces[face])
                add_faces = self.get_adjacent_triangles(faces, self.edge_to_face_map, face)
                cur_faces.extend(add_faces)
                cur_faces = list(np.unique(cur_faces))
                
                if len(cur_faces) >= self.load_only_sampled_points:
                    break

            back_1_mask = np.isin(cur_faces, faces_back_1)
            back_2_mask = np.isin(cur_faces, faces_back_2)
            back_3_mask = np.isin(cur_faces, faces_back_3)

            cur_faces = np.array(cur_faces)
            cur_faces_weights = {face: back_1_weight for face in cur_faces}
            cur_faces_weights.update({face: back_2_weight for face in cur_faces[back_1_mask]})
            cur_faces_weights.update({face: back_3_weight for face in cur_faces[back_2_mask]})
            cur_faces_weights.update({face: 1.0 for face in cur_faces[back_3_mask]})
            cur_faces = list(cur_faces)

            faces_back_3 = faces_back_3 + faces_back_2
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
                gt_pos = project_points_to_nearest_hyperplane(gt_pos, points[:, :3])
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

            distances[distances > 10] = 10
            distances[mask] = 10.05
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
                if "T17S2M5" in filename: # corrupt file
                    continue
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


    def test_loading(self, out_dir, idx: int, times=1) -> None:
        """
        Tests the loading of a data-label pair.

        Parameters
        ----------
        idx : int
            Index of the sample to be loaded.
        """
        from membrain_pick.optimization.plane_projection import make_2D_projection_scatter_plot
        for k in range(times):
            idx_dict = self.__getitem__(idx)
            make_2D_projection_scatter_plot(
                out_file=os.path.join(out_dir, "test%d_%d.png" % (idx, k)),
                point_cloud=idx_dict["membrane"][0, :, :3],
                color=idx_dict["membrane"][0, :, 7],
                s=150
            )