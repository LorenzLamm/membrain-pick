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



def compute_nearest_distances(point_data, PSII_pos):
    kd_tree = cKDTree(PSII_pos)
    distances, nn_idcs = kd_tree.query(point_data, k=1)
    return distances, nn_idcs


def get_array_hash(array):
    array_bytes = array.tobytes()
    hasher = hashlib.sha256()
    hasher.update(array_bytes)
    return hasher.hexdigest()

def brightness_transform(features, add_const):
    """Apply a brightness transform to the features of a point cloud."""
    return features + add_const


def apply_gradient_to_pointcloud(point_cloud, features, scale, loc=(-1, 2), strength=1.0, mean_centered=True):
    # Select a reference point for the "gradient center"
    # This example randomly selects a point within the bounding box defined by loc
    bounds_min, bounds_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    reference_point = np.array([np.random.uniform(bounds_min[i] + loc[0] * (bounds_max[i] - bounds_min[i]), 
                                                  bounds_min[i] + loc[1] * (bounds_max[i] - bounds_min[i])) for i in range(3)])
    
    # Calculate distances from each point to the reference point
    distances = np.linalg.norm(point_cloud - reference_point, axis=1)
    
    # Determine the gradient scale based on distances using a Gaussian-like formula
    # Adapt the scale and max_strength dynamically if needed
    gradient_values = np.exp(-0.5 * (distances / scale)**2)
    
    if mean_centered:
        gradient_values -= gradient_values.mean()
    
    max_gradient_val = max(np.max(np.abs(gradient_values)), 1e-8)
    gradient_values = gradient_values / max_gradient_val * strength
    
    # Apply the gradient to the feature
    modified_features = features + np.expand_dims(gradient_values, 1)
    
    return modified_features


def apply_random_contrast_to_pointcloud(features, contrast_factor, preserve_range=False):
    """
    Apply a random contrast transformation to a feature of a point cloud.

    :param features: N-length NumPy array of feature values for the point cloud.
    :param contrast_range: Tuple (min, max) specifying the range for the random contrast scaling factor.
    :param preserve_range: Boolean indicating whether to preserve the original feature value range.
    :param prob: Probability with which to apply the contrast adjustment.
    :return: The modified features.
    """

    # Apply the contrast transformation
    mean = np.mean(features)
    if preserve_range:
        minval, maxval = features.min(), features.max()
    features = mean + contrast_factor * (features - mean)
    if preserve_range:
        features = np.clip(features, minval, maxval)

    return features


def adjust_contrast_point_cloud(features, gamma):
    """
    Adjust the gamma of point cloud features using gamma correction.

    Args:
        features (np.ndarray): The feature values of the point cloud.
        gamma (float): The gamma value for contrast adjustment.

    Returns:
        np.ndarray: The features after contrast adjustment.
    """
    minval, maxval = features.min(), features.max()
    features = ((features - minval) / (maxval - minval)) ** gamma * (maxval - minval) + minval
    return features


def adjust_contrast_inversion_stats_point_cloud(features, gamma):
    """
    Adjust the contrast of point cloud features with inversion and stats preservation.

    Args:
        features (np.ndarray): The feature values of the point cloud.
        gamma (float): The gamma value for contrast adjustment.

    Returns:
        np.ndarray: The features after adjustment, inversion, and stats preservation.
    """
    original_mean = features.mean()
    original_std = features.std()
    
    # Adjust contrast
    minval, maxval = features.min(), features.max()
    adjusted_features = ((features - minval) / (maxval - minval)) ** gamma * (maxval - minval) + minval
    
    # Preserve original mean and standard deviation, then invert
    adjusted_mean = adjusted_features.mean()
    adjusted_std = adjusted_features.std()
    adjusted_features = (adjusted_features - adjusted_mean) / adjusted_std * original_std + original_mean
    adjusted_features *= -1  # Inversion
    
    return adjusted_features


def apply_local_gamma_to_pointcloud(point_cloud, features, scale=1.0, loc=(-.5, 1.5), gamma_range=(0.7, 1.4)):
    # Determine the bounds for the reference point selection
    bounds_min, bounds_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    reference_point = np.array([np.random.uniform(bounds_min[i] + loc[0] * (bounds_max[i] - bounds_min[i]), 
                                                  bounds_min[i] + loc[1] * (bounds_max[i] - bounds_min[i])) for i in range(3)])
    
    # Calculate distances from each point to the reference point
    distances = np.linalg.norm(point_cloud - reference_point, axis=1)
    weights = np.exp(-0.5 * (distances / scale)**2)
    
    # Generate a spatially varying gamma value based on distances
    # Simulate a Gaussian distribution for gamma values centered around the reference point
    # Note: This is a conceptual adaptation; adjust the distribution as needed
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    gamma_values = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * np.exp(-0.5 * (normalized_distances / scale)**2)
    
    # Apply gamma correction to each point's feature
    mn, mx = features.min(), features.max()
    normalized_features = (features - mn) / (mx - mn)
    modified_features = np.zeros_like(features)
    
    for i, gamma in enumerate(gamma_values):
        modified_features[i] = np.power(normalized_features[i], gamma)
    
    # Rescale modified features back to original range
    modified_features = modified_features * (mx - mn) + mn
    
    # interpolate based on weights
    modified_features = modified_features * np.expand_dims(weights, 1) + features * np.expand_dims(1 - weights, 1)
    
    return modified_features


def gaussian_weight(distance, sigma_squared):
    """Compute Gaussian weight using precomputed sigma squared."""
    return np.exp(-distance**2 / sigma_squared)

def smooth_feature_optimized(point_cloud, features, tree, radius=0.1, sigma=1.0):
    """Optimized feature smoothing for a point cloud."""
    sigma_squared = 2 * sigma**2
    smoothed_features = np.zeros_like(features)
    for i, point in enumerate(point_cloud):
        indices = tree.query_ball_point(point, r=radius)
        if not indices:
            continue
        
        distances = np.linalg.norm(point_cloud[indices] - point, axis=1)
        weights = gaussian_weight(distances, sigma_squared)
        weighted_features = features[indices] * np.expand_dims(weights, axis=1)
        smoothed_features[i] = np.sum(weighted_features, axis=0) / np.sum(weights)
    return smoothed_features

def apply_median_filter(point_cloud, features, tree, radius=0.1):
    """Apply a median filter to the features of a point cloud."""
    filtered_features = np.zeros_like(features)
    for i, point in enumerate(point_cloud):
        # Find indices of neighbors within the specified radius
        indices = tree.query_ball_point(point, r=radius)
        if not indices:
            continue
        
        # Extract the features of these neighbors
        neighbor_features = features[indices]
        
        # Calculate the median of the features
        filtered_features[i] = np.median(neighbor_features)
    
    return filtered_features


def random_erase(point_cloud, features, tree, patch_radius=0.1, num_patches=1):
    """Randomly erases points from a point cloud."""
    erased_features = features.copy()
    for _ in range(num_patches):
        center_idx = np.random.randint(point_cloud.shape[0])
        center_point = point_cloud[center_idx]
        indices = tree.query_ball_point(center_point, r=patch_radius)
        if not indices:
            continue
        erased_features[indices] = 0
    return erased_features


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

        self.patch_dicts = [None] * len(self)
        self._define_augment_params()
        
    def _define_augment_params(self):
        """
        Defines the parameters for augmentation.
        """
        # Gaussian noise
        self.noise_probability = .6
        self.noise_range = (0.0, 0.25)

        # Gaussian smoothing
        self.smoothing_probability = .6
        self.kdtrees = [KDTree(mb[:, :3]) for mb in (self.membranes if self.load_only_sampled_points is None else self.part_verts)]
        self.smoothing_radius_max = 4. / self.max_tomo_shape
        self.smoothing_sigma_max = 2.5 / self.max_tomo_shape
        self.smoothing_radius_min = 1. / self.max_tomo_shape
        self.smoothing_sigma_min = 0.

        # Median filter
        self.median_filter_probability = .6
        self.median_filter_radius_max = 2. / self.max_tomo_shape

        # Random erasing
        self.erase_probability = .6
        self.erase_radius_max = 5.5 / self.max_tomo_shape
        self.erase_radius_min = 1.5 / self.max_tomo_shape
        self.erase_patches_max = 5

        # Brightness transform
        self.brighness_probability = .6
        self.brightness_range = (-0.5, 0.5)

        # Brightness gradient
        self.brightness_gradient_probability = .6
        self.max_brightness_gradient_strength = .8
        self.brightness_gradient_scale_max = 200. / self.max_tomo_shape

        # Local brightness gamma
        self.local_brightness_gamma_probability = .6
        self.local_brightness_gamma_scale_max = 30. / self.max_tomo_shape

        # Contrast
        self.contrast_probability = .5
        self.contrast_range = (0.5, 2.)

        # Contrast with stats inversion
        self.contrast_with_stats_inversion_probability = .5







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
                "gt_pos": self.part_gt_pos[idx],
                "vert_weights": np.expand_dims(self.part_vert_weights[idx], 0)
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
                "gt_pos": self.gt_pos[idx],
                "vert_weights": np.expand_dims(np.ones(self.membranes[idx].shape[0]), 0)
            }
        
        idx_dict = self._augment_sample(idx_dict, idx)

        return idx_dict
    
    def _augment_sample(self, idx_dict: Dict[str, np.ndarray], idx: int) -> Dict[str, np.ndarray]:
        """
        Augments a sample by adding noise to the membrane.

        Parameters
        ----------
        idx_dict : Dict[str, np.ndarray]
            Dictionary containing the membrane to be augmented.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the augmented membrane.
        """
        global normalize_features_count, augment_noise_count, gaussian_smoothing_count, median_filter_count, random_erase_count, brightness_transform_count, brightness_gradient_count, local_brightness_gamma_count, contrast_count, contrast_with_stats_inversion_count
        if self.normalize_features:
            idx_dict["membrane"][:, :, 3:] = (idx_dict["membrane"][:, :, 3:] - idx_dict["membrane"][:, :, 3:].mean()) / idx_dict["membrane"][:, :, 3:].std()

        if np.random.rand() < (0.5 if self.gaussian_smoothing and self.median_filter else 1.0 if self.gaussian_smoothing else 0.0):
            if self.gaussian_smoothing and np.random.rand() < self.smoothing_probability:
                idx_dict["membrane"] = self._augment_gaussian_smoothing(idx_dict["membrane"].copy(), 
                                                                        self.kdtrees[idx])
        else:
            if self.median_filter and self.median_filter_probability and np.random.rand() < self.median_filter_probability:
                idx_dict["membrane"] = self._augment_median_filter(idx_dict["membrane"].copy(), 
                                                                self.kdtrees[idx])

        if self.augment_noise and np.random.rand() < self.noise_probability:
            idx_dict["membrane"] = self._augment_noise(idx_dict["membrane"].copy())

        if self.brighness_transform and np.random.rand() < self.brighness_probability:
            idx_dict["membrane"] = self._augment_brightness(idx_dict["membrane"].copy())

        if self.contrast and np.random.rand() < self.contrast_probability:
            idx_dict["membrane"] = self._augment_contrast(idx_dict["membrane"].copy())

        if self.contrast_with_stats_inversion and np.random.rand() < self.contrast_with_stats_inversion_probability:
            idx_dict["membrane"] = self._augment_contrast_with_stats_inversion(idx_dict["membrane"].copy())

        if self.random_erase and np.random.rand() < self.erase_probability:
            idx_dict["membrane"] = self._augment_random_erase(idx_dict["membrane"].copy(), 
                                                              self.kdtrees[idx])

        if self.brightness_gradient and np.random.rand() < self.brightness_gradient_probability:
            idx_dict["membrane"] = self._augment_brightness_gradient(idx_dict["membrane"].copy())

        if self.local_brightness_gamma and np.random.rand() < self.local_brightness_gamma_probability:
            idx_dict["membrane"] = self._augment_local_brightness_gamma(idx_dict["membrane"].copy())

        return idx_dict
    

    def _augment_brightness(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds brightness to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        brightness = np.random.uniform(*self.brightness_range)
        membrane[:, 3:] = brightness_transform(membrane[:, 3:], brightness)
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_brightness_gradient(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds brightness gradient to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        strength = np.random.uniform(-self.max_brightness_gradient_strength, self.max_brightness_gradient_strength)
        brightness_gradient_scale = np.random.uniform(0, self.brightness_gradient_scale_max)
        membrane[:, 3:] = apply_gradient_to_pointcloud(membrane[:, :3], membrane[:, 3:], scale=brightness_gradient_scale, strength=strength)
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_local_brightness_gamma(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds brightness gamma to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        scale = np.random.uniform(0, self.local_brightness_gamma_scale_max)
        membrane[:, 3:] = apply_local_gamma_to_pointcloud(membrane[:, :3], membrane[:, 3:], scale=scale)
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_contrast(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds contrast to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        contrast_factor = np.random.uniform(*self.contrast_range)
        preserve_range = np.random.random() < 0.5
        membrane[:, 3:] = apply_random_contrast_to_pointcloud(membrane[:, 3:], contrast_factor, preserve_range=preserve_range)
        membrane = np.expand_dims(membrane, 0)
        return membrane

    def _augment_contrast_with_stats_inversion(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds contrast to the membrane with stats inversion.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]

        # Apply contrast adjustment with inversion and stats preservation (2x)
        gamma = np.random.uniform(*self.contrast_range)
        membrane[:, 3:] = adjust_contrast_inversion_stats_point_cloud(membrane[:, 3:], gamma)
        gamma = np.random.uniform(*self.contrast_range)
        membrane[:, 3:] = adjust_contrast_inversion_stats_point_cloud(membrane[:, 3:], gamma)
        
        membrane = np.expand_dims(membrane, 0)
        return membrane

    def _augment_noise(self, membrane: np.ndarray) -> np.ndarray:
        """
        Adds noise to the membrane and label.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        noise_std = np.random.uniform(*self.noise_range)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=membrane[:, 3:].shape)
        membrane[:, 3:] += noise
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_gaussian_smoothing(self, membrane: np.ndarray, mb_tree: KDTree) -> np.ndarray:
        """
        Adds Gaussian smoothing to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]

        smoothing_radius = np.random.uniform(self.smoothing_radius_min, self.smoothing_radius_max)
        smoothing_sigma = np.random.uniform(self.smoothing_sigma_min, self.smoothing_sigma_max)
        membrane[:, 3:] = smooth_feature_optimized(membrane[:, :3], 
                                            membrane[:, 3:], 
                                            tree=mb_tree, 
                                            radius=smoothing_radius, 
                                            sigma=smoothing_sigma)
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_median_filter(self, membrane: np.ndarray, mb_tree: KDTree) -> np.ndarray:
        """
        Adds a median filter to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        filter_radius = np.random.uniform(0, self.median_filter_radius_max)
        membrane[:, 3:] = apply_median_filter(membrane[:, :3], 
                                              membrane[:, 3:], 
                                              tree=mb_tree, 
                                              radius=filter_radius)
        membrane = np.expand_dims(membrane, 0)
        return membrane
    
    def _augment_random_erase(self, membrane: np.ndarray, mb_tree: KDTree) -> np.ndarray:
        """
        Adds random erasing to the membrane.

        Parameters
        ----------
        membrane : np.ndarray
            The membrane to be augmented.

        Returns
        -------
        np.ndarray
            The augmented membrane.
        """
        if membrane.shape[0] == 1:
            membrane = membrane[0]
        num_patches = np.random.randint(1, self.erase_patches_max)
        patch_radius = np.random.uniform(self.erase_radius_min, self.erase_radius_max)
        membrane[:, 3:] = random_erase(membrane[:, :3], 
                                       membrane[:, 3:],
                                       mb_tree, 
                                       patch_radius=patch_radius, 
                                       num_patches=num_patches)
        membrane = np.expand_dims(membrane, 0)
        return membrane

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
                    face_candidates = self.exclude_faces_from_candidates(adj_faces, face_candidates, adj_faces_weights)
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



    def exclude_faces_from_candidates(self, face_list: np.ndarray, candidates: np.ndarray, faces_weights: np.ndarray):
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

        for k in range(times):
            idx_dict = self.__getitem__(idx)
            points = idx_dict["membrane"][0]
            labels = idx_dict["label"][0]
            faces = idx_dict["faces"][0]

            # store_point_and_vectors_in_vtp(
            #     out_path=os.path.join(out_dir, "test%d_%d.vtp" % (idx, k)),
            #     in_points=points[:, :3],
            #     in_scalars=[points[:, i] for i in range(4, points.shape[1])] + [labels]
            # )
        # create_and_store_mesh(faces, points[:, :3], os.path.join(out_dir, "test%d_mesh.vtp" % idx))


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