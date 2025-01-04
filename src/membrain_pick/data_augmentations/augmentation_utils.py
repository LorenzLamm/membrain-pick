""" augmentation functions that are used in point_cloud_augmentations.py """

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple


def gaussian_weight(distance: float, sigma_squared: float) -> float:
    """
    Compute Gaussian weight using precomputed sigma squared.

    Parameters
    ----------
    distance : float
        The distance value.
    sigma_squared : float
        The precomputed sigma squared value.

    Returns
    -------
    float
        The computed Gaussian weight.
    """
    return np.exp(-(distance**2) / sigma_squared)


def apply_gaussian_filter(
    point_cloud: np.ndarray,
    features: np.ndarray,
    tree: cKDTree,
    radius: float = 0.1,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Optimized feature smoothing for a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud data.
    features : np.ndarray
        The features associated with the point cloud.
    tree : cKDTree
        The KD-tree for spatial queries.
    radius : float, optional
        The radius within which to search for neighbors, by default 0.1.
    sigma : float, optional
        The standard deviation for the Gaussian filter, by default 1.0.

    Returns
    -------
    np.ndarray
        The smoothed features.
    """
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


def apply_median_filter(
    point_cloud: np.ndarray, features: np.ndarray, tree: cKDTree, radius: float = 0.1
) -> np.ndarray:
    """
    Apply a median filter to the features of a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud data.
    features : np.ndarray
        The features associated with the point cloud.
    tree : cKDTree
        The KD-tree for spatial queries.
    radius : float, optional
        The radius within which to search for neighbors, by default 0.1.

    Returns
    -------
    np.ndarray
        The filtered features.
    """
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


def scale_point_cloud_features(
    features: np.ndarray, scale_factor: float, preserve_range: bool = False
) -> np.ndarray:
    """
    Apply a random contrast transformation to a feature of a point cloud.

    Parameters
    ----------
    features : np.ndarray
        N-length NumPy array of feature values for the point cloud.
    scale_factor : float
        The scaling factor for the contrast adjustment.
    preserve_range : bool, optional
        Boolean indicating whether to preserve the original feature value range, by default False.

    Returns
    -------
    np.ndarray
        The modified features.
    """

    # Apply the contrast transformation
    mean = np.mean(features)
    if preserve_range:
        minval, maxval = features.min(), features.max()
    features = mean + scale_factor * (features - mean)
    if preserve_range:
        features = np.clip(features, minval, maxval)

    return features


def scale_point_cloud_stats_inversion(features, gamma):
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
    adjusted_features = ((features - minval) / (maxval - minval)) ** gamma * (
        maxval - minval
    ) + minval

    # Preserve original mean and standard deviation, then invert
    adjusted_mean = adjusted_features.mean()
    adjusted_std = adjusted_features.std()
    adjusted_features = (
        adjusted_features - adjusted_mean
    ) / adjusted_std * original_std + original_mean
    adjusted_features *= -1  # Inversion

    return adjusted_features


def random_erase(
    point_cloud: np.ndarray,
    features: np.ndarray,
    tree: cKDTree,
    patch_radii: List[float] = [1],
    num_patches: int = 1,
) -> np.ndarray:
    """
    Randomly erases points from a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud data.
    features : np.ndarray
        The features associated with the point cloud.
    tree : cKDTree
        The k-d tree for efficient neighbor search.
    patch_radii : List[float], optional
        The radii of the patches to erase, by default [1].
    num_patches : int, optional
        The number of patches to erase, by default 1.

    Returns
    -------
    np.ndarray
        The features with the points in the erased patches set to zero.
    """
    erased_features = features.copy()
    for patch_nr in range(num_patches):
        center_idx = np.random.randint(point_cloud.shape[0])
        center_point = point_cloud[center_idx]
        indices = tree.query_ball_point(center_point, r=patch_radii[patch_nr])
        if not indices:
            continue
        erased_features[indices] = 0
    return erased_features


def apply_gradient_to_pointcloud(
    point_cloud: np.ndarray,
    features: np.ndarray,
    scale: float,
    loc: Tuple[float, float] = (-1, 2),
    strength: float = 1.0,
    mean_centered: bool = True,
) -> np.ndarray:
    """
    Apply a gradient to the features of a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud data.
    features : np.ndarray
        The features associated with the point cloud.
    scale : float
        The scale factor for the gradient computation.
    loc : Tuple[float, float], optional
        The location bounds for the reference point selection, by default (-1, 2).
    strength : float, optional
        The strength of the gradient, by default 1.0.
    mean_centered : bool, optional
        Whether to mean center the gradient values, by default True.

    Returns
    -------
    np.ndarray
        The modified features with the applied gradient.
    """
    # Select a reference point for the "gradient center"
    # This example randomly selects a point within the bounding box defined by loc
    bounds_min, bounds_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    reference_point = np.array(
        [
            np.random.uniform(
                bounds_min[i] + loc[0] * (bounds_max[i] - bounds_min[i]),
                bounds_min[i] + loc[1] * (bounds_max[i] - bounds_min[i]),
            )
            for i in range(3)
        ]
    )

    # Calculate distances from each point to the reference point
    distances = np.linalg.norm(point_cloud - reference_point, axis=1)

    # Determine the gradient scale based on distances using a Gaussian-like formula
    # Adapt the scale and max_strength dynamically if needed
    gradient_values = np.exp(-0.5 * (distances / scale) ** 2)

    if mean_centered:
        gradient_values -= gradient_values.mean()

    max_gradient_val = max(np.max(np.abs(gradient_values)), 1e-8)
    gradient_values = gradient_values / max_gradient_val * strength

    # Apply the gradient to the feature
    modified_features = features + np.expand_dims(gradient_values, 1)

    return modified_features


def apply_local_gamma_to_pointcloud(
    point_cloud: np.ndarray,
    features: np.ndarray,
    scale: float = 1.0,
    loc: Tuple[float, float] = (-0.3, 1.3),
    gamma_range: Tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """
    Apply a local gamma correction to the features of a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud data.
    features : np.ndarray
        The features associated with the point cloud.
    scale : float, optional
        The scale factor for the Gaussian distribution, by default 1.0.
    loc : Tuple[float, float], optional
        The location bounds for the reference point selection, by default (-0.3, 1.3).
    gamma_range : Tuple[float, float], optional
        The range of gamma values, by default (0.5, 2.0).

    Returns
    -------
    np.ndarray
        The modified features with the applied gamma correction.
    """
    # Determine the bounds for the reference point selection
    bounds_min, bounds_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    reference_point = np.array(
        [
            np.random.uniform(
                bounds_min[i] + loc[0] * (bounds_max[i] - bounds_min[i]),
                bounds_min[i] + loc[1] * (bounds_max[i] - bounds_min[i]),
            )
            for i in range(3)
        ]
    )

    # Calculate distances from each point to the reference point
    distances = np.linalg.norm(point_cloud - reference_point, axis=1)
    weights = np.exp(-0.5 * (distances / scale) ** 2)

    # Generate a spatially varying gamma value based on distances
    # Simulate a Gaussian distribution for gamma values centered around the reference point
    # Note: This is a conceptual adaptation; adjust the distribution as needed
    normalized_distances = (distances - distances.min()) / (
        distances.max() - distances.min()
    )
    gamma_values = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * np.exp(
        -0.5 * (normalized_distances / scale) ** 2
    )

    # Apply gamma correction to each point's feature
    mn, mx = features.min(), features.max()
    normalized_features = (features - mn) / (mx - mn + 1e-10)
    modified_features = np.zeros_like(features)

    for i, gamma in enumerate(gamma_values):
        modified_features[i] = np.power(normalized_features[i], gamma)

    # Rescale modified features back to original range
    modified_features = modified_features * (mx - mn) + mn

    # interpolate based on weights
    modified_features = modified_features * np.expand_dims(
        weights, 1
    ) + features * np.expand_dims(1 - weights, 1)

    return modified_features
