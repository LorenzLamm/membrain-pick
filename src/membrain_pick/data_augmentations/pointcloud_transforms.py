"""
Data augmentation functions for point clouds.

These functions focus on the augmentation of features of point clouds, such as
the color or other scalar values associated with each point. They are not
concerned with the spatial arrangement of the points themselves.

"""

import numpy as np
from membrain_pick.data_augmentations.augmentation_utils import (
    apply_gaussian_filter,
    apply_median_filter,
    scale_point_cloud_features,
    scale_point_cloud_stats_inversion,
    random_erase,
    apply_gradient_to_pointcloud,
    apply_local_gamma_to_pointcloud,
)


# define a base class for point cloud augmentations
class PointCloudAugmentation:
    def __init__(self, has_positions=True, **kwargs):
        self.kwargs = kwargs
        self.has_positions = has_positions

    def __call__(self, point_cloud, *args, **kwargs):
        # assumes point_cloud is a dictionary
        # each key should be an array of shape (N, D)
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.kwargs})"

    def __str__(self):
        return self.__repr__()

    def _randomize():
        pass

    def extract_transform_array(self, point_cloud_dict, key):
        if self.has_positions:
            return point_cloud_dict[key][:, 3:]
        else:
            return point_cloud_dict[key]

    def insert_transform_array(self, point_cloud_dict, key, transform_array):
        if self.has_positions:
            point_cloud_dict[key][:, 3:] = transform_array
        else:
            point_cloud_dict[key] = transform_array
        return point_cloud_dict


class NormalizeFeatures(PointCloudAugmentation):
    def __init__(self, apply_on_separate_channels=False, **kwargs):
        super().__init__(**kwargs)
        self.apply_on_separate_channels = apply_on_separate_channels

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        for key in keys:
            transform_array = self.extract_transform_array(point_cloud_dict, key)
            if self.apply_on_separate_channels:
                for i in range(transform_array.shape[1]):
                    transform_array[:, i] = (
                        transform_array[:, i] - transform_array[:, i].mean()
                    ) / transform_array[:, i].std()
            else:
                transform_array = (
                    transform_array - transform_array.mean()
                ) / transform_array.std()
            point_cloud_dict = self.insert_transform_array(
                point_cloud_dict, key, transform_array
            )
        return point_cloud_dict


# Random Gaussian smoothing
class RandomGaussianSmoothing(PointCloudAugmentation):
    def __init__(
        self,
        apply_prob=0.5,
        smoothing_radius_range=(0.1, 0.5),
        smoothing_sigma_range=(0.1, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smoothing_radius_min, self.smoothing_radius_max = smoothing_radius_range
        self.smoothing_sigma_min, self.smoothing_sigma_max = smoothing_sigma_range
        self.apply_prob = apply_prob
        assert self.has_positions, "RandomGaussianSmoothing requires point positions"
        self._randomize()

    def _randomize(self):
        self.smoothing_radius = np.random.uniform(
            self.smoothing_radius_min, self.smoothing_radius_max
        )
        self.smoothing_sigma = np.random.uniform(
            self.smoothing_sigma_min, self.smoothing_sigma_max
        )

    def __call__(self, point_cloud_dict, keys, mb_tree, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = apply_gaussian_filter(
                    point_cloud_dict[key][:, :3],
                    transform_array,
                    tree=mb_tree,
                    radius=self.smoothing_radius,
                    sigma=self.smoothing_sigma,
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomMedianSmoothing(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, smoothing_radius_range=(0.0, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.smoothing_radius_min, self.smoothing_radius_max = smoothing_radius_range
        self.apply_prob = apply_prob
        assert self.has_positions, "RandomMedianSmoothing requires point positions"
        self._randomize()

    def _randomize(self):
        self.smoothing_radius = np.random.uniform(
            self.smoothing_radius_min, self.smoothing_radius_max
        )

    def __call__(self, point_cloud_dict, keys, mb_tree, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = apply_median_filter(
                    point_cloud_dict[key][:, :3],
                    transform_array,
                    tree=mb_tree,
                    radius=self.smoothing_radius,
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomFeatureDropout(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, dropout_prob_range=(0.0, 0.3), **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob_min, self.dropout_prob_max = dropout_prob_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.dropout_prob = np.random.uniform(
            self.dropout_prob_min, self.dropout_prob_max
        )

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                mask = np.random.rand(*transform_array.shape) > self.dropout_prob
                transform_array = transform_array * mask
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomFeatureNoise(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, noise_std_range=(0.0, 0.3), **kwargs):
        super().__init__(**kwargs)
        self.noise_std_min, self.noise_std_max = noise_std_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                noise = np.random.normal(0, self.noise_std, transform_array.shape)
                transform_array = transform_array + noise
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomFeatureShift(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, shift_range=(-0.5, 0.5), **kwargs):
        super().__init__(**kwargs)
        self.shift_min, self.shift_max = shift_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.shift = np.random.uniform(self.shift_min, self.shift_max)

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = transform_array + self.shift
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomFeatureScale(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, scale_range=(0.5, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.scale_min, self.scale_max = scale_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.scale = np.random.uniform(self.scale_min, self.scale_max)
        self.preserve_range = np.random.rand() < 0.5

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = scale_point_cloud_features(
                    transform_array, self.scale, self.preserve_range
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomFeatureScaleWithStatsInversion(PointCloudAugmentation):
    def __init__(self, apply_prob=0.5, gamma_range=(0.5, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.gamma_min, self.gamma_max = gamma_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.gamma = np.random.uniform(self.gamma_min, self.gamma_max)

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                # apply the contrast transformation twice s.t. result is not inverted
                self._randomize()
                transform_array = scale_point_cloud_stats_inversion(
                    transform_array, self.gamma
                )
                self._randomize()
                transform_array = scale_point_cloud_stats_inversion(
                    transform_array, self.gamma
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomErasing(PointCloudAugmentation):
    def __init__(
        self,
        apply_prob=0.5,
        patch_radius_range=(0.1, 0.5),
        num_patches_range=(1, 5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_radius_min, self.patch_radius_max = patch_radius_range
        self.num_patches_min, self.num_patches_max = num_patches_range
        self.apply_prob = apply_prob
        assert self.has_positions, "RandomErasing requires point positions"
        self._randomize()

    def _randomize(self):
        self.num_patches = np.random.randint(self.num_patches_min, self.num_patches_max)
        self.patch_radii = np.random.uniform(
            self.patch_radius_min, self.patch_radius_max, self.num_patches
        )

    def __call__(self, point_cloud_dict, keys, mb_tree, *args, **kwargs):
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = random_erase(
                    point_cloud_dict[key][:, :3],
                    transform_array,
                    tree=mb_tree,
                    patch_radii=self.patch_radii,
                    num_patches=self.num_patches,
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )
        return point_cloud_dict


class RandomBrightnessGradient(PointCloudAugmentation):
    def __init__(
        self,
        apply_prob=0.5,
        max_brightness_gradient_strength=1.0,
        brightness_gradient_scale_max=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_brightness_gradient_strength = max_brightness_gradient_strength
        self.brightness_gradient_scale_max = brightness_gradient_scale_max
        self.apply_prob = apply_prob
        assert self.has_positions, "RandomBrightnessGradient requires point positions"
        self._randomize()

    def _randomize(self):
        self.brightness_gradient_strength = np.random.uniform(
            -self.max_brightness_gradient_strength,
            self.max_brightness_gradient_strength,
        )
        self.brightness_gradient_scale = np.random.uniform(
            0, self.brightness_gradient_scale_max
        )

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = apply_gradient_to_pointcloud(
                    point_cloud_dict[key][:, :3],
                    transform_array,
                    scale=self.brightness_gradient_scale,
                    strength=self.brightness_gradient_strength,
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class RandomLocalBrightnessGamma(PointCloudAugmentation):
    def __init__(
        self, apply_prob=0.5, local_brightness_gamma_scale_range=(0.5, 2), **kwargs
    ):
        super().__init__(**kwargs)
        self.local_brightness_gamma_scale_min, self.local_brightness_gamma_scale_max = (
            local_brightness_gamma_scale_range
        )
        self.apply_prob = apply_prob
        assert self.has_positions, "RandomLocalBrightnessGamma requires point positions"
        self._randomize()

    def _randomize(self):
        self.local_brightness_gamma_scale = np.random.uniform(
            self.local_brightness_gamma_scale_min, self.local_brightness_gamma_scale_max
        )

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = apply_local_gamma_to_pointcloud(
                    point_cloud_dict[key][:, :3],
                    transform_array,
                    scale=self.local_brightness_gamma_scale,
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


def apply_gaussian_filter_along_features(features, radius, sigma):
    """Apply 1D Gaussian filter to smooth features along the feature dimension."""
    smoothed_features = np.zeros_like(features)
    for i in range(features.shape[1]):
        i_within_radius = np.arange(i - radius, i + radius + 1)
        i_within_radius = i_within_radius[
            (i_within_radius >= 0) & (i_within_radius < features.shape[1])
        ]
        weights = np.exp(-0.5 * ((i_within_radius - i) / sigma) ** 2)
        weights = weights / np.sum(weights)
        smoothed_features[:, i] = np.sum(features[:, i_within_radius] * weights, axis=1)
    return smoothed_features


class RandomSmoothPointFeatures(PointCloudAugmentation):
    def __init__(
        self,
        apply_prob=0.5,
        smoothing_range=(0, 5),
        smoothing_sigma_range=(0.1, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smoothing_min, self.smoothing_max = smoothing_range
        self.smoothing_sigma_min, self.smoothing_sigma_max = smoothing_sigma_range
        self.apply_prob = apply_prob
        self._randomize()

    def _randomize(self):
        self.smoothing = np.random.randint(self.smoothing_min, self.smoothing_max)
        self.smoothing_sigma = np.random.uniform(
            self.smoothing_sigma_min, self.smoothing_sigma_max
        )

    def __call__(self, point_cloud_dict, keys, *args, **kwargs):
        self._randomize()
        if np.random.rand() < self.apply_prob:
            for key in keys:
                transform_array = self.extract_transform_array(point_cloud_dict, key)
                transform_array = apply_gaussian_filter_along_features(
                    transform_array, self.smoothing, self.smoothing_sigma
                )
                point_cloud_dict = self.insert_transform_array(
                    point_cloud_dict, key, transform_array
                )

        return point_cloud_dict


class AnyAugmentation(PointCloudAugmentation):
    def __init__(self, augmentations, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = augmentations
        self.aug_len = len(augmentations)

    def __call__(self, point_cloud_dict, keys, mb_tree):
        aug_idx = np.random.randint(0, self.aug_len)
        point_cloud_dict = self.augmentations[aug_idx](point_cloud_dict, keys, mb_tree)
        return point_cloud_dict


class RandomFeatureAugmentation(PointCloudAugmentation):
    def __init__(self, augmentations, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = augmentations

    def __call__(self, point_cloud_dict, keys, mb_tree):
        for aug in self.augmentations:
            point_cloud_dict = aug(point_cloud_dict, keys, mb_tree)
        return point_cloud_dict
