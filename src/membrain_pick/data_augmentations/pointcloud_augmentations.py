from membrain_pick.data_augmentations.pointcloud_transforms import (
    RandomGaussianSmoothing,
    RandomBrightnessGradient,
    RandomErasing,
    RandomFeatureAugmentation,
    RandomFeatureDropout,
    RandomFeatureNoise,
    RandomFeatureScale,
    RandomFeatureScaleWithStatsInversion,
    RandomFeatureShift,
    RandomLocalBrightnessGamma,
    RandomSmoothPointFeatures,
    AnyAugmentation,
    NormalizeFeatures,
)


def get_training_transforms(tomo_shape_max, pixel_size, prob_to_one=False):
    tomo_shape_max /= pixel_size
    list_of_augs = [
        NormalizeFeatures(),
        AnyAugmentation(
            [
                RandomGaussianSmoothing(
                    apply_prob=(1.0 if prob_to_one else 0.3),
                    smoothing_radius_range=(1.5 / tomo_shape_max, 5.0 / tomo_shape_max),
                    smoothing_sigma_range=(0.0, 0.5 / tomo_shape_max),
                ),
                # RandomMedianSmoothing(
                #         apply_prob=(1.0 if prob_to_one else 0.5),
                #         smoothing_radius_range=(1.2 / tomo_shape_max, 2. / tomo_shape_max)
                #     ),
            ]
        ),
        RandomSmoothPointFeatures(
            apply_prob=(1.0 if prob_to_one else 0.3),
            smoothing_range=(1.0, 7.0),
            smoothing_sigma_range=(1.5, 5.0),
        ),
        RandomFeatureDropout(
            apply_prob=(1.0 if prob_to_one else 0.3), dropout_prob_range=(0.0, 0.15)
        ),
        RandomFeatureNoise(
            apply_prob=(1.0 if prob_to_one else 0.3), noise_std_range=(0.0, 0.25)
        ),
        RandomFeatureShift(
            apply_prob=(1.0 if prob_to_one else 0.3), shift_range=(-0.5, 0.5)
        ),
        RandomFeatureScale(
            apply_prob=(1.0 if prob_to_one else 0.3), scale_range=(0.5, 1.5)
        ),
        RandomFeatureScaleWithStatsInversion(
            apply_prob=(1.0 if prob_to_one else 0.3), gamma_range=(0.65, 1.7)
        ),
        RandomErasing(
            apply_prob=(1.0 if prob_to_one else 0.3),
            patch_radius_range=(1.5 / tomo_shape_max, 5.5 / tomo_shape_max),
            num_patches_range=(1, 7),
        ),
        RandomBrightnessGradient(
            apply_prob=(1.0 if prob_to_one else 0.3),
            max_brightness_gradient_strength=1.0,
            brightness_gradient_scale_max=0.75 * tomo_shape_max,
        ),
        RandomLocalBrightnessGamma(
            apply_prob=(1.0 if prob_to_one else 0.3),
            local_brightness_gamma_scale_range=(0.0, 1.0),
        ),
    ]
    combined_transform = RandomFeatureAugmentation(list_of_augs)
    return combined_transform


def get_test_transforms():
    list_of_augs = [
        NormalizeFeatures(),
    ]
    combined_transform = RandomFeatureAugmentation(list_of_augs)
    return combined_transform
