import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from scipy.spatial import KDTree
from membrain_pick.data_augmentations.pointcloud_augmentations import (
    get_test_transforms,
    get_training_transforms,
)
from membrain_pick.dataloading.mesh_partitioning import (
    precompute_partitioning,
    compute_nearest_distances,
)
from membrain_pick.dataloading.data_utils import (
    load_mesh_from_hdf5,
    convert_to_torch,
    read_star_file,
)
from membrain_pick.optimization.plane_projection import (
    project_points_to_nearest_hyperplane,
)

from membrain_pick.networks import diffusion_net


class MemSegDiffusionNetDataset(Dataset):
    """
    A custom Dataset for Cryo-ET membrane protein localization.

    This Dataset is designed to work with Mask-RCNN
    """

    def __init__(
        self,
        data_folder: str,
        train: bool = False,
        train_pct: float = 0.8,
        load_only_sampled_points: int = None,
        overfit: bool = False,
        is_single_mb: bool = False,
        cache_dir: str = "./cashe_dir/",
        force_recompute: bool = False,
        augment_all: bool = True,
        overfit_mb: bool = False,
        aug_prob_to_one: bool = False,
        position_tokens: list = None,
        input_pixel_size: float = 10.0,
        k_eig: int = 128,
        shuffle: bool = True,
        test_mb=None,
        diffusion_operator_params: Dict = {
            "k_eig": 128,  # 128
            "use_precomputed_normals": True,
            "normalize_verts": False,
            "hks_features": False,
            "augment_random_rotate": False,
            "aggregate_coordinates": False,
            "random_sample_thickness": False,
            "use_faces": True,
            "cache_dir": "./cache_dir/diffusion_operators",
        },
    ) -> None:
        """
        Constructs all the necessary attributes for the CryoETMemSegDataset object.
        """
        self.train = train
        self.train_pct = train_pct
        self.data_folder = data_folder
        self.load_only_sampled_points = load_only_sampled_points
        self.overfit = overfit
        self.overfit_mb = overfit_mb
        self.is_single_mb = is_single_mb
        self.force_recompute = force_recompute
        self.aug_prob_to_one = aug_prob_to_one
        self.augment_all = augment_all
        self.input_pixel_size = input_pixel_size
        self.process_pixel_size = 15.0  # hard-coded for now

        self.diffusion_operator_params = diffusion_operator_params
        self.position_tokens = position_tokens
        self.shuffle = shuffle

        self.diffusion_operator_params["k_eig"] = k_eig
        self.diffusion_operator_params["cache_dir"] = os.path.join(
            cache_dir, "diffusion_operators"
        )
        self.cache_dir = os.path.join(cache_dir, "mesh_partitioning")

        self.max_tomo_shape = 500.0  # TODO: rename and hard-code everywhere
        self.test_mb = test_mb
        self.initialize_data_paths()
        self.load_data()

        if self.load_only_sampled_points is not None:
            self._precompute_partitioning()

        # self.augment_all = False
        self.transforms = (
            get_training_transforms(
                tomo_shape_max=self.max_tomo_shape,
                pixel_size=self.process_pixel_size,
                prob_to_one=self.aug_prob_to_one,
            )
            if (self.train and self.augment_all)
            else get_test_transforms()
        )
        if self.train:
            self.kdtrees = [
                KDTree(mb[:, :3])
                for mb in (
                    self.membranes
                    if self.load_only_sampled_points is None
                    else self.part_verts
                )
            ]
        else:
            self.kdtrees = [None] * len(self)
        self.visited_flags = np.zeros(len(self), dtype=bool)

    def _precompute_partitioning(self) -> None:
        """
        Precomputes the partitioning of the mesh into smaller parts.
        """
        print("Precomputing partitioning of the mesh.")
        (
            self.part_verts,
            self.part_labels,
            self.part_faces,
            self.part_normals,
            self.part_mb_idx,
            self.part_gt_pos,
            self.part_vert_weights,
        ) = precompute_partitioning(
            membranes=self.membranes,
            faces=self.faces,
            labels=self.labels,
            vert_normals=self.vert_normals,
            gt_pos=self.gt_pos,
            max_sampled_points=self.load_only_sampled_points,
            overfit=self.overfit,
            overfit_mb=self.overfit_mb,
            cache_dir=self.cache_dir,
            force_recompute=self.force_recompute,
        )

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
                # "mb_token": os.path.basename(self.data_paths[self.part_mb_idx[idx]][0])[:-8],
                "mb_token": os.path.basename(self.data_paths[self.part_mb_idx[idx]][0])[
                    :-3
                ],
                "tomo_file": self.tomo_files[self.part_mb_idx[idx]],
                "gt_pos": self.part_gt_pos[idx] * self.max_tomo_shape,
                "vert_weights": self.part_vert_weights[idx],
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
                "mb_token": os.path.basename(self.data_paths[idx][0])[:-8],
                "tomo_file": self.tomo_files[idx],
                "gt_pos": self.gt_pos[idx],
                "vert_weights": np.ones(self.membranes[idx].shape[0]),
            }

        idx_dict = self.transforms(
            idx_dict, keys=["membrane"], mb_tree=self.kdtrees[idx]
        )
        if self.shuffle and self.train:
            assert idx_dict["membrane"].shape[1] >= 16 + 3
            feature_channels = idx_dict["membrane"][:, 3:].shape[1] - 16
            idx_start_range = range(feature_channels)
            idx_start = np.random.choice(idx_start_range) + 3
            idx_dict["membrane"] = np.concatenate(
                (
                    idx_dict["membrane"][:, :3],
                    idx_dict["membrane"][:, idx_start : idx_start + 16],
                ),
                axis=1,
            )
        elif self.shuffle and not self.train:
            assert idx_dict["membrane"].shape[1] >= 16 + 3
            center_feature_start = (idx_dict["membrane"][:, 3:].shape[1] - 16) // 2
            center_feature_start += 3
            idx_dict["membrane"] = np.concatenate(
                (
                    idx_dict["membrane"][:, :3],
                    idx_dict["membrane"][
                        :, center_feature_start : center_feature_start + 16
                    ],
                ),
                axis=1,
            )
        else:
            center_feature_start = (
                3  # this will run through all start indices and average the results
            )
            idx_dict["membrane"] = np.concatenate(
                (
                    idx_dict["membrane"][:, :3],
                    idx_dict["membrane"][:, center_feature_start:],
                ),
                axis=1,
            )
        idx_dict = convert_to_torch(idx_dict)
        overwrite_cache_flag = not self.visited_flags[idx] and self.force_recompute
        idx_dict = self._convert_to_diffusion_input(
            idx_dict, overwrite_cache_flag=overwrite_cache_flag
        )
        self.visited_flags[idx] = True
        return idx_dict

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
        return len(self.membranes)

    def get_parameter_len(self) -> int:
        """
        Returns the length of the parameters of the dataset.

        Returns
        -------
        int
            The length of the parameters of the dataset.
        """
        if self.shuffle:
            return 16
        return self.part_verts[0].shape[1] - 3

    def _get_GT_mask(self, gt_pos, gt_classes):
        gt_mask = np.zeros(gt_pos.shape[0], dtype=bool)
        if self.position_tokens is None:
            gt_mask = np.ones(gt_pos.shape[0], dtype=bool)
        else:
            for token in self.position_tokens:
                gt_mask = np.logical_or(gt_mask, gt_classes == token)
        return gt_mask

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
        self.tomo_files = []

        for entry in self.data_paths:
            hdf5_path = entry[0]
            gt_path = entry[1]

            mesh_data = load_mesh_from_hdf5(hdf5_path)
            points = mesh_data["points"]
            faces = mesh_data["faces"]
            vert_normals = mesh_data["normals"]
            normal_values = mesh_data["normal_values"]
            tomo_file = (
                "" if not "tomo_file" in mesh_data.keys() else mesh_data["tomo_file"]
            )

            points = np.concatenate([points, normal_values], axis=1)
            if os.path.isfile(gt_path):
                gt_data = read_star_file(gt_path)  # pandas dataframe
                # check size of gt_data
                if gt_data.shape[0] == 0:
                    gt_pos = np.zeros((1, 3))
                    gt_classes = np.ones(1) * -1
                else:
                    gt_pos = gt_data[
                        ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
                    ].values
                    assert (
                        "rlnCoordinateX" in gt_data.columns
                    ), "rlnCoordinateX not in columns"
                    if "rlnClassNumber" in gt_data.columns:
                        gt_classes = gt_data["rlnClassNumber"].values
                    else:
                        gt_classes = np.ones(gt_pos.shape[0])
                    gt_pos = project_points_to_nearest_hyperplane(gt_pos, points[:, :3])
            else:
                gt_pos = np.zeros((1, 3))
                gt_classes = np.ones(1) * -1

            gt_mask = self._get_GT_mask(gt_pos, gt_classes)
            gt_pos = gt_pos[gt_mask]

            points = points * self.input_pixel_size / self.process_pixel_size
            gt_pos = gt_pos * self.input_pixel_size / self.process_pixel_size

            distances, nn_idcs = compute_nearest_distances(points[:, :3], gt_pos)

            # Move GT along nearest normal
            _, nn_idcs_psii = compute_nearest_distances(gt_pos, points[:, :3])
            psii_normals = vert_normals[nn_idcs_psii]

            nearest_PSII_pos = gt_pos[nn_idcs] + psii_normals[nn_idcs] * 20
            connection_vectors = nearest_PSII_pos - (points[:, :3])

            angle_to_normal = np.einsum("ij,ij->i", connection_vectors, vert_normals)
            mask = angle_to_normal < 0

            distances[distances > 10] = 10
            distances[mask] = 10.05
            points[:, :3] /= self.max_tomo_shape
            points[:, :3] *= self.process_pixel_size
            gt_pos /= self.max_tomo_shape
            gt_pos *= self.process_pixel_size

            self.membranes.append(points)
            self.labels.append(distances)
            self.faces.append(faces)
            self.vert_normals.append(vert_normals)
            self.gt_pos.append(gt_pos)
            self.tomo_files.append(tomo_file)
            if self.overfit:
                break

    def _train_split(self) -> None:
        """
        Splits the dataset into training and validation sets.

        Parameters
        ----------
        train_pct : float
            The percentage of the dataset to be used for training.
        """
        if self.train:
            self.data_paths = self.data_paths[
                : int(len(self.data_paths) * self.train_pct)
            ]
        else:
            self.data_paths = self.data_paths[
                int(len(self.data_paths) * self.train_pct) :
            ]

    def initialize_data_paths(self) -> None:
        self.data_paths = []
        candidate_files = (
            os.listdir(self.data_folder)
            if not self.is_single_mb
            else [self.data_folder]
        )

        # Find all .h5 files in the data folder
        for filename in candidate_files:
            # if not "mem08" in filename:
            #     continue
            if filename.endswith(".h5"):
                self.data_paths.append(filename)
        self.data_paths.sort()

        # Find ground truth files in the form of .star files
        self.data_paths = [
            (
                os.path.join(self.data_folder, filename),
                os.path.join(self.data_folder, filename[:-3] + ".star"),
            )
            for filename in self.data_paths
        ]

        # Get current training split
        self._train_split()

    def _convert_to_diffusion_input(self, idx_dict, overwrite_cache_flag=False):

        faces = idx_dict["faces"]
        if not self.diffusion_operator_params["use_faces"]:
            faces = torch.zeros((0, 3))
        faces = faces.long()

        verts = idx_dict["membrane"][:, :3]
        features = idx_dict["membrane"][:, 3:]
        feature_len = 10
        if self.diffusion_operator_params["random_sample_thickness"]:
            start_sample = np.random.randint(0, features.shape[1] - feature_len)
            features = features[:, start_sample : start_sample + feature_len]

        verts_orig = verts.clone() * self.max_tomo_shape
        if self.diffusion_operator_params["normalize_verts"]:
            verts = diffusion_net.geometry.normalize_positions(
                verts.unsqueeze(0)
            ).squeeze(0)
        else:
            verts = verts.contiguous()
            verts -= verts.mean()
        # Get the geometric operators needed to evaluate DiffusionNet. This routine
        # automatically populates a cache, precomputing only if needed.
        try:
            _, mass, L, evals, evecs, gradX, gradY = (
                diffusion_net.geometry.get_operators(
                    verts=verts,
                    faces=faces,
                    k_eig=self.diffusion_operator_params["k_eig"],
                    op_cache_dir=self.diffusion_operator_params["cache_dir"],
                    normals=(
                        idx_dict["normals"].float()
                        if self.diffusion_operator_params["use_precomputed_normals"]
                        else None
                    ),
                    overwrite_cache=overwrite_cache_flag,
                )
            )
        except Exception as e:
            print(e)
            return idx_dict

        if self.diffusion_operator_params["augment_random_rotate"]:
            verts = diffusion_net.utils.random_rotate_points(verts)

        if self.diffusion_operator_params["hks_features"]:
            features_hks = diffusion_net.geometry.compute_hks_autoscale(
                evals, evecs, 16
            )
            features = torch.cat([features, features_hks], dim=1)

        if self.diffusion_operator_params["aggregate_coordinates"]:
            features = torch.cat([verts, features], dim=1)

        # Convert all inputs to float32 before passing them to the model
        features = features.float()
        mass = mass.float()
        L = L.float()
        evals = evals.float()
        evecs = evecs.float()
        gradX = gradX.float()
        gradY = gradY.float()
        faces = faces.float()

        diffusion_inputs = {
            "features": features,
            "mass": mass,
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
            "faces": faces,
        }

        idx_dict["diffusion_inputs"] = diffusion_inputs
        idx_dict["verts"] = verts
        idx_dict["verts_orig"] = verts_orig

        return idx_dict

    def test_loading(self, out_dir, idx: int, times=1) -> None:
        """
        Tests the loading of a data-label pair.

        Parameters
        ----------
        idx : int
            Index of the sample to be loaded.
        """
        from membrain_pick.optimization.plane_projection import (
            make_2D_projection_scatter_plot,
        )

        for k in range(times):
            print("Getting item %d for the %d-th time" % (idx, k))
            idx_dict = self.__getitem__(idx)
            mask = idx_dict["label"] != 10.05
            make_2D_projection_scatter_plot(
                out_file=os.path.join(out_dir, "test%d_%d.png" % (idx, k)),
                point_cloud=idx_dict["membrane"][:, :3][mask],
                color=idx_dict["membrane"][:, 7][mask],
                s=50,
            )
