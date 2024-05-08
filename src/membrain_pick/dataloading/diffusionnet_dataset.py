import os
from typing import Dict
import multiprocessing

import numpy as np
import torch
from torch.utils.data import Dataset

from membrain_seg.segmentation.dataloading.data_utils import get_csv_data

from scipy.spatial import KDTree
from membrain_pick.dataloading.pointcloud_augmentations import get_test_transforms, get_training_transforms
from membrain_pick.dataloading.mesh_partitioning import precompute_partitioning, compute_nearest_distances
from membrain_pick.optimization.plane_projection import project_points_to_nearest_hyperplane


from membrain_pick.networks import diffusion_net



# def process_operator(args):
#     i, verts_list_i, faces_list_i, k_eig, op_cache_dir, normals_i, verts_len, overwrite_cache_flag = args
#     print("get_all_operators() processing {} / {} {:.3f}%".format(i, verts_len, i / verts_len * 100))
#     verts_list_i = verts_list_i.contiguous()
#     faces_list_i = faces_list_i.long()
    
#     if normals_i is None:
#         outputs = diffusion_net.geometry.get_operators(verts_list_i[:, :3], faces_list_i, k_eig, op_cache_dir, overwrite_cache=overwrite_cache_flag)
#     else:
#         outputs = diffusion_net.geometry.get_operators(verts_list_i[:, :3], faces_list_i, k_eig, op_cache_dir, normals=normals_i, overwrite_cache=overwrite_cache_flag)
#     return outputs

# def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None, overwrite_cache_flag=False, max_cpu=16):
#     N = len(verts_list)
#     cpu_use = min(max_cpu, N)
#     cpu_use = min(cpu_use, multiprocessing.cpu_count())
#     pool = multiprocessing.Pool(processes=cpu_use)  # Use all available CPU cores
    
#     args_list = [(i, verts_list[i], faces_list[i], k_eig, op_cache_dir, normals[i] if normals else None, len(verts_list), overwrite_cache_flag) for i in range(N)]
#     results = pool.map(process_operator, args_list)
#     pool.close()
#     pool.join()
    
#     frames, massvec, L, evals, evecs, gradX, gradY = zip(*results)
#     return frames, massvec, L, evals, evecs, gradX, gradY


def convert_to_torch(data_dict: dict) -> dict:
    """
    Converts a list of numpy arrays to a list of torch tensors.

    Parameters
    ----------
    data_list : list
        A list of numpy arrays.

    Returns
    -------
    list
        A list of torch tensors.
    """
    out_dict = {}
    for key in data_dict:
        if isinstance(data_dict[key], np.ndarray):
            out_dict[key] = torch.from_numpy(data_dict[key]).float()
        else:
            out_dict[key] = data_dict[key]
    return out_dict


class MemSegDiffusionNetDataset(Dataset):
    """
    A custom Dataset for Cryo-ET membrane protein localization.

    This Dataset is designed to work with Mask-RCNN
    """

    def __init__(
        self,
        csv_folder: str,
        train: bool = False,
        train_pct: float = 0.8,
        max_tomo_shape: int = 1000, # should this be fixed?
        load_only_sampled_points: int = None,
        overfit: bool = False,
        is_single_mb: bool = False,
        cache_dir: str = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/diffusion_net_training/mesh_cache2",
        force_recompute: bool = False,
        augment_all: bool = True,
        overfit_mb: bool = False,
        aug_prob_to_one: bool = False,
        augment_noise: bool = False,
        gaussian_smoothing: bool = False,
        random_erase: bool = False,
        allpos: bool = False,
        use_psii: bool = True,
        use_b6f: bool = False,
        use_uk: bool = False,
        median_filter: bool = False,
        brightness_transform: bool = False,
        brightness_gradient: bool = False,
        local_brightness_gamma: bool = False,
        normalize_features: bool = False,
        contrast: bool = False,
        contrast_with_stats_inversion: bool = False,
        pixel_size: float = 10.0,
        k_eig: int = 128,
        test_mb=None,

        diffusion_operator_params: Dict = {
            "k_eig": 128, # 128
            "use_precomputed_normals": True,
            "normalize_verts": False,
            "hks_features": False,
            "augment_random_rotate": False,
            "aggregate_coordinates": False,
            "random_sample_thickness": False,
            "use_faces": True,
            "cache_dir": "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/Mesh_Detection/DiffusionNet/cache_dir",
        }

    
    ) -> None:
        """
        Constructs all the necessary attributes for the CryoETMemSegDataset object.
        """
        self.train = train
        self.train_pct = train_pct
        self.csv_folder = csv_folder
        self.max_tomo_shape = max_tomo_shape
        self.load_only_sampled_points = load_only_sampled_points
        self.overfit = overfit
        self.overfit_mb = overfit_mb
        self.cache_dir = cache_dir
        self.is_single_mb = is_single_mb
        self.force_recompute = force_recompute
        self.aug_prob_to_one = aug_prob_to_one
        self.augment_all = augment_all
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
        self.pixel_size = pixel_size
        self.diffusion_operator_params = diffusion_operator_params
        self.allpos = allpos
        self.use_psii = use_psii
        self.use_b6f = use_b6f
        self.use_uk = use_uk

        self.diffusion_operator_params["k_eig"] = k_eig
        self.diffusion_operator_params["cache_dir"] = cache_dir

        self.test_mb = test_mb
        self.initialize_csv_paths()
        self.load_data()
    


        if self.load_only_sampled_points is not None:
            self._precompute_partitioning()

        self.transforms = (
            get_training_transforms(tomo_shape_max=self.max_tomo_shape, 
                                    pixel_size=self.pixel_size,
                                    prob_to_one=self.aug_prob_to_one) if (self.train and self.augment_all) 
                                    else get_test_transforms()
        )
        if self.train:
            self.kdtrees = [KDTree(mb[:, :3]) for mb in (self.membranes if self.load_only_sampled_points is None else self.part_verts)]
        else:
            self.kdtrees = [None] * len(self)
        self.visited_flags = np.zeros(len(self), dtype=bool)


    def _move_to_torch(self):
        """
        Converts all numpy arrays to torch tensors.
        """
        self.membranes = [torch.from_numpy(mb).float() for mb in self.membranes]
        self.labels = [torch.from_numpy(l).float() for l in self.labels]
        self.faces = [torch.from_numpy(f).float() for f in self.faces]
        self.vert_normals = [torch.from_numpy(n).float() for n in self.vert_normals]
        self.gt_pos = [torch.from_numpy(p).float() for p in self.gt_pos]
        self.part_verts = [torch.from_numpy(mb).float() for mb in self.part_verts]
        self.part_labels = [torch.from_numpy(l).float() for l in self.part_labels]
        self.part_faces = [torch.from_numpy(f).float() for f in self.part_faces]
        self.part_normals = [torch.from_numpy(n).float() for n in self.part_normals]
        self.part_gt_pos = [torch.from_numpy(p).float() for p in self.part_gt_pos]
        self.part_vert_weights = [torch.from_numpy(w).float() for w in self.part_vert_weights]


    def _precompute_partitioning(self) -> None:
        """
        Precomputes the partitioning of the mesh into smaller parts.
        """
        print("Precomputing partitioning of the mesh.")
        self.part_verts, self.part_labels, self.part_faces, \
            self.part_normals, self.part_mb_idx, self.part_gt_pos, \
            self.part_vert_weights = precompute_partitioning(
                    membranes=self.membranes,
                    faces=self.faces,
                    labels=self.labels,
                    vert_normals=self.vert_normals,
                    gt_pos=self.gt_pos,
                    max_sampled_points=self.load_only_sampled_points,
                    overfit=self.overfit,
                    overfit_mb=self.overfit_mb,
                    cache_dir=self.cache_dir,
                    force_recompute=self.force_recompute
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
                "mb_token": os.path.basename(self.data_paths[self.part_mb_idx[idx]][0])[:-14],
                "gt_pos": self.part_gt_pos[idx] * self.max_tomo_shape,
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
                "mb_token": os.path.basename(self.data_paths[idx][0])[:-14],
                "gt_pos": self.gt_pos[idx],
                "vert_weights": np.ones(self.membranes[idx].shape[0])
            }
        idx_dict = self.transforms(idx_dict, keys=["membrane"], mb_tree=self.kdtrees[idx])
        idx_dict = convert_to_torch(idx_dict)
        idx_dict = self._convert_to_diffusion_input(idx_dict, overwrite_cache_flag=not self.visited_flags[idx])
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
        return self.part_verts[0].shape[1] - 3

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
            if isinstance(entry[1], list):
                gt_pos = np.zeros((0, 3))
                for gt_file in entry[1]:
                    if gt_file is None:
                        continue
                    if os.path.isfile(gt_file):
                        gt_pos = np.concatenate([gt_pos, np.array(get_csv_data(gt_file), dtype=float)], axis=0)
                    else:
                        print("Warning: GT file is not a file")
                gt_pos = project_points_to_nearest_hyperplane(gt_pos, points[:, :3])
                if gt_pos.shape[0] == 0:
                    gt_pos = np.zeros((1, 3))
            elif os.path.isfile(entry[1]):
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
            points[:, :3] *= self.pixel_size
            gt_pos /= self.max_tomo_shape
            gt_pos *= self.pixel_size

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

        candidate_files = (os.listdir(self.csv_folder) if not self.is_single_mb else [self.csv_folder])
        for filename in candidate_files:
            # if not filename.startswith("T17S1M13") and not filename.startswith("T17S1M11"):# and not filename.startswith("T17S2M5") and not filename.startswith("T17S1M7") and not filename.startswith("T17S1M10"):
            #     continue
            if filename.endswith("data.csv"):
                if "T17S2M5" in filename: # corrupt file
                    continue
                if self.test_mb is not None:
                    if self.test_mb not in filename:
                        continue
                self.data_paths.append(filename)

        self.data_paths.sort()
        if not self.is_single_mb:
            if not self.allpos:
                self.data_paths = [
                    (os.path.join(self.csv_folder, filename), 
                    [os.path.join(self.csv_folder, filename[:-14] + "_psii_pos.csv") if self.use_psii else None,
                        os.path.join(self.csv_folder, filename[:-14] + "_b6f_pos.csv") if self.use_b6f else None,
                        os.path.join(self.csv_folder, filename[:-14] + "_uk_pos.csv") if self.use_uk else None],
                    os.path.join(self.csv_folder, filename[:-14] + "_mesh_faces.csv"),
                    os.path.join(self.csv_folder, filename[:-14] + "_mesh_normals.csv"),)
                    for filename in self.data_paths
                ]
            else:
                self.data_paths = [
                    (os.path.join(self.csv_folder, filename), 
                    os.path.join(self.csv_folder, filename[:-14] + "_all_pos.csv"),
                    os.path.join(self.csv_folder, filename[:-14] + "_mesh_faces.csv"),
                    os.path.join(self.csv_folder, filename[:-14] + "_mesh_normals.csv"),)
                    for filename in self.data_paths
            ]
        else:  
            self.data_paths = [
                (filename,
                 filename[:-14] + "_psii_pos.csv",
                 filename[:-14] + "_mesh_faces.csv",
                 filename[:-14] + "_mesh_normals.csv")
            ]
        if self.train:
            self.data_paths = self.data_paths[:int(len(self.data_paths) * self.train_pct)]
        else:
            self.data_paths = self.data_paths[int(len(self.data_paths) * self.train_pct):]

    # def _precompute_operators(self,
    #                            overwrite_cache_flag=False):
    #     get_all_operators(
    #         verts_list = self.part_verts,
    #         faces_list = self.part_faces,
    #         k_eig = self.diffusion_operator_params["k_eig"],
    #         op_cache_dir = self.diffusion_operator_params["cache_dir"],
    #         normals = self.part_normals,
    #         overwrite_cache_flag = overwrite_cache_flag
    #     )

        

    def _convert_to_diffusion_input(self, 
                             idx_dict, 
                             overwrite_cache_flag=False):
        

        faces = idx_dict["faces"]
        if not self.diffusion_operator_params["use_faces"]:
            faces = torch.zeros((0, 3))
        faces = faces.long()

        verts = idx_dict["membrane"][:, :3] 
        features = idx_dict["membrane"][:, 3:]
        
        feature_len = 10
        if self.diffusion_operator_params["random_sample_thickness"]:
            start_sample = np.random.randint(0, features.shape[1] - feature_len)
            features = features[:, start_sample:start_sample + feature_len]
        
        verts_orig = verts.clone() * self.max_tomo_shape
        if self.diffusion_operator_params["normalize_verts"]:
            verts = diffusion_net.geometry.normalize_positions(verts.unsqueeze(0)).squeeze(0)
        else:
            verts = verts.contiguous()
            # verts *= self.pixel_size
            verts -= verts.mean()
        # Get the geometric operators needed to evaluate DiffusionNet. This routine 
        # automatically populates a cache, precomputing only if needed.
        try:
            _, mass, L, evals, evecs, gradX, gradY = \
                diffusion_net.geometry.get_operators(
                    verts=verts,
                    faces=faces,
                    k_eig=self.diffusion_operator_params["k_eig"],
                    op_cache_dir=self.diffusion_operator_params["cache_dir"],
                    normals=(idx_dict["normals"].float() if self.diffusion_operator_params["use_precomputed_normals"] else None),
                    overwrite_cache=overwrite_cache_flag
                )
        except Exception as e:
            print(e)
            return idx_dict
            
        
        if self.diffusion_operator_params["augment_random_rotate"]:
            verts = diffusion_net.utils.random_rotate_points(verts)
        
        if self.diffusion_operator_params["hks_features"]:
            features_hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
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
        from membrain_pick.optimization.plane_projection import make_2D_projection_scatter_plot
        for k in range(times):
            idx_dict = self.__getitem__(idx)
            mask = idx_dict["label"] != 10.05
            make_2D_projection_scatter_plot(
                out_file=os.path.join(out_dir, "test%d_%d.png" % (idx, k)),
                point_cloud=idx_dict["membrane"][:, :3][mask],
                color=idx_dict["membrane"][:, 7][mask],
                s=50
            )
            # print(np.unique(idx_dict["label"]))
            # make_2D_projection_scatter_plot(
            #     out_file=os.path.join(out_dir, "test%d_%d_lab.png" % (idx, k)),
            #     point_cloud=idx_dict["membrane"][:, :3][mask],
            #     color=idx_dict["label"][:][mask],
            #     s=7
            # )