import torch

from membrain_pick.networks.diffusion_net import DiffusionNet, DiffusionNetBlock
from membrain_pick.clustering.mean_shift_utils import MeanShiftForwarder

# from membrain_pick.networks.diffusion_net.ms_GPU_orig import MeanShiftEuc
# from torch_geometric.nn.pool import fps
# from torch_geometric.nn.pool import knn

from sklearn.neighbors import NearestNeighbors

# def remove_duplicate_clusters(clusters, combine_distance):


import torch


def farthest_point_sampling_single(
    point_cloud: torch.Tensor, num_samples: int
) -> torch.Tensor:
    """
    Perform Farthest Point Sampling on a single point cloud.

    Parameters
    ----------
    point_cloud : torch.Tensor
        The input tensor containing the point cloud with shape (N, D), where N is
        the number of points in the point cloud, and D is the dimensionality of
        each point.
    num_samples : int
        The number of points to sample from the point cloud.

    Returns
    -------
    torch.Tensor
        The indices of the sampled points with shape (num_samples).

    """
    N, D = point_cloud.shape
    centroids = torch.zeros(num_samples, dtype=torch.long, device=point_cloud.device)
    distance = torch.ones(N, device=point_cloud.device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=point_cloud.device)
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = point_cloud[farthest, :].view(1, D)
        dist = torch.sum((point_cloud - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def find_points_within_radius(
    point_cloud: torch.Tensor, centers_indices: torch.Tensor, radius: float
) -> torch.Tensor:
    """
    Find all points within a specified radius from each center (farthest point).

    Parameters
    ----------
    point_cloud : torch.Tensor
        The input tensor containing the point cloud with shape (N, D), where N is
        the number of points in the point cloud, and D is the dimensionality of
        each point.
    centers_indices : torch.Tensor
        The indices of the centers (farthest points) selected by FPS, with shape (num_centers,).
    radius : float
        The radius within which points will be considered close to a center.

    Returns
    -------
    list of torch.Tensor
        A list where each element is a tensor of indices of points within the radius
        from the corresponding center. The length of the list is equal to num_centers.

    """
    centers = point_cloud[centers_indices]  # Extract the centers using their indices
    squared_radius = radius**2

    # Calculate squared distances from each center to all points
    # Broadcasting allows us to subtract each center from all points in the point cloud
    dist_squared = torch.sum(
        (centers[:, None, :] - point_cloud[None, :, :]) ** 2, dim=2
    )

    # Determine which points are within the radius for each center
    within_radius_masks = dist_squared <= squared_radius

    # Extract indices of points within the radius for each center
    points_within_radius = [
        torch.nonzero(within_radius_masks[i], as_tuple=False).squeeze(1)
        for i in range(centers.size(0))
    ]

    return points_within_radius


def find_nearest_points(
    point_cloud: torch.Tensor, centers_indices: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Find N nearest points to each center.

    Parameters
    ----------
    point_cloud : torch.Tensor
        The input tensor containing the point cloud with shape (N, D), where N is
        the number of points in the point cloud, and D is the dimensionality of each point.
    centers_indices : torch.Tensor
        The indices of the centers (farthest points) selected by FPS, with shape (num_centers,).
    N : int
        The number of nearest points to find for each center.

    Returns
    -------
    list of torch.Tensor
        A list where each element is a tensor of indices of the N nearest points
        to the corresponding center. The length of the list is equal to num_centers.

    """
    centers = point_cloud[centers_indices]  # Extract the centers using their indices

    # Calculate squared distances from each center to all points
    # Broadcasting allows us to subtract each center from all points in the point cloud
    dist_squared = torch.sum(
        (centers[:, None, :] - point_cloud[None, :, :]) ** 2, dim=2
    )

    # Find indices of the N closest points for each center
    nearest_points_indices = torch.topk(dist_squared, N, largest=False, sorted=True)[1]

    # Convert to a list of tensors
    nearest_points = [nearest_points_indices[i] for i in range(centers.size(0))]

    nearest_points = torch.stack(nearest_points, dim=0)

    return nearest_points


class DiffusionNet_MSProposal(DiffusionNet):
    r"""
    Network for multi-scale proposal generation.
    """

    def __init__(
        self,
        C_in,
        C_out,
        C_width=128,
        N_block=4,
        last_activation=None,
        outputs_at="vertices",
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        lstm_first=False,
        mean_shift_clustering=False,
        ms_bandwidth=0.1,
        device="cuda:0",
    ):
        super().__init__(
            C_in,
            C_out=16,
            C_width=C_width,
            N_block=N_block,
            last_activation=last_activation,
            outputs_at=outputs_at,
            mlp_hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
            diffusion_method=diffusion_method,
            lstm_first=lstm_first,
            mean_shift_clustering=False,
            ms_bandwidth=ms_bandwidth,
            device=device,
        )

        ms_bandwidth = 0.1
        self.ms_bandwidth = ms_bandwidth
        self.ms_region_proposal = MeanShiftForwarder(
            ms_bandwidth, num_seeds=None, max_iter=10, device=device
        )
        self.combine_distance = 0.01 * ms_bandwidth

        self.point_block = DiffusionNetBlock(
            C_width=16,
            mlp_hidden_dims=[16, 16],
            dropout=dropout,
            diffusion_method="spectral",
            with_gradient_features=False,
            with_gradient_rotations=False,
        )
        self.point_block = self.point_block.to(self.device)

        self.post_mlp1 = torch.nn.Linear(512, 64)
        self.post_mlp2 = torch.nn.Linear(64, 16)
        self.post_mlp3 = torch.nn.Linear(16, 3)

    def forward(
        self,
        x_in,
        mass,
        L=None,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
        edges=None,
        faces=None,
    ):
        r"""
        Forward pass of the network.
        """
        out = super().forward(x_in, mass, L, evals, evecs, gradX, gradY, edges, faces)
        # self.point_block(out, mass, L, evals, evecs, gradX, gradY)
        from time import time

        start = time()

        fps_num = 50
        fps_out = farthest_point_sampling_single(out[:, :3], fps_num)
        points_with_rad = find_points_within_radius(
            out[:, :3], fps_out, self.ms_bandwidth * 2
        )
        points_closest_N = find_nearest_points(out[:, :3], fps_out, 32)

        closest_points_features = out[points_closest_N]
        closest_points_features = torch.reshape(
            closest_points_features, (closest_points_features.shape[0], -1)
        )

        points_shift1 = self.post_mlp1(closest_points_features)
        points_shift2 = self.post_mlp2(points_shift1)
        points_shift3 = self.post_mlp3(points_shift2)

        points_out = out[fps_out][:, :3] + points_shift3
        return out, points_out

        print(closest_points_features.shape)
        print(points_closest_N.shape)
        exit()
        fps_array = torch.cat()
        print(fps_array.shape)

        exit()

        fps_out = out[fps_out]
        out = out.to(self.device)
        mass = mass.to(self.device)
        L = L.to(self.device)
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)
        gradX = gradX.to(self.device)
        gradY = gradY.to(self.device)
        # edges = edges.to(self.device)
        faces = faces.to(self.device)

        shifts = []
        print()

        cur_points = torch.cat()
        fps_shifts = self.point_block(cur_points)

        exit()

        for i, points in enumerate(points_with_rad):
            print(i, "/", len(points_with_rad))
            points = points.to(self.device)

            points = torch.arange(out.shape[0])
            cur_points = out[points]
            print(points.shape)
            print(fps_out.shape)

            exit()
            cur_points = torch.cat(
                fps_out,
            )

            # cur_shift = self.point_block(cur_points, cur_mass, cur_L, cur_evals, cur_evecs, cur_gradX, cur_gradY, cur_edges, cur_faces)
            cur_shift = self.point_block(
                cur_points, cur_mass, cur_L, cur_evals, cur_evecs, cur_gradX, cur_gradY
            )
            shifts.append(cur_shift)

        # print(fps_out)
        # print(fps_out)
        exit()
        # self.ms_region_proposal.fit(x_in[:, :3], out)

        # clusters, _, cluster_weights = self.ms_region_proposal.mean_shift_forward(x_in[:, :3], out)
        # idc_order = torch.argsort(cluster_weights, descending=True)
        # clusters = clusters[idc_order]

        # unique_centers = torch.unique(clusters, dim=0)
        # unique = torch.ones(len(unique_centers), dtype=bool)
        # nbrs = NearestNeighbors(radius=self.combine_distance, metric='cosine').fit(unique_centers.cpu().detach().numpy())

        # for i, center in enumerate(unique_centers):
        #     if unique[i]:
        #         neighbor_idxs = nbrs.radius_neighbors([center.cpu().detach().numpy()],
        #                                         return_distance=False)[0]
        #         unique[neighbor_idxs] = 0
        #         unique[i] = 1  # leave the current point as unique
        # cluster_centers = unique_centers[unique]
        return cluster_centers
