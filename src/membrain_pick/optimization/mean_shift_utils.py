""" Implemntation of Mean Shift based on https://github.com/masqm/Faster-Mean-Shift/tree/master"""

import torch
from torch.nn.functional import normalize, pairwise_distance
import numpy as np
import scipy
# import KDTree
from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from time import time
from sklearn.cluster import DBSCAN

class MeanShiftForwarder():
    def __init__(self, bandwidth, num_seeds, max_iter, device, margin=2.):
        self.bandwidth = bandwidth
        self.num_seeds = num_seeds
        self.max_iter = max_iter
        self.margin = margin
        self.device = device

    def cos_batch(self, a, b):
            b = b.float()
            a = a.float()
            num = a @ b.T
            denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
            return num / denom

    def euclidean_dist(self, a, b):
        # dist_mat = torch.cdist(a.float(), b.float(), p=2)
        dist_mat = torch.cdist(a.float(), b.float(), p=2, compute_mode="donot_use_mm_for_euclid_dist")
        return dist_mat

    def get_weight(self, dist, nn_weights, bandwidth):
        # weights shape: [52, 52]
        # dists shape: [128, 2704] = [128, 52*52]
        nn_weights = nn_weights.reshape(1, -1) # [1, 2704]
        thr = bandwidth

        # normalize dists
        dist_norm = (thr - dist) 

        max = torch.tensor(1.0).float().to(self.device)
        min = torch.tensor(0.0).float().to(self.device)
        # dist = torch.from_numpy(dist).to(self.device)
        # dis = torch.where(dist < thr, max, min)
        # print(torch.sum(dis), "<--", dist.shape[0]*dist.shape[1])
        dis = torch.where(dist < thr, dist_norm, min)
        dis = dis.to(self.device)
        
        nn_weights = nn_weights.to(self.device)
        nn_weights = torch.clamp(10. - nn_weights, 1e-2, 10)
        dis *= (nn_weights + 0.05) # +0.05 as a regularization. Otherwise, most values will be close to zero!
        dis = normalize(dis, dim=1, p=2)
        return dis

    def mean_shift_for_seeds(self, coords, nn_weights, seeds):
        stop_thresh = 1e-3 * self.bandwidth
        iter = 0
        X = coords.to(self.device)
        S = seeds.to(self.device)
        B = torch.tensor(self.bandwidth).double().to(self.device)
        while True:
            weight = self.get_weight(self.euclidean_dist(S, X.float()), nn_weights, B)
            num = (weight[:, :, None] * X).sum(dim=1)
            S_old = S
            S_weights = num
            S = num / (weight.sum(1)[:, None] + 1e-6 * min(1., self.bandwidth))
            iter += 1
            print(torch.norm(S - S_old, dim=1).mean(), "<--", stop_thresh, iter, self.max_iter)
            if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter >= self.max_iter):
                break

        p_num = []
        for line in weight:
            p_num.append(line[line == 1].size()[0])

        return S, p_num, torch.norm(S_weights, dim=1)

    def initialize_seeds(self, x, sample_pos, sample_rad=7.5):
        # np.random.seed(999)
        shape = x.shape
        all_seeds = []
        if len(shape) == 4:
            s_max1 = shape[2]
            s_max2 = shape[3]
        elif len(shape) == 5:
            s_max1 = shape[2]
            s_max2 = shape[3]
            s_max3 = shape[4]
        else:
            raise IOError('Wrong input shape!')
        while len(all_seeds) < self.num_seeds:
            cur_sample_pt = len(all_seeds) % sample_pos.shape[0]
            cur_sample = sample_pos[cur_sample_pt]
            x_comp = np.random.uniform(max(self.margin, cur_sample[0] - sample_rad), min(s_max1-self.margin, cur_sample[0] + sample_rad))
            y_comp = np.random.uniform(max(self.margin, cur_sample[1] - sample_rad), min(s_max2-self.margin, cur_sample[1] + sample_rad))
            cur_point = np.array((x_comp, y_comp))
            if len(shape) == 5:
                z_comp = np.random.uniform(max(self.margin, cur_sample[2] - sample_rad), min(s_max3-self.margin, cur_sample[2] + sample_rad))
                cur_point = np.array((x_comp, y_comp, z_comp))
            dist = scipy.spatial.distance.cdist(np.expand_dims(cur_sample.numpy(), 0),np.expand_dims(cur_point, axis=0))
            if np.min(dist, axis=0) < sample_rad:
                all_seeds.append(cur_point)
        all_seeds = torch.from_numpy(np.stack(all_seeds))
        return all_seeds

    def initialize_seeds_deprec(self, x, sample_pos, sample_rad=7.5):
        np.random.seed(999)
        shape = x.shape
        all_seeds = []
        if len(shape) == 4:
            s_max1 = shape[2]
            s_max2 = shape[3]
        elif len(shape) == 5:
            s_max1 = shape[2]
            s_max2 = shape[3]
            s_max3 = shape[4]
        else:
            raise IOError('Wrong input shape!')
        while len(all_seeds) < self.num_seeds:
            x_comp = np.random.uniform(self.margin, s_max1-self.margin)
            y_comp = np.random.uniform(self.margin, s_max2-self.margin)
            cur_point = np.array((x_comp, y_comp))
            if len(shape) == 5:
                z_comp = np.random.uniform(self.margin, s_max3 - self.margin)
                cur_point = np.array((x_comp, y_comp, z_comp))
            dist = scipy.spatial.distance.cdist(sample_pos.numpy(),np.expand_dims(cur_point, axis=0))
            if np.min(dist, axis=0) < sample_rad:
                all_seeds.append(cur_point)
        all_seeds = torch.from_numpy(np.stack(all_seeds))
        return all_seeds


    def initialize_coords(self, x):
        shape = x.shape
        if len(shape) == 4:
            s_max1 = shape[2]
            s_max2 = shape[3]
        elif len(shape) == 5:
            s_max1 = shape[2]
            s_max2 = shape[3]
            s_max3 = shape[4]
        else:
            raise IOError('Wrong input shape!')
        coords_x = torch.linspace(0, s_max1-1, s_max1)
        coords_y = torch.linspace(0, s_max2-1, s_max2)

        if len(shape) == 5:
            coords_z = torch.linspace(0, s_max3-1, s_max3)
            grid_x, grid_y, grid_z = torch.meshgrid(coords_x, coords_y, coords_z)
            # coordinate_grid = torch.cat((grid_x.unsqueeze(3), grid_y.unsqueeze(3), grid_z.unsqueeze(3)), dim=3)
            coordinate_grid = torch.cat((grid_y.unsqueeze(3), grid_x.unsqueeze(3), grid_z.unsqueeze(3)), dim=3)

        else:
            grid_x, grid_y = torch.meshgrid(coords_x, coords_y)
            coordinate_grid = torch.cat((grid_y.unsqueeze(2), grid_x.unsqueeze(2)), dim=2) # TODO: changed order

        coordinate_grid = torch.reshape(coordinate_grid, (-1, (2 if len(shape) == 4 else 3)))
        return coordinate_grid

    def cleanup_cluster_centers(self, centers, threshold=0.5):
        # Compute pairwise Euclidean distances between centers
        dist_matrix = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Identify clusters that are within the threshold
        close_pairs = np.where((dist_matrix > 0) & (dist_matrix < threshold))
        
        # Use a set to keep track of indices already considered for merging
        merged_indices = set()
        new_centers = []
        
        for i, j in zip(close_pairs[0], close_pairs[1]):
            if i not in merged_indices and j not in merged_indices:
                # Average positions of the two centers
                new_center = (centers[i] + centers[j]) / 2
                new_centers.append(new_center)
                
                # Mark these indices as merged
                merged_indices.add(i)
                merged_indices.add(j)
        
        # Add remaining centers not close to any other
        for i, center in enumerate(centers):
            if i not in merged_indices:
                new_centers.append(center)
        
        return np.array(new_centers)
    
    def clean_cluster_centers_dbscan(self, centers, eps):
        # Apply DBSCAN to group centers
        clustering = DBSCAN(eps=eps, min_samples=5).fit(centers)
        unique_labels = set(clustering.labels_)
        
        new_centers = []
        for label in unique_labels:
            # Find all points belonging to the same cluster
            indices = np.where(clustering.labels_ == label)[0]
            cluster_points = centers[indices]
            
            # Compute the average position for the cluster
            new_center = np.mean(cluster_points, axis=0)
            new_centers.append(new_center)
        
        return np.array(new_centers)


    def mean_shift_forward(self, x, weights):
        coords = x.clone()
        seeds = x
        mean, p_num, mean_weights = self.mean_shift_for_seeds(coords.squeeze(), weights.squeeze(), seeds.squeeze())
        print("Number of clusters:", len(mean))
        mean = self.clean_cluster_centers_dbscan(mean.cpu().numpy(), eps=0.5)
        print("Number of clusters after cleanup:", len(mean))
        return mean, p_num, mean_weights