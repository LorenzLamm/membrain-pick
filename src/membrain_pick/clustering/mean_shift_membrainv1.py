import numpy as np
from membrain_pick.clustering.mean_shift_utils_membrainv1 import mean_shift
from sklearn.metrics.pairwise import euclidean_distances


class MeanShift_clustering(object):
    def __init__(self, pos_thres=-3.0, recluster_thres=None, recluster_bw=None):
        self.recluster_thres = recluster_thres
        self.recluster_bw = recluster_bw
        self.pos_thres = pos_thres

    def cluster_NN_output(self, all_coords, scores, bandwidth=20):
        pos_thres = self.pos_thres
        pos_mask = scores > 10.0 - pos_thres
        coords = all_coords[pos_mask]
        cluster_centers, cluster_labels = mean_shift(
            all_coords[pos_mask],
            scores[pos_mask],
            bandwidth=bandwidth,
            weighting="quadratic",
            kernel="gauss",
            n_pr=8,
        )

        if self.recluster_thres is not None:
            cluster_centers, cluster_labels = self.__refine_large_clusters(
                all_coords[pos_mask], cluster_centers, cluster_labels, scores[pos_mask]
            )

        cluster_centers = add_center_quantity(cluster_centers, cluster_labels, coords)
        cluster_centers, cluster_labels, all_coords_masked = exclude_small_centers(
            cluster_centers, cluster_labels, all_coords[pos_mask], threshold=3
        )
        cluster_centers = cluster_centers[:, :3]
        # if cluster_centers_with_normals.shape[0] == 0:
        #     cluster_centers_with_normals = np.ones((0, 10))
        return cluster_centers, cluster_labels

    def __refine_large_clusters(
        self, all_coords, cluster_centers, cluster_labels, scores
    ):
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            labelmask = cluster_labels == label
            cur_points = all_coords[labelmask]
            cur_scores = scores[labelmask]
            dist_mat = euclidean_distances(cur_points, cur_points, squared=False)
            max_val = np.amax(dist_mat)
            if max_val > self.recluster_thres:
                print("Refining one cluster because of its size.")
                new_cluster_centers, new_cluster_labels = mean_shift(
                    cur_points,
                    cur_scores,
                    bandwidth=self.recluster_bw,
                    weighting="quadratic",
                    kernel="gauss",
                    n_pr=1,
                )
                if (
                    new_cluster_centers.shape[0] == 1
                ):  ## This means that no more cluster centers were detected
                    print("-> No further splits.")
                    continue
                print("--> Split into", new_cluster_centers.shape[0], "clusters.")
                new_cluster_labels_BU = new_cluster_labels.copy()
                for i, center in enumerate(new_cluster_centers):
                    if i == 0:
                        cluster_centers[label] = center
                        new_cluster_labels[new_cluster_labels_BU == 0] = label
                    else:
                        cluster_centers = np.concatenate(
                            (cluster_centers, np.expand_dims(center, axis=0)), axis=0
                        )
                        new_cluster_labels[new_cluster_labels_BU == i] = (
                            cluster_centers.shape[0] - 1
                        )
                cluster_labels[labelmask] = new_cluster_labels
        return cluster_centers, cluster_labels


def extract_coords_and_scores(data, header):
    if header is not None:
        volume_x_id = np.squeeze(np.argwhere(np.array(header) == "posX"))
        volume_y_id = np.squeeze(np.argwhere(np.array(header) == "posY"))
        volume_z_id = np.squeeze(np.argwhere(np.array(header) == "posZ"))
        for k, entry in enumerate(header):
            if entry.startswith("predDist"):
                score_id = k
                break
    else:
        volume_x_id = 0
        volume_y_id = 1
        volume_z_id = 2
        score_id = 3
    x_coords = np.expand_dims(data[:, volume_x_id], 1)
    y_coords = np.expand_dims(data[:, volume_y_id], 1)
    z_coords = np.expand_dims(data[:, volume_z_id], 1)
    coords = np.concatenate((x_coords, y_coords, z_coords), 1)
    scores = data[:, score_id]
    return coords, scores


def add_center_quantity(cluster_centers, cluster_labels, coords):
    out_centers = np.concatenate(
        (cluster_centers, np.zeros((cluster_centers.shape[0], 1))), 1
    )
    for i in range(cluster_centers.shape[0]):
        labels_mask = cluster_labels == i
        temp_coords = coords[labels_mask]
        unique_coords = np.unique(temp_coords, axis=0)
        cen_quan = unique_coords.shape[0]
        out_centers[i, 3] = cen_quan
    return out_centers


def exclude_small_centers(cluster_centers, cluster_labels, all_coords, threshold):
    mask = cluster_centers[:, 3] > threshold
    keep_idcs = np.argwhere(mask)
    labels_mask = np.array(
        [cluster_labels[i] in keep_idcs for i in range(cluster_labels.shape[0])]
    )
    out_centers = cluster_centers[mask]
    out_labels = cluster_labels[labels_mask]
    out_coords = all_coords[labels_mask]
    print("Excluding", np.sum(1 - mask), "centers because they are too small.")
    return out_centers, out_labels, out_coords
