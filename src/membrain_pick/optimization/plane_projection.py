""" This is a collection of functions that are used to project a 3D point onto a 2D plane.

Using these functions and a roughly flat surface, we can generate images of the surface that are
roughly orthographic. This is useful to quickly take a look at the surface and see if there are any
obvious defects or if the surface is roughly as expected.
"""
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree



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



def find_best_fit_plane(point_cloud):
    # Center the point cloud
    point_cloud = np.array(point_cloud)
    point_cloud_mean = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - point_cloud_mean

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_point_cloud)

    # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = pca.components_[2]

    # The plane is defined by the normal vector and any point on the plane, for example, the mean point
    point_on_plane = point_cloud_mean

    return normal_vector, point_on_plane


def project_and_rotate_points(point_cloud, normal_vector, point_on_plane):
    # Step 1: Project points onto the plane
    # Calculate distance from each point to the plane
    distances = np.dot(point_cloud - point_on_plane, normal_vector)
    projected_points = point_cloud - np.outer(distances, normal_vector)

    # Step 2: Rotate the projected points
    # Calculate the rotation needed to align the normal vector with the z-axis
    z_axis = np.array([0, 0, 1])
    axis_of_rotation = np.cross(normal_vector, z_axis)
    angle_of_rotation = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))
    rotation_vector = axis_of_rotation / np.linalg.norm(axis_of_rotation) * angle_of_rotation
    rotation = R.from_rotvec(rotation_vector)
    rotated_points = rotation.apply(projected_points)

    # Ensure the z-component is zero by subtracting minimal z-value
    rotated_points[:, 2] -= np.min(rotated_points[:, 2])

    return rotated_points


def project_points_to_plane(point_cloud):
    normal_vector, point_on_plane = find_best_fit_plane(point_cloud)
    return project_and_rotate_points(point_cloud, normal_vector, point_on_plane)

def make_2D_projection_scatter_plot(out_file, point_cloud, color=None, s=7.5):
    projected_points = project_points_to_plane(point_cloud)
    
    plt.figure()
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=s, c=color, cmap="gray")
    plt.colorbar()
    plt.savefig(out_file)


def get_sample_point_cloud(fraction=None):
    from membrain_pick.dataloading.diffusionnet_dataset import MemSegDiffusionNetDataset
    ds = MemSegDiffusionNetDataset(
        csv_folder="/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data",
        load_only_sampled_points=fraction,
        mesh_data=True,
        overfit_mb=(True if not fraction else False),
        overfit=(True if fraction else False)
    )
    return ds


def test():
    from matplotlib import pyplot as plt
    from membrain_pick.dataloading.diffusionnet_dataset import MemSegDiffusionNetDataset
    ds = get_sample_point_cloud(fraction=1000)
    sample_idx = 2
    membrane_data = ds[sample_idx]["membrane"][0]
    labels = ds[sample_idx]["label"][0]

    mask = labels <= 10.
    membrane_data = membrane_data[mask]
    labels = labels[mask]

    vertices = membrane_data[:, :3]
    normal_vector, point_in_plane = find_best_fit_plane(vertices)
    projected_points = project_and_rotate_points(vertices, normal_vector, point_in_plane)
    out_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_tests/projection_test.png"
    plt.figure()
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=150, c=membrane_data[:,9], cmap="gray")
    plt.savefig(out_file)

if __name__ == "__main__":
    test()