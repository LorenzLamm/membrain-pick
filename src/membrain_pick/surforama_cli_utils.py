
import numpy as np
from matplotlib.pyplot import get_cmap
from membrain_seg.segmentation.dataloading.data_utils import load_tomogram
from surforama.app import QtSurforama
from membrain_pick.scalar_selection import ScalarSelectionWidget

def display_tomo(viewer, mesh_data, tomogram_path):
    if "tomo_file" in mesh_data.keys() and tomogram_path == "":
        tomogram_path = mesh_data["tomo_file"]
        if isinstance(tomogram_path, bytes):
            tomogram_path = tomogram_path.decode("utf-8")
    
    volume_layer = None
    if tomogram_path != "":
        tomogram = load_tomogram(tomogram_path)
        pixel_size = tomogram.voxel_size.x
        tomogram = tomogram.data
        tomogram = np.transpose(tomogram, (2, 1, 0))
        slice_number = tomogram.shape[0] // 2
        plane_properties = {
            'position': (slice_number, tomogram.shape[1] // 2, tomogram.shape[2] // 2),
            'normal': (1, 0, 0),
            'thickness': 1,
        }
        # do not show by default
        volume_layer = viewer.add_image(tomogram,
                                        name="tomogram",
                                        depiction="plane",
                                        blending="translucent",
                                        plane=plane_properties,
                                        visible=False,
                                        )
    return volume_layer

def get_pixel_size(mesh_data, pixel_size):
    pixel_size = None
    if "pixel_size" in mesh_data.keys():
        pixel_size = mesh_data["pixel_size"]
    if pixel_size is None:
        raise ValueError("Pixel size not found in the mesh data.")
    return pixel_size

def get_points_and_faces(mesh_data, pixel_size):
    points = mesh_data["points"] / pixel_size
    points = np.stack(points[:, [2, 1, 0]])
    faces = mesh_data["faces"]
    return points, faces


def display_scores(viewer, mesh_data, points, faces):
    if "scores" in mesh_data.keys():
        scores = mesh_data["scores"]
        normalized_scores = scores / 10.
        normalized_scores[normalized_scores < 0] = 0
        normalized_scores[normalized_scores > 1] = 1
        normalized_scores = 1 - normalized_scores
        cmap = get_cmap('RdBu') 
        colors = cmap(normalized_scores)[:, :3]  # Get RGB values and discard the alpha channel
        surface_layer = viewer.add_surface(
            (points, faces), vertex_colors=colors, name="Scores", shading="none"
        )
    return surface_layer

from surforama.gui.qt_point_io import QtPointIO
from surforama.constants import (
    NAPARI_NORMAL_0,
    NAPARI_NORMAL_1,
    NAPARI_NORMAL_2,
    NAPARI_UP_0,
    NAPARI_UP_1,
    NAPARI_UP_2,
    ROTATION,
)
def initialize_points(point_io, point_coordinates,):
    import numpy as np
    
    normal_data, up_data = (
        point_io._assign_orientations_from_nearest_triangles(
            point_coordinates=point_coordinates
        )
    )
    features_table = {
        NAPARI_NORMAL_0: normal_data[:, 1, 0],
        NAPARI_NORMAL_1: normal_data[:, 1, 1],
        NAPARI_NORMAL_2: normal_data[:, 1, 2],
        NAPARI_UP_0: up_data[:, 1, 0],
        NAPARI_UP_1: up_data[:, 1, 1],
        NAPARI_UP_2: up_data[:, 1, 2],
        ROTATION: np.zeros(normal_data.shape[0]) * 1.0,
    }

    # add the data to the viewer
    point_io.surface_picker.points_layer.data = point_coordinates
    point_io.surface_picker.points_layer.features = features_table
    point_io.surface_picker.points_layer.size = np.array([3] * point_coordinates.shape[0])

    point_io.surface_picker.normal_vectors_layer.data = normal_data
    point_io.surface_picker.up_vectors_layer.data = up_data

    point_io.surface_picker.normal_vectors_layer.edge_color = "purple"
    point_io.surface_picker.up_vectors_layer.edge_color = "orange"

    point_io.surface_picker.rotations = features_table[ROTATION]
    point_io.surface_picker.up_vectors = up_data[:, 1, :]
    point_io.surface_picker.normal_vectors = normal_data[:, 1, :]



def display_cluster_centers(viewer, mesh_data, pixel_size, surforama_widget):
    if "cluster_centers" in mesh_data.keys():
        cluster_centers = mesh_data["cluster_centers"] / pixel_size
        cluster_centers = np.stack(cluster_centers[:, [2, 1, 0]])

        surforama_widget.picking_widget.enabled = True
        point_io = surforama_widget.point_writer_widget
        initialize_points(
            point_io=point_io,
            point_coordinates=cluster_centers,
        )
        surforama_widget.picking_widget.enabled = False

def initialize_surforama_widget(points, faces, volume_layer, viewer):
    surface_layer_surf = viewer.add_surface(
        (points, faces), name="Surfogram", shading="none"
    )
    surforama_widget = QtSurforama(viewer,
        surface_layer=surface_layer_surf,
        volume_layer=volume_layer,)
    viewer.window.add_dock_widget(
        surforama_widget, area="right", name="Surforama"
    )
    return surforama_widget


def display_input_normal_values(viewer, mesh_data, points, faces):
    if "normal_values" in mesh_data.keys():
        normal_values = mesh_data["normal_values"]
        surface_layer_proj = viewer.add_surface(
            (points, faces), name="Projections", shading="none"
        )
        scalar_selection_widget = ScalarSelectionWidget(surface_layer_proj, normal_values)

        viewer.window.add_dock_widget(
            scalar_selection_widget, area="right", name="Scalar Selection"
        )