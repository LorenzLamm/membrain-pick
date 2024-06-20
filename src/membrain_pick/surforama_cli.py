"""Copied from https://github.com/teamtomo/fidder/blob/main/src/fidder/_cli.py."""

import typer
from click import Context
from typer.core import TyperGroup
from typing import List
from typer import Option

class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    MemBrain-pick's data conversion / mesh projection module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    membrain-pick --help

    -------

    Example:
    -------
    membrain-pick process-folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> 
        --output-folder <path-to-store-meshes>

    -------
    """

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

    point_io.surface_picker.normal_vectors_layer.data = normal_data
    point_io.surface_picker.up_vectors_layer.data = up_data

    point_io.surface_picker.normal_vectors_layer.edge_color = "purple"
    point_io.surface_picker.up_vectors_layer.edge_color = "orange"

    point_io.surface_picker.rotations = features_table[ROTATION]
    point_io.surface_picker.up_vectors = up_data[:, 1, :]
    point_io.surface_picker.normal_vectors = normal_data[:, 1, :]


@cli.command(name="surforama", no_args_is_help=True)
def surforama(
    h5_path: str = Option(  # noqa: B008
        ...,
        help="Path to the h5 container.",
        **PKWARGS,
    ),
    tomogram_path: str = Option(  # noqa: B008
        default="", help="Path to the tomogram to be projected", 
    ),
):
    import napari
    from surforama.app import QtSurforama
    import numpy as np
    from matplotlib.cm import get_cmap
    from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5
    from membrain_seg.segmentation.dataloading.data_utils import load_tomogram

    viewer = napari.Viewer(ndisplay=3)


    mesh_data = load_mesh_from_hdf5(h5_path)


    if "tomo_file" in mesh_data.keys() and tomogram_path == "":
        tomogram_path = mesh_data["tomo_file"]
        if isinstance(tomogram_path, bytes):
            tomogram_path = tomogram_path.decode("utf-8")
        
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
        volume_layer = viewer.add_image(tomogram,
                                        name="tomogram",
                                        depiction="plane",
                                        blending="translucent",
                                        plane=plane_properties,)
    if "pixel_size" in mesh_data.keys():
        pixel_size = mesh_data["pixel_size"]

    if pixel_size is None:
        raise ValueError("Pixel size not found in the mesh data or the tomogram.")

    points = mesh_data["points"] / pixel_size
    points = np.stack(points[:, [2, 1, 0]])

    faces = mesh_data["faces"]
    
    scores, labels, cluster_centers = None, None, None
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
    if "labels" in mesh_data.keys():
        labels = mesh_data["labels"]
    if "cluster_centers" in mesh_data.keys():
        cluster_centers = mesh_data["cluster_centers"] / pixel_size
        cluster_centers = np.stack(cluster_centers[:, [2, 1, 0]])
    
    surface_layer_surf = viewer.add_surface(
        (points, faces), name="Surfogram", shading="none"
    )

    surforama_widget = QtSurforama(viewer,
                                   surface_layer=surface_layer_surf,
                                   volume_layer=volume_layer,)

    if cluster_centers is not None:
        surforama_widget.picking_widget.enabled = True
        point_io = surforama_widget.point_writer_widget
        initialize_points(
            point_io=point_io,
            point_coordinates=cluster_centers,
        )

    viewer.window.add_dock_widget(
        surforama_widget, area="right", name="Surforama"
    )

    napari.run()

