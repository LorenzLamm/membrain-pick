import os
from typing import List

from typer import Option
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="surforama", no_args_is_help=True)
def surforama(
    h5_path: str = Option(  # noqa: B008
        ...,
        help="Path to the h5 container.",
        **PKWARGS,
    ),
    tomogram_path: str = Option(  # noqa: B008
        default="",
        help="Path to the tomogram to be projected",
    ),
    normal_offset: float = Option(  # noqa: B008
        0.0, help="Offset for the normal vectors."
    ),
    point_size: float = Option(5.0, help="Size of the points."),  # noqa: B008
    return_viewer: bool = Option(  # noqa: B008
        False, help="Should the viewer be returned?"
    ),
):
    """Visualize mesh containers in surforama.

    Example
    -------
    membrain_pick surforama --h5-path <path-to-your-h5-container>
    """
    import os
    import napari
    from membrain_pick.dataloading.data_utils import load_mesh_from_hdf5
    from membrain_pick.napari_utils.surforama_cli_utils import (
        display_tomo,
        get_pixel_size,
        get_points_and_faces,
        display_scores,
        display_cluster_centers,
        display_cluster_centers_as_points,
        initialize_surforama_widget,
        display_surforama_without_widget,
        display_input_normal_values,
    )

    viewer = napari.Viewer(ndisplay=3)

    mesh_files = None
    value_range = None
    if os.path.isdir(h5_path):
        mesh_files = [
            os.path.join(h5_path, f) for f in os.listdir(h5_path) if f.endswith(".h5")
        ]
    else:
        mesh_files = [h5_path]

    for h5_nr, h5_path in enumerate(mesh_files):
        mesh_data = load_mesh_from_hdf5(h5_path)

        if h5_nr == 0:
            volume_layer = display_tomo(viewer, mesh_data, tomogram_path)
            pixel_size = get_pixel_size(mesh_data, None)

        points, faces = get_points_and_faces(mesh_data, pixel_size)
        display_scores(viewer, mesh_data, points, faces)

        if h5_nr == 0:
            surforama_widget = initialize_surforama_widget(
                points, faces, volume_layer, viewer, normal_offset=normal_offset
            )
            display_cluster_centers(
                viewer, mesh_data, pixel_size, surforama_widget, point_size=point_size
            )
        else:
            value_range = display_surforama_without_widget(
                viewer, points, faces, value_range, normal_offset=normal_offset
            )
            display_cluster_centers_as_points(
                viewer, mesh_data, pixel_size, point_size=point_size
            )

        if h5_nr == 0:
            display_input_normal_values(viewer, mesh_data, points, faces)
    if return_viewer:
        return viewer
    napari.run()
