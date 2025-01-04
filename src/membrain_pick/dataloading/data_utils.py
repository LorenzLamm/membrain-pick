import lxml.etree as ET
import pandas as pd
import numpy as np
import vtk
import starfile
import h5py
import torch


def convert_to_torch(data_dict: dict) -> dict:
    """
    Converts all numpy arrays in a dictionary to torch tensors.

    Parameters
    ----------
    data_dict : dict
        A dictionary containing numpy arrays.

    Returns
    -------
    dict
        A dictionary containing torch tensors.
    """
    out_dict = {}
    for key in data_dict:
        if isinstance(data_dict[key], np.ndarray):
            out_dict[key] = torch.from_numpy(data_dict[key]).float()
        else:
            out_dict[key] = data_dict[key]
    return out_dict


def get_csv_data(
    csv_path: str,
    delimiter: str = ",",
    with_header: bool = False,
    return_header: bool = False,
) -> np.ndarray:
    """
    Reads CSV data into a numpy array.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    delimiter : str, optional
        Delimiter used in the CSV file, by default ",".
    with_header : bool, optional
        Whether to include the header in the returned data, by default False.
    return_header : bool, optional
        Whether to return the header separately, by default False.

    Returns
    -------
    np.ndarray
        The data from the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path, delimiter=delimiter)

    # If the user wants the header back
    if return_header:
        header = df.columns.values
        data = df.values
        return (data, header) if with_header else (np.vstack([header, data]), None)

    # Return the data as a numpy array, with or without the header
    return df.values


def store_array_in_csv(
    out_file: str, data: np.ndarray, out_del: str = ",", header: bool = False
) -> None:
    """
    Stores a numpy array in a CSV file.

    Parameters
    ----------
    out_file : str
        Path to the output CSV file.
    data : np.ndarray
        The data to store.
    out_del : str, optional
        Delimiter to use in the CSV file, by default ",".
    header : bool, optional
        Whether to include a header in the CSV file, by default False.

    Returns
    -------
    None
    """
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(data)

    # Store the DataFrame in a CSV file
    df.to_csv(out_file, index=False, header=header, sep=out_del)


def store_array_in_star(out_file: str, data: np.ndarray, header: list = None) -> None:
    """
    Stores a numpy array in a STAR file.

    Parameters
    ----------
    out_file : str
        Path to the output STAR file.
    data : np.ndarray
        The data to store.
    header : list, optional
        The header for the STAR file, by default ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"].

    Returns
    -------
    None
    """
    header = (
        header
        if header is not None
        else ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    )
    df = pd.DataFrame(data, columns=header)
    starfile.write(df, out_file)


def read_star_file(star_file):
    """
    Reads data from a STAR file.

    Parameters
    ----------
    star_file : str
        Path to the STAR file.

    Returns
    -------
    pd.DataFrame
        The data from the STAR file.
    """
    return starfile.read(star_file)


def store_array_in_npy(out_file, data):
    """
    Stores a numpy array in an NPY file.

    Parameters
    ----------
    out_file : str
        Path to the output NPY file.
    data : np.ndarray
        The data to store.

    Returns
    -------
    None
    """
    # Save the numpy array in an npy file
    np.save(out_file, data)


def store_mesh_in_hdf5(out_file: str, points: np.ndarray, faces: np.ndarray, **kwargs):
    """
    Store mesh data in an HDF5 file.

    The points and vertices will be stored in separate hdf5 datasets.
    Each kwargs key will be stored as a separate dataset.

    Parameters
    ----------
    out_file : str
        The path to the output HDF5 file.
    points : np.ndarray
        The points data.
    faces : np.ndarray
        The faces data.
    kwargs : dict
        Additional data to store in the HDF5 file.

    Returns
    -------
    None
    """

    with h5py.File(out_file, "w") as f:
        f.create_dataset("points", data=points)
        f.create_dataset("faces", data=faces)
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, str):
                    # Convert string to numpy array of variable-length UTF-8 strings
                    dt = h5py.string_dtype(encoding="utf-8")
                    f.create_dataset(key, data=np.array(value, dtype=dt))
                else:
                    f.create_dataset(key, data=value)


def load_mesh_from_hdf5(in_file: str):
    """
    Load mesh data from an HDF5 file.

    Parameters
    ----------
    in_file : str
        The path to the input HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the mesh data.
    """
    mesh_data = {}
    with h5py.File(in_file, "r") as f:
        for key in f.keys():
            mesh_data[key] = f[key][()]
    return mesh_data


def store_point_and_vectors_in_vtp(
    out_path: str,
    in_points: np.ndarray,
    in_vectors: np.ndarray = None,
    in_scalars: np.ndarray = None,
):
    """
    Store points and, optionally, their associated vectors into a VTP file.

    This function takes an array of points and an optional array of vectors
    corresponding to each point and stores them in a VTK PolyData format,
    which is then written to a VTP file.

    Parameters
    ----------
    out_path : str
        The path to the desired output VTP file.
    in_points : np.ndarray
        A Numpy array of points where each point is represented as [x, y, z].
        Shape should be (n_points, 3).
    in_vectors : np.ndarray, optional
        A Numpy array of vectors associated with each point, typically representing
        normals or other vector data. Shape should be (n_points, 3). If not provided,
        only point data is written to the VTP file.
    in_scalars : np.ndarray, optional
        A Numpy array of scalars associated with each point. Shape should be (n_points,).
        If not provided, only point and optional vector data are written to the VTP file.

    Returns
    -------
    None
        This function does not return a value. It writes directly to the specified file.

    Raises
    ------
    IOError
        If there is an error writing the file, an error message is printed.
    """
    points = vtk.vtkPoints()
    for point in in_points:
        points.InsertNextPoint(point[0], point[1], point[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    if in_vectors is not None:
        vectors = vtk.vtkDoubleArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("Normal")
        for vector in in_vectors:
            vectors.InsertNextTuple(vector)
        polydata.GetPointData().AddArray(vectors)
        polydata.GetPointData().SetActiveVectors(vectors.GetName())

    if in_scalars is not None:
        if not isinstance(in_scalars, list) and not isinstance(in_scalars, tuple):
            in_scalars = [in_scalars]
        for i, cur_scalars in enumerate(in_scalars):
            scalars = vtk.vtkFloatArray()
            scalars.SetName("Scalars%d" % i)
            for scalar in cur_scalars:
                scalars.InsertNextValue(scalar)
                polydata.GetPointData().AddArray(scalars)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        error_msg = "Error writing the file"
        print(error_msg)


def read_GT_data_membranorama_xml(
    gt_file_name: str, return_orientation: bool = False
) -> dict:
    """
    Reads ground truth data from a Membranorama XML file.

    Parameters
    ----------
    gt_file_name : str
        Path to the XML file.
    return_orientation : bool, optional
        Whether to return orientation data, by default False.

    Returns
    -------
    dict
        A dictionary containing the position data, and optionally the orientation data.
    """
    pos_dict = {}
    orientation_dict = {}
    tree = ET.parse(gt_file_name)
    root = tree.getroot()
    for i, elem in enumerate(root):
        if elem.tag == "PointGroups":
            coords_id = i
            break
    point_groups = root[coords_id]
    for particle_group in point_groups:
        positions = np.zeros((0, 3))
        orientations = np.zeros((0, 3))
        for point in particle_group:
            cur_pos = np.expand_dims(
                np.array(point.attrib["Position"].split(","), dtype=float), 0
            )
            positions = np.concatenate((positions, cur_pos), 0)
            if return_orientation:
                cur_orientation = np.expand_dims(
                    np.array(point.attrib["Orientation"].split(","), dtype=float), 0
                )
                orientations = np.concatenate((orientations, cur_orientation), 0)

        store_token = particle_group.attrib["Name"]
        if store_token not in pos_dict.keys():
            pos_dict[store_token] = np.zeros((0, 3))
        pos_dict[store_token] = np.concatenate(
            (pos_dict[store_token], positions), axis=0
        )

        if return_orientation:
            orientations = np.rad2deg(orientations)
            if store_token not in orientation_dict.keys():
                orientation_dict[store_token] = np.zeros((0, 3))
            orientation_dict[store_token] = np.concatenate(
                (orientation_dict[store_token], orientations), axis=0
            )

    if return_orientation:
        return pos_dict, orientation_dict
    return pos_dict
