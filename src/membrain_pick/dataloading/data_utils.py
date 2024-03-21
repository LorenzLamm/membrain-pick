import pandas as pd
import numpy as np
import vtk


def get_csv_data(csv_path, delimiter=",", with_header=False, return_header=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path, delimiter=delimiter)
    
    # If the user wants the header back
    if return_header:
        header = df.columns.values
        data = df.values
        return (data, header) if with_header else (np.vstack([header, data]), None)
    
    # Return the data as a numpy array, with or without the header
    return df.values


def store_array_in_csv(out_file, data, out_del=","):
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(data)
    
    # Store the DataFrame in a CSV file
    df.to_csv(out_file, index=False, header=False, sep=out_del)


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