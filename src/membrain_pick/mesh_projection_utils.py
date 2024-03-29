import numpy as np
from scipy.ndimage import label
from membrain_pick.bbox_utils import get_expanded_bounding_box, crop_array_with_bounding_box


def get_connected_components(seg, only_largest=True):
    seg = seg > 0
    seg, _ = label(seg)
    if only_largest:
        print(f"Only using the largest connected component (found {seg.max()} components)") 
        seg = seg == np.argmax(np.bincount(seg.flat)[1:]) + 1
    else:
        print(f"Found {seg.max()} connected components")
    return seg

def get_cropped_arrays(seg, tomo, expansion=20):
    bbox = get_expanded_bounding_box(seg, expansion)
    cur_seg = crop_array_with_bounding_box(seg, bbox)
    cur_tomo = crop_array_with_bounding_box(tomo, bbox)
    return cur_seg, cur_tomo


def face_coords(verts, faces):
    coords = verts[faces]
    return coords
def cross(vec_A, vec_B):
    return np.cross(vec_A, vec_B, axis=-1)

def normalize(x, divide_eps=1e-6):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if x.shape[-1] > 4:
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / np.expand_dims((norm(x) + divide_eps), axis=-1)

def norm(x):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return np.linalg.norm(x, axis=len(x.shape) - 1)

def assign_vertex_normals_from_face_normals(verts, faces, face_normals):
    """
    Assigns a normal to each vertex based on the average of the face normals that share the vertex
    """
    vertex_normals = np.zeros(verts.shape, dtype=float)
    for i in range(verts.shape[0]):
        faces_with_vertex = np.where(faces == i)[0]
        vertex_normals[i] = np.mean(face_normals[faces_with_vertex], axis=0)
    return vertex_normals

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def get_normals_from_face_order(mesh):
    """
    Get normals from face order

    This means that the normal is given by the cross product of the vectors from the 
    first vertex to the second and third vertex.
    This seems to be more accurate than the mesh.point_normals

    """
    faces = mesh.faces
    faces = np.reshape(faces, (-1, 4))
    faces = faces[:, 1:].copy()
    points = mesh.points

    # Get normals per triangle and assign back to vertices
    mesh_normals = np.array(face_normals(points, faces))
    vert_normals = assign_vertex_normals_from_face_normals(points, faces, mesh_normals)

    return points, faces, vert_normals