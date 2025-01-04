import numpy as np
from scipy.ndimage import label
from membrain_pick.mesh_projections.bbox_utils import get_expanded_bounding_box, crop_array_with_bounding_box


def remove_unused_vertices(points, faces, point_normals):
    used_vertices = np.unique(faces.flatten())
    # Create new points and normals arrays using the used vertices
    new_points = points[used_vertices]
    new_normals = point_normals[used_vertices]
    
    # Create a mapping from old vertex indices to new indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(used_vertices)}
    
    # Apply the mapping to the faces to update their vertex indices
    new_faces = np.vectorize(index_mapping.get)(faces)
    return new_points, new_faces, new_normals

def get_connected_components(seg, only_largest=True):
    seg = seg > 0
    seg, _ = label(seg)
    if only_largest:
        seg = seg == np.argmax(np.bincount(seg.flat)[1:]) + 1
    else:
        return seg
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


def assign_vertex_normals_from_face_normals(verts, faces, face_normals):
    """
    Assigns a normal to each vertex based on the average of the face normals that share the vertex
    """
    
    # Create an array to hold the sum of normals for each vertex
    vertex_normals = np.zeros_like(verts, dtype=float)
    
    # Create an array to count the number of faces each vertex is part of
    counts = np.zeros(verts.shape[0], dtype=int)
    
    # Add face normals to the vertex normals
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += face_normals[i]
            counts[vertex] += 1
    
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    
    # Compute the average by dividing by the number of faces
    vertex_normals /= counts[:, np.newaxis]

    # Normalize the normals
    vertex_normals = normalize(vertex_normals)
    
    return vertex_normals

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def get_normals_from_face_order(mesh, return_face_normals=False):
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
    if return_face_normals:
        return points, faces, mesh_normals
    
    vert_normals = assign_vertex_normals_from_face_normals(points, faces, mesh_normals)
    return points, faces, vert_normals