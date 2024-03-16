import os
import numpy as np

from membrain_seg.segmentation.dataloading.data_utils import load_tomogram, store_tomogram, store_array_in_csv, store_point_and_vectors_in_vtp
from membrain_seg.tomo_preprocessing.pixel_size_matching.match_pixel_size import match_pixel_size
from membrain_seg.tomo_preprocessing.pixel_size_matching.match_pixel_size_seg import match_segmentation_pixel_size_to_tomo
from membrain_pick.compute_mesh_projection import convert_seg_to_evenly_spaced_mesh, compute_values_along_normals
from membrain_pick.mesh_class import Mesh

from scipy.ndimage import label

from membrain_pick.bbox_utils import get_expanded_bounding_box, crop_array_with_bounding_box


def get_connected_components(seg, only_largest=True):
    seg = seg > 0
    seg, _ = label(seg)
    if only_largest:
        print(f"Only using the largest connected component in {mb_key} (found {seg.max()} components)") 
        seg = seg == np.argmax(np.bincount(seg.flat)[1:]) + 1
    else:
        print(f"Found {seg.max()} connected components in {mb_key}")
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
    return np.cross(vec_A, vec_B, dim=-1)

def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)

def norm(x):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return np.linalg.norm(x, dim=len(x.shape) - 1)

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



def process_mb_file(mb_file, 
                    tomo_file, 
                    out_folder, 
                    out_file=None, 
                    tomo=None, 
                    step_numbers=(-6, 7),
                    step_size=0.25,
                    mesh_smoothing=1000,
                    barycentric_area=1.0,
                    recompute_matching=False, 
                    match_size_flag=False, 
                    input_pixel_size=None, 
                    output_pixel_size=None, 
                    crop_box_flag=False,
                    only_largest_component=True, 
                    min_connected_size=1e4):
    print(f"Processing {mb_file}")
    mb_key = os.path.basename(mb_file).split(".")[0]
    seg = load_tomogram(mb_file).data

    seg = get_connected_components(seg, only_largest=only_largest_component)

    if tomo is None:
        print(f"Loading tomo {tomo_file}")
        tomo = load_tomogram(tomo_file).data

    sub_seg_count = 0
    for k in range(1, seg.max() + 1):
        if np.sum(seg == k) < min_connected_size:
            continue
        print(f"Processing sub-seg {k} of {mb_key}. Total seg count {seg.max()}")
        sub_seg_count += 1
        cur_mb_key = mb_key + f"_{sub_seg_count}"

        if crop_box_flag:
            cur_seg, cur_tomo = get_cropped_arrays(seg, tomo)
        else:
            cur_seg = seg == k
            cur_tomo = tomo

        mesh = convert_seg_to_evenly_spaced_mesh(seg=cur_seg,
                                                 smoothing=mesh_smoothing,
                                                 barycentric_area=barycentric_area)
        
        points, faces, point_normals = get_normals_from_face_order(mesh)

        normal_values = compute_values_along_normals(mesh=mesh, 
                                                     tomo=cur_tomo, 
                                                     steps=step_numbers, 
                                                     step_size=step_size, 
                                                     verts=points,
                                                     normals=point_normals)
        print("Computed values along normals")

        out_data = np.concatenate([mesh.points, normal_values], axis=1)
        out_file = os.path.join(out_folder, cur_mb_key + "_mesh_data.csv")
        out_file_pos = os.path.join(out_folder, cur_mb_key + "_psii_pos.csv")
        out_file_faces = os.path.join(out_folder, cur_mb_key + "_mesh_faces.csv")
        out_file_normals = os.path.join(out_folder, cur_mb_key + "_mesh_normals.csv")

mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/thylakoids/segs"
mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Sofie/subtomo/FaRLiP/FRL_PBS/seg/221024_FRL_tomo03/"
mb_folder = "/scicore/home/engel0006/GROUP/pool-visprot/Sofie/PSII_DragonFly/picking/drgnfly_picks/deepfinder/seg/"
mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/Zunlong_Covid/segmentations"
mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/221123_ETH_Krios2/Thala_WT/Amira/Tomo_02/"
# mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/4Lorenz/Tomo_1/membranes/"
# mb_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/pipeline_spinach/tomograms/Tomo_17/membranes"
# mb_folder = "/scicore/home/engel0006/GROUP/pool-visprot/Florent/Membrain_seg/Out/"
tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/data_Sofie/thylakoids/634.mrc"
tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/Zunlong_Covid/TS_004_dose-filt_lp50_bin8.rec"
tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/221123_ETH_Krios2/Thala_WT/tomo_02/tomo_02_cryocare.rec"
# tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/4Lorenz/Tomo_1/Tomo1L1_bin4.rec"
# tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/pipeline_spinach/tomograms/Tomo_17/tomo17_bin4_denoised.mrc"
# tomo_file = "/scicore/home/engel0006/GROUP/pool-visprot/Common/bin4_cryocare/179.mrc"
# tomo_file = "/scicore/home/engel0006/GROUP/pool-engel/Sofie/subtomo/FaRLiP/FRL_PBS/bin4_cryocare/3.rec"
out_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data/Manon_Pyshell"
temp_folder = "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/2D_projections/mesh_data/temp"
os.makedirs(out_folder, exist_ok=True)

match_size_flag = False
recompute_matching = False

mb_files = []
for filename in os.listdir(mb_folder):
    # if "179_Mito_labels" not in filename:
    #         continue
    if not filename.endswith(".mrc"):
        continue
    mb_file = os.path.join(mb_folder, filename)
    mb_files.append(mb_file)

def crop_file(filename, out_file):
    print(f"Cropping {filename}")
    tomo = load_tomogram(filename).data
    bbox = get_expanded_bounding_box(tomo > 0, 20)
    cropped_tomo = crop_array_with_bounding_box(tomo, bbox)
    store_tomogram(out_file, cropped_tomo)

if match_size_flag:
    tomo_file_out = os.path.join(temp_folder, "tomo_matched.mrc")
    print(f"Matching pixel size of {tomo_file} to 14.08 A")
    if not os.path.exists(tomo_file_out) or recompute_matching:
        match_pixel_size(
            input_tomogram=tomo_file,
            output_path=tomo_file_out,
            pixel_size_out=14.08,
            pixel_size_in=None,
            disable_smooth=False
        )
    # crop_file(tomo_file_out, tomo_file_out)
    tomo_file = tomo_file_out
    new_mb_files = []
    for mb_file in mb_files:
        print(f"Matching pixel size of {mb_file} to {tomo_file}")
        mb_out = os.path.join(temp_folder, os.path.basename(mb_file))
        new_mb_files.append(mb_out)
        if not os.path.exists(mb_out) or recompute_matching:
            match_segmentation_pixel_size_to_tomo(
                seg_path=mb_file,
                orig_tomo_path=tomo_file,
                output_path=mb_out
            )
            # crop_file(mb_out, mb_out)

    mb_files = new_mb_files

print(f"Loading tomo {tomo_file}")
tomo = load_tomogram(tomo_file).data

for mb_file in mb_files:
    
    # if "TS_004_" not in mb_file:
    #     continue
    print(f"Processing {mb_file}")
    # if "221024_Ctherm_FRL_tomo_03_cut_all_components_29.mrc" not in mb_file:
    #     continue
    if "_all.mrc" in mb_file or "_dirty_seg" in mb_file:
        continue
    mb_key = os.path.basename(mb_file).split(".")[0]
    seg = load_tomogram(mb_file).data

    # get connceted components
    seg = seg > 0
    seg, _ = label(seg)
    
    print(f"Found {seg.max()} connected components in {mb_key}")

    sub_seg_count = 0
    for k in range(1, seg.max() + 1):
        if np.sum(seg == k) < 10000:
            continue
        print(f"Processing sub-seg {k} of {mb_key}. Total seg count {seg.max()}")
        sub_seg_count += 1
        mb_key = mb_key + f"_{sub_seg_count}"

        bbox = get_expanded_bounding_box(seg == k, 20)
        cur_seg = crop_array_with_bounding_box(seg, bbox)
        cur_tomo = crop_array_with_bounding_box(tomo, bbox)

        mesh = convert_seg_to_evenly_spaced_mesh(seg=cur_seg,
                            smoothing=2000,
                            barycentric_area=1.5)
        print("Converted to mesh")
        mesh.compute_normals(point_normals=True)
        print("Computed normals")
        normal_values = compute_values_along_normals(mesh=mesh, tomo=cur_tomo, steps=(0, 10), step_size=0.5, normals=mesh.point_normals * -1)
        print("Computed values along normals")
        faces = mesh.faces
        faces = np.reshape(faces, (-1, 4))
        faces = faces[:, 1:]
        print(faces.shape, "<- faces shape")


        out_data = np.concatenate([mesh.points, normal_values], axis=1)
        out_file = os.path.join(out_folder, mb_key + "_mesh_data.csv")
        out_file_pos = os.path.join(out_folder, mb_key + "_psii_pos.csv")
        out_file_faces = os.path.join(out_folder, mb_key + "_mesh_faces.csv")
        out_file_normals = os.path.join(out_folder, mb_key + "_mesh_normals.csv")
        out_file_normals_vtp = os.path.join(out_folder, mb_key + "_mesh_normals.vtp")
        store_array_in_csv(out_file, out_data)
        store_array_in_csv(out_file_faces, faces)
        store_array_in_csv(out_file_normals, mesh.point_normals*(-1))

        store_point_and_vectors_in_vtp(out_file_normals_vtp, mesh.points, mesh.point_normals, in_scalars=[normal_values[:, k] for k in range(normal_values.shape[1])])

        mesh = Mesh(vertices=mesh.points, triangle_combos=faces+1)
        mesh.store_in_file(out_file.replace(".csv", ".obj"))
        store_tomogram(out_file.replace(".csv", ".mrc"), cur_tomo)
        store_tomogram(out_file.replace(".csv", "_seg.mrc"), cur_seg)
        print("HI")
