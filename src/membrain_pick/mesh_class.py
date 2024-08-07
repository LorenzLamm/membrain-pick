import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv



class Mesh(object):
    def __init__(self, vertices, triangle_combos):
        from time import time
        time_zero = time()
        self.vertices = vertices
        self.triangles = []
        self.triangle_combos = triangle_combos
        self.compute_triangle_centers()
        self.nn_entity = NearestNeighbors(n_neighbors=1).fit(self.triangle_centers)#.fit(self.triangle_centers)

    def __len__(self):
        return len(self.triangle_combos)

    def _reinit_nn(self):
        self.compute_triangle_centers()
        self.nn_entity = NearestNeighbors(n_neighbors=1).fit(self.triangle_centers)
    
    def compute_triangle_areas(self):
        ## Use Heron's formula
        a_vec = np.linalg.norm(self.vertices[self.triangle_combos[:, 0] - 1] - self.vertices[self.triangle_combos[:, 1] - 1], axis=1)
        b_vec = np.linalg.norm(self.vertices[self.triangle_combos[:, 1] - 1] - self.vertices[self.triangle_combos[:, 2] - 1], axis=1)
        c_vec = np.linalg.norm(self.vertices[self.triangle_combos[:, 2] - 1] - self.vertices[self.triangle_combos[:, 0] - 1], axis=1)
        s_vec = 0.5 * (a_vec + b_vec + c_vec)
        self.triangle_areas = np.sqrt(s_vec * (s_vec - a_vec) * (s_vec - b_vec) * (s_vec - c_vec))

    def compute_triangle_centers(self):
        self.triangle_combos = np.array(self.triangle_combos, dtype=int)
        self.vertices = np.array(self.vertices)
        tri_verts = self.vertices[self.triangle_combos - 1]
        self.triangle_centers = np.mean(tri_verts, axis=1)

    def make_triangles(self, triangle_combos):
        for combo in triangle_combos:
            triangle = Triangle(self.vertices[combo[0] - 1], self.vertices[combo[1] - 1], self.vertices[combo[2] - 1])
            self.triangles.append(triangle)

    def get_triangle(self, id):
        combo = self.triangle_combos[int(id)]
        triangle = Triangle(self.vertices[combo[0] - 1], self.vertices[combo[1] - 1], self.vertices[combo[2] - 1])
        return triangle

    def find_closest_triangle(self, position, return_dist=False):
        distances = np.linalg.norm(self.triangle_centers - position, axis=1)
        if return_dist:
            return np.argmin(distances), np.min(distances)
        else:
            return np.argmin(distances)

    def find_closest_triangles_batch(self, positions):
        # dists = self.query_tree.query(positions)
        dists = self.nn_entity.kneighbors(positions)
        return dists

    def store_in_file(self, out_file):
        with open(out_file, 'w') as out_csv:
            csv_writer = csv.writer(out_csv, delimiter=' ')
            for vert in self.vertices:
                row = ['v', str(vert[0]), str(vert[1]), str(vert[2])]
                csv_writer.writerow(row)
            for combo in self.triangle_combos:
                row = ['f', str(int(combo[0])), str(int(combo[1])), str(int(combo[2]))]
                csv_writer.writerow(row)
            



class Triangle(object):
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.normal = self.compute_normal()
    def compute_normal(self):
        normal = np.cross((self.v2 - self.v1), (self.v3 - self.v1))
        normal = normal / np.linalg.norm(normal)
        return normal
    def get_plane_matrix(self):
        m1 = self.v2 - self.v1
        m1 /= np.linalg.norm(m1)
        m2 = np.cross(self.normal, m1)
        m3 = self.normal
        return np.stack([m1, m2, m3])
    def get_center_position(self):
        return 1.0 / 3 * (self.v1 + self.v2 + self.v3)


def read_obj_file_to_triangles(filename):
    vertex_list = []
    combo_list = []
    with open(filename) as read_obj:
        for row in read_obj:
            if row.startswith('v'):
                coords = row.split(' ')[1:]
                if coords[2].endswith('\n'):
                    coords[2] = coords[2][:-1]
                coords = np.array(coords, dtype=float)
                vertex_list.append(coords)
            elif row.startswith('f'):
                combo = row.split(' ')[1:]
                if combo[2].endswith('\n'):
                    combo[2] = combo[2][:-1]
                combo = np.array(combo, dtype=np.int64)
                combo_list.append(combo)
    mesh = Mesh(vertex_list, combo_list)
    return mesh

