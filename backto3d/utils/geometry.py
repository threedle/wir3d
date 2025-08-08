from collections import defaultdict

import numpy as np
import potpourri3d as pp3d
import torch


def pairwise_geodesic_distances_mesh(verts, faces):
    """
    Pairwise geodesic distances of vertices of a mesh.

    :param verts: Vertices of mesh  (n x 3)
    :param faces: Faces of mesh (m x 3)
    :return: numpy array of size (n x n)
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(verts.numpy(), faces.numpy())
    geo_dists = np.zeros((verts.shape[0], verts.shape[0]))
    for i in range(verts.shape[0]):
        geo_dists[i, :] = solver.compute_distance(i)
    return geo_dists


def pairwise_geodesic_distances(points):
    """
    Pairwise geodesic distances of a set of points.

    :param points: Points (n x 3)
    :return: numpy array of size (n x n)
    """
    solver = pp3d.PointCloudHeatSolver(points.numpy())
    geo_dists = np.zeros((points.shape[0], points.shape[0]))
    for i in range(points.shape[0]):
        geo_dists[i, :] = solver.compute_distance(i)
    return geo_dists


def find_adjacent_faces(faces):
    """
    Find adjacent faces of faces in a mesh.

    :param faces: Faces of mesh (m x 3)
    :return: torch tensor of size (m x 3) with indices of adjacent faces, -1 if not 3 adjacent faces
    """
    device = faces.device
    faces = faces.tolist()
    # Mapping from edges to faces
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces map
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            edge_to_faces[edge].append(i)

    # Find adjacent faces
    adjacent_faces = []
    max_adjacent_faces = 0
    for i, face in enumerate(faces):
        neighbors = set()
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            for neighbor_face in edge_to_faces[edge]:
                if neighbor_face != i:
                    neighbors.add(neighbor_face)
        adjacent_faces.append(list(neighbors))
        max_adjacent_faces = max(max_adjacent_faces, len(neighbors))

    ret = torch.zeros(len(faces), max_adjacent_faces, dtype=torch.long, device=device) - 1
    for i, neighbors in enumerate(adjacent_faces):
        ret[i, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long, device=device)
    return ret
