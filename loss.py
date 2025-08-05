# New loss functions.
# Author: Daniel Fu
# Date created: 8/7/2024

import torch
from torch.nn.functional import relu
import networkx as nx

def viewlossfcn(vertices_ndc, mask=None):
    # vertices_endpoints = vertices_ndc[:,[0,-1],:]
    if mask is not None:
        return torch.sum((relu(vertices_ndc-1) + relu(-1*vertices_ndc)) * mask)
    else:
        return torch.sum(relu(vertices_ndc-1) + relu(-1*vertices_ndc))

def MSTlossfcn(initcurves):
    # create "startpoints" and "endpoints", each tensors of n (curves) x 3 (dimensions)
    startpoints = initcurves[:,0,:]
    endpoints = initcurves[:,-1,:]
    distances = torch.stack((torch.cdist(startpoints, startpoints),
                             torch.cdist(startpoints, endpoints),
                             torch.cdist(endpoints, endpoints)), dim=-1) # n x n x 3
    min_distances = torch.min(distances, dim=-1)[0]
    min_dist_matrix = min_distances.detach().cpu().numpy() # n x n
    g = nx.Graph(min_dist_matrix)
    mst = nx.minimum_spanning_edges(g, algorithm="prim")
    mst_weight = sum([min_distances[i][j] for i, j, _ in mst])
    return mst_weight


if __name__ == "__main__":
    # # Test functions
    # for x in [
    #     [1, 0, 2],
    #     [0, 0, 1],
    #     [0.5, 1, -0.5],
    #     [[0, -0.1, 2], [1, 0.9, 0.2]]
    # ]:
    #     print(x, "-->" ,viewlossfcn(torch.tensor(x)).item())
    test = torch.tensor([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ],
        [
            [1, 0, 0.5],
            [0, 0, 0],
            [0, 0.5, 0],
            [0, 0.9, 0]
        ],
        [
            [0.5, 0.1, 0],
            [0, 0, 0],
            [0, 0.5, 0],
            [0.1, 0.5, 0.2]
        ]
    ])
    print(MSTlossfcn(test))