import torch
import pytorch3d
from pytorch3d.io import IO, load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
objdir = "../data/textured_models/spot/spot_normalized.obj"
nviews = 10
ncurves = 6
verts, faces, aux = pytorch3d.io.load_obj(objdir)

# Load textures
import torchvision
texturedir = "../data/textured_models/spot/spot_texture.png"
textureimg = torchvision.io.read_image(texturedir).permute(1, 2, 0).float() / 255.0

textures = pytorch3d.renderer.mesh.textures.TexturesUV([textureimg], faces.textures_idx.unsqueeze(0), aux.verts_uvs.unsqueeze(0))

# Create mesh
mesh = pytorch3d.structures.Meshes([verts], [faces.verts_idx], textures).to(device)

import os
objname = os.path.basename(objdir).split(".")[0]

### Normalize mesh
center = mesh.verts_packed().mean(0)
mesh = mesh.offset_verts(-center)
maxnorm = torch.linalg.vector_norm(mesh.verts_packed(), dim=-1).max()
mesh = mesh.scale_verts(1/maxnorm.item())

from feature_backprojection import features_from_views, DINOWrapper
from utils.geometry import pairwise_geodesic_distances_mesh
from utils.rendering import setup_renderer, sample_view_points
import torch

# Compute features
renderer = setup_renderer(device)
model = DINOWrapper(device, small=True)
render_dist = 2.2
views = sample_view_points(render_dist, nviews)
geo_dists = pairwise_geodesic_distances_mesh(mesh.verts_packed().to("cpu"), mesh.faces_packed().to("cpu"))
features = features_from_views(
    renderer=renderer, model=model, mesh=mesh, views=views, render_dist=render_dist, batch_size=8,
    device=device, geo_dists=geo_dists, gaussian_sigma=0.001, only_visible=True,
    savedir="../scratch",
)

## Kmeans clustering for sparse keypoint detection
from sklearn.cluster import KMeans
import numpy as np

n_clusters = 30
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features.cpu().numpy())
keypoints = torch.tensor(kmeans.cluster_centers_).to(device)

# Get closest vertex to each cluster center
labels = kmeans.labels_
closest_vertex_idx = []
for i in range(n_clusters):
    cluster_verts = mesh.verts_packed()[labels == i]
    cluster_verts_idxs = np.where(labels == i)[0]
    center = cluster_verts.mean(dim=0).to(device)
    dists = torch.norm(cluster_verts - center[None], dim=-1)
    closest_vertex_idx.append(cluster_verts_idxs[torch.argmin(dists).item()])

# Save the detected keypoints
keypoints = mesh.verts_packed()[closest_vertex_idx]
torch.save(keypoints, f"../data/{objname}_keypoints.pt")

from sklearn.decomposition import PCA
from pytorch3d.vis.plotly_vis import plot_scene

# Perform PCA for visualization
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features.cpu().numpy())
features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0) - features_pca.min(axis=0))
mesh.textures = TexturesVertex(verts_features=torch.tensor(features_pca, dtype=torch.float32)[None].to(device))

# Plot mesh with PCA features and keypoints as pointcloud
from pytorch3d.structures import Pointclouds

keypoints = mesh.verts_packed()[closest_vertex_idx]
kp_pc = Pointclouds(points = [keypoints])

plot_scene({
    "mesh": {
        "mesh": mesh
        },
    "Pointcloud": {
        "keypoints": kp_pc,
    }
})


# TODO: keypoint spanning path initialization (heat geodesic based nearest neighbors -- same as FPS)
from igl import heat_geodesic
import copy
from meshing.mesh import Mesh
from meshing.io import PolygonSoup

soup = PolygonSoup.from_obj(objdir)
hemesh = Mesh(soup.vertices, soup.indices)

curves = []
tmp_sampled_points = np.array(closest_vertex_idx)
for _ in range(ncurves):
    source = np.random.choice(tmp_sampled_points)
    distances = heat_geodesic(hemesh.vertices, hemesh.faces, 1e-3, np.array([source]))[tmp_sampled_points]
    nn3 = tmp_sampled_points[np.argsort(distances)[1:4]]
    curves.append(hemesh.vertices[[source] + list(nn3)])

    tmp_sampled_points = np.array( [p for p in tmp_sampled_points if p not in list(nn3) + [source]])

initcurves = np.stack(curves)
initcurves = torch.from_numpy(initcurves).to(device)


# TODO: visualize the initialized curves
from optimize import render_curves
nviews = 10
elev = torch.tensor([30 * np.pi / 180] * nviews).to(device)
azim = torch.linspace(0, 360, nviews) * np.pi / 180

imgs = render_curves(initcurves, elev, azim, nviews, ncurves, 4, 400, 400, 0, r=5.5, fov=np.pi * 60 / 180).cpu().detach()

# Save
from pathlib import Path
objparentdir = os.path.dirname(objdir)
initsavedir = os.path.join(objparentdir, "kpinit")
Path(initsavedir).mkdir(parents=True, exist_ok=True)

def clear_directory(path):
    import shutil
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

clear_directory(initsavedir)

import torchvision
import sys
sys.path.append("..")
from make_gifs import compare_imgs, make_gif

for viewi in range(nviews):
    pred = torchvision.transforms.functional.to_pil_image(imgs[viewi, :3,])
    compare_imgs([pred], ["Init"]).save(os.path.join(initsavedir, f"{viewi:03d}.png"))

# Make gifs
make_gif(f"{initsavedir}/*.png", os.path.join(objparentdir, "kpinit.gif"))