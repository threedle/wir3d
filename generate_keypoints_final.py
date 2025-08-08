### Generate keypoints using pre-rendered Nerfs ###
import torch
import argparse
from optimize_utils import clear_directory
from torch.utils.data import Dataset, DataLoader

# Custom dataloader for renders
class RenderDataset(Dataset):
    def __init__(self, renders, positions, lookats):

        self.renders = renders
        self.positions = positions
        self.lookats = lookats

    def __len__(self):
        return len(self.renders)

    def __getitem__(self, idx):

        return {'renders': self.renders[idx], 'positions': self.positions[idx], 'lookats': self.lookats[idx]}

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("objdir", type=str)
parser.add_argument("rendername", type=str)
parser.add_argument("--savedir", type=str, default=None)
parser.add_argument("--npoints", type=int, default=30, help='Number of keypoints to generate from clustering')
parser.add_argument("--model", type=str, choices={"DINO", "SAM", "CLIP"}, default="DINO")
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--clustertype", type=str, choices={"kmeans"}, default="kmeans")
parser.add_argument("--weighttype", type=str, choices={"l2", "cosine"}, default="l2")
parser.add_argument("--modelpath", type=str, default=None)
parser.add_argument("--modeltype", type=str, default='ViT-L/14')
parser.add_argument("--overlaydir", type=str, default=None)
parser.add_argument("--zbufferdir", type=str, default=None)
parser.add_argument("--cachedir", type=str, default=None)
parser.add_argument("--savename", type=str, default=None)
parser.add_argument("--centertype", type=str, choices={"euclidean", "latent"}, default="euclidean")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--features_on_cpu", action='store_true')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--normalize", action='store_true')
parser.add_argument("--scale", type=float, default=1.0)

args = parser.parse_args()

# Load mesh
import igl
import numpy as np

meshpath = args.objdir
vertices, vt, n, faces, ftc, _ = igl.read_obj(meshpath)

if args.normalize:
    # Normalize based on bounding box mean
    from igl import bounding_box
    bb_vs, bf = bounding_box(vertices)
    vertices -= np.mean(bb_vs, axis=0)
    vertices /= (np.max(np.linalg.norm(vertices, axis=1)) / args.scale)

import os
from pathlib import Path

objname = os.path.basename(args.objdir).split(".")[0]

savedir = args.savedir
if savedir is None:
    savedir = os.path.dirname(args.objdir)
Path(savedir).mkdir(exist_ok=True, parents=True)

# Setup cache
cachedir = args.cachedir
if cachedir is None:
    cachedir = os.path.join(savedir, f"kp_cache")
Path(cachedir).mkdir(exist_ok=True, parents=True)

if args.overwrite:
    clear_directory(cachedir)

# If keypoints exist, then we're done
savename = args.savename
if savename is None:
    savename = f"keypoints.pt"

keypointsdir = os.path.join(savedir, savename)
if os.path.exists(keypointsdir):
    if args.overwrite:
        os.remove(keypointsdir)
    else:
        print(f"Already done with {keypointsdir}")
        exit(0)

from backto3d.feature_backprojection import DINOWrapper, SAMWrapper, CLIPWrapper
from backto3d.feature_backprojection.backprojection import features_from_renders
import torch

# Compute features
if args.model == "DINO":
    model = DINOWrapper(device, small=True)
    dim = 224
if args.model == "SAM":
    model = SAMWrapper(device, checkpointdir=args.modelpath)
    dim = 1024
if args.model == "CLIP":
    model = CLIPWrapper(device, modelpath=args.modelpath, modeltype=args.modeltype)
    dim = 224

vertices = torch.from_numpy(vertices).float().to(device)
faces = torch.from_numpy(faces).long().to(device)

### Load json transforms & renders ###
import json
from PIL import Image
import torchvision
import glob

renderdir = os.path.join(os.path.dirname(args.objdir), "renders", args.rendername)
imagepaths = sorted(glob.glob(os.path.join(renderdir, "*.png")))
renders = []
for imgpath in imagepaths:
    img = torch.from_numpy(np.array(Image.open(imgpath))) / 255.
    # img = torchvision.io.read_image(imgpath).float() / 255.
    renders.append(img)
renders = torch.stack(renders)[:,:3]

# Resize to model patch size
renders = torchvision.transforms.functional.resize(renders, (dim, dim))
renders = renders.to(device)

from optimize_twostage_final import load_renderset

positions, lookats, _, fov, _ = load_renderset(renderdir)
positions = torch.tensor(positions).to(device)
lookats = torch.tensor(lookats).to(device)

# Create dataloader
dataset = RenderDataset(renders, positions, lookats)
dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

feature_path = os.path.join(cachedir, "features.pt")
if os.path.exists(feature_path):
    features = torch.load(feature_path, weights_only=True, map_location=device)
else:
    import kaolin as kal

    dims = (512, 512)

    cam = kal.render.camera.Camera.from_args(
            eye=positions,
            at=lookats,
            up=torch.tensor([0., 1., 0.]).to(device),
            fov = fov,
            width=dims[0], height=dims[1],
            device=device
        )

    features = features_from_renders(
        dataloader=dataloader, model=model, vertices=vertices, faces=faces, fov=fov,
        epochs=args.epochs, device=device,
    )

    torch.save(features.cpu(), feature_path)

print("Done with feature backprojection")

## PCA Feature Visualization
from sklearn.decomposition import PCA
from new_renderer import Renderer
import math
import torchvision

if torch.cuda.is_available():
    # Perform PCA for visualization
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    features_pca = (features_pca - features_pca.min(axis=0)) / (features_pca.max(axis=0) - features_pca.min(axis=0)) # 0 to 1

    # Renderer settings
    colors = torch.from_numpy(features_pca).to(device).float()

    azim = torch.linspace(0, 2*np.pi, 8).to(device)
    elev = torch.zeros_like(azim)

    up = torch.tensor([0.0, 1.0, 0.0]).to(device)
    fov = math.radians(60)
    renderer = Renderer(device, dim=(512, 512), interpolation_mode = 'bilinear', fov=fov)

    totimgs = []
    for i in range(len(elev)):
        l_azim = [0., np.pi/2, np.pi, -np.pi/2, 0., 0.] + [azim[i].item()] * 2
        l_elev = [0.] * 4 + [np.pi/2, -np.pi/2] + [elev[i].item()] * 2
        imgs, mask = renderer.render_mesh(vertices, faces, colors,
                                        elev=elev[[i]], azim=azim[[i]], white_background=True,
                                        l_elev = l_elev, l_azim = l_azim,
                                        up = up, radius=2.2, rast_option=2,
                                        return_zbuffer=False)
        totimgs.append(imgs)
    totimgs = torch.cat(totimgs)
    for i in range(len(totimgs)):
        img = torchvision.transforms.functional.to_pil_image(totimgs[i].cpu().detach())
        img.save(os.path.join(savedir, f"pca{i}.png"))

    print(f"Done with PCA visualization")

#### Clustering ######
if args.clustertype == 'kmeans':
    ## Kmeans clustering for sparse keypoint detection
    from sklearn.cluster import KMeans

    n_clusters = args.npoints
    # Cache the labels and latent keypoints
    kmeans_path = os.path.join(cachedir, "kmeans.npz")
    if os.path.exists(kmeans_path):
        kmeans_results = np.load(kmeans_path)
        labels = kmeans_results['labels']
        keypoints = torch.from_numpy(kmeans_results['keypoints']).to(device)
        features = torch.cat([features, vertices], dim=1)
    else:
        features = torch.cat([features, vertices], dim=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features.detach().cpu().numpy())
        keypoints = torch.tensor(kmeans.cluster_centers_).to(device)
        labels = kmeans.labels_
        np.savez(kmeans_path, labels = labels, keypoints = keypoints.detach().cpu().numpy())

# Get closest face to each cluster center
print(f"Done with {args.clustertype} clustering")

# Two versions of finding cluster center: Euclidean vs Latent
face_centroids = vertices[faces].mean(dim=1)
closest_face_idx = []
for i in range(args.npoints):

    if args.centertype == 'euclidean':
        cluster_verts = vertices[labels == i]
        center = cluster_verts.mean(dim=0).to(device)
        dists = torch.norm(face_centroids - center[None], dim=-1)
        closest_face_idx.append(torch.argmin(dists).item())

    if args.centertype == 'latent':
        # Use cluster centers and find the closest faces
        face_centroid_features = features[faces].mean(dim=1)

        dists = torch.norm(face_centroid_features - keypoints[[i]], dim=-1)
        closest_face_idx.append(torch.argmin(dists).item())

# Save keypoints
final_keypoint = face_centroids[closest_face_idx].detach().cpu()
torch.save(final_keypoint, os.path.join(savedir, savename))

# Also save vertex indices
idxname = savename.replace(".pt", "_idxs.pt")
torch.save(torch.tensor(closest_face_idx), os.path.join(savedir, idxname))

print(f"Done with keypoint generation")

# Overlay the keypoints on a renderdir
basedir = os.path.dirname(args.objdir)
if args.overlaydir is not None:
    from render_kp_overlay import render_kp_overlay
    overlayname = f"kp_{os.path.basename(savedir)}"

    overlaydir = os.path.join(basedir, "renders", args.overlaydir)
    zbufferdir = args.zbufferdir
    if zbufferdir is None:
        zbufferdir = f"{overlaydir}_zbuffer"
    else:
        zbufferdir = os.path.join(basedir, "renders", zbufferdir)
    render_kp_overlay(overlaydir, os.path.join(savedir, savename), os.path.join(basedir, "renders", overlayname), kp_radius=0.01,
                      zbufferdir=zbufferdir, device=device,)

    print(f"Done with keypoint overlay")