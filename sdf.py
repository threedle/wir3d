from torch import nn
import torch
import matplotlib.pyplot as plt
import mesh_to_sdf
from mesh_to_sdf import mesh_to_voxels
import os
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, N, num_input_channels],
     returns a tensor of size [batches, N, num_input_channels + mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        B = torch.randn((num_input_channels, mapping_size)) * scale
        # NOTE: row-wise sorting (per-channel)
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # Expected input: B x N x C
        # B shape: C x C'

        channels = x.shape[-1]

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        res = x @ self._B.to(x.device).unsqueeze(0) # B x N x C'
        res = 2 * np.pi * res

        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=-1)

# Define the neural SDF model
class SDF(nn.Module):
    def __init__(self, positional=True, positional_dim=256):
        super(SDF, self).__init__()

        self.positional = positional
        self.input_dim = 3
        self.hidden_dim = 256

        # Positional encoding
        if self.positional:
            self.fourier = FourierFeatureTransform(3, mapping_size=positional_dim, scale=5)
            self.input_dim = 3 + positional_dim * 2
            self.hidden_dim = 3 + positional_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        if self.positional:
            x = self.fourier(x.unsqueeze(0)).squeeze()

        return self.mlp(x)

## ===== TRAINING STUFF =====
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def mesh_coordinates(mesh_size):
    """
    A helper function which takes in the mesh size and returns a
    tensor of coordinates for each voxel of the mesh. The coordinates
    are scaled to fit into [-1.0, 1.0]^3, meaning that the top-left corner
    has the coordinate (-1.0, -1.0, -1.0), and the bottom-right corner has the
    coordinate (1.0, 1.0, 1.0).

    Args:
        mesh_size: the size of the mesh of which we want the voxel
            coordinates

    Returns:
        A torch.Tensor of size [n_voxels, 3] where each row represents
            the *scaled* 3D coordinates of that voxel.
    """
    tensors = tuple([torch.linspace(-1, 1, steps=mesh_size)] * 3)
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, 3)

# input a mesh and 3d coordinate [x,y,z]
# output an sdf value
from torch import nn
from torch.utils.data import IterableDataset
import trimesh

class GeometryDataset(IterableDataset):
    def __init__(self, mesh, res=256, batch_size=256):
        super().__init__()

        self.batch_size = batch_size
        self.res = res
        self.coords = mesh_coordinates(res).to(device)
        self.values = torch.from_numpy(mesh_to_sdf.mesh_to_sdf(mesh, self.coords.detach().cpu().numpy(),
                                                               surface_point_method='scan',
                                                               scan_count=200, scan_resolution=800)).to(device)
        # self.values = torch.from_numpy(mesh_to_voxels(mesh, res, surface_point_method="sample", pad=False, check_result=True)).to(device)
        self.values = self.values.reshape(-1,1)

    def __iter__(self):
        while True:
            # Randomly sample a subset of the image coordinates:
            n_coords = self.coords.shape[0]
            coords_idcs = torch.randint(low=0, high=n_coords, size=(self.batch_size,))
            # Return the randomly sampled coordinates and corresponding values:
            yield self.coords[coords_idcs, :], self.values[coords_idcs, :]

def plot_pointcloud(coords, values=None, ax=None, vmin=0, vmax=1):
    """
    Shows a scatter plot of image coordinates (optionally with associated values).
    Note that we have to do some coordinate shuffling to have the plots displayed
    in the same coordinate system as the one matplotlib uses for images.
    """
    # NOTE: In MPL z and y axis are FLIPPED
    x = coords[:, 0]
    z = coords[:, 1]
    y = coords[:, 2]
    if ax is None:
        _, axes = plt.subplots(1, 1, figsize=(8, 8), squeeze=False, subplot_kw=dict(projection='3d'))
        ax = axes[0, 0]
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(x, y, z, c=values, vmin=vmin, vmax=vmax)

def train_geometry_representation(data, mlp, logdir, total_steps=3000, steps_til_summary=500):
    optim = torch.optim.Adam(lr=1e-3, params=mlp.parameters())

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', min_lr=1e-6, patience=100, factor=0.5, cooldown=100)

    # We use vmin and vmax are necessary to keep the visualizations in the
    # same color range, because otherwise they would not be comparable.
    vmin, vmax = data.values.min(), data.values.max()
    data_iter = iter(data)
    for step in range(total_steps):
        model_input, ground_truth = next(data_iter)
        model_output = mlp(model_input)

        # L1 loss
        loss = torch.nn.functional.l1_loss(model_output, ground_truth)

        # L2 loss
        # loss = ((model_output - ground_truth) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step(loss)

        # NOTE: Matplotlib doesn't work in cluster
        if (step % steps_til_summary) == 0:
            with torch.no_grad():
                current_model = []
                for i in range(0, len(data.coords), data.batch_size):
                    current_model.append(mlp(data.coords[i:i+data.batch_size]).detach().cpu().numpy())
                current_model = np.concatenate(current_model, axis=0)

            print(f"Step {step}: loss = {float(loss.detach().cpu()):.2f}")
            fig, axes = plt.subplots(1, 4, figsize=(18, 4), squeeze=False, subplot_kw=dict(projection='3d'))
            for ax in axes[0]:
                ax.view_init(elev=20, azim=45, vertical_axis='z')

            plot_pointcloud(
                model_input.detach().cpu().numpy(),
                values=model_output.detach().cpu().numpy(),
                ax=axes[0, 0],
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, 0].set_title("Trained MLP (Training Input)")
            plot_pointcloud(
                model_input.detach().cpu().numpy(),
                values=ground_truth.detach().cpu().numpy(),
                ax=axes[0, 1],
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, 1].set_title("Ground Truth (Training Input)")
            plot_pointcloud(
                (data.coords[abs(current_model[:, 0]) <= 0.01]).detach().cpu().numpy(),
                values=current_model[abs(current_model[:, 0]) <= 0.01],
                ax=axes[0, 2],
                vmin=vmin,
                vmax=vmax
            )
            axes[0, 2].set_title("Trained MLP (Zero Level Set)")
            plot_pointcloud(
                data.coords[abs(data.values[:, 0]) <= 0.01].detach().cpu().numpy(),
                values=data.values[abs(data.values[:, 0]) <= 0.01].detach().cpu().numpy(),
                ax=axes[0, 3],
                vmin=vmin,
                vmax=vmax
            )
            axes[0, 3].set_title("Ground Truth (Zero Level Set)")
            plt.savefig(os.path.join(logdir, f"step_{step:05}.png"))

            plt.close()

    return mlp

if __name__ == "__main__":
    import argparse
    from optimize_utils import clear_directory
    import trimesh
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--objdir", type=str, required=True)
    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--summarysteps", type=int, default=500)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--positional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    from pathlib import Path
    logdir = os.path.join(os.path.dirname(args.objdir), "sdf")
    Path(logdir).mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_directory(logdir)

    mesh = trimesh.load(args.objdir, force='mesh')

    # Normalize using bounding box
    from igl import bounding_box
    import numpy as np

    bb_vs, bf = bounding_box(mesh.vertices)

    mesh.vertices -= bb_vs.mean(axis=0)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1)) / args.scale
    mesh.vertices /= scale

    # Cache this dataset -- preprocessing takes long!!
    datasetdir = os.path.join(logdir, "sdfdata.pt")
    if not os.path.exists(datasetdir):
        print(f"Preprocessing the SDF training data ...")
        data = GeometryDataset(mesh, res=args.res, batch_size=args.batch_size)
        torch.save(data, datasetdir)
        print(f"Saved sdf data to cache: {datasetdir}")
    else:
        data = torch.load(datasetdir, weights_only=False)
        print("Loaded sdf data from cache")

        # If batch size is different, then change that
        if args.batch_size != data.batch_size:
            data.batch_size = args.batch_size

    objdir = os.path.dirname(args.objdir)
    if not args.eval:
        sdf = SDF(positional=args.positional)
        sdf.to(device)
        trained_sdf = train_geometry_representation(data, sdf, logdir=logdir, total_steps=args.nsteps,
                                                    steps_til_summary=args.summarysteps)

        torch.save(trained_sdf.state_dict(), os.path.join(objdir, "sdf.pt"))
        print(f"Saved trained SDF model to {os.path.join(objdir, 'sdf.pt')}")

        # Save zero level set prediction
        trained_sdf.eval()
        with torch.no_grad():
            model_input = data.coords.to(device)
            model_output = []
            for i in range(0, len(data.coords), data.batch_size):
                model_output.append(trained_sdf(data.coords[i:i+data.batch_size]))
            model_output = torch.cat(model_output, dim=0)
            level_set = (data.coords[abs(model_output[:, 0]) <= 0.01]).detach().cpu().numpy()
            np.save(os.path.join(objdir, "sdf_level_set.npy"), level_set)

    if args.eval:
        trained_sdf = SDF()
        trained_sdf.load_state_dict(torch.load(os.path.join(objdir, "sdf.pt"), weights_only=True))
        trained_sdf.to(device)
        trained_sdf.eval()
        with torch.no_grad():
            model_input = data.coords.to(device)
            model_output = trained_sdf(model_input)
            level_set = (data.coords[abs(model_output[:, 0]) <= 0.01]).detach().cpu().numpy()
            np.save(os.path.join(objdir, "sdf_level_set.npy"), level_set)