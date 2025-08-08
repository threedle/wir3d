import torch
import os

import torchvision.transforms.functional
import random
import numpy as np
import kaolin as kal
import pydiffvg
import shutil
import argparse
import re
import torch.multiprocessing as mp
import torchvision
import json
import glob
import time
import dill as pickle
import matplotlib.pyplot as plt
import copy

from loss import viewlossfcn
from pytorch3d.loss import chamfer_distance
from PIL import Image, ImageDraw
from pathlib import Path
from igl import heat_geodesic, uniformly_sample_two_manifold_at_vertices, uniformly_sample_two_manifold_internal

from clip_stuff import *
from make_gifs import compare_imgs, make_gif

def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate_renders(renders, return_valid=True):
    """Extracts elev and azim from filenames of renders.

    Assumes a format of <id>_<elev>_<azim>.png for all filenames, or <id>_<elev>_<azim>_textured.png for textured filenames.

    Deprecated as I am switching to storing the render angles in a dedicated file rather than the names of files.

    Args:
        renders (list of strings): List of render paths
        return_valid (bool, optional): Whether to return the list of valid renders. Defaults to True.

    Returns:
        Either (elev, azim, validrenders) or (elev, azim)
    """
    elev, azim = [], []
    if return_valid:
        validrenders = []
    for render in renders:
        research = re.search(r"(?:\d+_)?([-]*\d+\.\d+)_([-]*\d+\.\d+)(?:_textured)?.png", render)
        if research is None:
            print(f"{render} not a valid render")
        else:
            elev.append(float(research.group(1)))
            azim.append(float(research.group(2)))
            if return_valid:
                validrenders.append(render)
    if return_valid:
        return elev, azim, validrenders
    else:
        return elev, azim

def load_renderset(dirpath, load_zbuffer=False, zbuffer_dir=None):
    """Loads renders from a folder with a renderlist file.

    Expects a file named renderlist.json with the position, lookat, and fov information, and renders as sequentially numbered pngs.
    If such a file is not present, assumes angles are encoded in the filenames (using validate_renders) and defaults of r=2.2, fov=pi/3.

    Args:
        dirpath (string): path to the folder containing renders

    Returns:
        tuple of (positions, lookats, filepaths, fov, zpaths)
    """
    zpaths = None
    with open(os.path.join(dirpath, "renderlist.json")) as file:
        renderlist = json.load(file)

    positions = renderlist["positions"]
    lookats = renderlist["lookats"]
    fov = renderlist["fov"]

    imagepaths = sorted(glob.glob(os.path.join(dirpath, "*.png")))
    assert len(positions) == len(imagepaths), f"Number of elevations {len(positions)} does not match number of images {len(imagepaths)}"
    assert len(lookats) == len(imagepaths), f"Number of azimuths {len(lookats)} does not match number of images {len(imagepaths)}"

    if load_zbuffer:
        if zbuffer_dir is None:
            zpaths = sorted(glob.glob(os.path.join(dirpath + "_zbuffer", "*.pt")))
        else:
            zpaths = sorted(glob.glob(os.path.join(zbuffer_dir, "*.pt")))
        assert len(positions) == len(zpaths), f"Number of elevations {len(positions)} does not match number of zbuffer tensors {len(zpaths)}"
        assert len(lookats) == len(zpaths), f"Number of azimuths {len(lookats)} does not match number of zbuffer tensors {len(zpaths)}"

    return np.array(positions), np.array(lookats), np.array(imagepaths), fov, zpaths

def furthest_point_init(meshpath, ncurves, npoints=4, use_geodesics=False, scale=1.):
    import trimesh

    mesh = trimesh.load(meshpath, force="mesh")

    from igl import bounding_box
    bb_vs, bf = bounding_box(mesh.vertices)
    mesh.vertices -= np.mean(bb_vs, axis=0)
    mesh.vertices /= (np.max(np.linalg.norm(mesh.vertices, axis=1)) / scale)

    vertices = mesh.vertices
    faces = mesh.faces

    # Furthest point sampling
    if use_geodesics:
        sampled_points = torch.from_numpy(uniformly_sample_two_manifold_at_vertices(vertices, ncurves * npoints, 0.1))
    else:
        sampled_points = torch.from_numpy(uniformly_sample_two_manifold_internal(vertices, faces, ncurves, 0.1)).float()

    curves = []
    if use_geodesics:
        # Initialize curves by taking points with closest geodesic distances
        tmp_sampled_points = copy.deepcopy(sampled_points)
        for _ in range(ncurves):
            source = np.random.choice(tmp_sampled_points)
            distances = heat_geodesic(vertices, faces, 1e-3, np.array([source], dtype=int))[tmp_sampled_points]
            nn3 = tmp_sampled_points[np.argsort(distances)[1:4]]
            curves.append(torch.from_numpy(vertices[[source] + list(nn3)]))
            tmp_sampled_points = np.array([p for p in tmp_sampled_points if p not in list(nn3) + [source]])
    else:
        # Initialize curves by sampling gaussians centered at the sampled points
        for i in range(ncurves):
            curve = [sampled_points[i]]
            p0 = sampled_points[i]
            for _ in range(npoints - 1):
                p1 = p0 + torch.randn_like(p0) * 0.04
                curve.append(p1)
                p0 = p1
            curves.append(torch.stack(curve))

    return torch.stack(curves)

def random_vertex_init(meshpath, ncurves, npoints=4, scale=1.):
    import igl

    vertices, _, n, f, _, _ = igl.read_obj(meshpath)
    vertices = np.array(vertices)

    # Normalize the mesh
    from igl import bounding_box
    bb_vs, bf = bounding_box(vertices)
    vertices -= np.mean(bb_vs, axis=0)
    vertices /= (np.max(np.linalg.norm(vertices, axis=1)) / scale)

    # Randomly sample vertices
    sampled_points = np.random.choice(len(vertices), ncurves, replace=False)
    sampled_points = torch.from_numpy(vertices[sampled_points]).float()

    curves = []
    # Initialize curves by sampling gaussians centered at the sampled points
    for i in range(ncurves):
        curve = [sampled_points[i]]
        p0 = sampled_points[i]
        for _ in range(npoints - 1):
            p1 = p0 + torch.randn_like(p0) * 0.04
            curve.append(p1)
            p0 = p1
        curves.append(torch.stack(curve))

    return torch.stack(curves)

# Densely sample surface for coverage loss
def densely_sample_surface(meshpath, npoints=5000, scale=1.):
    import trimesh

    mesh = trimesh.load(meshpath, force="mesh")

    from igl import bounding_box
    bb_vs, bf = bounding_box(mesh.vertices)
    mesh.vertices -= np.mean(bb_vs, axis=0)
    mesh.vertices /= (np.max(np.linalg.norm(mesh.vertices, axis=1)) / scale)

    vertices = mesh.vertices
    faces = mesh.faces

    samples, face_idxs = trimesh.sample.sample_surface(mesh, npoints)
    sampled_points = torch.from_numpy(samples).float()

    return sampled_points

def kp_init(keypoints, ncurves):
    npoints = 4
    curves = []

    if len(keypoints.shape) == 1:
        keypoints = keypoints.unsqueeze(0)

    for keypoint in keypoints:
        for i in range(ncurves):
            curve = [keypoint]
            p0 = keypoint
            for _ in range(npoints - 1):
                p1 = p0 + torch.randn_like(p0) * 0.04
                curve.append(p1)
                p0 = p1
            curves.append(torch.stack(curve))

    return torch.stack(curves)

def render_curves(curves, cam_position, cam_lookat, nviews, ncurves, npoints, canvas_width, canvas_height,
                  iseed=0, return_ndc=False, colors=None, fov=np.pi * 60 / 180,
                  stroke_width=torch.tensor(1.5), keypoints=None, device = torch.device('cpu')):
    """Renders a set of curves into an image

    Args:
        curves (Tensor): Curves to be rendered
        cam_position (Tensor): Tensor of camera positions (assumes y axis is up)
        cam_lookat (Tensor): Tensor of camera lookats
        nviews (int): Number of views
        ncurves (int): Number of curves
        npoints (int): Number of points per curve (4 in most cases)
        canvas_width (int): Width of the canvas
        canvas_height (int): Height of the canvas
        iseed (int, optional): Seed used in the rendering process. Seems to be set to iteri+1. Defaults to 0.
        return_ndc (bool, optional): Whether to return vertices_ndc for use in viewbox loss. Defaults to False.
        colors (Tensor, optional): If set, uses as a tensor of colors for curves

    Returns:
        Stack of rendered images, with dimension B x 3 x H x W
    """
    eye = torch.tensor(cam_position).float().to(device)
    look_at = torch.tensor(cam_lookat).float().to(device)

    camera = kal.render.camera.Camera.from_args(
        eye=eye,
        at=look_at,
        up=torch.tensor([0., 1., 0.], dtype=torch.float),
        fov=fov,
        height=canvas_height, width=canvas_width,
        device=device
    )

    vertices_camera = camera.extrinsics.transform(curves.reshape(-1, 3).unsqueeze(0).repeat(nviews, 1, 1)) # Cameras x ncurves*points x 3
    vertices_clip = camera.intrinsics.transform(vertices_camera)
    vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_clip).reshape(nviews, ncurves, npoints, 2)
    vertices_ndc = (vertices_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

    # Render keypoints if available
    keypoints_ndc = None
    keypoints_clip = None
    if keypoints is not None:
        keypoints_camera = camera.extrinsics.transform(keypoints.unsqueeze(0).repeat(nviews, 1, 1)) # Cameras x nkeypoints x 3
        keypoints_clip = camera.intrinsics.transform(keypoints_camera)
        keypoints_ndc = kal.render.camera.intrinsics.down_from_homogeneous(keypoints_clip)
        keypoints_ndc = (keypoints_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

    imgs = []
    for vi in range(nviews):
        shapes = []
        shape_groups = []

        # Each curve is a single-segment bezier
        num_control_points = torch.zeros(1, dtype = torch.int32) + (npoints - 2) # the endpoints are base points for the curve

        for i in range(ncurves):
            points = vertices_ndc[vi, i]
            canvas_points = torch.stack((points[:, 0] * canvas_width, points[:, 1] * canvas_height), dim=-1)

            path = pydiffvg.Path(num_control_points = num_control_points,
                                    points = canvas_points,
                                    stroke_width = stroke_width,
                                    is_closed = False)

            shapes.append(path)
            stroke_color = colors[i,:] if colors is not None else torch.tensor([0.0, 0.0, 0.0, 1.0])
            # fill_color = colors[i,:] if colors is not None else None
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = None,
                                            stroke_color = stroke_color,
                                            )
            shape_groups.append(path_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, # width
                    canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    iseed,   # seed
                    None,
                    *scene_args)

        # Get white background
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = device) * (1 - opacity)
        imgs.append(img)

    imgs = torch.stack(imgs).permute(0, 3, 1, 2) # B x 3 x H x W
    imgs = torch.flip(imgs, dims=(2,))

    if return_ndc:
        return imgs, vertices_ndc, keypoints_ndc, keypoints_clip
    else:
        return imgs, keypoints_ndc, keypoints_clip

def load_init(initfile):
    """Loads a file for initial curves.

    Assumes it was output by optimization if it is a .pt and assumes it was exported from blender if it is a .npy

    Args:
        initfile (string): path to file

    Raises:
        Exception: when the given filename is neither a .pt or .npy

    Returns:
        Tensor of dimensions (curve) x (4 points per curve) x (3 coordinates per point)
    """
    if initfile.endswith(".pt"):
        return torch.load(initfile, weights_only=True)
    elif initfile.endswith(".npy"):
        res = np.load(initfile)
        # Assume a .npy file came from Blender and thus needs normalization and coordinate conversion (this assumption may change later)
        max_distance = np.sqrt(np.max(np.sum(np.square(res[:,[0, -1],:]), axis=-1)))
        res /= max_distance * 2
        # Rotation should be equivalent to a 90 degree CCW rotation about x
        res[:,:,1] *= -1 # Invert y
        res = res[:,:,[0,2,1]] # Swap y and z
        return torch.tensor(res).float()
    else:
        raise Exception(f"Unknown init file type: {initfile}")

def run(args, stage='geo', seed = 0, threadnum=None, setseed=False):
    # Set seed and device
    if setseed:
        set_seed(seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Directories
    meshpath = args.meshpath
    basepath = os.path.dirname(meshpath)
    renderdir = args.renderdir

    # Default renderdirs
    if len(renderdir) == 0:
        if stage == "geo":
            renderdir = [os.path.join(basepath, "renders", "freestyle_030_360")]
        else:
            renderdir = [os.path.join(basepath, "renders", "blender_030_360")]

    gtrenderdir = args.gtrenderdir
    if gtrenderdir is None:
        gtrenderdir = renderdir

    outputdir = os.path.join(args.outputdir, f"seed{seed}", stage)
    if args.overwrite:
        if os.path.exists(outputdir):
            clear_directory(outputdir)
    Path(outputdir).mkdir(exist_ok=True, parents=True)

    # Print thread numbers on every message if multithreading
    if threadnum is not None:
        threadstring = f"[{threadnum}]"
        def new_print(*x):
            print(threadstring, *x)
    else:
        def new_print(*x):
            print(*x)

    # Read or generate initial curves
    npoints = 4
    frozencurves = torch.tensor([]).to(device)

    if args.inits is None or (args.inits is not None and args.frozen_init):
        # Generate a new set of curves to optimize
        if args.init_type == "furthest":
            optimcurves = furthest_point_init(meshpath, args.ncurves, npoints, use_geodesics=args.fpgeodesics, scale=args.normalize_scale).to(device)
        elif args.init_type == "random":
            optimcurves = (torch.rand((args.ncurves, npoints, 3)).to(device) * 2 - 1)/3 # points in [-0.5, 0.5]
        elif args.init_type == "keypoint":
            init_keypoints = torch.load(os.path.join(basepath, args.init_keypoint_dir), weights_only=True).to(device)
            optimcurves = kp_init(init_keypoints, args.ncurves)
        elif args.init_type == "vertex":
            optimcurves = random_vertex_init(meshpath, args.ncurves, npoints).to(device)

    if args.inits is not None:
        inits = [load_init(initfile) for initfile in args.inits]
        if args.frozen_init:
            frozencurves = torch.cat(inits).to(device)
        else:
            optimcurves = torch.cat(inits).to(device)

    lr = args.lr
    optimcurves.requires_grad = True
    optimset = [{'params': [optimcurves], 'lr': lr}]
    optimizer = torch.optim.Adam(optimset)

    # Frozen init optimization
    if args.frozen_init_lr > 0 and len(frozencurves) > 0:
        frozencurves.requires_grad = True
        frozen_optimizer = torch.optim.Adam([{'params': [frozencurves], 'lr': args.frozen_init_lr}])

    ncurves = optimcurves.shape[0] + frozencurves.shape[0]

    # DiffVG scene parameters
    canvas_width, canvas_height = args.width, args.height

    # Load render sets
    cam_position, cam_lookat, totrenders, totzbuffers = [], [], [], []
    gt_position, gt_lookat, gtrenders = [], [], []

    # Keep track of which renders are keypoint-based (so we don't apply the view loss)
    viewloss_valid = []

    zbufferdirs = args.zbuffer_dir
    if len(zbufferdirs) == 0:
        zbufferdirs = [os.path.join(basepath, "renders", "surface_030_360_zbuffer")]

    for i in range(len(renderdir)):
        tmpdir = renderdir[i]
        zbufferdir = None
        if zbufferdirs is not None:
            zbufferdir = zbufferdirs[i]
        tmp_position, tmp_lookat, tmptotrenders, fov, tmptotzbuffers = load_renderset(tmpdir, load_zbuffer=zbufferdir is not None,
                                                                    zbuffer_dir = zbufferdir)
        cam_position.append(tmp_position)
        cam_lookat.append(tmp_lookat)
        totrenders.append(tmptotrenders)

        if tmptotzbuffers is not None:
            totzbuffers.extend(tmptotzbuffers)

        if "keypoints" in tmpdir:
            viewloss_valid.extend([False] * len(tmp_position))
        else:
            viewloss_valid.extend([True] * len(tmp_position))

    for tmpdir in gtrenderdir:
        tmp_gtposition, tmp_gtlookat, tmpgtrenders, gt_fov, _ = load_renderset(tmpdir)

        gt_position.append(tmp_gtposition)
        gt_lookat.append(tmp_gtlookat)
        gtrenders.append(tmpgtrenders)

    cam_position = np.concatenate(cam_position)
    cam_lookat = np.concatenate(cam_lookat)
    totrenders = np.concatenate(totrenders)
    viewloss_valid = torch.tensor(viewloss_valid, device=device)

    if len(totzbuffers) > 0:
        totzbuffers = np.array(totzbuffers)

    # Load the renders and zbuffers
    renders = []
    for imgpath in totrenders:
        img = torchvision.transforms.functional.pil_to_tensor(Image.open(imgpath)).float().to(device)/255.

        # Convert to white background
        if img.shape[0] > 3:
            img = img[:3] * img[[3]] + torch.ones_like(img[:3]) * (1 - img[[3]])

        renders.append(img)
    totrenders = torch.stack(renders)

    if len(totzbuffers) > 0:
        totzbuffers = torch.stack([torch.load(zbufferpath, weights_only=True) for zbufferpath in totzbuffers])

    gt_position = np.concatenate(gt_position)
    gt_lookat = np.concatenate(gt_lookat)
    gtrenders = np.concatenate(gtrenders)

    # Initialize progress variables
    inititer = 0
    bestloss = float("inf")
    bestiter = 0
    losses, cliplosses, lpipslosses, viewlosses, spherelosses, coveragelosses, sdflosses = \
        [], [], [], [], [], [], []
    if not args.overwrite:
        # Load the cached values from previous run
        if os.path.exists(os.path.join(outputdir, "latestiter")):
            with open(os.path.join(outputdir, "latestiter"), "rb") as f:
                inititer = pickle.load(f)

        if os.path.exists(os.path.join(outputdir, "optimcurves.pt")):
            optimcurves = torch.load(os.path.join(outputdir, "optimcurves.pt"), weights_only=True).to(device)
            optimcurves.requires_grad = True
            optimizer = torch.optim.Adam([optimcurves], lr=lr)
        elif os.path.exists(os.path.join(outputdir, "initcurves.pt")):
            optimcurves = torch.load(os.path.join(outputdir, "initcurves.pt"), weights_only=True).to(device)
            optimcurves.requires_grad = True
            optimizer = torch.optim.Adam([optimcurves], lr=lr)

        if os.path.exists(os.path.join(outputdir, "optimstate.pt")):
            state = torch.load(os.path.join(outputdir, "optimstate.pt"), weights_only=True)
            optimizer.load_state_dict(state)

        if args.frozen_init_lr > 0 and len(frozencurves) > 0:
            if os.path.exists(os.path.join(outputdir, "frozenoptcurves.pt")):
                frozencurves = torch.load(os.path.join(outputdir, "frozenoptcurves.pt"), weights_only=True).to(device)
                frozencurves.requires_grad = True
                frozen_optimizer = torch.optim.Adam([frozencurves], lr=args.frozen_init_lr)

            if os.path.exists(os.path.join(outputdir, "frozen_optimstate.pt")):
                state = torch.load(os.path.join(outputdir, "frozen_optimstate.pt"), weights_only=True)
                frozen_optimizer.load_state_dict(state)

            if os.path.exists(os.path.join(outputdir, "best_frozencurves.pt")):
                bestfrozencurves = torch.load(os.path.join(outputdir, "best_frozencurves.pt"), weights_only=True).to(device)

        if os.path.exists(os.path.join(outputdir, "bestcurves.pt")):
            bestcurves = torch.load(os.path.join(outputdir, "bestcurves.pt"), weights_only=True).to(device)

        if os.path.exists(os.path.join(outputdir, "bestloss")):
            with open(os.path.join(outputdir, "bestloss"), "rb") as f:
                bestloss = pickle.load(f)

        if os.path.exists(os.path.join(outputdir, "bestiter")):
            with open(os.path.join(outputdir, "bestiter"), "rb") as f:
                bestiter = pickle.load(f)

        if os.path.exists(os.path.join(outputdir, "losshistory.pkl")):
            try:
                with open(os.path.join(outputdir, "losshistory.pkl"), "rb") as f:
                    losses, cliplosses, lpipslosses, viewlosses, spherelosses, coveragelosses, sdflosses = pickle.load(f)
            except Exception as e:
                new_print(f"Error loading loss history: {e}")
                print("Resetting ...")
                losses, cliplosses, lpipslosses, viewlosses, spherelosses, coveragelosses, sdflosses = \
                        [], [], [], [], [], [], []


    niters = args.niters

    #### Load loss functions ####
    # CLIP loss
    lossfcn = CLIPConvLoss(clip_model_name = args.clip_model_name, clip_conv_layer_weights = args.clip_conv_layer_weights,
                           clip_fc_weight = args.clip_fc_weight, num_augs=args.clip_num_augs,
                           clip_fc_loss_type = args.clip_fc_losstype,
                           device=device)

    # LPIP loss
    if args.lambda_lpips > 0:
        from lpips_loss import JointLoss
        lpipsfcn = JointLoss(device=device, loss_type="LPIPS",
                             robust = True)

    # SDF loss
    if args.lambda_sdf > 0:
        # Load the SDF model with the weights from the path
        from sdf import SDF
        sdfmodel = SDF()
        sdfpath = os.path.join(basepath, "sdf.pt")
        sdfmodel.load_state_dict(torch.load(sdfpath, weights_only=True))
        sdfmodel.to(device)

    # Keypoint loss
    keypoints = None
    if args.spatial_keypoint_loss:
        keypointdir = os.path.join(basepath, args.spatial_keypoint_loss)
        keypoints = torch.load(keypointdir, weights_only=True).to(device)

        # assert len(totzbuffers) > 0, "Need zbuffers for spatial keypoint loss"

    # Coverage loss
    if args.lambda_coverage > 0:
        coverage_points = densely_sample_surface(meshpath, npoints = args.coverage_samples, scale = args.normalize_scale).to(device)

    # =============================

    # Get the GT images
    finalviews = len(gtrenders)
    gt = []
    for imgpath in gtrenders:
        gtimg = torchvision.transforms.functional.pil_to_tensor(Image.open(imgpath)).float().to(device)/255.

        # Convert to white background
        if gtimg.shape[0] > 3:
            gtimg = gtimg[:3] * gtimg[[3]] + torch.ones_like(gtimg[:3]) * (1 - gtimg[[3]])

        gt.append(gtimg)
    gt = torch.stack(gt).cpu().detach()

    # If have opacity, then convert to white background and remove the alpha channel
    if gt.shape[1] > 3:
        gt = gt[:, :3] * gt[:, [3]] + torch.ones_like(gt[:, :3]) * (1 - gt[:, [3]])

    # Resize to correct dimensions
    if gt.shape[2] != canvas_width or gt.shape[3] != canvas_height:
        gt = torchvision.transforms.functional.resize(gt, (canvas_width, canvas_height))

    ############ Render the initialization #####
    curve_colors = None
    if args.colorize:
        curve_colors = [[0, 0, 0, 1]] * frozencurves.shape[0] + [[1, 0, 0, 1]] * optimcurves.shape[0]
        curve_colors = torch.tensor(curve_colors, device=device)

    # Number of views to sample is min of nviews and # total renders
    nviews = min(args.nviews, len(totrenders))

    # Do 360 render of the initial curve points alongside the GT
    initviews = len(gtrenders)
    initdir = os.path.join(outputdir, "init")
    initcurves = torch.cat([frozencurves, optimcurves]).detach()
    Path(initdir).mkdir(exist_ok=True, parents=True)
    if not os.path.exists(os.path.join(initdir, "..", "init.gif")):
        if args.colorize:
            sample_colors = curve_colors
        else:
            sample_colors = None
        imgs = render_curves(initcurves, gt_position, gt_lookat, initviews, ncurves, npoints, canvas_width, canvas_height,
                             0, fov=gt_fov, colors=sample_colors, device=device)[0].cpu().detach()

        # Save
        for viewi in range(initviews):
            pred = torchvision.transforms.functional.to_pil_image(imgs[viewi, :3,])
            compare_imgs([pred], ["Init"]).save(os.path.join(initdir, f"{viewi:03d}.png"))

        # Make gifs
        make_gif(f"{initdir}/*.png", os.path.join(initdir, "..", "init.gif"))

    ################ TRAINING #############################
    if stage == "geo":
        logstr = " =============== Geometry Stage ================== "
    else:
        logstr = " =============== Semantics Stage ================== "
    new_print(logstr)

    logstr = "Iter | Time      | CLIP  | View "
    if args.lambda_lpips > 0:
        logstr += " | LPIPS "
    if args.lambda_sdf > 0:
        logstr += " | SDF  "
    if args.lambda_coverage > 0:
        logstr += " | Coverage"
    logstr += " | Total loss"

    new_print(logstr)

    # Load settings
    lambda_clip, lambda_view = args.lambda_clip, 1
    niters = args.niters

    # TODO tie these to arguments
    save_views = False
    save_progress_period = 50
    monitor_period = 50

    # Create monitoring directories
    vizdir = os.path.join(outputdir, "renders")
    if save_views or args.debug:
        Path(vizdir).mkdir(exist_ok=True, parents=True)

    monitordir = os.path.join(outputdir, "monitoring_renders")
    monitorcomparedir = os.path.join(outputdir, "monitoring_compare_renders")
    Path(monitordir).mkdir(exist_ok=True, parents=True)
    Path(monitorcomparedir).mkdir(exist_ok=True, parents=True)

    # Debug directory
    if args.debug:
        debugdir = os.path.join(outputdir, "debug")
        Path(debugdir).mkdir(exist_ok=True, parents=True)

    bestcurves = torch.clone(optimcurves).detach()

    bestfrozencurves = None
    if args.frozen_init_lr > 0 and len(frozencurves) > 0:
        bestfrozencurves = torch.clone(frozencurves).detach()

    # Keep track of curve lengths and prune if stays under some distance threshold for longer than 100 iters
    from collections import defaultdict
    from utils import bcurve_length
    curve_threshold = args.prune_curves
    thresholdcounts = defaultdict(int)
    keepcurves = torch.ones(len(optimcurves), dtype=torch.bool, device=device)
    bestkeep = torch.clone(keepcurves).detach()

    for iteri in range(inititer, niters):
        # Check for curve pruning
        if curve_threshold > 0:
            with torch.no_grad():
                lengths = bcurve_length(optimcurves)

            badcurves = torch.where(lengths < curve_threshold)[0]
            goodcurves = torch.where(lengths >= curve_threshold)[0]
            for badcurve in badcurves:
                badcurve = badcurve.item()
                thresholdcounts[badcurve] += 1
                if thresholdcounts[badcurve] > 400:
                    keepcurves[badcurve] = False
            for goodcurve in goodcurves:
                goodcurve = goodcurve.item()
                thresholdcounts[goodcurve] = 0

            badcurves_idx = torch.where(keepcurves == False)[0].detach().cpu().numpy().tolist()
            print(f"Curves {badcurves_idx} pruned")
            filtercurves = optimcurves[keepcurves]
        else:
            filtercurves = optimcurves

        start_time = time.time()

        ############ Optimization step ####################
        optimizer.zero_grad()

        if args.frozen_init_lr > 0 and len(frozencurves) > 0:
            frozen_optimizer.zero_grad()

        # Construct the full curves list
        curves = torch.cat([frozencurves, filtercurves])

        # Sample the GT renders
        chosen_i = np.random.choice(len(totrenders), nviews)
        iter_gt = totrenders[chosen_i].to(device)

        if args.spatial_keypoint_loss and len(totzbuffers) > 0:
            chosen_zbuffers = totzbuffers[chosen_i]

        # If have opacity, then convert to white background and remove the alpha channel
        if iter_gt.shape[1] > 3:
            iter_gt = iter_gt[:,:3] * iter_gt[:,[3]] + torch.ones_like(iter_gt[:,:3]) * (1 - iter_gt[:,[3]])

        # Get the sampled camera parameters
        chosen_position = cam_position[chosen_i]
        chosen_lookat = cam_lookat[chosen_i]

        # Render the current curves
        sample_colors = None
        imgs, vertices_ndc, keypoints_ndc, keypoints_clip = render_curves(curves, chosen_position, chosen_lookat, nviews, len(curves), npoints, canvas_width, canvas_height,
                                    iteri+1, fov=fov, return_ndc=True, colors=sample_colors, keypoints=keypoints, device=device
                                    )

        ## CLIP Loss
        # Generate semantic weights if spatial keypoints loss is enabled
        semantic_weights = None
        if args.spatial_keypoint_loss:
            semantic_weights = torch.ones((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3])).to(device) * args.spatial_weight_base
            pixel_y, pixel_x = torch.meshgrid(torch.linspace(1, 0, canvas_height), torch.linspace(0, 1, canvas_width))
            pixels_ndc = torch.stack((pixel_y, pixel_x), dim=0).to(device)
            sigma = args.spatial_keypoint_sigma

            for viewi, keypoint_ndc in enumerate(keypoints_ndc):
                kept_keypoints = []
                for ki, keypoint in enumerate(keypoint_ndc):
                    # Ignore everything outside the render window
                    if torch.any(keypoint > 1) or torch.any(keypoint < 0):
                        continue

                    if len(totzbuffers) > 0:
                        # Check z against loaded zbuffer
                        # NOTE: This only checks the keypoint CENTERS -- assumes 0 volume keypoints
                        kp_z = keypoints_clip[viewi, ki, 2] - 0.0005 # Slightly offset to avoid z fighting with the surface
                        kp_x, kp_y = torch.floor(keypoint * torch.tensor([chosen_zbuffers.shape[2], chosen_zbuffers.shape[1]], device=device).float())
                        check_z = chosen_zbuffers[viewi, kp_y.long(), kp_x.long()].item()

                        if kp_z > check_z:
                            continue

                    kept_keypoints.append(keypoint_ndc[ki])

                    # NOTE: Keypoint NDC is (width, height) starting from bottom left, while pixels_ndc is (y, x) starting from top left
                    # So need to swap
                    keypoint = keypoint.flip(0)

                    # Generate a gaussian mask around the keypoint
                    mask = torch.exp(-(torch.linalg.norm(pixels_ndc - keypoint.reshape(-1, 1, 1), dim=0))**2 / (2 * sigma**2)) # 1 at the center
                    # Threshold the mask under some epsilon to avoid precision nonsense
                    mask = torch.where(mask < 1e-2, torch.zeros_like(mask), mask)
                    semantic_weights[viewi] += mask.unsqueeze(0)

                if len(kept_keypoints) > 0:
                    kept_keypoints = torch.stack(kept_keypoints)

            # Save the semantic weights for monitoring
            if args.debug:
                for viewi, semantic_weight in enumerate(semantic_weights):
                    semantic_img = semantic_weight.squeeze().cpu().detach()
                    # Map to reds colormap
                    import matplotlib.pyplot as plt
                    import matplotlib.cm as cm

                    cmap = cm.get_cmap('Reds')

                    # Map to 0-1
                    normalized_img = ((semantic_img - semantic_img.min()) / (semantic_img.max() - semantic_img.min() + 1e-6)).detach().cpu().numpy()
                    colored_img = (cmap(normalized_img) * 255).astype(np.uint8)

                    # Remove alpha from areas where normalized img is 0
                    colored_img[normalized_img == 0, -1] = 0
                    colored_img[normalized_img != 0, -1] = 122

                    # Add back alpha to the gt render based on white background
                    iter_gt_img = iter_gt[viewi, :3]
                    iter_gt_alphas = torch.ones_like(iter_gt_img)[[0]]
                    iter_gt_alphas[:, iter_gt_img.sum(dim=0) == 3] = 0
                    iter_gt_img = torch.cat((iter_gt_img, iter_gt_alphas), dim=0)

                    # Alpha composite
                    iter_gt_img = torchvision.transforms.functional.to_pil_image(iter_gt_img.cpu().detach())
                    colored_img = Image.fromarray(colored_img)
                    colored_img.save(os.path.join(debugdir, f"semantic_weights_{iteri:03d}_{viewi:03d}.png"))

                    semantic_img = Image.alpha_composite(iter_gt_img, colored_img)
                    semantic_img.save(os.path.join(debugdir, f"semantic_weights_overlay_{iteri:03d}_{viewi:03d}.png"))

                    # Also: rasterize the keypoints before and after the zbuffer filtering
                    from new_renderer import draw_antialiased_circle

                    keypoint_img = Image.fromarray(np.ones_like(colored_img) * 255)
                    for keypoint in keypoints_ndc[viewi]:
                        keypoint_img = draw_antialiased_circle(keypoint_img, keypoint.cpu().numpy(), radius=0.02, scale_factor=4)
                    keypoint_img.save(os.path.join(debugdir, f"keypoints_{iteri:03d}_{viewi:03d}.png"))

                    if len(kept_keypoints) > 0:
                        filtered_keypoint_img = Image.fromarray(np.ones_like(colored_img) * 255)
                        for keypoint in kept_keypoints:
                            filtered_keypoint_img = draw_antialiased_circle(filtered_keypoint_img, keypoint.cpu().numpy(), radius=0.02, scale_factor=4)
                        filtered_keypoint_img.save(os.path.join(debugdir, f"filtered_keypoints_{iteri:03d}_{viewi:03d}.png"))

                    # Also: save the zbuffer
                    zbuffer = chosen_zbuffers[viewi]
                    zbuffer = (zbuffer - zbuffer.min()) / (zbuffer.max() - zbuffer.min())

                    # Flip upside down
                    zbuffer = torch.flip(zbuffer, dims=(0,))

                    zbuffer = Image.fromarray((torch.cat([zbuffer]*3, dim=-1) * 255).cpu().numpy().astype(np.uint8))
                    zbuffer.save(os.path.join(debugdir, f"zbuffer_{iteri:03d}_{viewi:03d}.png"))

                    # Also: save the GT and rendered images
                    gt_img = torchvision.transforms.functional.to_pil_image(iter_gt[viewi])
                    gt_img.save(os.path.join(debugdir, f"gt_{iteri:03d}_{viewi:03d}.png"))

                    rendered_img = torchvision.transforms.functional.to_pil_image(imgs[viewi])
                    rendered_img.save(os.path.join(debugdir, f"rendered_{iteri:03d}_{viewi:03d}.png"))

        cliploss = torch.tensor(0., device=device)
        if lambda_clip > 0:
            cliploss = lossfcn(imgs, iter_gt, semantic_weights, spatial_fc=args.spatial_fc, debug=args.debug)
            cliploss = sum(cliploss.values())

        # LPIPS loss
        lpipsloss = torch.tensor(0., device=device)
        if args.lambda_lpips > 0:
            lpipsloss = args.lambda_lpips * lpipsfcn(imgs, iter_gt)

        # Viewing box loss
        viewloss = torch.tensor(0., device=device)
        if args.lambda_view > 0:
            # Need to sample NDC points
            # Cubic bezier: (1-t)^3 P0 + (1-t)^2 3t P1 + 3(1-t)t^2 P2 + t^3 P3
            sampled_t = torch.rand(args.surfsamples).to(device).unsqueeze(0).unsqueeze(2)
            sampled_points = (1 - sampled_t)**3 * vertices_ndc[:, :, None, 0] + \
                3 * (1 - sampled_t)**2 * sampled_t * vertices_ndc[:, :, None, 1] + \
                3 * (1 - sampled_t) * sampled_t**2 * vertices_ndc[:, :, None, 2] + \
                    sampled_t**3 * vertices_ndc[:, :, None, 3]

            viewloss = viewlossfcn(sampled_points, mask = viewloss_valid[chosen_i, None, None, None])

        # Need to sample points along curves if either SDF loss or keypoint loss is enabled
        if args.lambda_sdf > 0 or args.lambda_coverage > 0:
            # Cubic bezier: (1-t)^3 P0 + (1-t)^2 3t P1 + 3(1-t)t^2 P2 + t^3 P3
            sampled_t = torch.rand(args.surfsamples).to(device).unsqueeze(0).unsqueeze(2)
            sampled_points = (1 - sampled_t)**3 * curves[:, None, 0] + \
                3 * (1 - sampled_t)**2 * sampled_t * curves[:, None, 1] + \
                3 * (1 - sampled_t) * sampled_t**2 * curves[:, None, 2] + \
                    sampled_t**3 * curves[:, None, 3]

        # SDF loss
        if args.lambda_sdf > 0:
            # Get the SDF values for the current curves
            sdfvals = sdfmodel(sampled_points.reshape(-1, 3))

            # Penalize SDF values greater than 0
            sdfloss = args.lambda_sdf * torch.nn.functional.relu(sdfvals).mean()
            # sdfloss = args.lambda_sdf * torch.nn.functional.mse_loss(sdfvals, torch.zeros_like(sdfvals), reduction='sum')

            sdflosses.append(sdfloss.detach().item())

        # Add losses together
        cliploss *= lambda_clip
        viewloss *= lambda_view

        loss = cliploss + viewloss + lpipsloss

        if args.lambda_sdf > 0:
            loss += sdfloss

        # Coverage loss
        if args.lambda_coverage > 0:
            coverageloss = chamfer_distance(coverage_points.unsqueeze(0), sampled_points.reshape(1, -1, 3), single_directional=True)[0]
            coverageloss *= args.lambda_coverage
            loss += coverageloss

            coveragelosses.append(coverageloss.item())

        loss.backward()
        assert optimcurves.grad is not None
        optimizer.step()

        if args.frozen_init_lr > 0 and len(frozencurves) > 0:
            frozen_optimizer.step()

        step_time = time.time() - start_time

        ############ Monitoring ####################
        import matplotlib.pyplot as plt

        # Record losses
        losses.append(loss.item())
        cliplosses.append(cliploss.item())
        lpipslosses.append(lpipsloss.item())
        viewlosses.append(viewloss.item())

        ## Check for best loss
        if loss.item() < bestloss:
            bestloss = loss.item()
            bestcurves = torch.clone(optimcurves).detach()
            bestkeep = torch.clone(keepcurves).detach()

            if args.frozen_init_lr > 0 and len(frozencurves) > 0:
                bestfrozencurves = torch.clone(frozencurves).detach()

            bestiter = iteri

        if save_views:
            ## Save pred and gt side by side
            for viewi in range(nviews):
                fig, axs = plt.subplots(1, 2)

                pred = torchvision.transforms.functional.to_pil_image(imgs[viewi, :3,].cpu().detach())
                gtimg = torchvision.transforms.functional.to_pil_image(iter_gt[viewi, :3,].cpu().detach())

                axs[0].imshow(pred)
                axs[0].axis('off')
                axs[0].set_title("Pred")

                axs[1].imshow(gtimg)
                axs[1].axis('off')
                axs[1].set_title("GT")

                plt.axis('off')
                fig.suptitle(f"Iter {iteri} View {viewi}")
                plt.savefig(os.path.join(vizdir, f"{iteri:03d}_view{viewi}.png"))
                plt.close(fig)
                plt.cla()

        # Save monitoring view
        if iteri % monitor_period == 0 or iteri == niters - 1:
            monitor_i = iteri % len(gtrenders)

            monitor_img = render_curves(curves, gt_position[[monitor_i]], gt_lookat[[monitor_i]], 1, len(curves),
                                        npoints, canvas_width, canvas_height,
                                        iteri+1, fov=gt_fov, colors=sample_colors, device=device)[0][0]
            monitor_img = torchvision.transforms.functional.to_pil_image(monitor_img.cpu().detach())

            gt_img = torchvision.transforms.functional.to_pil_image(gt[monitor_i,:,:,:].cpu().detach())
            monitor_compare_img = compare_imgs([monitor_img, gt_img], [f"Iter: {iteri:>5}", "GT"])
            monitor_compare_img = monitor_compare_img.convert("RGB")
            monitor_compare_img.save(os.path.join(monitorcomparedir, f"{iteri:05d}.png"))

        # Save latest iter values
        if iteri % save_progress_period == 0 or iteri == niters - 1:
            with open(os.path.join(outputdir, "latestiter"), "wb") as f:
                pickle.dump(iteri, f)

            torch.save(optimcurves.detach().cpu(), os.path.join(outputdir, "optimcurves.pt"))
            torch.save(optimizer.state_dict(), os.path.join(outputdir, "optimstate.pt"))
            torch.save(keepcurves.detach().cpu(), os.path.join(outputdir, "keepcurves.pt"))

            if args.frozen_init_lr > 0 and len(frozencurves) > 0:
                torch.save(frozencurves.detach().cpu(), os.path.join(outputdir, "frozenoptcurves.pt"))
                torch.save(frozen_optimizer.state_dict(), os.path.join(outputdir, "frozen_optimstate.pt"))

                torch.save(bestfrozencurves.detach().cpu(), os.path.join(outputdir, "best_frozencurves.pt"))

            with open(os.path.join(outputdir, "bestiter"), "wb") as f:
                pickle.dump(bestiter, f)

            torch.save(bestcurves.detach().cpu(), os.path.join(outputdir, "bestcurves.pt"))
            torch.save(bestkeep.detach().cpu(), os.path.join(outputdir, "bestkeep.pt"))
            torch.save(bestcurves.detach().cpu()[bestkeep.detach().cpu()], os.path.join(outputdir, "best_keepcurves.pt"))

            with open(os.path.join(outputdir, "bestloss"), "wb") as f:
                pickle.dump(bestloss, f)

            with open(os.path.join(outputdir, "losshistory.pkl"), "wb") as f:
                pickle.dump((losses, cliplosses, lpipslosses, viewlosses, spherelosses, coveragelosses, sdflosses), f)

        total_time = time.time() - start_time

        # Print info
        logstr = f"{iteri:4d} | {step_time:.2f}/{total_time:.2f} | {cliploss:0.4f} | {viewloss:0.4f} "

        if args.lambda_lpips > 0:
            logstr += f"| {lpipsloss.item():0.4f}"
        if args.lambda_sdf > 0:
            logstr += f"| {sdfloss.item():0.4f}"
        if args.lambda_coverage > 0:
            logstr += f"| {coverageloss.item():0.4f}"

        logstr += f"| {loss:0.4f}"
        new_print(logstr)

    # Save final curves
    torch.save(optimcurves.detach().cpu(), os.path.join(outputdir, f"finalcurves.pt"))
    torch.save(bestcurves.detach().cpu(), os.path.join(outputdir, f"bestcurves.pt"))

    if len(frozencurves) > 0:
        # Also save the tot curves
        torch.save(torch.cat([frozencurves, optimcurves]).detach().cpu(), os.path.join(outputdir, f"tot_finalcurves.pt"))
        torch.save(torch.cat([frozencurves, bestcurves]).detach().cpu(), os.path.join(outputdir, f"tot_bestcurves.pt"))

    if args.frozen_init_lr > 0 and len(frozencurves) > 0:
        torch.save(frozencurves.detach().cpu(), os.path.join(outputdir, "final_frozencurves.pt"))
        torch.save(bestfrozencurves.detach().cpu(), os.path.join(outputdir, "best_frozencurves.pt"))

    # Save final loss plot
    new_print("Generating loss plot")

    fig, axs = plt.subplots()

    axs.plot(np.arange(len(losses)), losses, "k-", label="Total loss")
    axs.plot(np.arange(len(cliplosses)), cliplosses, label="CLIP loss")
    axs.plot(np.arange(len(viewlosses)), viewlosses, label="Viewbox loss")

    if args.lambda_lpips > 0:
        axs.plot(np.arange(len(lpipslosses)), lpipslosses, label="LPIPS loss")
    if args.lambda_coverage > 0:
        axs.plot(np.arange(len(coveragelosses)), coveragelosses, label="Coverage loss")
    if args.lambda_sdf > 0:
        axs.plot(np.arange(len(sdflosses)), sdflosses, label="SDF loss")

    axs.set_title(f"Losses (Seed: {seed})")
    axs.legend()
    plt.savefig(os.path.join(outputdir, f"loss.png"))
    plt.close(fig)
    plt.cla()

    # Do 360 render of the final optimized curve points alongside the GT
    new_print("Generating final 360 renders")

    # Save gifs for best and final curves
    if bestfrozencurves is None:
        bestfrozencurves = frozencurves.detach()

    for finalname, finalcurves in [('best', torch.cat([bestfrozencurves, bestcurves[bestkeep]])), ('final', torch.cat([frozencurves, optimcurves[keepcurves]]))]:

        sample_colors = None

        imgs = render_curves(finalcurves, gt_position, gt_lookat, finalviews, len(finalcurves), npoints, 512, 512,
                             colors=sample_colors, device=device)[0]

        finaldir = os.path.join(outputdir, f"{finalname}")
        Path(finaldir).mkdir(exist_ok=True, parents=True)

        finalcomparedir = os.path.join(outputdir, f"{finalname}compare")
        Path(finalcomparedir).mkdir(exist_ok=True, parents=True)

        finalrawdir = os.path.join(outputdir, f"{finalname}raw")
        Path(finalrawdir).mkdir(exist_ok=True, parents=True)

        imgs = imgs.cpu().detach()

        for viewi in range(finalviews):
            pred = torchvision.transforms.functional.to_pil_image(imgs[viewi, :3,])
            gtimg = torchvision.transforms.functional.to_pil_image(gt[viewi, :3,])
            if finalname == 'best':
                title = f"Best Pred (Best iter: {bestiter}, Seed: {seed})"
            else:
                title = f"Final Pred (Num iter: {niters}, Seed: {seed})"

            # Save comparison
            combineimg = compare_imgs([pred, gtimg], [title, "GT"])
            combineimg.save(os.path.join(finalcomparedir, f"{viewi:03d}.png"))

            # Save just curves (with annotation)
            combineimg = compare_imgs([pred], [title])
            combineimg.save(os.path.join(finaldir, f"{viewi:03d}.png"))

            # Save raw curves image
            pred.save(os.path.join(finalrawdir, f"{viewi:03d}.png"))

        # Make gifs
        make_gif(f"{finalcomparedir}/*.png", os.path.join(outputdir, f"{finalname}_compare.gif"))
        make_gif(f"{finaldir}/*.png", os.path.join(outputdir, f"{finalname}.gif"))

    if args.colorize:
        for finalname, finalcurves in [('color_best', torch.cat([bestfrozencurves, bestcurves[bestkeep]])), ('color_final', torch.cat([frozencurves, optimcurves[keepcurves]]))]:

            imgs = render_curves(finalcurves, gt_position, gt_lookat, finalviews, len(finalcurves), npoints,
                                 512, 512, colors=curve_colors,
                                 device=device)[0]

            finaldir = os.path.join(outputdir, f"{finalname}")
            Path(finaldir).mkdir(exist_ok=True, parents=True)

            finalcomparedir = os.path.join(outputdir, f"{finalname}compare")
            Path(finalcomparedir).mkdir(exist_ok=True, parents=True)

            finalrawdir = os.path.join(outputdir, f"{finalname}raw")
            Path(finalrawdir).mkdir(exist_ok=True, parents=True)

            imgs = imgs.cpu().detach()

            for viewi in range(finalviews):
                pred = torchvision.transforms.functional.to_pil_image(imgs[viewi, :3,])
                gtimg = torchvision.transforms.functional.to_pil_image(gt[viewi, :3,])
                if finalname == 'color_best':
                    title = f"Best Pred (Best iter: {bestiter}, Seed: {seed})"
                else:
                    title = f"Final Pred (Num iter: {niters}, Seed: {seed})"

                # Save comparison
                combineimg = compare_imgs([pred, gtimg], [title, "GT"])
                combineimg.save(os.path.join(finalcomparedir, f"{viewi:03d}.png"))

                # Save just curves (with annotation)
                combineimg = compare_imgs([pred], [title])
                combineimg.save(os.path.join(finaldir, f"{viewi:03d}.png"))

                # Save raw curves image
                pred.save(os.path.join(finalrawdir, f"{viewi:03d}.png"))

            # Make gifs
            make_gif(f"{finalcomparedir}/*.png", os.path.join(outputdir, f"{finalname}_compare.gif"))
            make_gif(f"{finaldir}/*.png", os.path.join(outputdir, f"{finalname}.gif"))

    # Make progress gif
    # new_print("Making progress gifs")
    # make_gif(f"{monitordir}/*.png", os.path.join(outputdir, "progress.gif"), duration=10)
    # make_gif(f"{monitorcomparedir}/*.png", os.path.join(outputdir, "progress_compare.gif"), duration=10)

    new_print("Done")
    return bestloss, seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### Directory and input settings
    parser.add_argument("meshpath", type=str)
    parser.add_argument("--outputdir", type=str, required=True)
    parser.add_argument("--georenderdir", nargs="+", type=str, default=[])
    parser.add_argument("--semrenderdir", nargs="+", type=str, default=[])
    parser.add_argument("--geo_gtrenderdir", nargs="+", type=str)
    parser.add_argument("--sem_gtrenderdir", nargs="+", type=str)

    parser.add_argument("--overwrite", action='store_true', help="delete any existing result for the seed and restart (rather than continuing)")
    parser.add_argument("--overwrite-all", action='store_true', help="overwrites entire output directory (should NOT be used for seed-varying runs)")
    parser.add_argument("--setseed", action='store_true')
    parser.add_argument("--debug", action='store_true')

    ### Initialization
    parser.add_argument("--geoseed", type=int, default=0, help='seed to set for single run')
    parser.add_argument("--semseed", type=int, default=0, help='seed to set for single run')
    parser.add_argument("--inits", type=str, nargs="+", help="list of filepaths to initial curves, all listed curves will be loaded as the initialization", default=None)
    parser.add_argument("--frozen-init", action='store_true', help="keeps init curves fixed and adds randomized curves equal to ncurves to train")
    parser.add_argument("--fpgeodesics", action='store_true', help="furthest point init connect with geodesics")
    parser.add_argument("--normalize_scale", type=float, default=1.)

    ### Render parameters
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    ### Geometry curves ###
    parser.add_argument("--ngeocurves", type=int, default=30, help="number of randomly generated curves to initialize")
    parser.add_argument("--init-type-geo", type=str, choices={'random', 'furthest', 'keypoint', 'vertex'}, help="type of initialization to use", default='furthest')
    parser.add_argument("--init_keypoint_dir_geo", type=str, default=None, help="path to .pt file containing the initialization keypoints")
    parser.add_argument("--geo_max_init_keypoints", type=int, default=None)
    parser.add_argument("--geo_frozen_init_lr", type=float, help="small optimization on the frozen init", default=0)
    parser.add_argument("--geo-lambda-view",type=float, default=1)
    parser.add_argument("--geo_prune_curves", type=float, default=0.05)

    ### Geometry curve Losses
    parser.add_argument("--geo_clip_model_name", type=str, default='RN101', help="CLIP model name")
    parser.add_argument("--geo_clip_conv_layer_weights", type=float, nargs="+", default=[0,0,1.,1.,0], help="weight layers for the CLIPasso loss")
    parser.add_argument("--geo_clip_fc_weight", type=float, default=0.1, help="fully connected CLIPasso loss")
    parser.add_argument("--geo_clip_fc_losstype", type=str, choices={'L2', "Cos", "L1"}, default="L2", help="FC CLIP loss type")
    parser.add_argument("--geo_clip_num_augs",type=int, default=4)

    parser.add_argument("--geo-lambda-clip",type=float, default=1)

    parser.add_argument("--geo-lambda-lpips",type=float, default=0)

    parser.add_argument("--geo-lambda-coverage",type=float, default=0)

    parser.add_argument("--geo-lambda-sdf",type=float, default=0)

    parser.add_argument("--geo-lambda-sphere",type=float, default=1)

    parser.add_argument("--geo-spatial-keypoint-loss", type=str, default=None, help="path to .pt file containing the keypoints")
    parser.add_argument("--geo-spatial-weight-base",type=float, default=1)
    parser.add_argument("--geo-spatial-keypoint-sigma",type=float, default=0.05)
    parser.add_argument("--geo-spatial-fc", action="store_true", help='pool the spatial weights for the FC loss')
    parser.add_argument("--geo-zbuffer-dir", nargs="+", type=str, help='name of the folder containing the zbuffers', default=[])

    ### Semantic curves ###
    parser.add_argument("--nsemcurves", type=int, default=3, help="number of randomly generated curves to initialize")
    parser.add_argument("--init-type-sem", type=str, choices={'random', 'furthest', 'keypoint', 'vertex'}, help="type of initialization to use", default='keypoint')
    parser.add_argument("--init_keypoint_dir_sem", type=str, default=None, help="path to .pt file containing the initialization keypoints")
    parser.add_argument("--sem_max_init_keypoints", type=int, default=None)
    parser.add_argument("--sem_frozen_init_lr", type=float, help="small optimization on the frozen init", default=0)
    parser.add_argument("--sem-lambda-view",type=float, default=1)
    parser.add_argument("--sem_only_geo", action='store_true', help="if true, then only optimize the geocurves during sem stage without new curves")
    parser.add_argument("--sem_prune_curves", type=float, default=0.05)

    ### Semantic curve Losses
    parser.add_argument("--sem_clip_model_name", type=str, default='RN101', help="CLIP model name")
    parser.add_argument("--sem_clip_conv_layer_weights", type=float, nargs="+", default=[0,0,1,1,0], help="weight layers for the CLIPasso loss")
    parser.add_argument("--sem_clip_fc_weight", type=float, default=75., help="fully connected CLIPasso loss")
    parser.add_argument("--sem_clip_fc_losstype", type=str, choices={'L2', "Cos", "L1"}, default="L2", help="FC CLIP loss type")
    parser.add_argument("--sem-clip-num-augs",type=int, default=4)

    parser.add_argument("--sem-lambda-clip",type=float, default=1)

    parser.add_argument("--sem-lambda-coverage",type=float, default=0)

    parser.add_argument("--sem-lambda-lpips",type=float, default=1)

    parser.add_argument("--sem-lambda-sdf",type=float, default=1)

    parser.add_argument("--sem-lambda-sphere",type=float, default=1)

    parser.add_argument("--sem-spatial-keypoint-loss", type=str, default=None, help="path to .pt file containing the keypoints")
    parser.add_argument("--sem-spatial-weight-base",type=float, default=1)
    parser.add_argument("--sem-spatial-keypoint-sigma",type=float, default=0.05)
    parser.add_argument("--sem-spatial-fc", action="store_true", help='pool the spatial weights for the FC loss')
    parser.add_argument("--sem-zbuffer-dir", nargs="+", type=str, help='name of the folder containing the zbuffers', default=[])

    ### Optimization parameters
    parser.add_argument("--geonviews", type=int, default=1, help="number of views to use at each step (batch size)")
    parser.add_argument("--semnviews", type=int, default=1, help="number of views to use at each step (batch size)")
    parser.add_argument("--geoniters", type=int, default=20000, help="number of iterations to train to")
    parser.add_argument("--semniters", type=int, default=20000, help="number of iterations to train to")
    parser.add_argument("--nruns", type=int, default=1, help="number of runs to perform to select best")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

    ### Surface sampling
    parser.add_argument("--surfsamples", type=int, default=20)
    parser.add_argument("--coverage_samples", type=int, default=5000)

    ### Monitoring
    parser.add_argument("--colorize", action='store_true', help="differentiates frozen, geo, and sem curves by color")

    ### Multiprocessing
    parser.add_argument("--mp", action='store_true')

    args = parser.parse_args()

    # Delete entire directory if doing full overwrite
    if args.overwrite:
        if os.path.exists(args.outputdir):
            clear_directory(args.outputdir)

    # Stage 1
    geodir = os.path.join(args.outputdir, f"seed{args.geoseed}", "geo")
    geocurves_path = os.path.join(geodir, "bestcurves.pt")
    geocurves_result = os.path.join(geodir, "best_compare.gif")

    if args.overwrite and os.path.exists(geodir):
        clear_directory(geodir)

    geodone = False
    if os.path.exists(geocurves_path) and os.path.exists(geocurves_result):
        print(f"{geocurves_result} exists. Using this for the semantics initialization ...")
        geodone = True

    if args.ngeocurves > 0 and not geodone:

        # Set up all the things
        args.gtrenderdir = args.geo_gtrenderdir
        args.ncurves = args.ngeocurves
        args.nviews = args.geonviews
        args.niters = args.geoniters
        args.renderdir = args.georenderdir
        args.init_type = args.init_type_geo
        args.init_keypoint_dir = args.init_keypoint_dir_geo
        args.max_init_keypoints = args.geo_max_init_keypoints
        args.frozen_init_lr = args.geo_frozen_init_lr
        args.prune_curves = args.geo_prune_curves

        args.clip_model_name = args.geo_clip_model_name
        args.clip_conv_layer_weights= args.geo_clip_conv_layer_weights
        args.clip_fc_weight = args.geo_clip_fc_weight
        args.clip_fc_losstype = args.geo_clip_fc_losstype
        args.clip_num_augs = args.geo_clip_num_augs

        args.lambda_clip = args.geo_lambda_clip
        args.lambda_coverage = args.geo_lambda_coverage
        args.lambda_lpips = args.geo_lambda_lpips

        args.spatial_keypoint_loss = args.geo_spatial_keypoint_loss
        args.spatial_weight_base = args.geo_spatial_weight_base
        args.spatial_keypoint_sigma = args.geo_spatial_keypoint_sigma
        args.spatial_fc = args.geo_spatial_fc
        args.zbuffer_dir = args.geo_zbuffer_dir

        args.lambda_sdf = args.geo_lambda_sdf
        args.lambda_sphere = args.geo_lambda_sphere
        args.lambda_view = args.geo_lambda_view

        run(args, stage='geo', seed=args.geoseed, setseed=args.setseed)

    # Stage 2
    semdir = os.path.join(args.outputdir, f"seed{args.semseed}", "sem")
    semcurves_path = os.path.join(semdir, "bestcurves.pt")
    semcurves_result = os.path.join(semdir, "best_compare.gif")

    if args.overwrite and os.path.exists(semdir):
        clear_directory(semdir)

    if os.path.exists(semcurves_path) and os.path.exists(semcurves_result):
        print(f"{semcurves_result} exists. We're done.")
        exit(0)

    if args.nsemcurves > 0:

        # Set up all the things
        args.gtrenderdir = args.sem_gtrenderdir

        # Load and freeze the geometry curves if they exist
        if args.ngeocurves > 0:
            geocurves_path = os.path.join(args.outputdir, f"seed{args.geoseed}", "geo", "best_keepcurves.pt")
            assert os.path.exists(geocurves_path), f"{geocurves_path} not found!"

            args.inits = [geocurves_path]

            # If geofrozencurves, then load these as well
            frozengeocurves_path = os.path.join(geodir, "best_frozencurves.pt")
            if os.path.exists(frozengeocurves_path):
                args.inits.append(frozengeocurves_path)

            if args.sem_only_geo:
                args.frozen_init = False
            else:
                args.frozen_init = True

        args.ncurves = args.nsemcurves
        args.nviews = args.semnviews
        args.niters = args.semniters
        args.renderdir = args.semrenderdir
        args.init_type = args.init_type_sem
        args.init_keypoint_dir = args.init_keypoint_dir_sem
        args.max_init_keypoints = args.sem_max_init_keypoints
        args.frozen_init_lr = args.sem_frozen_init_lr
        args.prune_curves = args.sem_prune_curves

        args.clip_model_name = args.sem_clip_model_name
        args.clip_conv_layer_weights= args.sem_clip_conv_layer_weights
        args.clip_fc_weight = args.sem_clip_fc_weight
        args.clip_fc_losstype = args.sem_clip_fc_losstype
        args.clip_num_augs = args.sem_clip_num_augs

        args.lambda_clip = args.sem_lambda_clip
        args.lambda_coverage = args.sem_lambda_coverage
        args.lambda_lpips = args.sem_lambda_lpips

        args.spatial_keypoint_loss = args.sem_spatial_keypoint_loss
        args.spatial_weight_base = args.sem_spatial_weight_base
        args.spatial_keypoint_sigma = args.sem_spatial_keypoint_sigma
        args.spatial_fc = args.sem_spatial_fc
        args.zbuffer_dir = args.sem_zbuffer_dir

        args.lambda_sdf = args.sem_lambda_sdf
        args.lambda_sphere = args.sem_lambda_sphere
        args.lambda_view = args.sem_lambda_view

        run(args, stage='sem', seed=args.semseed, setseed=args.setseed)
