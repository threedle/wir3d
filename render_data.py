# Render 360 views of models
import numpy as np
import torch
import os

# This is basically a glorified meshgrid clone
def gen_elev_azim(elev_1, elev_2, elev_n, azim_1, azim_2, azim_n,
                  center_elev=None, center_azim=None,
                  device=torch.device('cpu')):

    if center_elev is not None and center_azim is not None:
        elev = []
        azim = []

        for ce, ca in zip(center_elev, center_azim):
            elev.append(torch.linspace(ce + elev_1, ce + elev_2, elev_n).repeat_interleave(azim_n).float().to(device))

            azim_steps = torch.linspace(ca + azim_1, ca + azim_2, azim_n+1)[:-1] # This excludes the last step to avoid overlap
            azim.append(azim_steps.tile(elev_n).float().to(device))
        elev = torch.cat(elev)
        azim = torch.cat(azim)
    else:
        elev = torch.linspace(elev_1, elev_2, elev_n).repeat_interleave(azim_n).float().to(device)

        azim_steps = torch.linspace(azim_1, azim_2, azim_n+1)[:-1] # This excludes the last step to avoid overlap
        azim = azim_steps.tile((elev_n,)).float().to(device)
    return elev, azim

def get_orthogonal_vector(v):
    # Get an orthogonal vector to v
    # Algorith: https://math.stackexchange.com/questions/133177/finding-a-unit-vector-perpendicular-to-another-vector
    if torch.allclose(v, torch.zeros_like(v)):
        raise ValueError("Cannot get orthogonal vector to zero vector")

    m = torch.where(~torch.isclose(v, torch.zeros_like(v)))[0][0]
    n = (m + 1) % 3

    y = torch.zeros_like(v)
    y[m] = -v[n]
    y[n] = v[m]

    return y / torch.linalg.norm(y)

def get_cross_product_matrix(v):
    # From: https://wikimedia.org/api/rest_v1/media/math/render/svg/e3ddca93f49b042e6a14d5263002603fc0738308
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

def get_rotation_from_axis_and_angle(axis, angle):
    # From: https://en.wikipedia.org/wiki/Rotation_matrix#:~:text=Rotation%20matrix%20from%20axis%20and%20angle
    cp = get_cross_product_matrix(axis).to(axis.device)
    return torch.cos(angle) * torch.eye(3, device=axis.device) + torch.sin(angle) * cp + (1 - torch.cos(angle)) * torch.outer(axis, axis)

def get_rotation(v1, v2):
    # Get rotation matrix to rotate v1 to v2
    # NOTE: v1 v2 must be unit vectors

    # Batched
    if len(v1.shape) > 1 and len(v2.shape) > 1:
        v = torch.linalg.cross(v1, v2, dim=1)
        s = torch.linalg.norm(v, dim=1)
        c = torch.einsum('ij,ij->i', v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        antiparallel_mask = torch.isclose(c, torch.tensor(-1., device=c.device))
        if torch.any(antiparallel_mask):
            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1[antiparallel_mask])
            R_antiparallel = get_rotation_from_axis_and_angle(ortho, np.pi)
            R = torch.eye(3).repeat(v1.size(0), 1, 1)
            R[antiparallel_mask] = R_antiparallel
        else:
            R = torch.eye(3).repeat(v1.size(0), 1, 1)

        # NOTE: When parallel, the answer is identity and is correct
        vx = torch.zeros((v1.size(0), 3, 3), device=v1.device)
        vx[:, 0, 1] = -v[:, 2]
        vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2]
        vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]
        vx[:, 2, 1] = v[:, 0]

        R[antiparallel_mask == False] += vx[antiparallel_mask == False] + torch.bmm(vx[antiparallel_mask == False], vx[antiparallel_mask == False]) * (1 / (1 + c[antiparallel_mask == False])).unsqueeze(1).unsqueeze(2)

        torch.testing.assert_close(torch.matmul(R, v1.unsqueeze(-1)).squeeze(), v2, rtol=1e-5, atol=1e-5)
    else:
        v = torch.linalg.cross(v1, v2)
        s = torch.linalg.norm(v)
        c = torch.dot(v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        if torch.allclose(c, torch.tensor(-1., device=c.device)):
            print("get_rotation: Antiparallel vectors detected")

            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1)
            return get_rotation_from_axis_and_angle(ortho, torch.tensor(np.pi, device=v1.device))

        # NOTE: When parallel, the answer is identity and is correct
        vx = torch.tensor([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]], device=v1.device)

        R = torch.eye(3, device=v1.device) + vx + vx @ vx * 1 / (1 + c)

        torch.testing.assert_close(torch.matmul(R, v1), v2, rtol=1e-5, atol=1e-5)

    return R

def get_pos_from_elev(elev, azim, r=3.0, origin=torch.zeros(3), origin_vector=None,
                      device=torch.device('cpu'), blender=False):
    """
    Convert tensor elevation/azimuth values into camera projections (with respect to origin/origin_vector)

    Base conversion assumes (1,0,0) vector as the origin vector.

    Args:
        elev (torch.Tensor): elevation
        azim (torch.Tensor): azimuth
        r (float, optional): radius. Defaults to 3.0.

    Returns:
        camera position vectors
    """
    if blender:
        # Y and Z axes are swapped, and rotation is opposite direction
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.cos(elev) * torch.sin(-azim)
        z = r * torch.sin(elev)
    else:
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)

    if len(x.shape) == 0:
        pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
    else:
        pos = torch.stack([x, y, z], dim=1).to(device)

    # Apply rotation matrix to origin vector
    if origin_vector is not None:
        origin_vector /= torch.linalg.norm(origin_vector)
        rotation_matrix = get_rotation(torch.tensor([1., 0., 0.], device=device), origin_vector.to(device))
        pos = torch.mm(rotation_matrix, pos.T).T

    return pos + origin.to(device)

def render(renderdir, meshdir, positions, lookats, fov, fp_out=None, texturedir=None, rast_option=0,
           opacity = False, keypointdir=None, keypoint_radius=0.01, keypoint_visibility=False,
           device=torch.device('cpu'), normalize=False, scale=1., up = torch.tensor([0.0, 1.0, 0.0]),
           resolution=(512, 512)):
    import torchvision
    from new_renderer import Renderer
    from pathlib import Path
    import json
    import shutil
    from tqdm import trange
    import igl

    # if os.path.exists(renderdir):
    #     shutil.rmtree(renderdir)
    Path(renderdir).mkdir(parents=True, exist_ok=True)

    # Zbufferdir
    zbufferdir = renderdir + "_zbuffer"
    if os.path.exists(zbufferdir):
        shutil.rmtree(zbufferdir)
    Path(zbufferdir).mkdir(parents=True, exist_ok=True)

    if keypointdir is not None:
        renderkpdir = renderdir + "_kp"
        if os.path.exists(renderkpdir):
            shutil.rmtree(renderkpdir)
        Path(renderkpdir).mkdir(parents=True, exist_ok=True)

    basedir = os.path.dirname(meshdir)
    meshname = os.path.basename(meshdir).split('.')[0]

    if texturedir is not None:
        assert os.path.exists(texturedir), f"{texturedir} not found"

    renderer = Renderer(device, dim=resolution, interpolation_mode = 'bilinear', fov=fov)
    vertices, vt, n, faces, ftc, _ = igl.read_obj(meshdir)

    if normalize:
        # Normalize based on bounding box mean
        from igl import bounding_box
        bb_vs, bf = bounding_box(vertices)
        vertices -= np.mean(bb_vs, axis=0)
        vertices /= (np.max(np.linalg.norm(vertices, axis=1)) / scale)

    vertices = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces).long().to(device)
    up = up.to(device)

    if texturedir is not None:
        tex = torchvision.io.read_image(texturedir).float().to(device) / 255.

        uvs = torch.from_numpy(vt).float().to(device)
        uvfs = torch.from_numpy(ftc).long().to(device)

        soupvs = vertices[faces].reshape(-1, 3)
        soupuvs = uvs[uvfs].reshape(-1, 2)
        assert len(soupvs) == len(soupuvs)
        soupfs = torch.from_numpy(np.arange(len(soupvs)).reshape(-1, 3)).to(device)
        vertices, faces, uvs, uvfs = soupvs, soupfs, soupuvs, soupfs

        soupuv = uvs[uvfs].reshape(-1, 3, 2)
    else:
        colors = torch.ones((len(vertices), 3)).to(device) * 0.7

    # Load keypoints if they exist
    keypoints = None
    if keypointdir is not None:
        keypoints = torch.load(keypointdir, map_location=device)

    # Default light positions: lights at each point of unit cube
    base_l_positions = torch.tensor([[1., 0., 0.],
                                     [0., 0., 1.],
                                     [-1., 0., 0.],
                                     [0., 0., -1.],
                                     [0., 1., 0.],
                                     [0., -1., 0.],], device=device)

    keypoint_masks = []

    # Debugging
    # lookats = torch.zeros_like(positions)
    # lookats[:, 2] = 1

    for i in trange(positions.shape[0]):
        l_position = torch.cat([positions[[i]], base_l_positions])

        # Compute view matrix manually
        # forward = lookats[i] - positions[i]
        # forward = forward / torch.linalg.norm(forward)
        # right = torch.linalg.cross(forward, up.to(device))
        # right = right / torch.linalg.norm(right)
        # up = torch.linalg.cross(right, forward)
        # up = up / torch.linalg.norm(up)

        # view_matrix = torch.eye(4, device=device)
        # view_matrix[0, :3] = right
        # view_matrix[1, :3] = up
        # view_matrix[2, :3] = -forward  # Negate forward to look down -Z in OpenGL-style conventions
        # view_matrix[:3, 3] = -torch.matmul(view_matrix[:3, :3], positions[i])
        # view_matrix = None

        # view_matrix = torch.stack([forward.squeeze(), right.squeeze(), up.squeeze(), -positions[i]], dim=1)
        # view_matrix = torch.cat([view_matrix, torch.tensor([[0., 0., 0., 1.]], device=device)], dim=0).unsqueeze(0)

        with torch.jit.optimized_execution(False):
            with torch.no_grad():
                if texturedir is not None:
                    renderout = renderer.render_texture(vertices, faces, soupuv, tex, uvs,
                                                        positions=positions[[i]], lookats=lookats[[i]],
                                                        l_position = l_position,
                                                        white_background=True,
                                                        up = up, mod=False, rast_option=rast_option,
                                                        keypoints=keypoints if keypointdir is not None else None,
                                                        keypoint_visibility=keypoint_visibility,
                                                        keypoint_radius=keypoint_radius,
                                                        return_zbuffer=True, clip_uv=True)
                else:
                    renderout = renderer.render_mesh(vertices, faces, colors,
                                                        positions=positions[[i]], lookats=lookats[[i]],
                                                        # view_matrix=view_matrix,
                                                        white_background=True,
                                                        l_position = l_position,
                                                        up = up, rast_option=rast_option,
                                                        keypoints=keypoints if keypointdir is not None else None,
                                                        keypoint_visibility=keypoint_visibility,
                                                        keypoint_radius=keypoint_radius,
                                                        return_zbuffer=True,
                                                        )

        if keypoints is not None:
            if keypoint_visibility:
                render, mask, keypoint_render, keypoint_mask, zbuffer = renderout
                keypoint_masks.append(keypoint_mask)
            else:
                render, mask, keypoint_render, zbuffer = renderout

            # NOTE: Keypoint renders are list of PIL images
            keypoint_render[0].save(os.path.join(renderkpdir, f"{i:04}.png"))
        else:
            render, mask, zbuffer = renderout

        if opacity:
            render = torch.cat([render, mask], dim=1)

        render = torchvision.transforms.functional.to_pil_image(render[0].cpu().detach())
        render.save(os.path.join(renderdir, f"{i:04}.png"))

        # Save zbuffer
        zbuffer = zbuffer[0].cpu().detach()
        torch.save(zbuffer, os.path.join(zbufferdir, f"{i:04}.pt"))

    renderlist = dict(
            positions=positions.tolist(),
            lookats=lookats.tolist(),
            fov = fov
        )

    if keypoints is not None and keypoint_visibility:
        keypoint_masks = torch.cat(keypoint_masks).tolist()
        renderlist['keypoint_dir'] = keypointdir
        renderlist["keypoint_masks"] = keypoint_masks

    with open(os.path.join(renderdir, "renderlist.json"), "w+") as file:
        json.dump(renderlist, file)
    print("Saved to", os.path.join(renderdir, "renderlist.json"))

    if fp_out is not None:
        make_gif(renderdir, fp_out)

        if keypoints is not None:
            make_gif(renderkpdir, fp_out.replace(".gif", "_kp.gif"))

def make_gif(renderdir, fp_out):
    import glob
    from PIL import Image

    fp_in = renderdir + "/*.png"
    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

    imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
            save_all=True, duration=30, loop=0, disposal=0)

def main():
    import os
    import numpy as np
    import argparse
    import math
    from optimize_utils import clear_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("modeldir")
    parser.add_argument("startelev", nargs="?", type=float, default=None)
    parser.add_argument("endelev", nargs="?", type=float, default=None)
    parser.add_argument("elevsamples", nargs="?", type=int, default=None)
    parser.add_argument("startazim", nargs="?", type=float, default=None)
    parser.add_argument("endazim", nargs="?", type=float, default=None)
    parser.add_argument("azimsamples", nargs="?", type=int, default=None)
    parser.add_argument("--rendername", type=str, default=None)
    parser.add_argument("--renderlistpath", type=str, default=None, help='path to renderlist.json for copying data from')
    parser.add_argument("--keypoints", type=str, default=None, help='path to keypoints if you want to visualize them over the renders')
    parser.add_argument("--keypoint_radius", type=float, default=0.01, help='radius in NDC of the visualized keypoints')
    parser.add_argument("--keypoint_visibility", action="store_true")
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--texturedir", type=str, default=None)
    parser.add_argument("--opacity", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--anchors", type=str, default=None, help='path to file with vertex-indexed anchors')
    parser.add_argument("--anchor_radius", type=float, default=0.5)
    parser.add_argument("--anchor_startelev", type=float, default=-15)
    parser.add_argument("--anchor_endelev", type=float, default=15)
    parser.add_argument("--anchor_elevsamples", type=int, default=3)
    parser.add_argument("--anchor_startazim", type=float, default=-15)
    parser.add_argument("--anchor_endazim", type=float, default=15)
    parser.add_argument("--anchor_azimsamples", type=int, default=3)

    args = parser.parse_args()

    meshdir = args.modeldir
    fov = math.radians(args.fov)
    dirname = os.path.dirname(meshdir)
    rendername = args.rendername
    renderdir = os.path.join(dirname, "renders", rendername)
    gifdir = os.path.join(renderdir, "..", f"{rendername}.gif")

    if os.path.exists(renderdir) and args.overwrite:
        clear_directory(renderdir)

        if os.path.exists(gifdir):
            os.remove(gifdir)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

    assert (args.startelev is not None and args.endelev is not None and args.elevsamples is not None and \
        args.startazim is not None and args.endazim is not None and args.azimsamples is not None) \
            or args.renderlistpath is not None, f"Must provide either elev/azim, or renderlist path"

    if args.startelev is not None:
        start_elev = math.radians(args.startelev)
        end_elev = math.radians(args.endelev)
        elev_samples = args.elevsamples
        start_azim = math.radians(args.startazim)
        end_azim = math.radians(args.endazim)
        azim_samples = args.azimsamples

        if args.rendername is None:
            rendername = f"{elev_samples}x{azim_samples}_elev_{args.start_elev}_{args.end_elev}_azim_{args.start_azim}_{args.end_azim}"
        else:
            rendername = args.rendername
        r = args.radius

        elev, azim = gen_elev_azim(start_elev, end_elev, elev_samples, start_azim, end_azim, azim_samples,
                               device=device)

        # If anchors are provided, generate positions/lookats for them
        if args.anchors is not None:
            from igl import per_vertex_normals
            from igl import read_triangle_mesh
            anchors = torch.load(os.path.join(dirname, args.anchors), weights_only=True, map_location=device)
            vertices, faces = read_triangle_mesh(meshdir)
            vertex_normals = torch.tensor(per_vertex_normals(vertices, faces), device=device).float()

            positions = []
            blender_positions = []
            blender_lookats = []
            for anchor in anchors:
                # Get closest vertex
                closest_vertex = np.argmin(np.linalg.norm(vertices - anchor.cpu().numpy(), axis=1))
                origin_normal = vertex_normals[closest_vertex]
                anchor_positions = get_pos_from_elev(elev, azim, r, blender=False,
                                                   origin_vector=origin_normal/torch.linalg.norm(origin_normal),
                                                   origin=anchor, device=device)
                positions.append(anchor_positions)

                # Swap y and z axes for blender coordinate anchors
                blender_positions.append(torch.stack([anchor_positions[:, 0],
                                                     -anchor_positions[:, 2],
                                                     anchor_positions[:, 1]], dim=1).float())
                blender_anchor = torch.tensor([anchor[0], -anchor[2], anchor[1]], device=device).float()
                blender_lookats.append(torch.stack([blender_anchor] * len(elev), dim=0))

                # NOTE: This is the SAME as doing the swap with the end positions/lookats
                # blender_normal = torch.tensor([origin_normal[0], -origin_normal[2], origin_normal[1]], device=device)
                # blender_anchor = torch.tensor([anchor[0], -anchor[2], anchor[1]], device=device).float()
                # blender_positions.append(get_pos_from_elev(elev, azim, r, blender=True,
                #                                    origin_vector=blender_normal/torch.linalg.norm(blender_normal),
                #                                    origin=blender_anchor, device=device))
                # blender_lookats.append(torch.cat([blender_anchor] * len(elev), dim=0))

            positions = torch.cat(positions, dim=0)
            blender_positions = torch.cat(blender_positions, dim=0)
            lookats = anchors.repeat_interleave(len(elev), dim=0)
            blender_lookats = torch.cat(blender_lookats, dim=0)
        else:
            # Convert elev/azims to pos/lookats (if anchors, then COB to normal/up direction)
            positions = get_pos_from_elev(elev, azim, r, blender=False, device=device)
            lookats = torch.zeros_like(positions)

            blender_positions = get_pos_from_elev(elev, azim, r, blender=True, device=device)
            blender_lookats = torch.zeros_like(blender_positions)

        # Debugging
        # lookats[:, 1] = 2.

    if args.renderlistpath is not None:
        import json
        with open(args.renderlistpath, "r") as file:
            data = json.load(file)
        positions = torch.tensor(data['positions']).float().to(device)
        lookats = torch.tensor(data['lookats']).float().to(device)
        fov = data['fov']
        rendername = args.rendername

        # Convert everything to blender coordinates and save
        blender_mat = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float().to(device)
        blender_positions = (blender_mat @ positions.T).T
        blender_lookats = (blender_mat @ lookats.T).T

    renderlist = dict(
        positions=blender_positions.tolist(),
        lookats=blender_lookats.tolist(),
        fov = fov
    )

    from pathlib import Path
    meshdir = args.modeldir
    dirname = os.path.dirname(meshdir)
    renderdir = os.path.join(dirname, "renders", rendername)
    Path(renderdir).mkdir(parents=True, exist_ok=True)

    import json
    with open(os.path.join(renderdir, "blender_renderlist.json"), "w+") as file:
        json.dump(renderlist, file)
    print("Saved to", os.path.join(renderdir, "blender_renderlist.json"))

    ## Render parameters
    if args.renderlistpath is None:
        fov = math.radians(args.fov)

    if os.path.exists(gifdir):
        print(f"Already done with {renderdir}")
        exit(0)

    # if args.anchors is not None:
    #     # Also generate for blender coordinates
    #     # NOTE: Conversion is simply blender y = -z and blender z = y
    #     blender_mat = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float().to(device)
    #     blender_lookats = (blender_mat @ anchor_lookats.T).T
    #     blender_normals = (blender_mat @ anchor_normals.T).T

    #     blender_positions = get_pos_from_elev(anchor_elev[valid], anchor_azim[valid], args.anchor_radius, blender=True,
    #                                         origin=blender_lookats, origin_vector=blender_normals,
    #                                         device=device)

    #     renderlist = dict(
    #         positions=blender_positions.tolist(),
    #         lookats=blender_lookats.tolist(),
    #         fov = fov
    #     )

    #     from pathlib import Path
    #     Path(renderdir).mkdir(parents=True, exist_ok=True)

    #     import json
    #     with open(os.path.join(renderdir, "blender_renderlist.json"), "w+") as file:
    #         json.dump(renderlist, file)
    #     print("Saved to", os.path.join(renderdir, "blender_renderlist.json"))

    render(meshdir=meshdir, renderdir=renderdir, positions=positions, lookats=lookats,
           fov=fov, rast_option=2, texturedir=args.texturedir, opacity=args.opacity, device=device,
           keypointdir=args.keypoints, keypoint_radius=args.keypoint_radius,
           keypoint_visibility=args.keypoint_visibility, scale=args.scale,
           normalize=args.normalize, resolution=(args.resolution, args.resolution),)
    make_gif(renderdir=renderdir, fp_out = os.path.join(renderdir, "..", f"{rendername}.gif"))

    if args.keypoints is not None:
        make_gif(renderdir=renderdir + "_kp", fp_out = os.path.join(renderdir, "..", f"{rendername}_kp.gif"))

if __name__ == "__main__":
    main()
