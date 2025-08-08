from new_renderer import draw_antialiased_circle
from optimize_twostage_final import load_renderset
import torch
import kaolin as kal
from PIL import Image
from make_gifs import make_gif
from pathlib import Path
import os

def render_kp_overlay(renderdir, keypoints_path, kpdir, zbufferdir=None, device=torch.device("cpu"),
                      kp_radius=0.01):
    positions, lookats, totrenders, fov, totzbuffers = load_renderset(renderdir, load_zbuffer=zbufferdir is not None, zbuffer_dir = zbufferdir)
    positions = torch.from_numpy(positions).to(device)
    lookats = torch.from_numpy(lookats).to(device)

    B = positions.shape[0]

    camera = kal.render.camera.Camera.from_args(
            eye=positions,
            at=lookats,
            up=torch.tensor([0., 1., 0.], dtype=torch.float),
            fov=fov,
            height=512, width=512,
            device=device
        )

    # Load keypoints and project
    keypoints = torch.load(keypoints_path, map_location=device, weights_only=True)

    keypoints_camera = camera.extrinsics.transform(keypoints.unsqueeze(0).repeat(B, 1, 1)) # Cameras x nkeypoints x 3
    keypoints_clip = camera.intrinsics.transform(keypoints_camera)
    keypoints_ndc = kal.render.camera.intrinsics.down_from_homogeneous(keypoints_clip)
    keypoints_ndc = (keypoints_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

    Path(kpdir).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for viewi, render in enumerate(tqdm(totrenders)):
        img = Image.open(render)

        if zbufferdir is not None:
            zbuffer = torch.load(totzbuffers[viewi], map_location=device, weights_only=True)

        for ki, keypoint in enumerate(keypoints_ndc[viewi]):
            if zbufferdir is not None:
                # Ignore everything outside the render window
                if torch.any(keypoint > 1) or torch.any(keypoint < 0):
                    continue

                kp_z = keypoints_clip[viewi, ki, 2] - 0.0005 # Slightly offset to avoid z fighting with the surface
                kp_x, kp_y = torch.floor(keypoint * torch.tensor([512, 512], device=device).float())

                check_z = zbuffer[kp_y.long(), kp_x.long()].item()
                if kp_z > check_z:
                    continue
                else:
                    img = draw_antialiased_circle(img, keypoint.cpu().numpy(), radius=kp_radius, scale_factor=4)
            else:
                img = draw_antialiased_circle(img, keypoint.cpu().numpy(), radius=kp_radius, scale_factor=4)

        img.save(f"{kpdir}/{viewi:04}.png")
    make_gif(f"{kpdir}/*.png", f"{kpdir}.gif", duration=10)

if __name__ == "__main__":
    import argparse
    from optimize_utils import clear_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help='model name')
    parser.add_argument("modeldir", type=str, help='path to the model directory')
    parser.add_argument("--zbuffer", type=str, default=None)
    parser.add_argument("--keypoints_path", type=str, default=None)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--rendername", type=str, default=None)
    parser.add_argument("--renderset", type=str, default='surface_0elev_360azim')
    parser.add_argument("--radius", type=float, default=0.01)

    args = parser.parse_args()

    renderset = args.renderset
    model = args.model
    modeldir = args.modeldir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renderdir = os.path.join(modeldir, model, "renders", renderset)

    kpname = args.rendername
    if kpname is None:
        kpname = f"{renderset}"

    if args.zbuffer:
        kpname += "_occ"
    else:
        kpname += "_noocc"

    kpdir = os.path.join(modeldir, model, "renders", kpname)
    Path(kpdir).mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        clear_directory(kpdir)

        if os.path.exists(f"{kpdir}.gif"):
            os.remove(f"{kpdir}.gif")

    zbufferdir = None
    if args.zbuffer:
        zbufferdir = args.zbuffer

    if os.path.exists(f"{kpdir}.gif"):
        print(f"Already done with {model} {renderset}")
        exit()

    keypoint_path = args.keypoints_path
    if keypoint_path is None:
        keypoint_path = os.path.join(modeldir, model, f"{model}_keypoints.pt")

    render_kp_overlay(renderdir, keypoint_path, kpdir, kp_radius=args.radius,
                      zbufferdir=zbufferdir, device=device)

    print(f"Done with {model} {renderset}")
