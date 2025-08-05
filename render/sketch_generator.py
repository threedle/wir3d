#!/usr/bin/env python3

import sys
import bpy
from mathutils import Vector
import math
import mathutils
import argparse
import numpy as np
import os
import json
import glob
import pip

pip.main(['install', 'torch'])
pip.main(['install', 'tqdm'])
pip.main(['install', 'pyyaml'])
pip.main(['install', 'pillow'])
pip.main(['install', 'numpy'])

from PIL import Image

from pathlib import Path
import importlib

def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)

import os
import shutil

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

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from common.bpy_util import normalize_scale, look_at, del_obj, clean_scene

"""
Shader references:
    pencil shader - https://www.youtube.com/watch?v=71KGlu_Yxtg
    white background (compositing) - https://www.youtube.com/watch?v=aegiN7XeLow
    creating transparent object - https://www.katsbits.com/codex/transparency-cycles/
"""

from tqdm import tqdm
import sys
sys.path.append("/share/data/pals/guanzhi/Stroke3D")
sys.path.append("/net/projects/ranalab/guanzhi/Stroke3D")
sys.path.append("/Users/guanzhi/Documents/Graphics/Stroke3D")
from render_data import gen_elev_azim, get_pos_from_elev

def prepare():
    # hide the main collections (if it is already hidden, there is no effect)
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].hide_viewport = True
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].exclude = True

    clean_scene()

    # setup to avoid rendering surfaces and only render the freestyle curves
    bpy.context.scene.view_layers["View Layer"].use_pass_z = False
    bpy.context.scene.view_layers["View Layer"].use_pass_combined = False
    bpy.context.scene.view_layers["View Layer"].use_sky = False
    bpy.context.scene.view_layers["View Layer"].use_solid = False
    bpy.context.scene.view_layers["View Layer"].use_volumes = False
    bpy.context.scene.view_layers["View Layer"].use_strand = True  # freestyle curves

def render_sketch(filepath, outputpath, positions, lookats, fov=np.pi/3, normalize=True, startiter=0,
                  resolution=400):

    # Import scene based on the extension
    if filepath.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=str(filepath), axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl")
    elif filepath.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=str(filepath))

    obj = bpy.context.selected_objects[0]

    # normalize the object
    if normalize:
        normalize_scale(obj)

    name="Material_Name"

    # Get the material
    blenderMat = bpy.data.materials.get(name)
    if blenderMat is None:
        # create material
        blenderMat = bpy.data.materials.new(name=name)

    # get the nodes
    blenderMat.use_nodes=True
    nodes = blenderMat.node_tree.nodes

    # clear all nodes to start clean
    for node in nodes:
        nodes.remove(node)

    # link nodes
    links = blenderMat.node_tree.links

    #create the basic material nodes
    node_output  = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = 400,0
    node_pbsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_pbsdf.location = 0,0
    node_pbsdf.inputs['Base Color'].default_value = (0.8, 0.05, 0.05, 1.0)
    node_pbsdf.inputs['Alpha'].default_value = 1 # 1 is opaque, 0 is invisible
    node_pbsdf.inputs['Roughness'].default_value = 0.2
    node_pbsdf.inputs['Specular'].default_value = 0.5
    node_pbsdf.inputs['Transmission'].default_value = 0.5 # 1 is fully transparent

    link = links.new(node_pbsdf.outputs['BSDF'], node_output.inputs['Surface'])

    blenderMat.blend_method = 'HASHED'
    blenderMat.shadow_method = 'HASHED'
    blenderMat.use_screen_refraction = True

    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_ssr_refraction = True

    # Render with transparent images
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True

    # Add material to object
    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = blenderMat
    else:
        # no slots
        obj.data.materials.append(blenderMat)

    # NOTE: Both elev and azim are flipped b/w Blender and nvdiffrast
    # eulers = [mathutils.Euler((-elev, 0.0, -azim+np.pi/2), 'XYZ') for elev, azim in zip(elevs, azims)]
    # for i, eul in enumerate(eulers):

    for i, (position, lookat) in enumerate(zip(positions, lookats)):
        target_file_name = f"{i+startiter:04}.png"
        target_file = os.path.join(outputpath, target_file_name)

        # camera setting
        # cam_pos = mathutils.Vector((0.0, -radius, 0.0))
        # cam_pos.rotate(eul)

        scene = bpy.context.scene
        bpy.ops.object.camera_add(enter_editmode=False, location=position)
        new_camera = bpy.context.active_object
        new_camera.name = "camera_tmp"
        new_camera.data.name = "camera_tmp"
        new_camera.data.lens_unit = 'FOV'
        new_camera.data.angle = fov
        look_at(new_camera, Vector(tuple(lookat)))

        # render
        scene.camera = new_camera
        scene.render.filepath = str(target_file)
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        bpy.context.scene.cycles.samples = 30
        bpy.ops.render.render(write_still=True)

        # prepare for the next camera
        del_obj(new_camera)

    # delete the obj to prepare for the next one
    del_obj(obj)

def make_gif(renderdir, fp_out):
    fp_in = renderdir + "/*.png"
    imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]

    imgs[0].save(fp=fp_out, format='GIF', append_images=imgs[1:],
            save_all=True, duration=30, loop=0, disposal=0)

if __name__ == "__main__":
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=str)
    parser.add_argument("startelev", type=float, default=None)
    parser.add_argument("endelev", type=float, default=None)
    parser.add_argument("elevsamples", type=int, default=None)
    parser.add_argument("startazim", type=float, default=None)
    parser.add_argument("endazim", type=float, default=None)
    parser.add_argument("azimsamples", type=int, default=None)
    parser.add_argument("--anchor_views", type=str, default=None, help='path to json file with anchor pos/lookat/fov')
    parser.add_argument("--rendername", type=str, required=True)
    parser.add_argument("--objname", type=str, default=None)
    parser.add_argument("--outputpath", type=str, default=None)
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args(argv)

    modelpath = args.modelpath

    start_elev = math.radians(args.startelev)
    end_elev = math.radians(args.endelev)
    elev_samples = args.elevsamples
    start_azim = math.radians(args.startazim)
    end_azim = math.radians(args.endazim)
    azim_samples = args.azimsamples
    r = args.radius
    fov = math.radians(args.fov)

    anchor_views = args.anchor_views

    assert (start_elev is not None and end_elev is not None and elev_samples is not None and \
        start_azim is not None and end_azim is not None and azim_samples is not None) or \
            anchor_views is not None, f"Must provide either elev/azim or anchor views"

    model = os.path.basename(modelpath).split(".")[0]
    modeldir = os.path.dirname(modelpath)
    rendername = args.rendername

    if args.outputpath is None:
        renderdir = f"{modeldir}/renders/{rendername}"
    else:
        renderdir = os.path.join(args.outputpath, "renders", rendername)
        Path(renderdir).mkdir(parents=True, exist_ok=True)

    if args.outputpath is None:
        gifdir = f"{modeldir}/renders"
    else:
        gifdir = os.path.join(args.outputpath, "renders")
    giffile = os.path.join(gifdir, f"{rendername}.gif")

    if args.overwrite and os.path.exists(renderdir):
        clear_directory(renderdir)

        if os.path.exists(giffile):
            os.remove(giffile)

    if anchor_views is not None:
        with open(anchor_views, "r") as file:
            data = json.load(file)
        positions = np.array(data['positions'])
        lookats = np.array(data['lookats'])
        fov = data['fov']
    else:
        # Generate the elevs and azims
        elevs, azims = gen_elev_azim(start_elev, end_elev, elev_samples, start_azim, end_azim, azim_samples)

        # Convert elev/azims to pos/lookats (if anchors, then COB to normal/up direction)
        positions = get_pos_from_elev(elevs, azims, r, blender=True).detach().cpu().numpy()
        lookats = np.zeros_like(positions)

    elevs, azims = gen_elev_azim(start_elev, end_elev, elev_samples, start_azim, end_azim, azim_samples)

    prepare()

    latest_positions = positions
    latest_lookats = lookats

    # Subset based on the most recent render
    import glob
    imagepaths = sorted(glob.glob(os.path.join(renderdir, "*.png")))
    if len(imagepaths) > 0:
        latest_positions = positions[len(imagepaths):]
        latest_lookats = lookats[len(imagepaths):]
        print(f"Resuming from render frame {len(imagepaths)}.")

    if len(positions) > 0 and len(lookats) > 0:
        render_sketch(modelpath, renderdir, latest_positions, latest_lookats, fov, normalize=args.normalize,
                      resolution=args.resolution)
    else:
        print(f"Already done with {renderdir}.")

    # Save renderlist without blender coordinatess
    positions = get_pos_from_elev(elevs, azims, r, blender=False).detach().cpu().numpy()
    with open(os.path.join(renderdir, "renderlist.json"), "w+") as file:
        json.dump(dict(
            positions = positions.tolist(),
            lookats = lookats.tolist(),
            fov = fov
        ), file)
    print("Saved to", os.path.join(renderdir, "renderlist.json"))

    if args.outputpath is None:
        gifdir = f"./data/{model}/renders"
    else:
        gifdir = os.path.join(args.outputpath, "renders")

    if os.path.exists(giffile):
        print(f"Already done with {renderdir}")
    else:
        make_gif(renderdir=renderdir, fp_out = os.path.join(gifdir, f"{rendername}.gif"))
