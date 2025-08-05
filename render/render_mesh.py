import os
import sys
import numpy as np
import bpy
from mathutils import Vector
import json
import argparse
import subprocess
import math
from pathlib import Path

try:
    import pip
    pip.main(['install', '--upgrade', 'pip'])
    pip.main(['install', 'imageio==2.23.0'])
    pip.main(['install', 'imageio[ffmpeg]==2.23.0'])
    pip.main(['install', 'libigl==2.6.1'])
except ModuleNotFoundError:
    print("No module named 'pip'. Skip installation of 'imageio' and 'libigl'.")


# add path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from common.file_util import read_obj, make_gif
from common.bpy_util import normalize_scale


def gen_elev_azim(elev_1, elev_2, elev_n, azim_1, azim_2, azim_n,
                  center_elev=None, center_azim=None):

    if center_elev is not None and center_azim is not None:
        elev = []
        azim = []

        for ce, ca in zip(center_elev, center_azim):
            elev.append(np.repeat(np.linspace(ce + elev_1, ce + elev_2, elev_n), azim_n))
            azim.append(np.tile(np.linspace(ca + azim_1, ca + azim_2, azim_n+1)[:-1], elev_n))
        elev = np.concatenate(elev)
        azim = np.concatenate(azim)
    else:
        elev = np.repeat(np.linspace(elev_1, elev_2, elev_n), azim_n)
        azim = np.tile(np.linspace(azim_1, azim_2, azim_n+1)[:-1], elev_n)
    return elev, azim

def get_pos_from_elev(elev, azim, r=3.0, origin=np.zeros(3), origin_vector=None,
                      blender=False):
    """
    Convert tensor elevation/azimuth values into camera projections (with respect to origin/origin_vector)

    Base conversion assumes (1,0,0) vector as the origin vector.

    Args:
        elev (np.array): elevation
        azim (np.array): azimuth
        r (float, optional): radius. Defaults to 3.0.

    Returns:
        camera position vectors
    """
    if blender:
        # Y and Z axes are swapped, and rotation is opposite direction
        x = r * np.cos(elev) * np.cos(azim)
        y = r * np.cos(elev) * np.sin(-azim)
        z = r * np.sin(elev)

        init_vector = np.array([np.cos(0) * np.cos(0), -np.cos(0) * np.sin(0), np.sin(0)])
    else:
        x = r * np.cos(elev) * np.cos(azim)
        y = r * np.sin(elev)
        z = r * np.cos(elev) * np.sin(azim)

        init_vector = np.array([np.cos(0) * np.cos(0), np.sin(0), np.cos(0) * np.sin(0)])

    if len(x.shape) == 0:
        pos = np.array([x,y,z]).reshape(1, 3)
    else:
        pos = np.stack([x, y, z], axis=1)

    # Apply rotation matrix to origin vector
    if origin_vector is not None:
        # Make sure origin vector is a unit vector
        origin_vector /= np.linalg.norm(origin_vector)
        rotation_matrix = get_rotation(init_vector, origin_vector)
        pos = np.dot(rotation_matrix, pos.T).T

    return pos + origin

def get_orthogonal_vector(v):
    # Get an orthogonal vector to v
    # Algorith: https://math.stackexchange.com/questions/133177/finding-a-unit-vector-perpendicular-to-another-vector
    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Cannot get orthogonal vector to zero vector")

    m = np.where(~np.isclose(v, np.zeros_like(v)))[0][0]
    n = (m + 1) % 3

    y = np.zeros_like(v)
    y[m] = -v[n]
    y[n] = v[m]

    return y / np.linalg.norm(y)

def get_cross_product_matrix(v):
    # From: https://wikimedia.org/api/rest_v1/media/math/render/svg/e3ddca93f49b042e6a14d5263002603fc0738308
    return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

def get_rotation_from_axis_and_angle(axis, angle):
    # From: https://en.wikipedia.org/wiki/Rotation_matrix#:~:text=Rotation%20matrix%20from%20axis%20and%20angle
    cp = get_cross_product_matrix(axis)
    return np.cos(angle) * np.eye(3) + np.sin(angle) * cp + (1 - np.cos(angle)) * np.outer(axis, axis)

def get_rotation(v1, v2):
    # Get rotation matrix to rotate v1 to v2
    # NOTE: v1 v2 must be unit vectors

    # Batched
    if len(v1.shape) > 1 and len(v2.shape) > 1:
        v = np.cross(v1, v2, axis=1)
        s = np.linalg.norm(v, axis=1)
        c = np.einsum('ij,ij->i', v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        antiparallel_mask = np.isclose(c, -1.)
        if np.any(antiparallel_mask):
            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1[antiparallel_mask])
            R_antiparallel = get_rotation_from_axis_and_angle(ortho, np.pi)
            R = np.eye(3).reshape(1, 3, 3).repeat(v1.shape[0], axis=0)
            R[antiparallel_mask] = R_antiparallel
        else:
            R = np.eye(3).reshape(1, 3, 3).repeat(v1.shape[0], axis=0)

        # NOTE: When parallel, the answer is identity and is correct
        vx = np.zeros((v1.shape[0], 3, 3))
        vx[:, 0, 1] = -v[:, 2]
        vx[:, 0, 2] = v[:, 1]
        vx[:, 1, 0] = v[:, 2]
        vx[:, 1, 2] = -v[:, 0]
        vx[:, 2, 0] = -v[:, 1]
        vx[:, 2, 1] = v[:, 0]

        R[antiparallel_mask == False] += vx[antiparallel_mask == False] + np.matmul(vx[antiparallel_mask == False], vx[antiparallel_mask == False]) * (1 / (1 + c[antiparallel_mask == False])).reshape(-1, 1, 1)

        np.testing.assert_allclose(np.matmul(R, v1.reshape(-1, 1, 3)).squeeze(), v2, rtol=1e-5, atol=1e-5)
    else:
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)

        # Edge case: antiparallel vectors
        # NOTE: Precision gets worse the closer the vectors are to anti-parallel
        if np.allclose(c, -1.):
            print("get_rotation: Antiparallel vectors detected")

            # 180 rotation about some orthogonal vector
            ortho = get_orthogonal_vector(v1)
            return get_rotation_from_axis_and_angle(ortho, np.pi)

        # NOTE: When parallel, the answer is identity and is correct
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

        R = np.eye(3) + vx + np.matmul(vx, vx) * 1 / (1 + c)

        np.testing.assert_allclose(np.matmul(R, v1), v2, rtol=1e-5, atol=1e-5)

    return R

def hex_to_vec(hex_color):
    """
    Examples:
        F72585 --> Vector((0.969, 0.145, 0.522))
        f72585 --> Vector((0.969, 0.145, 0.522))
        #f72585 --> Vector((0.969, 0.145, 0.522))
        #f72585ff --> Vector((0.969, 0.145, 0.522, 1.0))
    """
    if hex_color[0] == '#':
        hex_color = hex_color[1:]
    r = hex_color[:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    if len(hex_color) == 8:
        a = hex_color[-2:]
        return Vector((int(r, 16), int(g, 16), int(b, 16), int(a, 16))) / 255
    else:
        return Vector((int(r, 16), int(g, 16), int(b, 16))) / 255


def select_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def copy(obj):
    dup_obj = obj.copy()
    dup_obj.data = obj.data.copy()
    dup_obj.animation_data_clear()
    bpy.context.collection.objects.link(dup_obj)
    # set active
    select_obj(dup_obj)
    return dup_obj


class Blender():
    def __init__(self, args):
        self.args = args
        self.basepath = Path(__file__).resolve().parent
        self.cfg = self.get_config(Path(os.path.join(self.basepath, "config", args.config_file)))
        self.light_type = self.cfg["light_type"]
        self.num_points = self.cfg["num_pc"]
        self.pc_colors = None
        self.has_pc_color = False
        self.background_color_hex = args.background_color_hex
        self.mesh_basepath = Path(os.path.abspath(args.obj_file)).resolve().parent
        self.textureimg = None
        if args.textureimg is not None:
            self.textureimg = os.path.join(self.mesh_basepath, args.textureimg)
        self.fps = args.animate_fps
        self.clip_texture = args.clip_texture

        obj_file = args.obj_file
        if obj_file is None:
            # use active object within Blender
            self.use_mesh_color = False
            print("Using active object within Blender")
            self.mesh = bpy.context.active_object
        elif os.path.splitext(obj_file)[1] == '.obj':
            self.use_mesh_color = False
            self.mesh = self.load_obj(obj_file)
        elif os.path.splitext(obj_file)[1] == '.ply':
            self.use_mesh_color = True
            self.mesh = self.load_ply(obj_file)
        elif os.path.splitext(obj_file)[1] == '.glb':
            self.use_mesh_color = False
            bpy.ops.import_scene.gltf(filepath=obj_file)
            self.mesh = bpy.context.selected_objects[0]
        if self.args.freestyle:
            self.mark_all_edges_freestyle(self.mesh)
        object_to_rotate = self.mesh_settings(self.mesh)
        self.render_settings(args, object_to_rotate)

    def get_config(self, config_file_path: Path):
        base_config_file_path = Path(__file__).resolve().parent / 'config' / 'config.json'
        cfg = {}
        if base_config_file_path.is_file():
            print(f"Loading base config file [{base_config_file_path}]")
            with open(base_config_file_path, 'r') as base_config_file:
                cfg = json.load(base_config_file)
        if config_file_path:
            config_file_path = config_file_path.resolve()
            if config_file_path != base_config_file_path:
                print(f"Loading user specified config file [{config_file_path}]")
                with open(config_file_path, 'r') as config_file:
                    override_cfg = json.load(config_file)
                    cfg.update(override_cfg)
        print("Configuration for rendering:")
        print(cfg)
        return cfg

    @staticmethod
    def use_gpu_if_available():
        """
        allow Blender to use all available GPUs
        """
        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected!')
        except Exception:
            print('No Nvidia GPU available!')
            return
        bpy.data.scenes['Scene'].render.engine = "CYCLES"
        # set the device_type
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        # set device to GPU
        bpy.context.scene.cycles.device = "GPU"
        # get_devices detects GPU devices
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1  # using all devices, include GPU and CPU
            print(d["name"], d["use"])

    @staticmethod
    def look_at(obj_camera, point):
        loc_camera = obj_camera.matrix_world.to_translation()
        print("CAMERA LOCATION:", loc_camera)
        print("LOOK AT POINT:", point)
        direction = point - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()

    # def setup_animation_keyframes(self, object_to_rotate, fps=10, duration=8.):
    #     """
    #     set the keyframes for an animation that rotates the object 360 degrees at a constant speed
    #     """
    #     select_obj(object_to_rotate)
    #     # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    #     animation_duration_sec = duration
    #     frames_per_second = fps
    #     # -1 is needed so that the animation will loop seamlessly
    #     end_frame = int(animation_duration_sec * frames_per_second - 1)
    #     object_to_rotate.rotation_mode = 'AXIS_ANGLE'
    #     object_to_rotate.rotation_axis_angle = [object_to_rotate.rotation_axis_angle[0], 0.0, 0.0, 1.0]
    #     value_start = object_to_rotate.rotation_axis_angle[0]
    #     value_end = object_to_rotate.rotation_axis_angle[0] + 2 * math.pi
    #     object_to_rotate.animation_data_create()
    #     object_to_rotate.animation_data.action = bpy.data.actions.new(name="RotateObject")
    #     fcurve = object_to_rotate.animation_data.action.fcurves.new(data_path="rotation_axis_angle", index=0)
    #     keyframe_one = fcurve.keyframe_points.insert(frame=0.0, value=value_start)
    #     keyframe_one.interpolation = "LINEAR"
    #     keyframe_two = fcurve.keyframe_points.insert(frame=end_frame, value=value_end)
    #     keyframe_two.interpolation = "LINEAR"
    #     bpy.context.scene.frame_end = end_frame

    def setup_animation_keyframes(self, args, object_to_rotate, r=2.2, fov=60):
        """
        set the keyframes for an animation that rotates the object 360 degrees at a constant speed
        """
        import mathutils
        select_obj(object_to_rotate)

        if args.use_renderlist is not None:
            outputdir = args.out_file_path
            import shutil
            Path(outputdir).mkdir(parents=True, exist_ok=True)

            # Use the renderlists indicated by the path
            renderdir = os.path.join(os.path.dirname(outputdir), args.use_renderlist)
            with open(os.path.join(renderdir, "blender_renderlist.json"), "r") as file:
                renderlist = json.load(file)
                positions = np.array(renderlist["positions"], dtype=np.float32)
                lookats = np.array(renderlist["lookats"], dtype=np.float32)

            end_frame = len(positions) - 1

            # Copy renderlist to current directory
            shutil.copyfile(os.path.join(renderdir, "blender_renderlist.json"), os.path.join(outputdir, "blender_renderlist.json"))

            # Also copy opengl renderlist if it exists
            if os.path.exists(os.path.join(renderdir, "renderlist.json")):
                shutil.copyfile(os.path.join(renderdir, "renderlist.json"), os.path.join(outputdir, "renderlist.json"))

            fov = math.radians(fov)
        else:
            start_azim, end_azim, azim_samples, start_elev, end_elev, elev_samples = \
                math.radians(args.startazim), math.radians(args.endazim), args.azimsamples, \
                math.radians(args.startelev), math.radians(args.endelev), args.elevsamples

            renderdir = args.out_file_path
            Path(renderdir).mkdir(parents=True, exist_ok=True)

            # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Generate the position samples
            elevs, azims = gen_elev_azim(start_elev, end_elev, elev_samples, start_azim, end_azim, azim_samples)

            # Get anchor points if exist
            if args.anchors is not None:
                import torch
                from igl import per_vertex_normals

                # Get object vertices and faces as numpy arrays
                import bmesh
                bm = bmesh.new()
                bm.from_mesh(object_to_rotate.data)
                bm.verts.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

                vertices = np.array([(object_to_rotate.matrix_world @ v.co) for v in object_to_rotate.data.vertices], dtype=np.float32)

                # vertices = np.array([v.co[:] for v in bm.verts], dtype=np.float32)
                faces = []
                for face in bm.faces:
                    if len(face.verts) == 3:  # Only get triangles
                        faces.append([v.index for v in face.verts])
                faces = np.array(faces, dtype=np.int32)
                vertex_normals = per_vertex_normals(vertices, faces)

                bm.free()

                anchors = torch.load(os.path.join(self.mesh_basepath, args.anchors), weights_only=True).cpu().numpy()
                blender_anchors = np.stack((anchors[:, 0], -anchors[:, 2], anchors[:, 1]), axis=1)  # Swap y and z axes for blender coordinates

                positions = []
                opengl_positions = []
                for blender_anchor in blender_anchors:
                    closest_vertex = np.argmin(np.linalg.norm(vertices - blender_anchor, axis=1))
                    blender_normal = vertex_normals[closest_vertex]

                    # Swap y and z axes for blender coordinate anchors
                    positions.append(get_pos_from_elev(elevs, azims, r, blender=True,
                                                    origin_vector=blender_normal/np.linalg.norm(blender_normal),
                                                    origin=blender_anchor))

                    # Use the original normal for OpenGL coordinates
                    origin_normal = np.array([blender_normal[0], blender_normal[2], -blender_normal[1]])  # Swap y and z axes for OpenGL coordinates
                    anchor = np.array([blender_anchor[0], blender_anchor[2], -blender_anchor[1]])  # Swap y and z axes for OpenGL coordinates
                    opengl_positions.append(get_pos_from_elev(elevs, azims, r, blender=False,
                                                            origin_vector=origin_normal/np.linalg.norm(origin_normal),
                                                            origin=anchor))
                positions = np.concatenate(positions, axis=0)
                opengl_positions = np.concatenate(opengl_positions, axis=0)
                lookats = np.repeat(anchors, len(elevs), axis=0)
                fov = math.radians(fov)
                end_frame = azim_samples * elev_samples * len(anchors) - 1
            else:
                positions = get_pos_from_elev(elevs, azims, r, blender=True)
                opengl_positions = get_pos_from_elev(elevs, azims, r, blender=False)
                lookats = np.zeros_like(positions)
                fov = math.radians(fov)
                end_frame = azim_samples * elev_samples - 1

            # Save renderlist.json to output path
            if not os.path.exists(os.path.join(renderdir, "renderlist.json")):

                with open(os.path.join(renderdir, "renderlist.json"), "w+") as file:
                    json.dump(dict(
                        positions = opengl_positions.tolist(),
                        lookats = lookats.tolist(),
                        fov = fov
                    ), file)
                print("Saved to", os.path.join(renderdir, "renderlist.json"))

            # Save blender_renderlist.json to output path
            if not os.path.exists(os.path.join(renderdir, "blender_renderlist.json")):
                with open(os.path.join(renderdir, "blender_renderlist.json"), "w+") as file:
                    json.dump(dict(
                        positions = positions.tolist(),
                        lookats = lookats.tolist(),
                        fov = fov
                    ), file)
                print("Saved to", os.path.join(renderdir, "blender_renderlist.json"))

        # Set camera keyframe at every azimuth and elevation sample
        scene = bpy.context.scene
        scene.render.resolution_x = self.args.resolution[0]
        scene.render.resolution_y = self.args.resolution[1]
        bpy.context.scene.cycles.samples = 30

        # Search for camera objects
        cameras_obj = [cam for cam in bpy.data.objects if cam.type == 'CAMERA']

        # If none exist, then we define a new camera
        if len(cameras_obj) == 0:
            bpy.ops.object.camera_add(enter_editmode=False, location=mathutils.Vector((0, 0, 0)))
            camera = bpy.context.active_object
        else:
            camera = cameras_obj[0]  # use the first camera found

        # Set FOV
        camera.data.lens_unit = 'FOV'
        camera.data.angle = fov

        # Set camera to the scene
        scene.camera = camera

        for i, position in enumerate(positions):
            cam_pos = mathutils.Vector(position)

            camera.location = cam_pos
            camera.keyframe_insert(data_path="location", frame=i)

            # Set the lookat
            direction = mathutils.Vector(lookats[i]) - cam_pos

            # point the cameras '-Z' and use its 'Y' as up
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            camera.keyframe_insert(data_path="rotation_euler", frame=i)

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = end_frame

    def setup_lights(self):
        objs = [x.name for x in bpy.data.objects]
        if 'light' in objs:
            bpy.data.objects['light'].data.materials['Material'].node_tree.nodes['Emission'].inputs['Strength'].default_value = 1.8
            bpy.data.objects['light'].data.materials['Material'].node_tree.nodes['Emission'].inputs['Color'].default_value = [0.8, 0.77, 0.8, 1]
            bpy.data.objects['light'].scale = bpy.data.objects['light'].scale * 1.5
            bpy.data.objects['light'].location.x = 6
        # bpy.data.objects['light'].select_set(True)
        # bpy.ops.object.delete()
        # bpy.context.scene.world.light_settings.use_ambient_occlusion = True
        # bpy.context.scene.world.light_settings.ao_factor = 0.1
        # bpy.context.scene.world.light_settings.distance = 100
        if self.light_type == 'SUN':
            bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 5.24564)) #  "SUN"/"POINT"/"SPOT"/"AREA"
            bpy.data.objects['Sun'].data.color = (1.0, 0.95, 0.9626210331916809)
            bpy.data.objects['Sun'].data.angle = 0.1745329201221466
            bpy.data.objects['Sun'].data.energy = 2.0
        elif self.light_type != 'POINT':
            NotImplementedError()

    def render_settings(self, args, object_to_rotate):
        scene = bpy.data.scenes['Scene']
        # engine
        scene.render.engine = self.args.engine
        # device
        bpy.context.scene.cycles.device = self.args.device
        if self.args.device == 'GPU':
            self.use_gpu_if_available()
        # denoising
        if self.args.denoise_type:
            scene.cycles.use_denoising = True
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = self.args.denoise_type
        scene.cycles.samples = self.args.samples
        blender_render = scene.render
        blender_render.resolution_x = self.args.resolution[0]
        blender_render.resolution_y = self.args.resolution[1]
        blender_render.resolution_percentage = 100
        # freestyle (non-photorealistic rendering)
        blender_render.use_freestyle = self.args.freestyle is not None
        if self.args.freestyle:
            bpy.data.linestyles["LineStyle"].thickness = self.args.freestyle
        scene.use_nodes = True

        # camera settings
        cam = bpy.data.scenes["Scene"].objects['Camera']
        if self.args.camera_location is not None:
            cam.location.x = self.args.camera_location[0]
            cam.location.y = self.args.camera_location[1]
            cam.location.z = self.args.camera_location[2]
        bpy.context.view_layer.update()
        # assuming the objects are normalized to fit a unit sphere size
        look_at_z_offset = 0
        if "look_at_z_offset" in self.cfg:
            look_at_z_offset = self.cfg["look_at_z_offset"]
        Blender.look_at(cam, Vector((0.0, 0.0, self.mesh.location.z + look_at_z_offset)))

        # lighting
        # soft_shadow.blend already has the lights set up
        if 'blank' in bpy.data.filepath:
            self.setup_lights()

        background_color = hex_to_vec(self.background_color_hex)
        if len(background_color) == 3:
            # add alpha channel
            background_color = Vector((background_color[0], background_color[1], background_color[2], 1.0))
        if background_color[3] == 0.0:
            # no alpha over
            composite_node_tree = bpy.data.scenes["Scene"].node_tree
            composite_node_tree.links.new(composite_node_tree.nodes['Render Layers'].outputs['Image'], composite_node_tree.nodes['Composite'].inputs['Image'])
        else:
            bpy.data.scenes["Scene"].node_tree.nodes["Alpha Over"].inputs[1].default_value = background_color

        if self.args.animate:
            self.setup_animation_keyframes(args, object_to_rotate, args.radius)

    def render(self, args):
        # bpy.data.scenes['Scene'].render.filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_file)
        out_path = args.out_file_path
        if args.animate:
            out_path = f"{out_path}{os.sep}"
        bpy.data.scenes['Scene'].render.filepath = out_path
        bpy.ops.render.render(write_still=not self.args.animate, animation=self.args.animate)
        print(f"Done saving to [{out_path}]")

    def mark_all_edges_freestyle(self, mesh):
        for edge in mesh.data.edges:
            edge.use_freestyle_mark = True

    def load_obj(self, obj_file):
        if self.args.pc:
            _, self.pc_colors, _ = read_obj(obj_file)
            self.has_pc_color = len(self.pc_colors) > 0
            bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl",
                                     use_edges=False, use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                     use_groups_as_vgroups=False, use_image_search=True, split_mode='OFF')
        else:
            bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl",
                                     use_edges=True, use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                     use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
        obj = bpy.context.selected_objects[0]
        return obj

    def load_ply(self, ply_file):
        bpy.ops.import_mesh.ply(filepath=ply_file)
        ob = bpy.context.selected_objects[0]
        return ob

    def apply_modifier(self, mesh, cfg, type):
        if cfg[type]:
            bpy.context.view_layer.objects.active = mesh
            mod = mesh.modifiers.new(name='temp', type=type)
            for key, value in cfg[type].items():
                setattr(mod, key, value)
            bpy.ops.object.modifier_apply(modifier=mod.name)

    def create_mat(self, col, hsv=(0.6, 1.3, 1.0, 1.0), indiv_color=False):
        mat = bpy.data.materials.new(name="sphereMat")
        mat.use_nodes = True
        tree = mat.node_tree
        HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
        HSVNode.inputs['Color'].default_value = col
        HSVNode.inputs['Hue'].default_value = hsv[0]
        HSVNode.inputs['Saturation'].default_value = hsv[1]
        HSVNode.inputs['Value'].default_value = hsv[2]
        HSVNode.inputs['Fac'].default_value = hsv[3]
        #
        # set color brightness/contrast
        BCNode = tree.nodes.new('ShaderNodeBrightContrast')
        BCNode.inputs['Bright'].default_value = 0
        BCNode.inputs['Contrast'].default_value = 0
        tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
        tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

        # add color input node:
        if indiv_color:
            ObjInfo = tree.nodes.new('ShaderNodeObjectInfo')
            tree.links.new(ObjInfo.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

        return mat

    def add_spheres(self, mesh):
        mesh.data.update()
        bpy.context.view_layer.update()
        vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])

        print('num points: ', self.num_points)
        if self.num_points > -1:
            ids = np.random.permutation(len(vertices))
            vertices = vertices[ids[:self.num_points], :]

        m = bpy.data.meshes.new('pc')
        m.from_pydata(vertices, [], [])

        # create mesh object and link to scene collection
        o = bpy.data.objects.new('pc', m)

        # my add
        scene = bpy.context.scene
        scene.collection.objects.link(o)
        o = bpy.data.objects['pc']

        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.scene.cursor.location = self.mesh.location
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        bpy.ops.object.select_all(action='DESELECT')

        # Add minimal icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=self.cfg["sph_subdiv"], radius=self.cfg["sph_radius"])

        # Add material
        sph = bpy.context.view_layer.objects.active
        sph.data.materials.clear()
        mat = self.create_mat((0.2, 0.29, 0.3, 1), indiv_color=self.has_pc_color)
        sph.data.materials.append(mat)
        sph.active_material = mat

        # Set instancing props
        for ob in [sph, o]:
            ob.instance_type = 'VERTS'

        # set instance parenting (parent icosphere to verts)
        bpy.ops.object.select_all(action='DESELECT')
        sph.select_set(True)
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.parent_set(type='VERTEX_TRI')  # this allows correct rotation in animation mode

        obj_to_rotate = o

        # this is only if we want to render per-point colors
        if self.has_pc_color:
            # now duplicate all the spheres
            select_obj(o)
            print('duplicating all spheres to add colors...')
            bpy.ops.object.duplicates_make_real()
            bpy.context.view_layer.objects.active = bpy.data.objects[f'Icosphere.{1:03d}']

            # for the animation (in animation mode), we first create an empty
            # the spheres will have a common parent which is that empty
            # the animation will turn the empty slowly, and all the spheres will rotate in relation to it
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            obj_to_rotate = bpy.context.active_object
            obj_to_rotate.location = self.mesh.location

            # iterate through all vertices and color them
            print('Adding colors...')
            o.instance_type = 'NONE'
            bpy.ops.object.select_all(action='DESELECT')
            for i in range(len(o.data.vertices)):
                icosphere_obj = bpy.data.objects[f'Icosphere.{i+1:03d}']
                icosphere_obj.color = (self.pc_colors[i, 0], self.pc_colors[i, 1], self.pc_colors[i, 2], 1)
                icosphere_obj.select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[f'Icosphere.{1:03d}']

            if self.args.animate:
                # note that parenting a lot of points may take some time
                obj_to_rotate.select_set(True)
                bpy.context.view_layer.objects.active = obj_to_rotate
                bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            bpy.ops.object.select_all(action='DESELECT')

        # return the cursor to the origin of the scene
        bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
        return obj_to_rotate

    def import_colors(self, mesh):
        # mat = mesh.active_material
        # mat = bpy.data.materials.new(name="colorMat")
        mesh.data.materials.clear()
        mat = bpy.data.materials[self.cfg["material"]]
        mesh.data.materials.append(mat)
        mesh.active_material = mat
        # mat.use_nodes = True
        tree = mat.node_tree
        AttrNode = tree.nodes.new('ShaderNodeAttribute')
        AttrNode.attribute_name = 'Col'
        tree.links.new(AttrNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
        try:
            tree.links.new(AttrNode.outputs['Alpha'], tree.nodes['Principled BSDF'].inputs['Alpha'])
        except KeyError:
            print('bpy_prop_collection[key]: key "Alpha" not found.\n'
                  'The "Alpha" property will not be used for "Principled BSDF".')

    def mesh_settings(self, mesh):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)

        # move origin to the center of the object's bounding box
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        if self.args.normalize:
            print("Normalizing object's to fit inside a unit sphere")
            normalize_scale(mesh)

        # apply subdiv modifier
        if not self.args.pc:
            self.apply_modifier(mesh, self.cfg, "SUBSURF")
            # apply all modifiers
            bpy.ops.object.convert(target='MESH')

        vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])

        # Shift ground plane to be at the bottom of the object (min z-axis)
        floor = bpy.data.objects.get("floor")
        if floor:
            floor.location = Vector((0, 0, min(vertices[:, 2])))

        # shift bottom vertex to sit on zero (first update)
        # mesh.data.update()
        # bpy.context.view_layer.update()
        # vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])
        # mesh.location = Vector((0, 0, mesh.location.z - min(vertices[:, 2])))
        # mesh.location.x += self.cfg["obj_location_offset_x"]
        # mesh.location.y += self.cfg["obj_location_offset_y"]
        # mesh.location.z += self.cfg["obj_location_offset_z"]

        object_to_rotate = self.mesh
        # now let's handle importing the colors
        if self.use_mesh_color:
            print('adding mesh_colors...')
            self.import_colors(mesh)
            print('added mesh colors!')
        elif self.args.pc:
            print('adding spheres (pc colors)...')
            object_to_rotate = self.add_spheres(mesh)
            print('added spheres (pc colors)!')
        elif self.args.keep_material:
            pass
        else:
            # remove any material assigned to the object
            mesh.data.materials.clear()
            print('adding a material...')
            mat = bpy.data.materials[self.cfg["material"]]
            mat.use_nodes = True

            if self.textureimg is None:
                base_color = Vector((0.7, 0.7, 0.7, 1.0)) if "blank" in bpy.data.filepath else Vector((0.4, 0.4, 0.4, 1.0))
                mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = base_color
            else:
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                tex = mat.node_tree.nodes.new('ShaderNodeTexImage')

                tex.image = bpy.data.images.load(self.textureimg)

                if not self.clip_texture:
                    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])
                else:
                    mat.node_tree.nodes["Image Texture"].extension = 'CLIP'

                    # NOTE: This is a hack to set the clipped values to a base color
                    vectormath = mat.node_tree.nodes.new('ShaderNodeVectorMath')
                    # bpy.ops.node.add_node(type="ShaderNodeVectorMath", use_transform=True)
                    vectormath.operation = 'ADD'

                    math0 = mat.node_tree.nodes.new('ShaderNodeMath')
                    # bpy.ops.node.add_node(type="ShaderNodeMath", use_transform=True)
                    math0.operation = 'COMPARE'
                    math0.inputs[1].default_value = 0.0
                    math0.inputs[2].default_value = 0.001

                    mat.node_tree.links.new(tex.outputs['Color'], vectormath.inputs[0])
                    mat.node_tree.links.new(tex.outputs['Color'], math0.inputs[0])

                    math1 = mat.node_tree.nodes.new('ShaderNodeMath')
                    # bpy.ops.node.add_node(type="ShaderNodeMath", use_transform=True)
                    math1.operation = 'MULTIPLY'
                    math1.inputs[1].default_value = 0.7
                    mat.node_tree.links.new(math0.outputs['Value'], math1.inputs[0])
                    mat.node_tree.links.new(math1.outputs['Value'], vectormath.inputs[1])

                    mat.node_tree.links.new(bsdf.inputs['Base Color'], vectormath.outputs['Vector'])

            # assign the material to object
            if mesh.data.materials:
                # assign to 1st material slot
                mesh.data.materials[0] = mat
            else:
                # no slots
                mesh.data.materials.append(mat)

            print('added a material!')

        if not self.args.disable_smooth_shading and not self.args.pc:
            print("Applying smooth shading to object")
            mesh.data.use_auto_smooth = 1
            bpy.ops.object.shade_smooth()

        return object_to_rotate

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

def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser()
    parser.add_argument('obj_file')
    parser.add_argument('config_file', help="name of config file in the config directory, e.g. 'config.json'")
    parser.add_argument('out_file', help='name of render output directory')
    parser.add_argument("startelev", type=float, nargs="?", default=None)
    parser.add_argument("endelev", type=float, nargs="?", default=None)
    parser.add_argument("elevsamples", type=int, nargs="?", default=None)
    parser.add_argument("startazim", type=float, nargs="?", default=None)
    parser.add_argument("endazim", type=float, nargs="?", default=None)
    parser.add_argument("azimsamples", type=int, nargs="?", default=None)

    parser.add_argument("--use_renderlist", type=str, default=None)
    parser.add_argument('--background-color-hex', type=str, default="#FFFFFF00", help='Compositor background color, alpha channel is optional (defaults to transparent background)')
    parser.add_argument('--animate', action='store_true', default=False, help='Render a loop video of the object, result is saved as a directory with images and a gif file. Consider reducing the resolution or the rendering samples for faster results.')
    parser.add_argument('--animate_fps', type=int, default=33)
    parser.add_argument('--animate_sec', type=int, default=15)
    parser.add_argument('--radius', type=float, default=3)
    parser.add_argument("--anchors", type=str, default=None, help='path to file with anchors')

    parser.add_argument('--pc', action='store_true', default=False, help='Render point cloud')
    parser.add_argument('--samples', type=int, default=50, help='Increase this if the result of the render is noisy or inaccurate, this will increase rendering time')
    parser.add_argument('--denoise-type', type=str, default="OPTIX", help='Render denoise type, either None (no denoising), OPTIX, or OPENIMAGEDENOISE. For NVIDIA GPU, OPTIX is preferable. On MacOS, only options are NONE and METAL.')
    parser.add_argument('--device', type=str, default="GPU", help='Preferred rendering device, either CPU or GPU')
    parser.add_argument('--engine', type=str, default="CYCLES", help='Rendering engine, only CYCLES is supported.')
    parser.add_argument('--resolution', nargs=2, type=int, metavar=('width', 'height'), default=[600, 600], help='Resolution for rendering given in width (x) and height (y)')
    parser.add_argument('--object-rotation', nargs=3, type=float, metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0], help='Rotate the object upon import in degrees')
    parser.add_argument('--object-scale', type=float, default=1.0, help='Modify object scale upon import')
    parser.add_argument('--camera-location', nargs=3, type=float, metavar=('x', 'y', 'z'), default=None, help='Rotate the object upon import')
    parser.add_argument('--freestyle', type=float, default=None, help='Render mesh as wireframe, the parameter specifies the line thickness, 0.8 is a good starting point')
    parser.add_argument('--normalize', action='store_true', default=False, help='Normalize imported objects or point clouds')
    parser.add_argument('--debug-file-path', type=str, default=None, help='.blend file that is saved after the program was executed for debugging purposes')
    parser.add_argument('--disable-smooth-shading', action='store_true', default=False, help='Disable smooth shading')
    parser.add_argument('--mp4', action='store_true', default=False, help='Save animation as .mp4 (default is .gif)')

    parser.add_argument('--keep_material', action='store_true', default=False)
    parser.add_argument('--clip_texture', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--textureimg', type=str, default=None, help='texture file name')

    args = parser.parse_args(argv)

    print(args)
    # sanity checks
    if args.mp4:
        assert args.animate
    if not args.animate:
        assert args.out_file[:-4] == '.png'
    if args.animate and args.out_file[:-4] == '.png':
        # when animating, the output will be a directory with all the images, as well as the .gif file of those images
        # for that reason, we remove the '.png' extension if it exists
        args.out_file = args.out_file[:-4]
    assert args.device == 'GPU' or args.device == 'CPU'
    # NOTE: Below is only valid on certian architectures with GPU. For MacOS, only METAL is available
    # assert not args.denoise_type or args.denoise_type == 'OPTIX' or args.denoise_type == 'OPENIMAGEDENOISE'
    if os.path.splitext(args.obj_file)[1] == '.ply':
        assert not args.pc, "Use point cloud rendering only to render a collection of points, not for a mesh"
    assert args.engine == 'CYCLES', "Currently only CYCLES rendering engine is supported"
    assert not (args.pc and args.freestyle), "Freestyle cannot be used while rendering point clouds"

    args.out_file_path = os.path.join(os.path.dirname(os.path.abspath(args.obj_file)), "renders", args.out_file)
    if args.overwrite and os.path.exists(args.out_file_path):
        clear_directory(args.out_file_path)

    blender = Blender(args)
    blender.render(args)
    animation_dir_path = Path(args.out_file_path)
    if args.animate:
        make_gif(animation_dir_path, extension="mp4" if args.mp4 else "gif")

    # debug
    if args.debug_file_path:
        print(f'Saving .blend file for debug under the path [{args.debug_file_path}]')
        filepath = Path(args.debug_file_path)
        assert filepath.suffix == '.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


if __name__ == '__main__':
    main()
