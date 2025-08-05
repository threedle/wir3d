try:
    import imageio
except ModuleNotFoundError:
    print("No module named 'imageio'. For creating an animation gif, please install imageio:\n"
          "pip install --upgrade pip\n"
          "pip install imageio==2.23.0\n"
          "pip install imageio[ffmpeg]==2.23.0\n"
          )

import numpy as np
from pathlib import Path
import yaml
import hashlib
from typing import Union

def get_recipe_yml_obj(recipe_file_path: Union[str, Path]):
    with open(recipe_file_path, 'r') as recipe_file:
        recipe_yml_obj = yaml.load(recipe_file, Loader=yaml.FullLoader)
    return recipe_yml_obj

def hash_file_name(file_name):
    return int(hashlib.sha1(file_name.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

def read_obj(file):
    vs, normals, col, fs = [], [], [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
            if splitted_line[4:]:
                col.append([float(c) for c in splitted_line[4:]])
        elif splitted_line[0] == 'vn':
            normals.append([float(n) for n in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            fs.append([int(f) for f in splitted_line[1:4]])
    f.close()
    vs = np.asarray(vs)
    col = np.asarray(col)
    normals = np.asarray(normals)
    faces = np.asarray(fs)
    if len(normals) > 0:
        if np.max(normals) > 1:
            normals = normals / 255
        points = np.concatenate((vs, normals), axis=1)
    else:
        points = vs
    return points, col, faces


def write_obj(file, verts, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        # vertices
        for vi, v in enumerate(verts):
            # color
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            # normal
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))

        # faces
        for face in faces:
            f.write("f %d %d %d\n" % (face[0], face[1], face[2]))


def write_ply(file, verts, faces, color=None):
    with open(file, 'w+') as f:
        # header
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(verts.shape[0]))
        f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
        if color is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uint8 int32 vertex_index\n")
        f.write("end_header\n")

        # vertices
        for vi, v in enumerate(verts):
            if color is not None:
                f.write("%f %f %f %d %d %d\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            else:
                f.write("%f %f %f\n" % (v[0], v[1], v[2]))

        # faces
        for face in faces:
            f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))


def make_gif(image_seq_dir_path: Path, extension='gif'):
    assert image_seq_dir_path.is_dir(), f"No such directory [{image_seq_dir_path}]"
    print(image_seq_dir_path)
    file_paths = sorted(image_seq_dir_path.glob("*.png"))
    images = []
    for file_path in file_paths:
        try:
            images.append(imageio.imread(str(file_path)))
        except:
            images.append(imageio.v2.imread(str(file_path)))
    # assert len(images) > 100
    target_file_path = image_seq_dir_path.with_suffix(f'.{extension}')
    imageio.mimsave(target_file_path, images, fps=30)
    print(f"Animation saved to [{target_file_path}]")
