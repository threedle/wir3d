import numpy as np
import torch
from pytorch3d import _C
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    PointLights, MeshRendererWithFragments, HardFlatShader
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation as R

from .geometry import find_adjacent_faces


def setup_renderer(device, **kwargs):
    """
    Setup a standard PyTorch3D renderer for rendering meshes.
    """
    res = kwargs.get("res", 224)
    R = look_at_rotation([[-1, 0, -1]], device=device)
    T = torch.tensor([0, 0, 1], device=device).repeat(1, 1)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(
        image_size=res,
        # max_faces_per_bin=5,
        faces_per_pixel=kwargs.get("faces_per_pixel", 5),
        bin_size=kwargs.get("bin_size", 0),
        cull_backfaces=kwargs.get("cull_backfaces", True)
    )
    lights = PointLights(ambient_color=((0.6, 0.6, 0.6),), location=-T, device=device)

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    return renderer


def sample_view_points(radius, partition):
    """
    Views from top and bottom are given, partition determines level of even subdivision between these points
    """
    points = []

    phi = np.linspace(0, 2 * np.pi, (partition + 1) * 2, endpoint=False)
    theta = np.linspace(0, np.pi, (partition + 1), endpoint=False)

    for i, p in enumerate(phi):
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.cos(t)
            z = radius * np.sin(t) * np.sin(p)
            if t == 0:
                continue
            points.append([x, y, z])

    points.append([0, radius, 0])
    points.append([0, -radius, 0])
    rotated_points = rotate_points(points, 0.001, axis=[1, 0, 0])

    return np.array(rotated_points)


def rotate_points(points, angle, axis=None):
    """
    Rotate a set of points around an axis by a specified angle.
    """
    # Create rotation matrix around the axis
    if axis is None:
        axis = [0, 1, 0]
    rotation_matrix = R.from_rotvec(angle * np.array(axis)).as_matrix()

    # Apply rotation to each point
    rotated_points = []
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_points.append(rotated_point)

    return rotated_points


########################################################################################################################

def check_visible_vertices(pix_to_face, mesh, adjacent_faces=True):
    """
    Find all visible vertices in the rendered images using the pix_to_face tensor returned in the fragments when running
    MeshRendererWithFragments.

    :param pix_to_face: pix_to_face tensor returned in the fragments when running MeshRendererWithFragments
    :param mesh: pytorch3d.structures.Meshes object
    :param adjacent_faces: If True, also mark adjacent faces as visible (helps in dealing with small triangles)
    :return: Boolean array of shape (V, num_vertices) where V is the number of rendered views
    """

    num_views = pix_to_face.shape[0]
    visible_faces_per_view = pix_to_face.view(num_views, -1) % len(mesh.faces_packed())
    visible_vertices_per_view = torch.zeros(num_views, len(mesh.verts_packed()), dtype=torch.bool, device=mesh.device)
    for view in range(num_views):
        visible_faces = visible_faces_per_view[view]
        visible_faces = visible_faces[visible_faces >= 0]
        if adjacent_faces:
            # also mark adjacent faces as visible (helps in dealing with small triangles)
            adjacent_faces_ = find_adjacent_faces(mesh.faces_packed())
            visible_faces = torch.cat([visible_faces, adjacent_faces_[visible_faces].view(-1)], dim=0)
        visible_vertices = mesh.faces_packed()[visible_faces.type(torch.long)].view(-1)
        visible_vertices_per_view[view, visible_vertices] = True
    return visible_vertices_per_view


def check_visible_points(pix_to_face, mesh, points, adjacent_faces=True):
    """
    Find all visible points in the rendered images using the pix_to_face tensor returned in the fragments when running
    MeshRendererWithFragments.

    :param pix_to_face: pix_to_face tensor returned in the fragments when running MeshRendererWithFragments
    :param mesh: pytorch3d.structures.Meshes object
    :param points: List of points to check (m, 3)
    :param adjacent_faces: If True, also mark adjacent faces as visible (helps in dealing with small triangles)
    :return: Boolean array of shape (N, m) where N is the number of rendered views
    """
    pcls = Pointclouds(points[None].to(mesh.device))
    points = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    verts_packed = mesh.verts_packed()
    faces_packed = mesh.faces_packed()
    tris = verts_packed[faces_packed]
    tris_first_idx = mesh.mesh_to_faces_packed_first_idx()

    _, face_ids = _C.point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points, 1e-5
    )

    num_views = pix_to_face.shape[0]
    visible_faces_per_view = pix_to_face.view(num_views, -1) % len(mesh.faces_packed())
    visible_points_per_view = torch.zeros(num_views, len(points), dtype=torch.bool, device=mesh.device)
    if adjacent_faces:
        # also mark adjacent faces as visible (helps in dealing with small triangles)
        adjacent_faces_ = find_adjacent_faces(mesh.faces_packed())
    for view in range(num_views):
        visible_faces = visible_faces_per_view[view]
        visible_faces = visible_faces[visible_faces >= 0]
        if adjacent_faces:
            visible_faces = torch.cat([visible_faces, adjacent_faces_[visible_faces].view(-1)], dim=0)
        try:
            visible_points_per_view[view] = (face_ids.unsqueeze(0) == visible_faces.unsqueeze(1)).any(dim=0)
        except RuntimeError as e:
            print(e, visible_faces.shape, face_ids.shape)
            for j in range(len(points)):
                if torch.any(face_ids[j] == visible_faces):
                    visible_points_per_view[view, j] = True
    return visible_points_per_view
