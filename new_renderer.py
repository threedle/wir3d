# Copied over from the rendering folder
import torch
import numpy as np
import math
import copy

if torch.cuda.is_available():
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

# Function to draw an antialiased red circle on an image
def draw_antialiased_circle(image, center, radius, scale_factor=4):
    from PIL import Image, ImageDraw
    # Scale up the image for supersampling
    width, height = image.size
    large_image = image.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
    large_draw = ImageDraw.Draw(large_image)

    # Scale up the circle center and radius
    # NOTE: We flip the cy because PIL draws from the top left corner
    cx, cy = center[0] * scale_factor * width, scale_factor * height - center[1] * scale_factor * height
    r = radius * scale_factor * np.mean([width, height])

    # Draw the red circle on the large (supersampled) image
    large_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill='red', outline='black')

    # Downscale the image back to the original size (this applies the antialiasing effect)
    small_image = large_image.resize((width, height), Image.LANCZOS)

    return small_image

def get_camera_from_position(positions, lookats, up=torch.tensor([0.0, 1.0, 0.0]),
                             fov=math.radians(60), dims=(512,512), return_cam=False, device=torch.device('cpu')):

    import kaolin as kal

    assert len(positions) == len(lookats), f"Positions {len(positions)} and lookats {len(lookats)} must have the same length"

    B = len(positions)
    up = up.type(positions.dtype).unsqueeze(0).repeat(B, 1).to(device)
    camera_transform = kal.render.camera.generate_transformation_matrix(positions, lookats, up).to(device)

    # If return camera, then create camera object
    if return_cam:
        cams = kal.render.camera.Camera.from_args(
                eye=positions,
                at=lookats,
                up=up,
                fov=fov,
                near=1e-2, far=1e2,
                width=dims[0], height=dims[1], device=device
            )
        return camera_transform, cams

    return camera_transform

class Renderer:
    # from https://github.com/eladrich/latent-nerf
    def __init__(
        self,
        device,
        fov,
        dim=(224, 224),
        interpolation_mode='bilinear',
        # Light Tensor (positive first): [ambient, right/left, front/back, top/bottom, ...]
        lights=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        import kaolin as kal
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(fov).to(device)

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.fov = fov
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()
        self.lights = lights.to(device)

    def get_camera_from_view(self, elev, azim, up=torch.tensor([0.0, 1.0, 0.0]), r=3.0, look_at_height=0.0,
                             fov = math.radians(60), return_cam=False, device=torch.device('cpu')):
        """
        Convert tensor elevation/azimuth values into camera projections

        Args:
            elev (torch.Tensor): elevation
            azim (torch.Tensor): azimuth
            r (float, optional): radius. Defaults to 3.0.

        Returns:
            Camera projection matrix (B x 4 x 3)
        """
        import kaolin as kal
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)
        B = elev.shape[0]

        if len(x.shape) == 0:
            pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
        else:
            pos = torch.stack([x, y, z], dim=1)
        # look_at = -pos
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height

        up = up.type(pos.dtype).unsqueeze(0).repeat(B, 1).to(device)
        camera_transform = kal.render.camera.generate_transformation_matrix(pos, look_at, up).to(device)

        # If return camera, then create camera object
        if return_cam:
            cams = kal.render.camera.Camera.from_args(
                    eye=pos,
                    at=look_at,
                    up=up,
                    fov = self.fov,
                    width=self.dim[0], height=self.dim[1], device=self.device
                )
            return camera_transform, cams

        return camera_transform

    def render_texture(
        self, verts, faces, uv_face_attr, texture_map, uvs,
        elev=None, azim=None, radius=2.2, look_at_height=0.0,
        positions=None, lookats=None,
        up=torch.tensor([0.0, 1.0, 0.0]), dims=None, white_background=False, vertexnormals=None,
        mod = False, specular = False, clip_uv=False,
        keypoints=None, keypoint_radius=0.01, keypoint_visibility=False, return_zbuffer=True,
        l_position = None,
        l_azim = [0., np.pi/2, np.pi, -np.pi/2, 0., 0.],
        l_elev = [0.] * 4 + [np.pi/2, -np.pi/2], amplitude = 1., sharpness = 3., rast_option=0,
        view_matrix = None,
    ):
        """ rast options: 0 - kaolin, 1 - kaolin with nvdiffrast backend, 2 - nvdiffrast"""

        # uv face attr: B x F x 3 x 2
        # NOTE: Pytorch coordinates -1 to 1, yaxis from top to bottom -- circular filtering NOT supported
        import kaolin as kal
        dims = self.dim if dims is None else dims

        assert (azim is not None and elev is not None) or (positions is not None and lookats is not None), "Either provide elev/azim or positions/lookats"

        if positions is not None:
            B = len(positions)
        else:
            B = len(elev)

        if view_matrix is not None:
            cam = kal.render.camera.Camera.from_args(
                    view_matrix=view_matrix,
                    width=dims[0], height=dims[1], device=self.device
            )
            camera_transform = None
        elif positions is not None:
            camera_transform, cam = get_camera_from_position(positions, lookats, up=up, fov=self.fov, dims=dims,
                                                            return_cam=True, device=self.device)
        else:
            camera_transform, cam = self.get_camera_from_view(elev, azim, up=up, r=radius, look_at_height=look_at_height,
                                                        return_cam=True, device=self.device)

        # UV: F x 3 x 2
        uv_face_attr = uv_face_attr.repeat(B, 1, 1, 1)

        if vertexnormals is None:
            fnormals = kal.ops.mesh.face_normals(verts[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces, fnormals.unsqueeze(2).repeat(1,1,3,1))

        # Project the keypoints
        if keypoints is not None:
            keypoints_camera = cam.extrinsics.transform(keypoints.unsqueeze(0).repeat(B, 1, 1)) # Cameras x nkeypoints x 3
            keypoints_clip = cam.intrinsics.transform(keypoints_camera)
            keypoints_ndc = kal.render.camera.intrinsics.down_from_homogeneous(keypoints_clip)
            keypoints_ndc = (keypoints_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

        if rast_option == 2:
            import nvdiffrast.torch as dr

            vertices_camera = cam.extrinsics.transform(verts)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[dims[1], dims[0]])
            face_idx = rast[:,:,:,3].long()-1

            uv_features, _ = dr.interpolate(attr=uvs, rast=rast, tri=faces.int())
            normal_features, _ = dr.interpolate(attr=vertexnormals, rast=rast, tri=faces.int())

            # NOTE: zbuffer is aligned properly with y-axis indexing (0) corresponds to the bottom
            # Replace 0s with high z value
            zbuffer = rast[:,:,:,-2]
            zbuffer[zbuffer == 0] = 1000

            face_idx = torch.flip(face_idx, dims=(1,))
            uv_features = torch.flip(uv_features, dims=(1,))
            normal_features = torch.flip(normal_features, dims=(1,))
        else:
            # Vertices in camera coordinates (B x F x 3 x XYZ), vertices in image coordinates (B x F x 3 x 2),
            # face normals (B x F x 3)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

            normal_face_attr = vertexnormals[0, faces].repeat(B, 1, 1, 1).to(self.device)

            # TODO: sanity check -> single triangle render and check against a pixel loss
            # NOTE: We rasterize both UVs and normals per-pixel for correct shading
            if rast_option == 1:
                image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                    face_vertices_image, [uv_face_attr, normal_face_attr], backend="nvdiffrast")
            else:
                image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                    face_vertices_image, [uv_face_attr, normal_face_attr])
            uv_features, normal_features = image_features

        # Apply mod function to UVs if set (don't grad this)
        if mod:
            with torch.no_grad():
                floorint = torch.floor(uv_features)
            uv_features = uv_features - floorint

        mask = (face_idx != -1)
        albedo = kal.render.mesh.utils.texture_mapping(uv_features, texture_map.repeat(B, 1, 1, 1),
                                                        mode=self.interpolation_mode)

        # Replace all albedo values where UVs are outside the image with white
        if clip_uv:
            uv_mask = ((uv_features[..., 0] >= 0) & (uv_features[..., 0] <= 1)) & ((uv_features[..., 1] >= 0) & (uv_features[..., 1] <= 1))
            albedo[~uv_mask] = 0.7

        albedo = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        ### Add lighting
        if l_position is None:
            # NOTE: Base lighting -- 6 lights from each primary direction
            l_azimuth = torch.tensor(l_azim, device=self.device).float()
            l_elevation = torch.tensor(l_elev, device=self.device).float()
            base_amplitude = torch.full((l_azimuth.shape[0], 3), amplitude, device=self.device).float()
            base_sharpness = torch.full((l_azimuth.shape[0],), sharpness, device=self.device).float()
        else:
            base_amplitude = torch.full((l_position.shape[0], 3), amplitude, device=self.device).float()
            base_sharpness = torch.full((l_position.shape[0],), sharpness, device=self.device).float()

        # If specular, then need to construct camera and generate pinhole rays + additional material params
        rays_d = base_spec = base_roughness = None
        if specular:
            base_spec = mask.unsqueeze(-1) * torch.tensor([1., 1., 1.], device=self.device)
            base_roughness = torch.full((B, *dims), 0.1, device=self.device)

            # Compute the rays
            rays_d = []
            for c in cam:
                rays_d.append(generate_pinhole_rays_dir(c, height=self.dim[0], width=self.dim[1]))
            # Rays must be toward the camera
            rays_d = -torch.cat(rays_d, dim=0)

        im_world_normal = torch.nn.functional.normalize(normal_features.detach(), p=2, dim=-1)

        if l_position is None:
            img = add_lighting(mask, base_amplitude, base_sharpness, im_world_normal,
            albedo, l_azimuth = l_azimuth, l_elevation = l_elevation, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)
        else:
            img = add_lighting(mask, base_amplitude, base_sharpness, im_world_normal,
                albedo, l_position = l_position, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)

        if white_background:
            img = img + (1 - mask.unsqueeze(-1).int())

        if keypoints is not None:
            from torchvision.transforms.functional import to_pil_image
            keypoints_img = []
            keypoints_mask = []
            radius = keypoint_radius

            for viewi in range(B):
                tmp_img = to_pil_image(img[viewi].permute(2, 0, 1).cpu())
                tmp_mask = torch.zeros(len(keypoints_ndc[viewi]), dtype=torch.bool)

                for ki, keypoint in enumerate(keypoints_ndc[viewi]):
                    # Ignore everything outside the render window
                    if torch.any(keypoint > 1) or torch.any(keypoint < 0):
                        continue

                    # If visibility: check the z value of keypoint_clip against the z buffers
                    if keypoint_visibility:
                        kp_z = keypoints_clip[viewi, ki, 2] - 0.0005 # Slightly offset to avoid z fighting with the surface
                        kp_x, kp_y = torch.floor(keypoint * torch.tensor(dims, device=self.device).float())

                        check_z = zbuffer[viewi, kp_y.long(), kp_x.long()].item()
                        if kp_z > check_z:
                            continue
                        else:
                            tmp_mask[ki] = True
                            tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)
                    else:
                        tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)

                keypoints_img.append(tmp_img)

                if keypoint_visibility:
                    keypoints_mask.append(tmp_mask)

            if keypoint_visibility:
                keypoints_mask = torch.stack(keypoints_mask, dim=0)

                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask
            else:
                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img

        if return_zbuffer:
            return img.permute(0, 3, 1, 2), mask.unsqueeze(1), zbuffer

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

    def render_mesh(self, vertices, faces, colors,
                    elev=None, azim=None, radius=2.2, look_at_height=0.0,
                    positions=None, lookats=None,
                    up=torch.tensor([0.0, 1.0, 0.0]),
                    dims=None, white_background=True, vertexnormals=None,
                    l_position = None,
                    l_azim = [0., np.pi/2, np.pi, -np.pi/2, 0., 0.],
                    l_elev = [0.] * 4 + [np.pi/2, -np.pi/2], amplitude = 1., sharpness = 3.,
                    keypoints=None, keypoint_radius=0.01, keypoint_visibility=False, return_zbuffer=True,
                    specular = False, rast_option=0, view_matrix=None):
        """ rast options: 0 - kaolin, 1 - kaolin with nvdiffrast backend, 2 - nvdiffrast"""

        # uv face attr: B x F x 3 x 2
        # NOTE: Pytorch coordinates -1 to 1, yaxis from top to bottom -- circular filtering NOT supported
        import kaolin as kal
        dims = self.dim if dims is None else dims

        assert (azim is not None and elev is not None) or (positions is not None and lookats is not None) or (view_matrix is not None), "Either provide elev/azim, positions/lookats, or view matrix"

        if positions is not None:
            B = len(positions)
        elif elev is not None:
            B = len(elev)
        elif view_matrix is not None:
            B = view_matrix.shape[0]

        if view_matrix is not None:
            cam = kal.render.camera.Camera.from_args(
                    view_matrix=view_matrix,
                    width=dims[0], height=dims[1], device=self.device,
                    fov = self.fov,
            )
            camera_transform = None
        elif positions is not None:
            camera_transform, cam = get_camera_from_position(positions, lookats, up=up, fov=self.fov, dims=dims,
                                                            return_cam=True, device=self.device)
        else:
            camera_transform, cam = self.get_camera_from_view(elev, azim, up=up, r=radius, look_at_height=look_at_height,
                                                        return_cam=True, device=self.device)

        # Need normals for shading
        if vertexnormals is None:
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces.long(), fnormals.unsqueeze(2).repeat(1,1,3,1))

        # Project the keypoints
        if keypoints is not None:
            keypoints_camera = cam.extrinsics.transform(keypoints.unsqueeze(0).repeat(B, 1, 1)) # Cameras x nkeypoints x 3
            keypoints_clip = cam.intrinsics.transform(keypoints_camera)
            keypoints_ndc = kal.render.camera.intrinsics.down_from_homogeneous(keypoints_clip)
            keypoints_ndc = (keypoints_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

        if rast_option == 2:
            import nvdiffrast.torch as dr

            vertices_camera = cam.extrinsics.transform(vertices)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[dims[1], dims[0]])
            face_idx = rast[:,:,:,3].long()-1 # Do we need long?
            albedo, _ = dr.interpolate(attr=colors, rast=rast, tri=faces.int())
            normal_features, _ = dr.interpolate(attr=vertexnormals, rast=rast, tri=faces.int())

            # NOTE: zbuffer is aligned properly with y-axis indexing (0) corresponds to the bottom
            # Replace 0s with high z value
            zbuffer = rast[:,:,:,-2]
            zbuffer[zbuffer == 0] = 1000

            face_idx = torch.flip(face_idx, dims=(1,))
            albedo = torch.flip(albedo, dims=(1,))
            normal_features = torch.flip(normal_features, dims=(1,))
        else:
            import kaolin

            face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
                                    colors.unsqueeze(0),
                                    faces.long()
                                )
            face_attributes = face_attributes.repeat(B, 1, 1, 1).to(self.device)

            normal_face_attr = vertexnormals[0, faces].repeat(B, 1, 1, 1).to(self.device)

            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                vertices, faces, self.camera_projection, camera_transform=camera_transform)

            if rast_option == 1:
                image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                                     face_vertices_image, [face_attributes, face_vertices_camera[:, :, :, [-1]], normal_face_attr],
                                                                     backend="nvdiffrast")
            else:
                image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                                     face_vertices_image, [face_attributes, face_vertices_camera[:, :, :, [-1]], normal_face_attr])
            albedo, zbuffer, normal_features = image_features

            # Flip z sign and replace 0s with high z value
            zbuffer = -zbuffer
            zbuffer[zbuffer == 0] = 1000

        mask = (face_idx != -1)
        albedo = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        ### Add lighting
        if l_position is None:
            # NOTE: Base lighting -- 6 lights from each primary direction
            l_azimuth = torch.tensor(l_azim, device=self.device).float()
            l_elevation = torch.tensor(l_elev, device=self.device).float()
            base_amplitude = torch.full((l_azimuth.shape[0], 3), amplitude, device=self.device).float()
            base_sharpness = torch.full((l_azimuth.shape[0],), sharpness, device=self.device).float()
        else:
            base_amplitude = torch.full((l_position.shape[0], 3), amplitude, device=self.device).float()
            base_sharpness = torch.full((l_position.shape[0],), sharpness, device=self.device).float()

        # If specular, then need to construct camera and generate pinhole rays + additional material params
        rays_d = base_spec = base_roughness = None
        if specular:
            base_spec = mask.unsqueeze(-1) * torch.tensor([1., 1., 1.], device=self.device)
            base_roughness = torch.full((B, *dims), 0.1, device=self.device)

            # Compute the rays
            rays_d = []
            for c in cam:
                rays_d.append(generate_pinhole_rays_dir(c, height=self.dim[0], width=self.dim[1]))
            # Rays must be toward the camera
            rays_d = -torch.cat(rays_d, dim=0)

        im_world_normal = torch.nn.functional.normalize(normal_features.detach(), p=2, dim=-1)

        if l_position is None:
            img = add_lighting(mask, base_amplitude, base_sharpness, im_world_normal,
            albedo, l_azimuth = l_azimuth, l_elevation = l_elevation, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)
        else:
            img = add_lighting(mask, base_amplitude, base_sharpness, im_world_normal,
                albedo, l_position = l_position, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)

        if white_background:
            img = img + (1 - mask.unsqueeze(-1).int())

        if keypoints is not None:
            from torchvision.transforms.functional import to_pil_image
            keypoints_img = []
            keypoints_mask = []
            radius = keypoint_radius

            for viewi in range(B):
                tmp_img = to_pil_image(img[viewi].permute(2, 0, 1).cpu())
                tmp_mask = torch.zeros(len(keypoints_ndc[viewi]), dtype=torch.bool)

                for ki, keypoint in enumerate(keypoints_ndc[viewi]):
                    # Ignore everything outside the render window
                    if torch.any(keypoint > 1) or torch.any(keypoint < 0):
                        continue

                    # If visibility: check the z value of keypoint_clip against the z buffers
                    if keypoint_visibility:
                        kp_z = keypoints_clip[viewi, ki, 2] - 0.0005 # Slightly offset to avoid z fighting with the surface
                        kp_x, kp_y = torch.floor(keypoint * torch.tensor(dims, device=self.device).float())
                        check_z = zbuffer[viewi, kp_y.long(), kp_x.long()].item()
                        if kp_z > check_z:
                            continue
                        else:
                            tmp_mask[ki] = True
                            tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)
                    else:
                        tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)

                keypoints_img.append(tmp_img)

                if keypoint_visibility:
                    keypoints_mask.append(tmp_mask)

            if keypoint_visibility:
                keypoints_mask = torch.stack(keypoints_mask, dim=0)

                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask
            else:
                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img

        if return_zbuffer:
            return img.permute(0, 3, 1, 2), mask.unsqueeze(1), zbuffer

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

    def render_features(self, vertices, faces, features,
                    elev=None, azim=None, radius=2.2, look_at_height=0.0,
                    positions=None, lookats=None,
                    up=torch.tensor([0.0, 1.0, 0.0]),
                    dims=None, white_background=True,
                    keypoints=None, keypoint_radius=0.01, keypoint_visibility=False, return_zbuffer=True,
                    view_matrix=None):
        """ rast options: 0 - kaolin, 1 - kaolin with nvdiffrast backend, 2 - nvdiffrast"""

        # uv face attr: B x F x 3 x 2
        # NOTE: Pytorch coordinates -1 to 1, yaxis from top to bottom -- circular filtering NOT supported
        import kaolin as kal
        dims = self.dim if dims is None else dims

        assert (azim is not None and elev is not None) or (positions is not None and lookats is not None), "Either provide elev/azim or positions/lookats"

        if positions is not None:
            B = len(positions)
        else:
            B = len(elev)

        if view_matrix is not None:
            cam = kal.render.camera.Camera.from_args(
                    view_matrix=view_matrix,
                    width=dims[0], height=dims[1], device=self.device
            )
            camera_transform = None
        elif positions is not None:
            camera_transform, cam = get_camera_from_position(positions, lookats, up=up, fov=self.fov, dims=dims,
                                                            return_cam=True, device=self.device)
        else:
            camera_transform, cam = self.get_camera_from_view(elev, azim, up=up, r=radius, look_at_height=look_at_height,
                                                        return_cam=True, device=self.device)


        # Project the keypoints
        if keypoints is not None:
            keypoints_camera = cam.extrinsics.transform(keypoints.unsqueeze(0).repeat(B, 1, 1)) # Cameras x nkeypoints x 3
            keypoints_clip = cam.intrinsics.transform(keypoints_camera)
            keypoints_ndc = kal.render.camera.intrinsics.down_from_homogeneous(keypoints_clip)
            keypoints_ndc = (keypoints_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

        import nvdiffrast.torch as dr

        vertices_camera = cam.extrinsics.transform(vertices)
        vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
        vertices_image = torch.nn.functional.pad(
            vertices_image,
            (0, 1), mode='constant', value=1
        )

        rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[dims[1], dims[0]])
        face_idx = rast[:,:,:,3].long()-1 # Do we need long?
        albedo, _ = dr.interpolate(attr=features, rast=rast, tri=faces.int())

        # NOTE: zbuffer is aligned properly with y-axis indexing (0) corresponds to the bottom
        zbuffer = rast[:,:,:,-2]

        # Replace 0s with high z value
        zbuffer[zbuffer == 0] = 1000

        face_idx = torch.flip(face_idx, dims=(1,))
        albedo = torch.flip(albedo, dims=(1,))

        mask = (face_idx != -1)
        img = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        if white_background:
            img = img + (1 - mask.unsqueeze(-1).int())

        if keypoints is not None:
            from torchvision.transforms.functional import to_pil_image
            keypoints_img = []
            keypoints_mask = []
            radius = keypoint_radius

            for viewi in range(B):
                tmp_img = to_pil_image(img[viewi].permute(2, 0, 1).cpu())
                tmp_mask = torch.zeros(len(keypoints_ndc[viewi]), dtype=torch.bool)

                for ki, keypoint in enumerate(keypoints_ndc[viewi]):
                    # Ignore everything outside the render window
                    if torch.any(keypoint > 1) or torch.any(keypoint < 0):
                        continue

                    # If visibility: check the z value of keypoint_clip against the z buffers
                    if keypoint_visibility:
                        kp_z = keypoints_clip[viewi, ki, 2] - 0.0005 # Slightly offset to avoid z fighting with the surface
                        kp_x, kp_y = torch.floor(keypoint * torch.tensor(dims, device=self.device).float())
                        check_z = zbuffer[viewi, kp_y.long(), kp_x.long()].item()
                        if kp_z > check_z:
                            continue
                        else:
                            tmp_mask[ki] = True
                            tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)
                    else:
                        tmp_img = draw_antialiased_circle(tmp_img, keypoint.cpu().numpy(), radius, scale_factor=4)

                keypoints_img.append(tmp_img)

                if keypoint_visibility:
                    keypoints_mask.append(tmp_mask)

            if keypoint_visibility:
                keypoints_mask = torch.stack(keypoints_mask, dim=0)

                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, keypoints_mask
            else:
                if return_zbuffer:
                    return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img, zbuffer

                return img.permute(0, 3, 1, 2), mask.unsqueeze(1), keypoints_img

        if return_zbuffer:
            return img.permute(0, 3, 1, 2), mask.unsqueeze(1), zbuffer

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

#----------------------------------------------------------------------------
# Render Helpers
#----------------------------------------------------------------------------

def generate_pinhole_rays_dir(camera, height, width, device='cuda'):
    """Generate centered grid.

    This is a utility function for specular reflectance with spherical gaussian.
    """
    import kaolin as kal
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    pixel_x = pixel_x + 0.5  # scale and add bias to pixel center
    pixel_y = pixel_y + 0.5  # scale and add bias to pixel center

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    # Convert to NDC
    pixel_x = 2 * (pixel_x / width) - 1.0
    pixel_y = 2 * (pixel_y / height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1).float()

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

    return ray_dir[0].reshape(1, height, width, 3)

# Given albedo and lighting parameters, add lighting to the render
def add_lighting(hard_mask, amplitude, sharpness, im_world_normal, albedo,
                 l_azimuth = None, l_elevation = None,
                 l_position = None,
                 specular = False, rays_d=None, spec_albedo=None, roughness=None):
    """Render diffuse and specular components.

    Use spherical gaussian fitted approximation for the diffuse component"""
    import kaolin as kal

    assert (l_azimuth is not None and l_elevation is not None) or l_position is not None, "Either provide l_azimuth/l_elevation or l_positions"

    # Add lighting components broadcasted over batch dimension
    if l_position is not None:
        directions = l_position
    else:
        directions = torch.stack(kal.ops.coords.spherical2cartesian(l_azimuth, l_elevation), dim=-1)
    img = torch.zeros(im_world_normal.shape, device='cuda', dtype=torch.float32)

    # NOTE: May need to repeat amp/dir/sharp over batch size
    # Render diffuse component
    diffuse_effect = kal.render.lighting.sg_diffuse_inner_product(
        amplitude,
        directions,
        sharpness,
        im_world_normal[hard_mask],
        albedo[hard_mask]
    ).float()

    if specular:
        assert rays_d is not None
        assert spec_albedo is not None
        assert roughness is not None

        # Render specular component
        specular_effect = kal.render.lighting.sg_warp_specular_term(
            amplitude,
            directions,
            sharpness,
            im_world_normal[hard_mask],
            roughness[hard_mask].float(),
            rays_d[hard_mask],
            spec_albedo[hard_mask].float()
        ).float()
        img[hard_mask] = diffuse_effect + specular_effect
    else:
        img[hard_mask] = diffuse_effect

    # HDR: Rescale to [0, 1]
    if torch.max(img) > 1:
        img = img / torch.max(img)

    return img

def texture_mapping(texture_coordinates, texture_maps, mode='nearest',
                    padding_mode='zeros'):
    r"""Interpolates texture_maps by dense or sparse texture_coordinates.
    This function supports sampling texture coordinates for:
    1. An entire 2D image
    2. A sparse point cloud of texture coordinates.

    Args:
        texture_coordinates(torch.FloatTensor):
            dense image texture coordinate, of shape :math:`(\text{batch_size}, h, w, 2)` or
            sparse texture coordinate for points, of shape :math:`(\text{batch_size}, \text{num_points}, 2)`
            Coordinates are expected to be normalized between [0, 1].
            Note that opengl tex coord is different from pytorch's coord.
            opengl coord ranges from 0 to 1, y axis is from bottom to top
            and it supports circular mode(-0.1 is the same as 0.9)
            pytorch coord ranges from -1 to 1, y axis is from top to bottom and does not support circular
            filtering is the same as the mode parameter for torch.nn.functional.grid_sample.
        texture_maps(torch.FloatTensor):
            textures of shape :math:`(\text{batch_size}, \text{num_channels}, h', w')`.
            Here, :math:`h'` & :math:`w'` are the height and width of texture maps.

            If ``texture_coordinates`` are image texture coordinates -
            For each pixel in the rendered image of height we use the coordinates in
            texture_coordinates to query corresponding value in texture maps.
            Note that height :math:`h` and width :math:`w` of the rendered image could be different from
            :math:`h'` & :math:`w'`.

            If ``texture_coordinates`` are sparse texture coordinates -
            For each point in ``texture_coordinates`` we query the corresponding value in ``texture_maps``.

    Returns:
        (torch.FloatTensor):
        interpolated texture of shape :math:`(\text{batch_size}, h, w, \text{num_channels})` or
        interpolated texture of shape :math:`(\text{batch_size}, \text{num_points}, \text{num_channels})`
    """
    batch_size = texture_coordinates.shape[0]
    num_channels = texture_maps.shape[1]
    _texture_coordinates = texture_coordinates.reshape(batch_size, -1, 1, 2)

    # convert coord mode from ogl to pytorch
    # some opengl texture coordinate is larger than 1 or less than 0
    # in opengl it will be normalized by remainder
    # we do the same in pytorch
    _texture_coordinates = torch.clamp(_texture_coordinates, 0., 1.)
    _texture_coordinates = _texture_coordinates * 2 - 1  # [0, 1] to [-1, 1]
    _texture_coordinates[:, :, :, 1] = -_texture_coordinates[:, :, :, 1]  # reverse y

    # sample
    texture_interpolates = torch.nn.functional.grid_sample(texture_maps,
                                                           _texture_coordinates,
                                                           mode=mode,
                                                           align_corners=False,
                                                           padding_mode=padding_mode)
    texture_interpolates = texture_interpolates.permute(0, 2, 3, 1)
    return texture_interpolates.reshape(batch_size, *texture_coordinates.shape[1:-1], num_channels)