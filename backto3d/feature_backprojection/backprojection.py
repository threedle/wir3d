import numpy as np
import torch
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, PointLights
from tqdm import tqdm

import sys
sys.path.append("..")
from backto3d.utils import check_visible_points, check_visible_vertices, VERBOSE


def compute_kp_dists_features(dataset, feature_function):
    """
    :param dataset: List of few-shot samples / small few-shot dataset as given from KeypointNetDataset
    :param feature_function: 'features_from_views' with only remaining arguments: 'mesh', 'keypoints', 'geo_dists'
    :return: Keypoint features, Pointwise normalized distances (np.array): (num_keypoints, emb_dim), (num_keypoints, num_keypoints)
    """

    kp_max_id = max(
        [kp["semantic_id"] for mesh, keypoints, class_title, mesh_id, pcd in dataset[:] for kp in keypoints])

    keypoint_dists = np.zeros((kp_max_id + 1, kp_max_id + 1))
    keypoint_features_dict = {kp_id: [] for kp_id in range(kp_max_id + 1)}

    for mesh, keypoints, class_title, mesh_id, pcd in tqdm(dataset):
        pcd_np = pcd.points_packed().cpu().numpy()

        kp_indices = [kp["semantic_id"] for kp in keypoints]
        kp_pcd_indices = [kp["pcd_info"]["point_index"] for kp in keypoints]
        from backto3d.utils.evaluation import gen_geo_dists
        dists_between_points = gen_geo_dists(pcd_np).astype(np.float32)
        dists_between_points[np.isinf(dists_between_points)] = np.max(
            dists_between_points[~np.isinf(dists_between_points)])
        dists_normalized = dists_between_points / np.max(dists_between_points)
        kp_dists = dists_normalized[kp_pcd_indices, :][:, kp_pcd_indices]

        keypoint_features = feature_function(mesh, keypoints, kp_dists)

        keypoint_dists[np.ix_(kp_indices, kp_indices)] += kp_dists
        for kp_id, kp_features in zip(kp_indices, keypoint_features):
            keypoint_features_dict[kp_id].append(kp_features)

    empty_kp_ids = [kp_id for kp_id, kp_features in keypoint_features_dict.items() if len(kp_features) == 0]
    keypoint_features = torch.stack(
        [torch.mean(torch.stack(kp_features), dim=0) for kp_features in keypoint_features_dict.values() if
         len(kp_features) > 0])
    keypoint_dists = keypoint_dists[~np.isin(np.arange(kp_max_id + 1), empty_kp_ids)][:,
                     ~np.isin(np.arange(kp_max_id + 1), empty_kp_ids)]

    return keypoint_features, keypoint_dists

# Rendered features per pixel NOT patches!!
def features_from_featurerenders(dataloader, vertices, faces, fov,
                                epochs=5, device="cpu",
                            ):
    """
    :param renderer: kaolin renderer
    :param vertices, faces: torch.tensors
    :param renders: rendered images
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    import kaolin as kal

    # Initialize vertex features
    featuremaps = next(iter(dataloader))['featuremaps'] # B x C x W x H
    featuremaps = torch.load(featuremaps[0], map_location=device, weights_only=True)
    emb_dim = featuremaps.shape[0]
    res = featuremaps.shape[-1]

    vertex_features = torch.zeros(len(vertices), emb_dim).to(device)
    vertex_features.requires_grad_()
    optim = torch.optim.Adam([vertex_features], lr=0.1)

    overall_visibility = torch.zeros(len(vertices), device=device)

    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(dataloader):
            featuremaps = []
            featurepaths = data['featuremaps']
            for path in featurepaths:
                featuremaps.append(torch.load(path, map_location=device, weights_only=True).float())
            featuremaps = torch.stack(featuremaps)
            featuremaps = featuremaps.to(device)
            positions = data['positions']
            lookats = data['lookats']

            # Define camera
            cam = kal.render.camera.Camera.from_args(
                eye=positions,
                at=lookats,
                up=torch.tensor([0., 1., 0.]).to(device),
                fov = fov,
                width=res, height=res,
                device=device
            )

            import nvdiffrast.torch as dr
            glctx = dr.RasterizeCudaContext()

            #### Render the obj ####
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)

            vertices_camera = cam.extrinsics.transform(vertices)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )
            zimage = vertices_image[...,[-2]].contiguous()

            vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)

            # Flip y-axis values for grid_sample (-1, -1) is top-left
            vertices_ndc = vertices_ndc * torch.tensor([1, -1, 1]).to(device)

            # Y and X also need to be swapped in the indexing
            vertices_ndc = vertices_ndc[..., [1, 0, 2]]

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[res, res])

            # Replace 0s with high z value
            zbuffer = rast[:,:,:,-2]
            zbuffer[zbuffer == 0] = 1000
            zbuffer = torch.flip(zbuffer, dims=(1,))

            #### Determine visibility ####
            # Sample the zbuffer
            zbuffer_samples = torch.nn.functional.grid_sample(zbuffer.unsqueeze(1), vertices_ndc[:,:,None, :2],
                                                              align_corners=False).squeeze()
            visible_mask = (zbuffer_samples + 0.0005 >= zimage.squeeze()) # B x V

            # Map vertex features to the image coordinates
            vertex_features_rendered, _ = dr.interpolate(attr=vertex_features, rast=rast, tri=faces.int())
            vertex_features_rendered = torch.flip(vertex_features_rendered, dims=(1,))
            vertex_features_rendered = vertex_features_rendered.permute(0, 3, 1, 2)

            # If zbuffer exists in the dataloader, then use it to mask the features
            if 'zbuffer' in data:
                gt_zbuffer = []
                for path in data['zbuffer']:
                    gt_zbuffer.append(torch.load(path, map_location=device, weights_only=True).float())
                gt_zbuffer = torch.stack(gt_zbuffer)
                gt_zbuffer = gt_zbuffer.to(device)
                # NOTE: Saved zbufferes will have y-axis indexing starting from bottom-left
                gt_zbuffer = torch.flip(gt_zbuffer, dims=(1,))

                zbuffer_mask = (zbuffer.squeeze() - 0.001 < gt_zbuffer.squeeze())

                ## Debugging
                # from PIL import Image
                # import torchvision

                # for i in range(len(zbuffer)):
                #     zbuffer_img = zbuffer[i].clone()
                #     zbuffer_img[zbuffer_img != 1000] -= zbuffer_img[zbuffer_img != 1000].min()
                #     zbuffer_img[zbuffer_img != 1000] /= (zbuffer_img[zbuffer_img != 1000].max() * 1.5)
                #     zbuffer_img[zbuffer_img == 1000] = 1 # Background is white
                #     zbuffer_pil = torchvision.transforms.functional.to_pil_image(zbuffer_img.detach().cpu())
                #     zbuffer_pil.save(f'zbuffer{i}.png')

                #     # Also save masked zbuffer
                #     zbuffer_masked = zbuffer_img.clone()
                #     zbuffer_masked[~zbuffer_mask[i]] = 1 # Background is white

                #     zbuffer_pil = torchvision.transforms.functional.to_pil_image(zbuffer_masked.detach().cpu())
                #     zbuffer_pil.save(f'zbuffer_masked{i}.png')

            overall_visibility += visible_mask.sum(dim=0)

            #### Optimize
            optim.zero_grad()
            if 'zbuffer' in data:
                loss = torch.mean(torch.sum((featuremaps - vertex_features_rendered) ** 2 * zbuffer_mask.unsqueeze(1), dim=(1, 2, 3)))
            else:
                loss = torch.mean(torch.sum((featuremaps - vertex_features_rendered) ** 2, dim=(1, 2, 3)))
            loss.backward()
            optim.step()

            print(f"Epoch {epoch} Batch {i}: Initial loss: {loss.detach().cpu().item()}")

        print(f"Epoch {epoch}: Loss: {loss.detach().cpu().item()}")

        nonvisible = torch.sum(overall_visibility == 0)
        if nonvisible > 0:
            print(f"WARNING: {nonvisible} vertices are not visible in any view! ")

    return vertex_features

def features_from_featuremaps(dataloader, vertices, faces, fov,
                              res=224,
                              epochs=5, device="cpu",
                            ):
    """
    :param renderer: kaolin renderer
    :param vertices, faces: torch.tensors
    :param renders: rendered images
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    import kaolin as kal

    # TODO: backproject render features to the mesh
    #   1. maintain z-buffer for tracking face visibility
    #   2. backproject features to the mesh (look into the inverse code)
    #       2.5 Get feature for pixel is the key function (inputs 2D vertex location and returns V x N feature map)
    #   3. optimization over the vertex features

    # Initialize vertex features
    tmp_features = next(iter(dataloader))['featuremaps'] # B x Num Patches x emb_dim
    npatches = tmp_features.shape[1]
    emb_dim = tmp_features.shape[-1]

    vertex_features = torch.zeros(len(vertices), emb_dim).to(device)
    vertex_features.requires_grad_()
    optim = torch.optim.Adam([vertex_features], lr=0.1)

    overall_visibility = torch.zeros(len(vertices), device=device)

    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(dataloader):
            featuremaps = data['featuremaps']
            featuremaps = featuremaps.to(device)
            positions = data['positions']
            lookats = data['lookats']

            # Define camera
            cam = kal.render.camera.Camera.from_args(
                eye=positions,
                at=lookats,
                up=torch.tensor([0., 1., 0.]).to(device),
                fov = fov,
                width=res, height=res,
                device=device
            )

            import nvdiffrast.torch as dr
            glctx = dr.RasterizeCudaContext()

            #### Render the obj ####
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)

            vertices_camera = cam.extrinsics.transform(vertices)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )

            vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)

            # Flip y-axis values for grid_sample (-1, -1) is top-left
            vertices_ndc = vertices_ndc * torch.tensor([1, -1, 1]).to(device)

            # Y and X also need to be swapped in the indexing
            vertices_ndc = vertices_ndc[..., [1, 0, 2]]

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[res, res])

            zimage = vertices_image[...,[-2]].contiguous()
            zbuffer, _ = dr.interpolate(attr=zimage, rast=rast, tri=faces.int())

            # Replace 0s with high z value
            zbuffer[zbuffer == 0] = 1000
            zbuffer = zbuffer.permute(0, 3, 1, 2) # for grid sample indexing

            #### Determine visibility ####
            tmp_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)
            tmp_ndc = (tmp_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]
            tmp_ndc = tmp_ndc[..., [1, 0, 2]]
            vertices_pixels = torch.floor(tmp_ndc[:,:,:2] * res).long() # B x V x 2

            # Sample the zbuffer
            zbuffer_samples = torch.nn.functional.grid_sample(zbuffer, vertices_ndc[:,:,None, :2]).squeeze()
            visible_mask = (zbuffer_samples + 0.0005 >= zimage.squeeze()) # B x V

            overall_visibility += visible_mask.sum(dim=0)

            #### Get features ####
            features_per_view = get_feature_for_pixel_location(
                featuremaps, vertices_pixels,
                image_size=res,
                patch_size=int(res / np.sqrt(npatches)),
                )

            #### Optimize
            optim.zero_grad()
            loss = torch.sum((features_per_view - vertex_features * visible_mask.unsqueeze(-1)) ** 2)
            loss.backward()
            optim.step()

            print(f"Epoch {epoch}: Initial loss: {loss.detach().cpu().item()}")

        print(f"Epoch {epoch}: Loss: {loss.detach().cpu().item()}")

        nonvisible = torch.sum(overall_visibility == 0)
        if nonvisible > 0:
            print(f"WARNING: {nonvisible} vertices are not visible in any view! ")

    return vertex_features

def train_mlp(dataloader, model, featuremlp, vertices, faces, fov,
                            epochs=5, device="cpu", lr=1e-3,
                            ):
    """
    Compute the features extracted by 'model' from rendered images rendered with 'renderer (deprecated)' for the keypoints

    :param renderer: kaolin renderer
    :param model: 2D ViT pipeline (including pre- and post-processing) in: (N, H, W, C), out: (N, num_patches, emb_dim)
    :param vertices, faces: torch.tensors
    :param renders: rendered images
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    import kaolin as kal

    # TODO: backproject render features to the mesh
    #   1. maintain z-buffer for tracking face visibility
    #   2. backproject features to the mesh (look into the inverse code)
    #       2.5 Get feature for pixel is the key function (inputs 2D vertex location and returns V x N feature map)
    #   3. optimization over the vertex features

    # Initialize vertex features
    tmp_renders = next(iter(dataloader))['renders']
    res = tmp_renders.shape[-1]
    with torch.no_grad():
        tmp_processed = model(tmp_renders[[0]])
        emb_dim = tmp_processed.shape[-1]

    optim = torch.optim.Adam([vertex_features], lr=lr)

    overall_visibility = torch.zeros(len(vertices), device=device)

    num_views = len(dataloader.dataset)

    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(dataloader):
            renders = data['renders']
            positions = data['positions']
            lookats = data['lookats']

            # Define camera
            cam = kal.render.camera.Camera.from_args(
                eye=positions,
                at=lookats,
                up=torch.tensor([0., 1., 0.]).to(device),
                fov = fov,
                width=renders.shape[3], height=renders.shape[2],
                device=device
            )

            import nvdiffrast.torch as dr
            glctx = dr.RasterizeCudaContext()

            #### Render the obj ####
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces.long(), fnormals.unsqueeze(2).repeat(1,1,3,1))

            vertices_camera = cam.extrinsics.transform(vertices)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )

            vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)

            # Flip y-axis values for grid_sample (-1, -1) is top-left
            vertices_ndc = vertices_ndc * torch.tensor([1, -1, 1]).to(device)

            # Y and X also need to be swapped in the indexing
            vertices_ndc = vertices_ndc[..., [1, 0, 2]]

            # vertices_ndc = (vertices_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[res, res])

            zimage = vertices_image[...,[-2]].contiguous()
            zbuffer, _ = dr.interpolate(attr=zimage, rast=rast, tri=faces.int())
            # Replace 0s with high z value
            zbuffer[zbuffer == 0] = 1000
            zbuffer = zbuffer.permute(0, 3, 1, 2) # for grid sample indexing

            #### Determine visibility ####
            # TODO: Below is for sanity checking
            tmp_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)
            tmp_ndc = (tmp_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]
            tmp_ndc = tmp_ndc[..., [1, 0, 2]]
            vertices_pixels = torch.floor(tmp_ndc[:,:,:2] * res).long() # B x V x 2

            # Sample the zbuffer
            zbuffer_samples = torch.nn.functional.grid_sample(zbuffer, vertices_ndc[:,:,None, :2]).squeeze()
            visible_mask = (zbuffer_samples + 0.0005 >= zimage.squeeze()) # B x V

            overall_visibility += visible_mask.sum(dim=0)

            #### Get features ####
            with torch.no_grad():
                processed_images = model(renders)  # (N, num_patches, emb_dim)

            from .model_wrappers import SAMWrapper
            features_per_view = get_feature_for_pixel_location(
                processed_images, vertices_pixels, image_size=res,
                patch_size=int(res / np.sqrt(processed_images.shape[1])),
                use_sam=isinstance(model, SAMWrapper)) # B x V x emb_dim

            #### Optimize
            optim.zero_grad()
            loss = torch.sum((features_per_view - vertex_features * visible_mask.unsqueeze(-1)) ** 2)
            loss.backward()
            optim.step()

            print(f"Epoch {epoch}: Initial loss: {loss.detach().cpu().item()}")

        print(f"Epoch {epoch}: Loss: {loss.detach().cpu().item()}")

        nonvisible = torch.sum(overall_visibility == 0)
        if nonvisible > 0:
            print(f"WARNING: {nonvisible} vertices are not visible in any view! ")

    return vertex_features

def features_from_renders(dataloader, model, vertices, faces, fov,
                            epochs=5, device="cpu",
                            ):
    """
    Compute the features extracted by 'model' from rendered images rendered with 'renderer (deprecated)' for the keypoints

    :param renderer: kaolin renderer
    :param model: 2D ViT pipeline (including pre- and post-processing) in: (N, H, W, C), out: (N, num_patches, emb_dim)
    :param vertices, faces: torch.tensors
    :param renders: rendered images
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    import kaolin as kal

    # TODO: backproject render features to the mesh
    #   1. maintain z-buffer for tracking face visibility
    #   2. backproject features to the mesh (look into the inverse code)
    #       2.5 Get feature for pixel is the key function (inputs 2D vertex location and returns V x N feature map)
    #   3. optimization over the vertex features

    # Initialize vertex features
    tmp_renders = next(iter(dataloader))['renders']
    res = tmp_renders.shape[-1]
    with torch.no_grad():
        tmp_processed = model(tmp_renders[[0]])
        emb_dim = tmp_processed.shape[-1]
    vertex_features = torch.zeros(len(vertices), emb_dim).to(device)
    vertex_features.requires_grad_()
    optim = torch.optim.Adam([vertex_features], lr=0.1)

    overall_visibility = torch.zeros(len(vertices), device=device)

    num_views = len(dataloader.dataset)

    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(dataloader):
            renders = data['renders']
            positions = data['positions']
            lookats = data['lookats']

            # Define camera
            cam = kal.render.camera.Camera.from_args(
                eye=positions,
                at=lookats,
                up=torch.tensor([0., 1., 0.]).to(device),
                fov = fov,
                width=renders.shape[3], height=renders.shape[2],
                device=device
            )

            import nvdiffrast.torch as dr
            glctx = dr.RasterizeCudaContext()

            #### Render the obj ####
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces.long(), fnormals.unsqueeze(2).repeat(1,1,3,1))

            vertices_camera = cam.extrinsics.transform(vertices)
            vertices_image = cam.intrinsics.transform(vertices_camera) # B x V x 3
            vertices_image = torch.nn.functional.pad(
                vertices_image,
                (0, 1), mode='constant', value=1
            )

            vertices_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)

            # Flip y-axis values for grid_sample (-1, -1) is top-left
            vertices_ndc = vertices_ndc * torch.tensor([1, -1, 1]).to(device)

            # Y and X also need to be swapped in the indexing
            vertices_ndc = vertices_ndc[..., [1, 0, 2]]

            # vertices_ndc = (vertices_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]

            rast, _ = dr.rasterize(glctx=glctx, pos=vertices_image, tri=faces.int(), resolution=[res, res])

            zimage = vertices_image[...,[-2]].contiguous()
            zbuffer, _ = dr.interpolate(attr=zimage, rast=rast, tri=faces.int())
            # Replace 0s with high z value
            zbuffer[zbuffer == 0] = 1000
            zbuffer = zbuffer.permute(0, 3, 1, 2) # for grid sample indexing

            #### Determine visibility ####
            # TODO: Below is for sanity checking
            tmp_ndc = kal.render.camera.intrinsics.down_from_homogeneous(vertices_image)
            tmp_ndc = (tmp_ndc + 1) / 2 # Map from [-1, 1] to [0, 1]
            tmp_ndc = tmp_ndc[..., [1, 0, 2]]
            vertices_pixels = torch.floor(tmp_ndc[:,:,:2] * res).long() # B x V x 2

            # Sample the zbuffer
            zbuffer_samples = torch.nn.functional.grid_sample(zbuffer, vertices_ndc[:,:,None, :2],
                                                              padding_mode="zeros").squeeze()
            visible_mask = (zbuffer_samples + 0.0005 >= zimage.squeeze()) # B x V

            overall_visibility += visible_mask.sum(dim=0)

            #### Get features ####
            with torch.no_grad():
                processed_images = model(renders)  # (N, num_patches, emb_dim)

            from .model_wrappers import SAMWrapper
            features_per_view = get_feature_for_pixel_location(
                processed_images, vertices_pixels, image_size=res,
                patch_size=model.patch_size(),
                use_sam=isinstance(model, SAMWrapper)) # B x V x emb_dim

            #### Optimize
            optim.zero_grad()
            loss = torch.sum((features_per_view - vertex_features * visible_mask.unsqueeze(-1)) ** 2)
            loss.backward()
            optim.step()

            print(f"Epoch {epoch}: Initial loss: {loss.detach().cpu().item()}")

        print(f"Epoch {epoch}: Loss: {loss.detach().cpu().item()}")

        nonvisible = torch.sum(overall_visibility == 0)
        if nonvisible > 0:
            print(f"WARNING: {nonvisible} vertices are not visible in any view! ")

    return vertex_features

def features_from_views(renderer, model, mesh, views, keypoints=None, only_visible=True, render_dist=1.0,
                        batch_size=None, device="cpu", geo_dists=None, gaussian_sigma=0.1,
                        verbose=False):
    """
    Compute the features extracted by 'model' from rendered images rendered with 'renderer (deprecated)' for the keypoints

    :param renderer: pytorch3d MeshRendererWithFragments
    :param model: 2D ViT pipeline (including pre- and post-processing) in: (N, H, W, C), out: (N, num_patches, emb_dim)
    :param mesh: pytorch3d Mesh
    :param views: viewpoints (e.g. from 'views_around_object')
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    mesh = mesh.to(device)
    points = torch.tensor(np.array([kp["xyz"] for kp in keypoints]), dtype=torch.float32).to(
        device) if keypoints is not None else mesh.verts_packed()
    res = renderer.rasterizer.raster_settings.image_size

    if geo_dists is not None:
        # NOTE: We keep this as a np array because it is too memory intensive to keep it as a torch tensor
        reweight = np.exp(-geo_dists ** 2 / (2 * gaussian_sigma ** 2))
        # reweight = torch.tensor(np.exp(-geo_dists ** 2 / (2 * gaussian_sigma ** 2)), device=device)

    if batch_size is None:
        batch_size = len(views)

    num_views = len(views)

    # camera transform
    R = look_at_rotation(views, device=device)
    T = torch.tensor([0, 0, render_dist], device=device).repeat(len(views), 1)

    ret_array = None  # initialize when we know the embedding size
    point_values_counts = torch.zeros(len(points)).to(device)

    overall_visibility = torch.zeros(len(points))
    while len(views) > 0:
        batch_views = views[:batch_size]
        batch_R = R[:batch_size]
        batch_T = T[:batch_size]

        views = views[batch_size:]
        R = R[batch_size:]
        T = T[batch_size:]

        okay = False
        import time

        while not okay:
            try:
                # 1. Render the mesh
                camera = FoVPerspectiveCameras(R=batch_R, T=batch_T, device=device)
                light = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=batch_views, device=device)

                start = time.time()
                with torch.no_grad():
                    images, fragments = renderer(mesh.extend(len(batch_views)), cameras=camera, lights=light)
                    images = images[..., :3]
                end = time.time()
                if verbose:
                    print(end - start, "for rendering")

                pixel_coords_all_points = camera.transform_points_screen(points,
                                                                         image_size=(res, res)).cpu()  # (V, N, 3)

                # 2. Determine visibility
                start = time.time()
                if only_visible:
                    if keypoints is not None:
                        visible_points = check_visible_points(fragments.pix_to_face, mesh, points)
                    else:
                        visible_points = check_visible_vertices(fragments.pix_to_face, mesh)
                    overall_visibility += visible_points.cpu().sum(dim=0)
                end = time.time()
                if verbose:
                    print(end - start, "for visibility")

                # 3. Extract features
                start = time.time()

                with torch.no_grad():
                    processed_images = model(images)  # (N, num_patches, emb_dim)
                end = time.time()
                if verbose:
                    print(end - start, "for model")
                start = time.time()
                from .model_wrappers import SAMWrapper
                features_per_view = get_feature_for_pixel_location(
                    processed_images, pixel_coords_all_points, image_size=res,
                    patch_size=int(res / np.sqrt(processed_images.shape[1])),
                    use_sam=isinstance(model, SAMWrapper))  # (V, N, emb_dim)
                end = time.time()
                if verbose:
                    print(end - start, "for getting features for pixel location")

                # 4. Aggregate features
                if ret_array is None:
                    ret_array = torch.zeros(len(points), features_per_view.shape[-1]).to(device)

                start = time.time()
                if only_visible:
                    ret_array += torch.sum(features_per_view * visible_points[..., None], dim=0)
                    point_values_counts += visible_points.sum(dim=0)
                else:
                    ret_array += torch.sum(features_per_view, dim=0)
                    point_values_counts += features_per_view.size(0)  # Increment by the number of batch_views

                if only_visible and geo_dists is not None:
                    value_array = ret_array
                    point_counts = point_values_counts
                    point_values_counts = torch.zeros(len(points)).to(device)
                    ret_array = torch.zeros(len(points), features_per_view.shape[-1]).to(device)
                    for i in range(len(points)):
                        if point_counts[i] <= 0:
                            continue
                        reweighted_features = torch.outer(torch.from_numpy(reweight[i]).to(device),
                                                          value_array[i])
                        ret_array += reweighted_features
                        point_values_counts += torch.from_numpy(reweight[i]).to(device) * point_counts[i]
                end = time.time()
                if verbose:
                    print(end - start, "for adding features")

                okay = True

            except AssertionError as e:
                print(e)
                batch_T = batch_T + 0.05

    if only_visible:
        if verbose:
            print(f"Number of views: {num_views}",
                  f"Median visibility of points: {overall_visibility.median().item()}",
                  f"Mean visibility of points: {overall_visibility.mean().item()}")
        if torch.any(overall_visibility == 0):
            print(f"WARNING: {torch.sum(overall_visibility == 0)} points are not visible in any view! ")

    ret_array[point_values_counts > 0] /= point_values_counts[point_values_counts > 0][:, None]
    return ret_array


def get_feature_for_pixel_location(feature_map, pixel_locations, image_size=224, patch_size=14, use_sam=False):
    """
    :param feature_map: (V, (image_size / patch_size) ** 2, emb_dim)
    :param pixel_locations: (V, N, 3) from camera.transform_points_screen
    :param image_size: Size of the rendered image
    :param patch_size: Size of the patches in the feature map
    :param use_sam: Whether to use the SAM model
    :return: (V, N, emb_dim)
    """

    if use_sam:
        image_size = 1024
        patch_size = 16

    def transform_px_to_patch_id(pixel_locations, image_size=224, patch_size=14):
        """
        :param pixel_locations: (V, N, 3) from camera.transform_points_screen
        :param image_size: Size of the rendered image
        :param patch_size: Size of the patches in the feature map
        :return: (V, N) patch_id
        """
        if len(pixel_locations.shape) == 2:
            pixel_locations = pixel_locations.unsqueeze(0)
        # assert pixel_locations.max() <= image_size, "Pixel locations must be in [0, image_size], but max is {}".format(
        #     pixel_locations.max())
        return (pixel_locations[:, :, 1] // patch_size * (image_size / patch_size) + pixel_locations[:, :,
                                                                                     0] // patch_size).long()

    patch_id = transform_px_to_patch_id(pixel_locations, image_size=image_size, patch_size=patch_size)  # (V, N)

    patch_id[torch.any(pixel_locations < 0, dim=-1)] = 0
    patch_id[torch.any(pixel_locations >= image_size, dim=-1)] = 0

    ret = torch.stack([feature_map[i, patch_id[i], :] for i in range(len(feature_map))])

    # All pixel locations which fall outside of image are set to 0
    ret[torch.any(pixel_locations < 0, dim=-1),:] = 0
    ret[torch.any(pixel_locations >= image_size, dim=-1),:] = 0

    return ret
