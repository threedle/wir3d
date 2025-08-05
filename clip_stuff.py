import clip
import collections
import torch
import torch.nn as nn
from torchvision import models, transforms

# NOTE: We need to raise the dtype of these otherwise go to inf too easily
def l2_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [torch.square((x_conv - y_conv) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [torch.abs((x_conv - y_conv) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, weights=None):
    if weights:
        return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1) * w).mean() for x_conv, y_conv, w in
                zip(xs_conv_features, ys_conv_features, weights)]
    else:
        return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        # NOTE: This may not be true always!! Need to check if use intermediate layers
        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps

class CLIPConvLoss(torch.nn.Module):
    def __init__(self, clip_model_name="RN101", clip_conv_loss_type='L2',
                 clip_conv_layer_weights = [0,0,1.,1.,0],
                 clip_fc_weight = 0.1, num_augs = 4, clip_fc_loss_type='Cos',
                 device=torch.device("cuda:0")):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = clip_model_name
        # assert self.clip_model_name in [
        #     "RN50",
        #     "RN101",
        #     "RN50x4",
        #     "RN50x16",
        #     "ViT-B/32",
        #     "ViT-B/16",
        # ]

        self.clip_conv_loss_type = clip_conv_loss_type
        self.clip_fc_loss_type = clip_fc_loss_type  # clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, self.clip_preprocess = clip.load(
            self.clip_model_name, device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            # init_layers = torch.nn.Sequential(*layers)[:8]
            # self.layer1 = layers[8]
            # self.layer2 = layers[9]
            # self.layer3 = layers[10]
            # self.layer4 = layers[11]
            # self.att_pool2d = layers[12]

            self.layer1 = self.visual_model.layer1
            self.layer2 = self.visual_model.layer2
            self.layer3 = self.visual_model.layer3
            self.layer4 = self.visual_model.layer4
            self.att_pool2d = self.visual_model.attnpool

        self.img_size = self.clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            self.clip_preprocess.transforms[0],  # Resize
            self.clip_preprocess.transforms[1],  # CenterCrop
            self.clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = device
        self.num_augs = num_augs

        augmentations = []
        augmentations.append(transforms.RandomPerspective(
            fill=0, p=1.0, distortion_scale=0.5))
        augmentations.append(transforms.RandomResizedCrop(
            self.clip_preprocess.transforms[0].size, scale=(0.8, 0.8), ratio=(1.0, 1.0), antialias=True))
        augmentations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augmentations)

        self.clip_fc_layer_dims = None  # self.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.clip_conv_layer_dims
        self.clip_fc_loss_weight = clip_fc_weight
        self.counter = 0
        self.clip_conv_layer_weights = clip_conv_layer_weights

    def forward(self, sketch, target, semantic_weights=None, spatial_fc=False, debug=False):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        # sketch_augs, img_augs = [self.normalize_transform(x)], [
        #     self.normalize_transform(y)]
        sketch_augs, img_augs = [], []

        # if mode == "eval":
        #     # for regular clip distance, no augmentations
        #     with torch.no_grad():
        #         sketches = self.preprocess(sketch).to(self.device)
        #         sketches_features = self.model.encode_image(sketches)
        #         return 1. - torch.cosine_similarity(sketches_features, self.targets_features)

        # NOTE: First transform from clip preprocess calls resize
        semantic_augs = None
        if semantic_weights is not None:
            # semantic_augs = [self.clip_preprocess.transforms[0](semantic_weights)]
            semantic_augs = []
            for n in range(self.num_augs):
                # Sample the parameters of the augmentations
                perspective_params = self.augment_trans.transforms[0].get_params(self.clip_preprocess.transforms[0].size, self.clip_preprocess.transforms[0].size, 0.5)

                augmented_pair = torch.cat([self.clip_preprocess.transforms[0](x),
                                            self.clip_preprocess.transforms[0](y)])

                crop_params = self.augment_trans.transforms[1].get_params(augmented_pair[0], scale=(0.8, 0.8), ratio=(1.0, 1.0))

                augmented_pair = transforms.functional.perspective(augmented_pair, *perspective_params, fill=0.)
                augmented_pair = transforms.functional.resize(transforms.functional.crop(augmented_pair, *crop_params),
                                                              (self.clip_preprocess.transforms[0].size, self.clip_preprocess.transforms[0].size))
                augmented_pair = self.augment_trans.transforms[-1](augmented_pair)

                augmented_weight = self.clip_preprocess.transforms[0](semantic_weights)
                augmented_weight = transforms.functional.perspective(augmented_weight, *perspective_params, fill=1.)
                augmented_weight = transforms.functional.resize(transforms.functional.crop(augmented_weight, *crop_params),
                                                                (self.clip_preprocess.transforms[0].size, self.clip_preprocess.transforms[0].size))

                # augmented_pair = self.augment_trans(torch.cat([self.clip_preprocess.transforms[0](x),
                #                                             self.clip_preprocess.transforms[0](y),
                #                                             self.clip_preprocess.transforms[0](semantic_weights)]))
                sketch_augs.append(augmented_pair[:len(x)])
                img_augs.append(augmented_pair[len(y):])
                semantic_augs.append(augmented_weight)
            semantic_augs = torch.cat(semantic_augs, dim=0).to(self.device)
        else:
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([self.clip_preprocess.transforms[0](x),
                                                            self.clip_preprocess.transforms[0](y)]))
                sketch_augs.append(augmented_pair[:len(x)])
                img_augs.append(augmented_pair[len(y):])

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())
        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        # Keep only the features with non-zero conv layer weights
        keep_indices = [i for i in range(len(self.clip_conv_layer_weights)) if self.clip_conv_layer_weights[i] > 0]
        xs_conv_features = [xs_conv_features[i] for i in keep_indices]
        ys_conv_features = [ys_conv_features[i] for i in keep_indices]

        # Resize the semantic weights to each layer
        spatial_weights = None
        if semantic_augs is not None:
            spatial_weights = []
            for i, features in enumerate(xs_conv_features):
                spatial_weights.append(
                    torch.nn.functional.interpolate(semantic_augs, size=features.shape[-2:], mode='bilinear'))

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, spatial_weights)

        for layer, w in enumerate(self.clip_conv_layer_weights):
            if layer in keep_indices:
                lossidx = keep_indices.index(layer)
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[lossidx] * w

        if self.clip_fc_loss_weight:
            weights = None
            if semantic_augs is not None and spatial_fc:
                # Mean pool the semantic weights
                weights = torch.mean(semantic_augs, dim=[1,2,3])

            if self.clip_fc_loss_type == "Cos":
                if weights is None:
                    fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features)).mean()
                else:
                    fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features) * weights).mean()
                # fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                #         ys_fc_features, dim=1)).mean()

            elif self.clip_fc_loss_type == "L2":
                if weights is None:
                    fc_loss = torch.square(xs_fc_features - ys_fc_features).mean(1).mean(0)
                else:
                    fc_loss = (torch.square(xs_fc_features - ys_fc_features).mean(1) * weights).mean(0)

            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        if debug:
            from sklearn.decomposition import PCA
            from PIL import Image
            import numpy as np
            from pathlib import Path
            from optimize_utils import clear_directory

            Path("./outputs/test/debug").mkdir(parents=True, exist_ok=True)
            # clear_directory("./outputs/test/debug")

            ## PCA and save the feature maps for visualisation
            for i in range(len(xs_conv_features)):
                pca = PCA(n_components=3)

                # TODO: Likely need to squeeze out the batch dimension
                xs_pca = pca.fit_transform(xs_conv_features[i][0].permute(1, 2, 0).reshape(-1, xs_conv_features[i][0].shape[0]).detach().cpu().numpy())
                ys_pca = pca.fit_transform(ys_conv_features[i][0].permute(1, 2, 0).reshape(-1, ys_conv_features[i][0].shape[0]).detach().cpu().numpy())

                # Normalize to [0, 1]
                xs_pca = (xs_pca - xs_pca.min(axis=0)) / (xs_pca.max(axis=0) - xs_pca.min(axis=0))
                ys_pca = (ys_pca - ys_pca.min(axis=0)) / (ys_pca.max(axis=0) - ys_pca.min(axis=0))

                # Convert to PIL image and save
                img = Image.fromarray((xs_pca * 255).reshape(xs_conv_features[i][0].shape[1], xs_conv_features[i][0].shape[2], -1).astype(np.uint8))
                img.save(f"./outputs/test/debug/ours_pca_layer_{i}.png")

                img = Image.fromarray((ys_pca * 255).reshape(ys_conv_features[i][0].shape[1], ys_conv_features[i][0].shape[2], -1).astype(np.uint8))
                img.save(f"./outputs/test/debug/gt_pca_layer_{i}.png")

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu1(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
