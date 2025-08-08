"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
import collections

class ModelWrapper(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, interpolation, images):
        pass

    @abstractmethod
    def patch_size(self):
        pass


class DINOWrapper(ModelWrapper):
    def __init__(self, device='cpu', small=False):
        super().__init__()
        if not small:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
        self.model.eval()

        self.image_transforms = T.Compose([
            T.Resize((224, 224)), # Resize to 224
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images, interpolation=False):
        images = self.image_transforms(images)
        out = self.model.forward_features(images)
        if interpolation:
            features = out['x_norm_patchtokens']
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out['x_norm_patchtokens']

    def patch_size(self):
        return 14


class CLIPWrapper(ModelWrapper):
    def __init__(self, device='cpu', modelpath=None, modeltype='ViT-L/14'):
        super().__init__()
        import clip
        if modelpath is None:
            self.model, self.image_transforms = clip.load(modeltype, device=device, jit=False)
        else:
            self.model, self.image_transforms = clip.load(modelpath, device=device, jit=False)
        self.model.eval()

        self.image_transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True, max_size=None),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images, interpolation=False):
        images = self.image_transforms(images).type(self.model.dtype)
        x = self.model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                   dtype=x.dtype, device=x.device), x],
                      dim=1)
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)

        out = self.model.visual.ln_post(x[:, :, :])

        if self.model.visual.proj is not None:
            out = out @ self.model.visual.proj

        if interpolation:
            features = out[:, 1:, :]
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out[:, 1:, :]

    def patch_size(self):
        return 14

# class CLIPVisualEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.clip_model = clip_model
#         self.featuremaps = None

#         # NOTE: This may not be true always!! Need to check if use intermediate layers
#         for i in range(12):  # 12 resblocks in VIT visual transformer
#             self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
#                 self.make_hook(i))

#     def make_hook(self, name):
#         def hook(module, input, output):
#             if len(output.shape) == 3:
#                 self.featuremaps[name] = output.permute(
#                     1, 0, 2)  # LND -> NLD bs, smth, 768
#             else:
#                 self.featuremaps[name] = output

#         return hook

#     def forward(self, x):
#         self.featuremaps = collections.OrderedDict()
#         fc_features = self.clip_model.encode_image(x).float()
#         featuremaps = [self.featuremaps[k] for k in range(12)]

#         return fc_features, featuremaps

# class CLIPWrapper(ModelWrapper):
#     def __init__(self, device='cpu', modelpath=None, modeltype='ViT-L/14'):
#         super(CLIPWrapper, self).__init__()
#         self.clip_model_name = modeltype
#         # assert self.clip_model_name in [
#         #     "RN50",
#         #     "RN101",
#         #     "RN50x4",
#         #     "RN50x16",
#         #     "ViT-B/32",
#         #     "ViT-B/16",
#         # ]
#         import clip
#         from torchvision import transforms

#         if modelpath is None:
#             self.model, self.clip_preprocess = clip.load(modeltype, device=device, jit=False)
#         else:
#             self.model, self.clip_preprocess = clip.load(modelpath, device=device, jit=False)
#         self.model.eval()

#         if self.clip_model_name.startswith("ViT"):
#             self.visual_encoder = CLIPVisualEncoder(self.model)

#         else:
#             self.visual_model = self.model.visual
#             layers = list(self.model.visual.children())
#             # init_layers = torch.nn.Sequential(*layers)[:8]
#             # self.layer1 = layers[8]
#             # self.layer2 = layers[9]
#             # self.layer3 = layers[10]
#             # self.layer4 = layers[11]
#             # self.att_pool2d = layers[12]

#             self.layer1 = self.visual_model.layer1
#             self.layer2 = self.visual_model.layer2
#             self.layer3 = self.visual_model.layer3
#             self.layer4 = self.visual_model.layer4
#             self.att_pool2d = self.visual_model.attnpool

#         self.img_size = self.clip_preprocess.transforms[1].size
#         self.model.eval()
#         self.device = device

#         self.normalize_transform = transforms.Compose([
#             self.clip_preprocess.transforms[0],  # Resize
#             self.clip_preprocess.transforms[-1],  # Normalize
#         ])

#     def forward(self, images, layer='fc'):
#         xs = self.normalize_transform(images.to(self.device))

#         if self.clip_model_name.startswith("RN"):
#             xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
#                 xs.contiguous())
#         else:
#             xs_fc_features, xs_conv_features = self.visual_encoder(xs)

#         if layer == "fc":
#             return xs_fc_features.unsqueeze(1)
#         else:
#             return xs_conv_features[layer].reshape(xs_fc_features.shape[0], xs_conv_features[layer].shape[1], -1).permute(0, 2, 1)

#     def forward_inspection_clip_resnet(self, x):
#         def stem(m, x):
#             for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
#                 x = m.relu1(bn(conv(x)))
#             x = m.avgpool(x)
#             return x

#         x = x.type(self.visual_model.conv1.weight.dtype)
#         x = stem(self.visual_model, x)
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         y = self.att_pool2d(x4)
#         return y, [x, x1, x2, x3, x4]

#     def patch_size(self):
#         return 1

class SAMWrapper(ModelWrapper):
    def __init__(self, device='cpu', checkpointdir=None):
        super().__init__()
        from segment_anything import sam_model_registry
        if checkpointdir is None:
            checkpointdir = "/share/data/pals/guanzhi/SAMmodels/sam_vit_l_0b3195.pt"
        self.model = sam_model_registry["vit_l"](checkpoint=checkpointdir).to(device)
        self.model.eval()

        self.image_transforms = T.Compose(
            [T.Resize((self.model.image_encoder.img_size, self.model.image_encoder.img_size),
                      interpolation=T.InterpolationMode.BICUBIC, antialias=True),
             T.Normalize(torch.tensor((123.675, 116.28, 103.53)) / torch.tensor(
                (58.395, 57.12, 57.375)), (1., 1., 1.))]
        )

    def preprocess(self, images):
        images = self.image_transforms(images)
        h, w = images.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        return F.pad(images, (0, padw, 0, padh))

    def forward(self, images, interpolation=False):
        images = self.preprocess(images)
        x = self.model.image_encoder.patch_embed(images)
        if self.model.image_encoder.pos_embed is not None:
            x = x + self.model.image_encoder.pos_embed
        for blk in self.model.image_encoder.blocks:
            x = blk(x)

        out = x.view(x.shape[0], -1, x.shape[-1])

        if interpolation:
            features = out
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out

    def patch_size(self):
        return 1


class EffNetWrapper(ModelWrapper):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        self.model.eval().to(device)
        self.image_transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def forward(self, images, interpolation=False):
        images = self.image_transforms(images)
        out = self.model.extract_features(images)["layer5"]
        out = torch.permute(out, (0, 2, 3, 1)).view(out.shape[0], -1, out.shape[1])
        if interpolation:
            features = out
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out

    def patch_size(self):
        return 14  # TODO: CHECK THIS
