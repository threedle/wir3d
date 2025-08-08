from .backprojection import compute_kp_dists_features, features_from_views
from .model_wrappers import CLIPWrapper, DINOWrapper, EffNetWrapper, SAMWrapper

models = {
    "clip": CLIPWrapper,
    "dino": DINOWrapper,
    "effnet": EffNetWrapper,
    "sam": SAMWrapper
}
