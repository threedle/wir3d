# WIR3D (ICCV 2025 Oral)
![teaser](assets/teaser.png)

[Project Page](https://threedle.github.io/wir3d/)

Official implementation of "WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction"

## Installation
**Requirements**
- Blender 3.6
- GPU >= 24GB
- CUDA >= 12.4

```bash
conda env create -f wir3d.yml
conda activate wir3d
```
```bash
pip install git+https://github.com/openai/CLIP.git
```
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
```bash
pip install git+https://github.com/jonbarron/robust_loss_pytorch
```
```bash
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu124.html --no-dependencies
```
```bash
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```
Install [nvdiffrast](https://nvlabs.github.io/nvdiffrast/#installation) according to the instructions.

Code has been tested on Ubuntu 20.04 with an A40 GPU.
## Usage
WIR3D will work for any triangle mesh, with optional texture and keypoints (must be in the same directory as the obj file). To preprocess the input mesh, you must first call `preprocess.sh` as follows.
```bash
BLENDER36={path to blender executable} ./preprocess.sh {path to obj file} -t {name of texture file} -k {name of keypoints file}
```
The preprocessing may take anywhere from 1-3 hours depending on your system.

After preprocessing, call `generate_keypoints_final.py` to run WIR3D.
```bash
python optimize_twostage_final.py {path to obj file} --sem_clip_model_name RN50x16 --sem-spatial-keypoint-loss keypoints.pt --sem-spatial-weight-base 0 --init_keypoint_dir_sem keypoints.pt --outputdir {path to output directory}
```
Changing the CLIP model (--sem_clip_model_name) will result in abstractions of different quality. Our experiments show that RN50x16 and ViT-L/14 generally work well.

The best output curves from the optimization will be saved in `{output directory}/seed0/sem/tot_bestcurves.pt`. The results are visualized in `{output directory}/seed0/sem/best_compare.gif`.

## Examples
We've provided 3 example meshes from our paper to run.
### Spot
```
BLENDER36={path to blender executable} ./preprocess.sh ./data/spot/spot.obj -t spot_texture.png -k spot_manual_keypoints.pt
```
```
python optimize_twostage_final.py ./data/spot/spot.obj --sem_clip_model_name RN50x16 --sem-spatial-keypoint-loss spot_manual_keypoints.pt --sem-spatial-weight-base 1 --sem-lambda-lpips 2 --geonviews 5 --semnviews 5 --geoniters 10000 --semniters 10000 --geoseed 50 --semseed 50000 --setseed --init_keypoint_dir_sem spot_manual_keypoints.pt --outputdir ./outputs/wir3d/spot3
```
![spot](assets/spot_example.gif)
### Nefertiti
```
BLENDER36={path to blender executable} ./preprocess.sh ./data/nefertiti/nefertiti.obj -t nefertiti_texture.png -k nefertiti_manual_keypoints.pt
```
```
python optimize_twostage_final.py ./data/nefertiti/nefertiti.obj --sem_clip_model_name ViT-L/14 --sem-spatial-keypoint-loss nefertiti_manual_keypoints.pt --sem-spatial-weight-base 0 --sem-lambda-lpips 2 --geonviews 5 --semnviews 5 --geoniters 10000 --semniters 10000 --geoseed 281 --semseed 281000 --setseed --init_keypoint_dir_sem nefertiti_manual_keypoints.pt --outputdir ./outputs/wir3d/nefertiti2
```
![nefertiti](assets/nefertiti_example.gif)
### Dragon
```
BLENDER36={path to blender executable} ./preprocess.sh ./data/dragon/dragon.obj -t dragon_texture.png
```
```
python optimize_twostage_final.py ./data/dragon/dragon.obj --init-type-sem furthest --sem_clip_model_name ViT-L/14 --sem-spatial-weight-base 0 --sem_frozen_init_lr 0.0001 --geoseed 22 --semseed 23 --setseed --init_keypoint_dir_sem ./data/dragon/keypoints.pt --nsemcurves 30 --outputdir ./outputs/wir3d/dragon
```
![dragon](assets/dragon_example.gif)
## Citation
```
@InProceedings{Liu_2025_ICCV,
    author    = {Liu, Richard and Fu, Daniel and Tan, Noah and Lang, Itai, and Hanocka, Rana},
    title     = {WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```