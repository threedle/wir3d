#!/bin/bash
# First check that BLENDER36 is set
if [ -z "$BLENDER36" ]; then
  echo "Error: BLENDER36 environment variable is not set. Please set it to the path of Blender 3.6 executable."
  exit 1
fi

# Command line args: model path, -t/--texture path to texture, -k/--keypoints path to keypoints
MODELPATH=$1; shift 1;
TEXTUREFILE=
KPFILE=
while true; do
  case "$1" in
    -t | --texture ) TEXTUREFILE="$2"; shift 2 ;;
    -k | --keypoints ) KPFILE="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

# Render standard blender renders
echo "=========================="
echo "Rendering blender renders for model: $MODELPATH"
echo "==========================="
rendercmd="$BLENDER36 ./render/blend_files/soft_shadow_light.blend --background --python ./render/render_mesh.py -- $MODELPATH config.json blender_030_360 0 30 3 0 360 50 --animate --resolution 400 400 --background-color-hex=#FFFFFFFF --radius 2.2 --normalize --overwrite"
if [ -n "$TEXTUREFILE" ]; then
  rendercmd="$rendercmd --textureimg $TEXTUREFILE"
fi
${rendercmd}

# Render the sketch views
echo "==========================="
echo "Rendering sketch renders for model: $MODELPATH"
echo "==========================="
$BLENDER36 ./render/blend_files/freestyle.blend --background --python ./render/sketch_generator.py -- $MODELPATH 0 30 3 0 360 50 --rendername freestyle_030_360 --radius 2.2 --fov 60 --normalize --resolution 400 --overwrite

# Render the surface views (zbuffer for blender renders)
echo "==========================="
echo "Rendering surface renders for model: $MODELPATH"
echo "==========================="
python render_data.py $MODELPATH 0 30 3 0 360 50 --rendername surface_030_360 --radius 2.2 --normalize --resolution 400 --overwrite

# Generate keypoints if keypoints file not given
if [ -z "$KPFILE" ]; then
  echo "==========================="
  echo "Generating keypoints for model: $MODELPATH"
  echo "==========================="
  python generate_keypoints_final.py $MODELPATH blender_030_360 --npoints 15 --model DINO --normalize --batchsize 4 --centertype latent --overlaydir surface_030_360 --modeltype RN50x16 --weighttype l2 --clustertype kmeans
  KPFILE="keypoints.pt"
else
  echo "==========================="
  echo "Using provided keypoints file: $KPFILE"
  echo "==========================="
fi

# Fit SDF
python sdf.py --objdir $MODELPATH --nsteps 20000 --res 256 --batch_size 1024 --positional --overwrite
