# Dynamic Splatting with multiple cameras

## Download multicam data

`ns-download-data dynerf --capture-name flame-steak`

## Install COLMAP

```
# Slightly modified from https://colmap.github.io/install.html#linux
# had to add CMAKE_CUDA_ARCHITECTURES based on error message
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
cd ~
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="native" -DCUDA_ENABLED=ON #-DGUI_ENABLED=OFF 
ninja
sudo ninja install
```

## Process multicam data locally
`ns-process-data multicam-video --data data/dynerf/flame-steak/ --output-dir data/dynerf/flame-steak-proc`
\
\
limit frames with:
`--num-frames-target 10`

## Train Single Frame
`ns-train splatfacto --data data/dynerf/flame-steak-proc --vis viewer+tensorboard`
\
\
Look for the unique string output in outputs/ and inspect with `tensorboard --logdir outputs/flame-steak-proc/splatfacto/2024-04-20_XYZ` or visit the viewer at `localhost:7007`

## Train Dynamic Sequence
`ns-train dynamic-splatfacto --data data/dynerf/flame-steak-proc --vis viewer+tensorboard`
\
\
with downscale factor
`ns-train dynamic-splatfacto --vis viewer+tensorboard multicam-nerfstudio-data --downscale-factor 8 --data data/dynerf/flame-steak-proc`

