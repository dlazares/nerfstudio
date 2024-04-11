# Dynamic Splatting with multiple cameras

## Download multicam data

`ns-download-data dynerf --capture-name flame-steak`

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


## View the Dynamic Sequence after training
Run the viewer and move the time slider in the control panel.
`ns-viewer --load-config outputs/XXYYZZ/config.yml`
