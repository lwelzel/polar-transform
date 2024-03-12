# polar-transform
Library for polar transforms of images using PyTorch and Kornia. The module can be resued if the mappings (pixel flow) are the same, e.g. if you have a number of images from the same camera etc. No idea if it propagates gradients. Very small features (~1 pixel) might not be preserved exactly, depending on the upscaling.

## Noiseless example
![polar_transform_no_noise](https://github.com/lwelzel/polar-transform/assets/29613344/8e2781d8-3727-4b04-ba87-9a91c793540c)

## Noisy example
![polar_transform_with_noise](https://github.com/lwelzel/polar-transform/assets/29613344/ebd1e8f3-a98e-464e-9dec-d05f3bb2672f)
