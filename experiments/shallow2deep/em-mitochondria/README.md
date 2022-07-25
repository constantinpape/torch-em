# Shallow2Deep for Mitochondria in EM

## Evaluation

Evaluation of different shallow2deep setups for mitochondria segmentation in EM.
The enhancers are (potentially) trained on multiple datasets, evaluation is done on the Kasthuri dataset (which is not part of the training set except for one last version that will be the (for now) final one to be uploaded to bioimagei.io).
All scores are measured with a soft dice score.

## Datasets

- Mito-EM
- VNC
- Lucchi
- UroCell
- Kasthuri

TODO: check if we can add the mito data from platy into the mix! This could be very good for evaluation!


### V4

- 2d enhancer: trained on Mito-EM and VNC
- anisotropic enhancer: random forests are trained in 2d, enhancer trained in 3d, trained on Mito-EM
- 3d enhancer: random forests trained in 3d, enhancer trained in 3d, trained on Kasthuri
- direct-nets: 2d and anisotropic networks trained on Mito-EM, 3d network trained on Kasthuri
- different strategies for training the initial rfs:
    - `vanilla`: random forests are trained on randomly sampled dense patches
    - `worst_points`: initial stage of forests (25 forests) are trained on random samples, forests in the next stages add worst predictions from prev. stage to their training set
    - `uncertain_worst_points`: same as `worst_points`, but points are selected based on linear combination of uncertainty and worst predictions
    - `random_points`: random points sampled in each stage, points are accumulated over the stages
    - `worst_tiles`: training samples are taken from worst tile predictions

| method                             |   few-labels |   medium-labels |   many-labels |
|:-----------------------------------|-------------:|----------------:|--------------:|
| rf3d                               |        0.326 |           0.328 |         0.385 |
| 2d-random_points                   |        0.593 |           0.693 |         0.782 |
| 2d-uncertain_worst_points          |        0.613 |           0.777 |         0.794 |
| 2d-vanilla                         |        0.639 |           0.717 |         0.764 |
| 2d-worst_points                    |        0.549 |           0.711 |         0.730 |
| 2d-worst_tiles                     |        0.661 |           0.796 |         0.828 |
| direct_2d                          |        0.849 |         nan     |       nan     |
| anisotropic-random_points          |        0.521 |           0.566 |         0.671 |
| anisotropic-uncertain_worst_points |        0.530 |           0.616 |         0.711 |
| anisotropic-vanilla                |        0.576 |           0.660 |         0.749 |
| anisotropic-worst_points           |        0.458 |           0.568 |         0.600 |
| anisotropic-worst_tiles            |        0.614 |           0.728 |         0.788 |
| direct_anisotropic                 |        0.467 |         nan     |       nan     |
| 3d-random_points                   |        0.344 |           0.381 |         0.353 |
| 3d-worst_tiles                     |        0.385 |           0.472 |         0.504 |


### V5

TODO: (only best sampling from V4)
- train 2d on Mito-EM, VNC, Kasthuri and UroCell
- train anisotropic on Mito-EM, Kasthuri and UroCell
- train 3d on Kasthuri and UroCell

## V6

TODO same as V5, but train everything on Lucchi as well and upload the one with best sampling strategy to bioimage.io


## Old evaluation

Evaluation of older set-ups.

### V1

Enhancers and direct model trained on VNC and applied to crops of MITO-EM; everything fully 2d.

| enhancer          |   few-labels |   many-labels |
|:------------------|-------------:|--------------:|
| advanced-enhancer |     0.319515 |      0.453895 |
| rf-score          |     0.163359 |      0.165454 |
| vanilla-enhancer  |     0.263941 |      0.436766 |
| direct-net        |     0.178033 |      0.32553  |
Raw net evaluation: 0.43931060004562367

### V2

Enhancers and direct model trained on Mito-EM and applied to VNC; everything fully in 2d.

Evaluation results:
| enhancer          |   few-labels |   medium-labels |   many-labels |
|:------------------|-------------:|----------------:|--------------:|
| advanced-enhancer |    0.703895  |        0.735011 |     0.752428  |
| rf-score          |    0.336755  |        0.288951 |     0.298257  |
| vanilla-enhancer  |    0.728594  |        0.732537 |     0.738573  |
| direct-net        |    0.0538231 |        0.227444 |     0.0402941 |
Raw net evaluation: 0.5664974146646325


### V3

Enhancer and direct model trained on Mito-EM and applied to VNC; everything fully in 3d

Evaluation results:
| enhancer          |   few-labels |   medium-labels |   many-labels |
|:------------------|-------------:|----------------:|--------------:|
| vanilla-enhancer  |     0.439439 |        0.464081 |      0.45908  |
| advanced-enhancer |     0.461565 |        0.495362 |      0.574192 |
| rf-score          |     0.259479 |        0.223419 |      0.223723 |
Raw net evaluation: 0.20031713226486814
