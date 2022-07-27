# Shallow2Deep for Mitochondria in EM

## Set-up

Evaluation of different shallow2deep setups for mitochondria segmentation in EM.
The enhancers are (potentially) trained on multiple datasets, evaluation is done on datasets not part of the enhancer training set.
All scores are measured with a soft dice score.

## Datasets

- Mito-EM
- VNC
- Lucchi
- UroCell
- Platy
- Kasthuri <- this is weirdly aligned, so don't use it for now (this is likely why the 3d one is so bad in V4)


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

Evaluation on Lucchi:

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

Evaluation on Platy:

| method                             |   few-labels |   medium-labels |   many-labels |
|:-----------------------------------|-------------:|----------------:|--------------:|
| rf3d                               |        0.354 |           0.298 |         0.255 |
| 2d-random_points                   |        0.185 |           0.242 |         0.237 |
| 2d-uncertain_worst_points          |        0.202 |           0.255 |         0.229 |
| 2d-vanilla                         |        0.243 |           0.297 |         0.253 |
| 2d-worst_points                    |        0.250 |           0.294 |         0.280 |
| 2d-worst_tiles                     |        0.209 |           0.277 |         0.251 |
| direct_2d                          |        0.299 |         nan     |       nan     |
| anisotropic-random_points          |        0.203 |           0.221 |         0.246 |
| anisotropic-uncertain_worst_points |        0.208 |           0.283 |         0.294 |
| anisotropic-vanilla                |        0.244 |           0.316 |         0.344 |
| anisotropic-worst_points           |        0.272 |           0.317 |         0.307 |
| anisotropic-worst_tiles            |        0.237 |           0.324 |         0.331 |
| direct_anisotropic                 |        0.235 |         nan     |       nan     |
| 3d-random_points                   |        0.190 |           0.184 |         0.168 |
| 3d-worst_tiles                     |        0.172 |           0.200 |         0.213 |


### V5

- 2d: Mito-EM, VNC, Lucchi
- anisotropic: Mito-EM, Lucchi, and UroCell
- 3d: Lucchi and UroCell

| method                  |   few-labels |   medium-labels |   many-labels |
|:------------------------|-------------:|----------------:|--------------:|
| rf3d                    |        0.354 |           0.298 |         0.255 |
| 2d-worst_tiles          |        0.223 |           0.289 |         0.261 |
| direct_2d               |        0.299 |         nan     |       nan     |
| anisotropic-worst_tiles |        0.281 |           0.325 |         0.291 |
| direct_anisotropic      |        0.235 |         nan     |       nan     |
| 3d-worst_tiles          |        0.338 |           0.285 |         0.258 |

Intermediate summary and TODOs:
- 3d one gets much better with proper training data
- the others don't change much (mito em is probably dominating performance, but hard to measure with the bad overall performacne)
- bad random forest predictions are def. the main problem
- TODO (here and in v4): measure performance of enhancer on source and source net performance on soruce to find the gap between prediction quality there


## V6

TODO same as V5, but train everything on Platy as well and upload the one with best sampling strategy to bioimage.io


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
