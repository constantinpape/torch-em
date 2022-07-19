# Shallow2Deep for Mitochondria in EM

## Evaluation

Evaluation of different shallow2deep setups for mitochondria segmentation in EM.
The enhancers are (potentially) trained on multiple datasets, evaluation is always on the EPFL dataset (which is ofc not part of the training set).
All scores are measured with a soft dice score.


### V4

- 2d enhancer: trained on mito-em and vnc
- anisotropic enhancer: random forests are trained in 2d, enhancer trained in 3d, trained on mito-em
- direct-nets: 2d and 3d networks trained on mito-em
- different strategies for training the initial rfs:
    - `vanilla`: random forests are trained on randomly sampled dense patches
    - `worst_points`: initial stage of forests (25 forests) are trained on random samples, forests in the next stages add worst predictions from prev. stage to their training set
    - `uncertain_worst_points`: same as `worst_points`, but points are selected based on linear combination of uncertainty and worst predictions

a


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
