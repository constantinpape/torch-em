# Shallow2Deep for mitochondria

## Evaluation

Evaluation of different shallow2deep setups on EM-Mitochondria. All scores are measured with a soft dice score.

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
