# Shallow2Deep for mitochondria

## Evaluation

### V1

Enhancers and direct model trained on VNC and applied to crops of MITO-EM; everything fully 2d.

|                     |few-labels | many-labels|
|---------------------|-----------|------------|
|vanilla-enhancer     |0.263941   |   0.436766 |
|advanced-enhancer    |0.319515   |   0.453895 |
|rf-score             |0.163359   |   0.165454 |
|direct-net           |0.178033   |   0.325530 |

VNC model applied to the MITO-EM data: 0.035003

### V2

TODO: train on (MITO-EM) in 2d (to have bigger variety of shapes), use VNC for validation

### V3

TODO: train on mito-em in 3d (should work now, since fixing several bugs in training), find another 3d dataset for validation
