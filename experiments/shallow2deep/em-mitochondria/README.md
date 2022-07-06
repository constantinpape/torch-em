# Shallow2Deep for mitochondria

## Evaluation

Evaluation of different shallow2deep setups on EM-Mitochondria. All scores are measured with a soft dice score.

### V1

Enhancers and direct model trained on VNC and applied to crops of MITO-EM; everything fully 2d.

|                     |few-labels | many-labels|
|---------------------|----------:|-----------:|
|vanilla-enhancer     |0.263941   |   0.436766 |
|advanced-enhancer    |0.319515   |   0.453895 |
|rf-score             |0.163359   |   0.165454 |
|direct-net           |0.178033   |   0.325530 |
Raw net evaluation: 0.439310

### V2

Enhancers and direct model trained on Mito-EM and applied to VNC; everything fully in 2d.

| enhancer          |   few-labels |   medium-labels |   many-labels |
|:------------------|-------------:|----------------:|--------------:|
| advanced-enhancer |    0.209421  |       0.334881  |     0.44396   |
| rf-score          |    0.0611923 |       0.0896972 |     0.0996913 |
| vanilla-enhancer  |    0.136944  |       0.329187  |     0.330667  |
| direct-net        |    0.101433  |       0.0624726 |     0.0431661 |
Raw net evaluation: 0.5664974438773702

### V3

TODO: train on mito-em in 3d (should work now, since fixing several bugs in training), find another 3d dataset for validation
