# Shallow2Deep for Mitochondria in EM

TODO:
- fix rf training for livecell + covid-if
- explore vs training jointly on (livecell, covid-if) and (mouse-embryo, ovules, root) makes sense
- compare to plantseg results
- figure out what's going wrong in v1

## Datasets

- Ovules
- Root
- PNAS
- LiveCell
- CovidIf

## V1

trained on (mouse-embryo, ovules, root), evaluated on PNAS

| method               |   few-labels |   medium-labels |   many-labels |
|:---------------------|-------------:|----------------:|--------------:|
| rf3d                 |        0.645 |           0.630 |         0.563 |
| enhancer2d           |        0.354 |           0.341 |         0.356 |
| enhancer_anisotropic |        0.338 |           0.324 |         0.317 |
| enhancer3d           |        0.387 |           0.375 |         0.366 |


-> something is still going wrong!


## V2

trained on (ovules, root, PNAS), evaluated on mouse-embryo
