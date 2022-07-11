# Shallow2Deep

Enhancer networks trained to improve Random Forest predictions. Can be used interactively in the ilastik `Pixel Classification Enhancer` workflow (still in beta!).
For more details on the shallow2deep method, check out [From Shallo to Deep: Exploiting Feature-Based Classifiers for Domain Adaptation in Semantic Segmentation](https://doi.org/10.3389/fcomp.2022.805166). Please also cite this paper if you are using any of these networks in your research.

## Enhancers

The subfolders contain training scripts for different enhancer setups:
- `em-boundaries`: enhancer trained for segmentation of boundaries/membranes in 3D EM data. The enhancer is available on [bioimage.io](https://bioimage.io/#/?tags=shallow2deep&id=10.5281%2Fzenodo.6808325&type=model). (It's not trained with the script here, but using more training dataset, by @JonasHell)
- `em-mitochondria`: enhancers trained for segmentation of boundaries in 2D/3D  EM data. The enahncers trained on the MitoEM dataset are available on bioimagei.io ([2D](https://bioimage.io/#/?tags=shallow2deep&id=10.5281%2Fzenodo.6406756), [3D](https://bioimage.io/#/?tags=shallow2deep&id=10.5281%2Fzenodo.6811491 )).
- `lm-membrane`: enhancers trained for segmentation of cell membranes in LM data. WIP
- `lm-nuclei`: enhancers trained for segmentation of nuclei in LM data. WIP
