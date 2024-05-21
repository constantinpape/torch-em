# Experiments for Mini-Batch Normalzation Schemes

> NOTE 1: The discussions below are for semantic segmentation.

> NOTE 2: The chosen evaluation metric is `Dice Score (DSC)` for semantic segmentation,
and `Segmentation Accuracy over IoU50 (SA50)` for instance segmentation.

## Quantitative Results:

### LIVECell:
- Binary Segmentation (DSC)
    - `OldDefault`: 0.928867041453946
    - `InstanceNorm`: 0.9139200411471561

- Boundary Segmentation (mSA | SA50)
    - `OldDefault`: 0.3596574029856674 | 0.5706313962830515
    - `InstanceNorm`: 0.21136391380220415 | 0.32975049087380925


### Mouse Embryo (Nuclei):
- Binary Segmentation (DSC)
    - `OldDefault`: 0.773449732035369
    - `InstanceNorm`: 0.7413715897174049

- Boundary Segmentation (mSA | SA50)
    - `OldDefault`: 0.2089494345163684 | 0.396692895729518
    - `InstanceNorm`: 0.15872795687689387 | 0.3200512111903994


## PlantSeg (Root):
- Binary Segmentation (DSC)
    - `OldDefault`: 0.9928809913563437
    - `InstanceNorm`: 0.981061366019915

- Boundary Segmentation (mSA | SA50)
    - `OldDefault`: 
    - `InstanceNorm`: 


## MitoEM (Human and Rat):
- Binary Segmentation (DSC)
    - `OldDefault`: 
    - `InstanceNorm`: 

- Boundary Segmentation (mSA | SA50)
    - `OldDefault`: 
    - `InstanceNorm`: 