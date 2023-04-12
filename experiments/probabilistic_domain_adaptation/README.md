# Probabilistic Domain Adaption

Implemention of [Probabilistic Domain Adaptation for Biomedical Image Segmentation](https://arxiv.org/abs/2303.11790) in `torch_em`.
Please cite the paper if you are using these approaches in your research.

## Self-Training Approaches

The subfolders contain the training scripts for both separate and joint training setups:

- `unet_source.py` (UNet Source Training): 
```
python unet_source.py -p [check / train / evaluate]
                      -c <CELL-TYPE>
                      -i <PATH-TO-DATA>
                      -s <PATH-TO-SAVE-MODEL-WEIGHTS>
                      -o <PATH-FOR-SAVING-PREDICTIONS>
```

- `unet_mean_teacher.py` (UNet Mean-Teacher Separate Training):
```
python unet_mean_teacher.py -p [check / train / evaluate]
                            -c <CELL-TYPE>
                            -i <PATH-TO-DATA>
                            -s <PATH-TO-SAVE-MODEL-WEIGHTS>
                            -o <PATH-FOR-SAVING-PREDICTIONS>
                            [(optional) --target_ct <TARGET-DOMAIN-CELL-TYPE>]
                            [(optional) --confidence_threshold <THRESHOLD-FOR-COMPUTING-FILTER-MASK>]
```

- `unet_adamt.py` (UNet Mean-Teacher Joint Training):
```
python unet_adamt.py -p [check / train / evaluate]
                     -c <CELL-TYPE>
                     -i <PATH-TO-DATA>
                     -s <PATH-TO-SAVE-MODEL-WEIGHTS>
                     -o <PATH-FOR-SAVING-PREDICTIONS>
                     [(optional) --target_ct <TARGET-DOMAIN-CELL-TYPE>]
                     [(optional) --confidence_threshold <THRESHOLD-FOR-COMPUTING-FILTER-MASK>]
```

- `unet_fixmatch.py` (UNet FixMatch Separate Training):
```
python unet_fixmatch.py -p [check / train / evaluate]
                        -c <CELL-TYPE>
                        -i <PATH-TO-DATA>
                        -s <PATH-TO-SAVE-MODEL-WEIGHTS>
                        -o <PATH-FOR-SAVING-PREDICTIONS>
                        [(optional) --target_ct <TARGET-DOMAIN-CELL-TYPE>]
                        [(optional) --confidence_threshold <THRESHOLD-FOR-COMPUTING-FILTER-MASK>]
                        [(optional) --distribution_alignment <ACTIVATES-DISTRIBUTION-ALIGNMENT>]
```

- `unet_adamatch.py` (UNet FixMatch Joint Training):
```
python unet_adamatch.py -p [check / train / evaluate]
                        -c <CELL-TYPE>
                        -i <PATH-TO-DATA>
                        -s <PATH-TO-SAVE-MODEL-WEIGHTS>
                        -o <PATH-FOR-SAVING-PREDICTIONS>
                        [(optional) --target_ct <TARGET-DOMAIN-CELL-TYPE>]
                        [(optional) --confidence_threshold <THRESHOLD-FOR-COMPUTING-FILTER-MASK>]
                        [(optional) --distribution_alignment <ACTIVATES-DISTRIBUTION-ALIGNMENT>]
```
