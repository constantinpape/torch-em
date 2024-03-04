# UNETR on LiveCELL for Instance Segmentation

The combination of target objectives for performing instance segmentation:
- Foreground - Boundary Segmentation (`--experiment_name boundaries`)
- Affinities (`--experiment_name affinities`)
- Distance Maps (`--experiment_name distances`)

```bash
python train_livecell [--train / --predict / --evaluate]
                      --experiment_name <NAME>  # see above for details
                      -i <PATH_TO_LIVECELL_DATA>
                      -m <VIT_SIZE>  # supported for vit_b / vit_l / vit_h
                      -s <SAVE_ROOT>
                      --save_dir <PREDICTION_DIR>
                      [(OPTIONAL) --do_sam_ini]  # use Segment Anything's pretrained weights of image encoder
```
