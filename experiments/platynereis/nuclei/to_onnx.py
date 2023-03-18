import torch
from torch_em.util import get_trainer
from elf.io import open_file

model = get_trainer("./checkpoints/binary_model", device="cpu").model

# load input and expected output data
input_path = "/scratch/pape/platy/nuclei/train_data_nuclei_01.h5"
with open_file(input_path, "r") as f:
    ds = f["volumes/raw"]
    shape = ds.shape
    halo = [16, 128, 128]
    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(shape, halo))
    input_data = ds[bb]
input_tensor = torch.from_numpy(input_data[None, None].astype("float32"))

output_path = "./weights.onnx"
opset_version = 10

with torch.no_grad():
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        verbose=True,
        opset_version=opset_version,
        # example_outputs=expected_outputs,
    )
