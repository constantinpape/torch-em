import argparse
import torch
import onnx
from train_affinities_2d import get_model


def export_model(ckpt, output):
    model = get_model()
    state = torch.load(ckpt)['model_state']
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.rand((1, 1, 256, 256))
    msg = torch.onnx.export(model, dummy_input, output,
                            verbose=True, opset_version=9)
    print(msg)

    loaded_model = onnx.load(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    export_model(args.input, args.output)
