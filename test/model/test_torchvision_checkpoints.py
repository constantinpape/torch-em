import os
import tempfile
import unittest

import torch
from torchvision.models import vit_b_16

from torch_em.model import UNETR
from torch_em.util import get_constructor_arguments
from torch_em.model.vit import ViT_Torchvision
from torch_em.model.torchvision_unet import TorchvisionUNet2d, TorchvisionUNet3d


class TestTorchvisionCheckpoints(unittest.TestCase):
    def test_unet_reconstruction(self):
        for model_class, backbone, shape in (
            (TorchvisionUNet2d, "resnet18", (1, 1, 32, 32)),
            (TorchvisionUNet3d, "r3d_18", (1, 1, 8, 32, 32)),
        ):
            with self.subTest(model=model_class.__name__):
                kwargs = dict(
                    backbone=backbone, out_channels=2, in_channels=1, depth=2,
                    initial_features=4, gain=3, pretrained=False, perform_range_checks=False,
                    final_activation="Sigmoid", postprocessing=None, check_shape=False, norm="BatchNorm",
                )
                model = model_class(**kwargs).eval()
                constructor_args = get_constructor_arguments(model)
                self.assertEqual(constructor_args, kwargs)
                restored = model_class(**constructor_args).eval()
                restored.load_state_dict(model.state_dict())
                x = torch.rand(shape)
                with torch.no_grad():
                    torch.testing.assert_close(restored(x), model(x))

    def test_encoder_checkpoints(self):
        source = vit_b_16(weights=None, image_size=384)
        state = source.state_dict()
        del source
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "encoder.pt")
            torch.save({"state_dict": state}, path)
            for checkpoint in (state, {"state_dict": state}, path):
                with self.subTest(checkpoint=type(checkpoint).__name__):
                    model = UNETR(
                        backbone="torchvision", encoder="vit_b_16", img_size=384,
                        encoder_checkpoint=checkpoint, pretrained=False, initial_features=4,
                    )
                    torch.testing.assert_close(model.encoder.encoder.pos_embedding, state["encoder.pos_embedding"])
                    self.assertEqual(model.encoder.encoder.pos_embedding.shape[1], 577)
                    del model
        self.assertIn("heads.head.weight", state)
        self.assertIn("heads.head.bias", state)

    def test_nested_encoder_reload(self):
        source = vit_b_16(weights=None, image_size=384)
        state = {k: v for k, v in source.state_dict().items() if not k.startswith("heads.")}
        del source
        encoder = ViT_Torchvision("vit_b_16", pretrained=False)
        parent = torch.nn.ModuleDict({"encoder": encoder})
        parent.load_state_dict({f"encoder.{k}": v for k, v in state.items()})
        torch.testing.assert_close(encoder.encoder.pos_embedding, state["encoder.pos_embedding"])
        with torch.no_grad():
            output, _ = encoder.eval()(torch.rand(1, 3, 224, 224))
        self.assertEqual(output.shape, (1, 768, 14, 14))


if __name__ == "__main__":
    unittest.main()
