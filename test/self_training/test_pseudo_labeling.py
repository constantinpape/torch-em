import unittest

import torch

import torch_em.self_training as self_training


class TestPseudoLabeling(unittest.TestCase):
    def _check_mask_channel(self, pseudo_labeler_class):
        # Channel 0 is confident everywhere and channel 1 nowhere, so the masks of the two channels differ.
        input_ = torch.stack([torch.full((4, 32, 32), 0.99), torch.full((4, 32, 32), 0.6)], dim=1)
        teacher = torch.nn.Identity()

        for mask_channel, expected in ((0, 1.0), (1, 0.0), (-1, 0.0)):
            pseudo_labeler = pseudo_labeler_class(confidence_threshold=0.9, mask_channel=mask_channel)
            pseudo_labels, label_mask = pseudo_labeler(teacher, input_)
            self.assertEqual(label_mask.shape, pseudo_labels.shape)
            self.assertTrue(torch.all(label_mask == expected))

    def test_default_pseudo_labeler_mask_channel(self):
        self._check_mask_channel(self_training.DefaultPseudoLabeler)

    def test_scheduled_pseudo_labeler_mask_channel(self):
        self._check_mask_channel(self_training.ScheduledPseudoLabeler)


if __name__ == "__main__":
    unittest.main()
