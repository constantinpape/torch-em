import os
import tempfile
import unittest

import numpy as np
import tifffile


class TestImageRead(unittest.TestCase):
    def test_read_memap(self):
        from torch_em.util.image import load_image, supports_memmap

        with tempfile.TemporaryDirectory() as td:
            tifffile.imwrite(os.path.join(td, "test.tif"), np.zeros((10, 10, 2)))
            self.assertTrue(supports_memmap(os.path.join(td, "test.tif")))
            data = load_image(os.path.join(td, "test.tif"))
            self.assertEqual(data.shape, (10, 10, 2))

    def test_read_copressed(self):
        from torch_em.util.image import load_image

        with tempfile.TemporaryDirectory() as td:
            tifffile.imwrite(os.path.join(td, "test.tif"), np.zeros((10, 10, 2)), compression="ADOBE_DEFLATE")
            data = load_image(os.path.join(td, "test.tif"))
            self.assertEqual(data.shape, (10, 10, 2))


if __name__ == "__main__":
    unittest.main()
