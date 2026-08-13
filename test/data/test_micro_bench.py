import unittest

from torch_em.data.datasets.light_microscopy.micro_bench import (
    _get_sample_id, _get_source, _rasterize, _validate_source,
)


class TestMicroBench(unittest.TestCase):
    def test_sources(self):
        polygon = [{"className": "H2B-mCherry", "points": [1, 1, 2, 1, 2, 2]}]
        self.assertEqual(_get_source({"dataset": None, "polygon": polygon}), "cellcognition")
        self.assertEqual(_get_source({"dataset": "opencell", "polygon": polygon}), "opencell")
        self.assertEqual(_get_source({"dataset": "sirinukunwattana_et_al_2016", "polygon": polygon}), None)

    def test_sample_id_includes_source_dataset(self):
        row = {"dataset": "burgess_et_al_2024_contour", "image_id": "reused-id"}
        contour_id = _get_sample_id(row, "burgess")
        row["dataset"] = "burgess_et_al_2024_texture"
        texture_id = _get_sample_id(row, "burgess")
        self.assertNotEqual(contour_id, texture_id)

    def test_rasterize_burgess_targets(self):
        polygons = [
            {"className": "cell", "points": [0, 0, 6, 0, 6, 6, 0, 6]},
            {"className": "nucleus", "points": [2, 2, 4, 2, 4, 4, 2, 4]},
        ]
        cells = _rasterize(polygons, (7, 7), "cell")
        nuclei = _rasterize(polygons, (7, 7), "nucleus")
        self.assertGreater(cells.sum(), nuclei.sum())
        self.assertEqual(cells.max(), 1)
        self.assertEqual(nuclei.max(), 1)

    def test_validate_source(self):
        self.assertEqual(_validate_source("burgess", None), "cell")
        self.assertEqual(_validate_source("burgess", "nucleus"), "nucleus")
        self.assertEqual(_validate_source("cellcognition", None), "instances")
        with self.assertRaises(ValueError):
            _validate_source("cellcognition", "nucleus")


if __name__ == "__main__":
    unittest.main()
