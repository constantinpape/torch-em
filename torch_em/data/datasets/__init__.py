"""Read-to-use Datasets and DataLoaders for many different bio-medical datasets.

The datasets are separated into four categories:
- `torch_em.data.datasets.light_microscopy` contains light microscopy datasets.
- `torch_em.data.datasets.electron_microscopy` contains electron microscopy datasets.
- `torch_em.data.datasets.histopathology` contains histopathology datasets.
- `torch_em.data.datasets.medical` contains medical imaging datasets.
"""
from .electron_microscopy import *
from .histopathology import *
from .light_microscopy import *
from .medical import *
from .util import get_bioimageio_dataset_id
