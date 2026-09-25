"""
.. include:: ../doc/start_page.md
.. include:: ../doc/datasets_and_dataloaders.md
"""
from .segmentation import (
    default_segmentation_dataset,
    default_segmentation_loader,
    default_segmentation_trainer,
    get_data_loader,
)
from .__version__ import __version__
