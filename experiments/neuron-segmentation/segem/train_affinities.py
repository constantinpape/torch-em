import argparse
import os
import numpy as np

import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask


def train_affinities():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_affinities()
