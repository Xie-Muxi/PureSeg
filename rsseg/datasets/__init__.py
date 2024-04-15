# from .datasets import WiderFaceDataset
# from .transformers import *
from .aerorit import AeroRITDataSet
from .nwpu import NWPUInsSegDataset
from .isaid import iSAIDDataset
from .potsdam import PotsdamDataset

__all__ = ["AeroRITDataSet", "NWPUInsSegDataset", "iSAIDDataset", "PotsdamDataset"]
