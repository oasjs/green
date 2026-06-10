from enum import Enum
import torch
from numpy.typing import NDArray
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from line_profiler import profile as time_profile
from memory_profiler import profile as mem_profile


class Algorithm(Enum):
    BM = 1
    SGBM = 2


class ClassicalPipeline:
    def __init__(self):
        pass

    def __call__(self, left_image, right_image):
        pass
