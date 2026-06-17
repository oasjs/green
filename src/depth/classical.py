from enum import Enum


class Algorithm(Enum):
    BM = 1
    SGBM = 2


class ClassicalPipeline:
    def __init__(self):
        pass

    def __call__(self, left_image, right_image):
        pass
