import torch
from numpy.typing import NDArray
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from line_profiler import profile as time_profile
from memory_profiler import profile as mem_profile


from camera import StereoCamera


class DepthEstimator:
    def __init__(self):
        pass

    def run(
        self, left_image: NDArray, right_image: NDArray, camera_info: StereoCamera.Info
    ):
        raise NotImplementedError(
            "DepthEstimator is an abstract class and must not be instantiated."
        )


class BM(DepthEstimator):
    def __init__(self):
        super().__init__()


class SGBM(DepthEstimator):
    def __init__(self):
        super().__init__()

@time_profile
@mem_profile
class StereoRaftBenchmark(DepthEstimator):
    def __init__(self, model, weights, device="cpu",repeats = 10):
        super().__init__()

        self._model =model 
        self._transforms = weights.transforms 


        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = Raft_Large_Weights.DEFAULT
            self._transforms = weights.transforms()
        
            self._model = raft_large(weights=weights).to(device).eval()
        except Exception as e:
            print(f"RAFT unavailable: {e}")

    @time_profile
    @mem_profile
    def _run(self):
        pass

