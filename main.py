import argparse
from stereo_matching import StereoOutput
from src.camera import StereoCameraLive
from src.depth.neural import NeuralPipeline
import depthai as dai
import cv2
from typing import cast
import numpy as np


class TerminalUserInterface:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            prog="green",
            description="A tool for live testing and comparing yellow and blue clone detection and depth estimation.",
        )

        self._parser.add_argument("-i", "--interactive", action="store_true")

        self.running = True

    def start(self):
        args = self._parser.parse_args()
        if args.interactive:
            self.welcome()
            return True
        else:
            self._parser.print_help()
            return False

    def welcome(self):
        print("Welcome to green!\n")

    def ask_depth_estimation_approach(self):
        print("Choose a depth estimation approach:\n")
        print("  [1] Classical approach")
        print("  [2] Neural networks approach\n")
        choice = self._prompt("Enter choice: ", valid={"1", "2"})

        if choice == "1":
            return self.classical_options()
        elif choice == "2":
            return self.neural_network_options()

    def classical_options(self):
        pass

    def neural_network_options(self):
        print("\nNeural Network Options\n")
        print("  [1] Raft-Stereo")
        print("  [2] CREStereo")
        # print("  [3] AANet")
        # print("  [4] FoundationStereo")
        # print("  [5] IGEV-Stereo")
        # print("  [6] IGEV++")
        # print("  [7] S2M2")
        # print("  [8] UniMatch")
        print("  [9] DepthAi\n")
        choice = self._prompt(
            "Enter choice: ", valid={"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        )

        match choice:
            case "1":
                return self.raft_stereo()
            case "2":
                return "crestereo"
            case "3":
                return self.aa_net()
            case "4":
                return self.foundation_stereo()
            case "5":
                return self.igev_stereo()
            case "6":
                return self.igev_plusplus()
            case "7":
                return self.s2m2()
            case "8":
                return self.unimatch()
            case "9":
                return self.depthai()
            case _:
                return self.raft_stereo()

    def raft_stereo(self):
        print("\nRaft Stereo Options\n")
        print("  [1] Raft-Stereo")
        print("  [2] Raft-Stereo-Middlebury")
        print("  [3] Raft-Stereo-Eth3d")
        print("  [4] Raft-Stereo-Realtime\n")
        choice = self._prompt("Enter choice: ", valid={"1", "2", "3", "4"})

        match choice:
            case "1":
                return "raft-stereo"
            case "2":
                return "raft-stereo-realtime"
            case "3":
                return "raft-stereo-eth3d"
            case "4":
                return "raft-stereo-realtime"

    def depthai(self):
        print("\nDepthAi Options\n")
        print("  [1] Embedded\n")
        choice = self._prompt("Enter choice: ", valid={"1"})

        match choice:
            case "1":
                return "depthai-embedded"

    def aa_net(self):
        print("\nAAnet Options\n")
        print("  [1] AAnet")
        print("  [2] AAnet-KITTI2012")
        print("  [3] AAnet-Sceneflow\n")
        choice = self._prompt("Enter choice: ", valid={"1", "2", "3"})

        match choice:
            case "1":
                return "aanet"
            case "2":
                return "aanet-kitti2012"
            case "3":
                return "annet-sceneflow"

    def foundation_stereo(self):
        print("\nFoundationStereo Options\n")
        print("  [1] FoundationStereo")
        print("  [2] FoundationStereo Large")
        choice = self._prompt("Enter choice: ", valid={"1", "2", "3"})

        match choice:
            case "1":
                return "foundation-stereo"
            case "2":
                return "foundation-stereo-large"

    def igev_stereo(self):
        raise NotImplementedError()

    def igev_plusplus(self):
        raise NotImplementedError()

    def s2m2(self):
        raise NotImplementedError()

    def unimatch(self):
        raise NotImplementedError()

    def _prompt(self, message: str, valid: set) -> str:
        while True:
            choice = input(message).strip()
            if choice in valid:
                return choice
            print(f"Invalid choice. Options: {', '.join(sorted(valid))}")


class DepthMapVisualizer:
    def __init__(self):
        self._color_map = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
        )
        self._color_map[0] = [0, 0, 0]
        self._max_disparity = 1

    def show(self, disparity):
        max_disparity = max(self._max_disparity, np.max(disparity))
        cv2.imshow(
            "disparity",
            cv2.applyColorMap(
                ((disparity / max_disparity) * 255).astype(np.uint8),
                self._color_map,
            ),
        )

    @property
    def color_map(self):
        return self._color_map


def main():
    print("here")
    tui = TerminalUserInterface()
    if not tui.start():
        exit(0)

    while tui.running:
        depth_estimator_approach = tui.ask_depth_estimation_approach()
        if depth_estimator_approach is None:
            exit(0)

        depthai_pipeline = dai.Pipeline()
        camera = StereoCameraLive(depthai_pipeline, 640, 400, 30)
        camera.stereo.setRectification(True)
        # camera.stereo.setExtendedDisparity(True)
        camera.stereo.setLeftRightCheck(True)

        depth_estimator = (
            camera.stereo.disparity.createOutputQueue()
            if depth_estimator_approach == "depthai-embedded"
            else NeuralPipeline(model=depth_estimator_approach)
        )

        deapth_visualizer = DepthMapVisualizer()

        with depthai_pipeline:
            depthai_pipeline.start()

            while depthai_pipeline.isRunning():
                if depth_estimator_approach == "depthai-embedded":
                    assert isinstance(depth_estimator, dai.MessageQueue)
                    dai_disparity = cast(dai.ImgFrame, depth_estimator.get())
                    disparity = dai_disparity.getFrame()
                else:
                    left_image, right_image = camera.next_rectified_pair()
                    left_image, right_image = (
                        left_image.getCvFrame(),
                        right_image.getCvFrame(),
                    )
                    if left_image.ndim == 2 and right_image.ndim == 2:
                        left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
                        right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

                    assert isinstance(depth_estimator, NeuralPipeline)
                    result = depth_estimator(left_image, right_image)
                    result = cast(StereoOutput, result)
                    disparity = result.disparity

                deapth_visualizer.show(disparity)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    depthai_pipeline.stop()
                    break


if __name__ == "__main__":
    main()
