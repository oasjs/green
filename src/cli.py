"""Command Line Interface for the green program"""

from src.depth.estimator import DepthEstimator

import argparse
from .app import App


class CommandLineInterface:
    def __init__(self, app: App):
        self._parser = argparse.ArgumentParser(
            prog="green",
            description="A tool for live testing and comparing yellow and blue clone detection and depth estimation.",
        )
        self._parser.add_argument(
            "-dp",
            "--disparity_algorithm",
            choices=DepthEstimator.NN_MODELS + DepthEstimator.CLASSICAL_ALGORITHMS,
        )
        self._parser.add_argument("-c", "--cone_detector")
        self._parser.add_argument("--recorded", action="store_true")

        self._start(app)

    def _start(self, app: App):
        args = self._parser.parse_args()
        if args.disparity_algorithm is None and args.cone_detector is None:
            self._parser.error(
                "At least one of --disparity_algorithm or --cone_detector is required"
            )

        if args.recorded:
            app.run_recorded(args.disparity_algorithm)
        else:
            app.run_live(args.disparity_algorithm, 640, 400)
