"""Command Line Interface for the green program"""

from src.depth.estimator import MODELS

import argparse
from .app import App


class CommandLineInterface:
    def __init__(self, app: App):
        self._parser = argparse.ArgumentParser(
            prog="green",
            description="A tool for live testing and comparing yellow and blue clone detection and depth estimation.",
        )
        self._parser.add_argument(
            "--disparity_algorithm",
            choices=MODELS,
        )
        self._parser.add_argument("--cone_detector")
        self._parser.add_argument("--recorded", action="store_true")

        self._start(app)

    def _start(self, app: App):
        args = self._parser.parse_args()
        if args.recorded:
            app.run_recorded()
        else:
            if args.disparity_algorithm is None and args.cone_detector is None:
                self._parser.error(
                    "At least one of --disparity_algorithm or --cone_detector is required"
                )

            app.run_live(args.disparity_algorithm)
