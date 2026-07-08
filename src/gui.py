"""Stereo depth estimation demo: upload OR live camera, BM / SGBM / RAFT-Stereo."""

import csv
import os
import time

import cv2
import numpy as np
import psutil
import gradio as gr
import depthai as dai

import metrics
from estimators import (
    colorize,
    BMEstimator,
    SGBMEstimator,
    RaftStereoEstimator,
    DepthAiSgbmEstimator,
    MonoDepthEstimator,
    RAFT_AVAILABLE,
    DEPTHAI_AVAILABLE,
    MONO_AVAILABLE,
)
from camera_source import LiveStereoSource


class DepthEstimationApp:
    """Builds and serves the Gradio UI, one tab per estimator."""

    MONO_ESTIMATOR_NAME = "Depth Anything V2 (mono)"

    def __init__(
        self,
        camera_source: LiveStereoSource | None = None,
        pipeline: dai.Pipeline | None = None,
    ):
        self._camera_source = camera_source
        self._pipeline = pipeline
        self._estimators = {
            "StereoBM": BMEstimator(),
            "StereoSGBM": SGBMEstimator(),
            "RAFT-Stereo": RaftStereoEstimator(),
            self.MONO_ESTIMATOR_NAME: MonoDepthEstimator(),
        }
        if camera_source is not None and DEPTHAI_AVAILABLE:
            # Must be constructed before pipeline.start(): DepthAI v3 requires
            # initialConfig to be set pre-start.
            self._estimators["DepthAI-SGBM"] = DepthAiSgbmEstimator(camera_source.camera)
        self._process = psutil.Process(os.getpid())
        self._history: dict[str, list[dict]] = {name: [] for name in self._estimators}

        # Scale-alignment state for the monocular estimator: relative depth models
        # only recover depth up to an unknown affine transform (depth = a*raw + b).
        # We solve for a, b using a stereo estimator's disparity as the reference.
        self._last_stereo_raw: np.ndarray | None = None
        self._last_stereo_name: str | None = None
        self._last_mono_raw: np.ndarray | None = None
        self._mono_scale: float = 1.0
        self._mono_shift: float = 0.0
        self._mono_calibration_source: str = "none (showing raw relative depth)"

    # ---- helpers ----
    def _capture_live(self):
        if self._camera_source is None:
            raise gr.Error("No live camera configured.")
        gray_left, gray_right = self._camera_source.get_pair()
        center = self._camera_source.get_center_frame()
        return gray_left, gray_right, gray_left, gray_right, center

    @staticmethod
    def _to_gray(img):
        """Convert whatever the user uploaded (or captured) to grayscale so the
        estimators always run on gray, even if the displayed image is in color."""
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            gray = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        return np.stack([gray] * 3, axis=-1)

    def _build_param_widgets(self, estimator, extra: dict | None = None):
        widgets = {}
        for key, (lo, hi, default, step) in estimator.param_spec().items():
            widgets[key] = gr.Slider(lo, hi, value=default, step=step, label=key)
        if extra:
            widgets.update(extra)
        return widgets

    def _save_frame_pair(self, left, right):
        if left is None or right is None:
            raise gr.Error("No frames to save yet.")
        out_dir = "/tmp/stereo_captures"
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        left_path = os.path.join(out_dir, f"left_{ts}.png")
        right_path = os.path.join(out_dir, f"right_{ts}.png")
        cv2.imwrite(left_path, cv2.cvtColor(np.asarray(left), cv2.COLOR_RGB2BGR))
        cv2.imwrite(right_path, cv2.cvtColor(np.asarray(right), cv2.COLOR_RGB2BGR))
        return [left_path, right_path]

    def _run(self, estimator_name, left, right, param_values, param_keys):
        estimator = self._estimators[estimator_name]
        if estimator.requires_images and left is None:
            raise gr.Error("Provide a left image (upload or capture live).")
        if estimator.requires_images and estimator.requires_right and right is None:
            raise gr.Error(
                "Provide both left and right images (upload or capture live)."
            )
        params = dict(zip(param_keys, param_values))

        start = time.perf_counter()
        colored, raw = estimator.compute(np.asarray(left), np.asarray(right), params)
        elapsed = time.perf_counter() - start

        calibration_text = None
        if estimator_name == self.MONO_ESTIMATOR_NAME:
            self._last_mono_raw = raw
            if self._mono_scale != 1.0 or self._mono_shift != 0.0:
                raw = raw * self._mono_scale + self._mono_shift
                colored = colorize(raw)
            calibration_text = (
                f"Scale reference: {self._mono_calibration_source}\n"
                f"depth = {self._mono_scale:.4g} * raw + {self._mono_shift:.4g}   "
                f"(mean: {float(np.mean(raw)):.3g}, median: {float(np.median(raw)):.3g})"
            )
        elif estimator.requires_right:
            # Any stereo estimator's output can be used as the mono calibration reference.
            self._last_stereo_raw = raw
            self._last_stereo_name = estimator_name

        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        mem_mb = self._process.memory_info().rss / (1024 * 1024)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fps": fps,
            "elapsed_ms": elapsed * 1000,
            "mem_mb": mem_mb,
        }

        if left is not None and right is not None:
            quality = metrics.compute(np.asarray(left), np.asarray(right), raw)
            quality_text = quality.as_text()
            record.update(
                reprojection_l1=quality.reprojection_l1,
                reprojection_ssim=quality.reprojection_ssim,
                valid_ratio=quality.valid_ratio,
                edge_alignment=quality.edge_alignment,
            )
        else:
            quality_text = "N/A (no reference images to compare against)"

        self._history[estimator_name].append(record)

        return (
            colored,
            f"{fps:.2f} FPS ({elapsed * 1000:.0f} ms)",
            f"{mem_mb:.0f} MB",
            quality_text,
            self._average_text(estimator_name),
            calibration_text or "",
        )

    def _calibrate_mono(self):
        if self._last_mono_raw is None:
            raise gr.Error("Run the Depth Anything V2 tab at least once first.")
        if self._last_stereo_raw is None:
            raise gr.Error("Run a stereo tab (BM/SGBM/RAFT/DepthAI) at least once first.")

        mono_raw = self._last_mono_raw
        stereo_raw = self._last_stereo_raw
        if stereo_raw.shape != mono_raw.shape:
            stereo_raw = cv2.resize(
                stereo_raw.astype(np.float32), (mono_raw.shape[1], mono_raw.shape[0])
            )

        valid = (stereo_raw > 0) & np.isfinite(stereo_raw) & np.isfinite(mono_raw)
        if valid.sum() < 100:
            raise gr.Error("Not enough valid overlapping pixels to calibrate against.")

        x = mono_raw[valid].astype(np.float64)
        y = stereo_raw[valid].astype(np.float64)
        # Least-squares fit of y = a*x + b (i.e. scale + shift to align mono's
        # relative depth with the stereo estimator's disparity-like units).
        a, b = np.polyfit(x, y, 1)

        self._mono_scale = float(a)
        self._mono_shift = float(b)
        self._mono_calibration_source = f"{self._last_stereo_name} ({valid.sum()} px)"

        return (
            f"Calibrated: scale={a:.4g}, shift={b:.4g} "
            f"against {self._last_stereo_name} using {valid.sum()} pixels."
        )

    def _average_text(self, estimator_name: str) -> str:
        records = self._history[estimator_name]
        if not records:
            return "No runs yet."

        n = len(records)

        def avg(key):
            vals = [r[key] for r in records if key in r and not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        lines = [f"Average over {n} run(s):"]
        lines.append(f"  FPS: {avg('fps'):.2f}   Compute: {avg('elapsed_ms'):.0f} ms   Memory: {avg('mem_mb'):.0f} MB")
        if any("reprojection_l1" in r for r in records):
            lines.append(
                f"  Reprojection L1: {avg('reprojection_l1'):.2f}   SSIM: {avg('reprojection_ssim'):.3f}"
            )
            lines.append(
                f"  Valid pixels: {avg('valid_ratio') * 100:.1f}%   Edge alignment: {avg('edge_alignment'):.3f}"
            )
        return "\n".join(lines)

    def _export_metrics(self, estimator_name: str):
        records = self._history[estimator_name]
        if not records:
            raise gr.Error("No runs recorded yet for this estimator.")

        out_dir = "/tmp/stereo_metrics"
        os.makedirs(out_dir, exist_ok=True)
        safe_name = estimator_name.replace(" ", "_").replace("/", "_")
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{safe_name}_metrics_{ts}.csv")

        fieldnames = [
            "timestamp", "fps", "elapsed_ms", "mem_mb",
            "reprojection_l1", "reprojection_ssim", "valid_ratio", "edge_alignment",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        return path

    def _reset_metrics(self, estimator_name: str):
        self._history[estimator_name] = []
        return "No runs yet."

    def _build_tab(
        self,
        estimator_name: str,
        left_img,
        right_img,
        left_gray,
        right_gray,
        center_img,
        extra_widgets_fn=None,
        live_widgets=None,
    ):
        estimator = self._estimators[estimator_name]
        is_mono = estimator_name == self.MONO_ESTIMATOR_NAME

        with gr.Tab(estimator.name) as tab:
            with gr.Row():
                with gr.Column():
                    extra_widgets = extra_widgets_fn() if extra_widgets_fn else None
                    widgets = self._build_param_widgets(estimator, extra_widgets)
                    run_btn = gr.Button(f"Run {estimator.name}", variant="primary")

                with gr.Column():
                    out_img = gr.Image(label="Disparity map", height=400)

                    fps_box = gr.Textbox(label="Compute speed", interactive=False)
                    mem_box = gr.Textbox(label="Memory usage", interactive=False)
                    quality_box = gr.Textbox(
                        label="Quality metrics (no ground truth)",
                        interactive=False,
                        lines=2,
                    )
                    avg_box = gr.Textbox(
                        label="Running average",
                        interactive=False,
                        lines=3,
                        value="No runs yet.",
                    )

                    calibration_box = None
                    if is_mono:
                        calibration_box = gr.Textbox(
                            label="Scale calibration (metric depth requires a stereo reference)",
                            interactive=False,
                            lines=2,
                            value=(
                                "Not calibrated — showing raw relative depth. Run a stereo "
                                "tab, then click Calibrate below."
                            ),
                        )
                        calibrate_btn = gr.Button("Calibrate against last stereo run")
                        calibrate_btn.click(self._calibrate_mono, outputs=calibration_box)

                    with gr.Row():
                        export_btn = gr.Button("Export metrics to CSV")
                        reset_btn = gr.Button("Reset history")
                    metrics_file = gr.File(label="Metrics CSV", visible=True, height=100)

                    export_btn.click(
                        lambda _name=estimator_name: self._export_metrics(_name),
                        outputs=metrics_file,
                    )
                    reset_btn.click(
                        lambda _name=estimator_name: self._reset_metrics(_name),
                        outputs=avg_box,
                    )

                    param_keys = list(widgets.keys())
                    run_outputs = [out_img, fps_box, mem_box, quality_box, avg_box]
                    if is_mono:
                        run_outputs.append(calibration_box)

                    if self._camera_source is not None:
                        live_toggle, interval = live_widgets

                        timer = gr.Timer(1.0, active=False)

                        def _live_tick(*vals, _name=estimator_name, _keys=param_keys, _mono=is_mono):
                            gray_left, gray_right, _, _, center = self._capture_live()
                            result = self._run(_name, gray_left, gray_right, vals, _keys)
                            colored, fps_text, mem_text, quality_text, avg_text, calib_text = result
                            out = [
                                gray_left, gray_right, gray_left, gray_right, center,
                                colored, fps_text, mem_text, quality_text, avg_text,
                            ]
                            if _mono:
                                out.append(calib_text)
                            return out

                        timer.tick(
                            _live_tick,
                            inputs=list(widgets.values()),
                            outputs=[
                                left_img, right_img, left_gray, right_gray, center_img,
                                *run_outputs,
                            ],
                            concurrency_limit=1,
                        )

                        live_toggle.change(
                            lambda active: gr.Timer(active=active),
                            inputs=live_toggle,
                            outputs=timer,
                        )
                        interval.change(
                            lambda secs: gr.Timer(value=secs),
                            inputs=interval,
                            outputs=timer,
                        )

            def _on_run(l, r, *vals, _name=estimator_name, _keys=param_keys, _mono=is_mono):
                result = self._run(_name, l, r, vals, _keys)
                return result if _mono else result[:5]

            run_btn.click(
                _on_run,
                inputs=[left_gray, right_gray, *widgets.values()],
                outputs=run_outputs,
            )

        return tab

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Stereo Depth Estimation Demo") as demo:
            gr.Markdown(
                "# Stereo Depth Estimation — BM / SGBM / RAFT-Stereo\n"
                "Upload a rectified stereo pair, or capture one from a live camera, "
                "then tune each algorithm's parameters."
            )

            # Shared across all tabs: one capture, all estimators run on the same pair.
            with gr.Row():
                left_img = gr.Image(label="Left image", type="numpy", height=400)
                right_img = gr.Image(label="Right image", type="numpy", height=400)
                center_img = gr.Image(
                    label="Center camera (RGB)", type="numpy", height=400, interactive=False
                )

            # Hidden grayscale copies: the estimators always run on these, regardless
            # of whether the displayed image above is a color live-feed preview or a
            # color file the user uploaded.
            left_gray = gr.State()
            right_gray = gr.State()
            left_img.upload(self._to_gray, inputs=left_img, outputs=left_gray)
            right_img.upload(self._to_gray, inputs=right_img, outputs=right_gray)

            estimator_names = list(self._estimators.keys())
            live_widgets_by_name = {}
            if self._camera_source is not None:
                with gr.Row():
                    live_btn = gr.Button("Capture from camera")
                    save_btn = gr.Button("Save current frame pair")
                    for i, name in enumerate(estimator_names):
                        toggle = gr.Checkbox(label="Live feed", value=False, visible=(i == 0))
                        interval = gr.Slider(
                            0.1, 5.0, value=1.0, step=0.1,
                            label="Update interval (s)", visible=(i == 0),
                        )
                        live_widgets_by_name[name] = (toggle, interval)

                saved_files = gr.File(label="Download frames", file_count="multiple", height=100)
                live_btn.click(
                    self._capture_live,
                    outputs=[left_img, right_img, left_gray, right_gray, center_img],
                )
                save_btn.click(
                    self._save_frame_pair,
                    inputs=[left_img, right_img],
                    outputs=saved_files,
                )

            def _live_widgets(name):
                return live_widgets_by_name.get(name)

            with gr.Tabs():
                tabs = []
                tabs.append(self._build_tab(
                    "StereoBM", left_img, right_img, left_gray, right_gray, center_img,
                    live_widgets=_live_widgets("StereoBM"),
                ))
                tabs.append(self._build_tab(
                    "StereoSGBM",
                    left_img,
                    right_img,
                    left_gray,
                    right_gray,
                    center_img,
                    extra_widgets_fn=lambda: {
                        "mode": gr.Dropdown(
                            ["SGBM", "HH", "SGBM_3WAY"], value="SGBM_3WAY", label="mode"
                        )
                    },
                    live_widgets=_live_widgets("StereoSGBM"),
                ))

                def _raft_extra_widgets():
                    if not RAFT_AVAILABLE:
                        gr.Markdown(
                            "⚠️ RAFT-Stereo tab needs the `stereo_matching` package "
                            "(`pip install stereo_matching`)."
                        )
                    return {
                        "model_id": gr.Dropdown(
                            RaftStereoEstimator.MODEL_VARIANTS,
                            value=RaftStereoEstimator.MODEL_VARIANTS[0],
                            label="model_id",
                        ),
                    }

                tabs.append(self._build_tab(
                    "RAFT-Stereo", left_img, right_img, left_gray, right_gray, center_img,
                    extra_widgets_fn=_raft_extra_widgets,
                    live_widgets=_live_widgets("RAFT-Stereo"),
                ))

                def _mono_extra_widgets():
                    if not MONO_AVAILABLE:
                        gr.Markdown(
                            "⚠️ Depth Anything V2 needs `transformers` + `torch` "
                            "(`pip install transformers torch`)."
                        )
                    return {
                        "model_id": gr.Dropdown(
                            MonoDepthEstimator.MODEL_VARIANTS,
                            value=MonoDepthEstimator.MODEL_VARIANTS[0],
                            label="model_id",
                        ),
                    }

                tabs.append(self._build_tab(
                    "Depth Anything V2 (mono)",
                    left_img,
                    right_img,
                    left_gray,
                    right_gray,
                    center_img,
                    extra_widgets_fn=_mono_extra_widgets,
                    live_widgets=_live_widgets("Depth Anything V2 (mono)"),
                ))

                if "DepthAI-SGBM" in self._estimators:
                    tabs.append(self._build_tab(
                        "DepthAI-SGBM", left_img, right_img, left_gray, right_gray, center_img,
                        live_widgets=_live_widgets("DepthAI-SGBM"),
                    ))

            # Show only the active tab's live-feed controls up top.
            if self._camera_source is not None:
                all_live_components = [w for pair in live_widgets_by_name.values() for w in pair]
                for name, tab in zip(estimator_names, tabs):
                    toggle, interval = live_widgets_by_name[name]

                    def _select_visibility(_toggle=toggle, _interval=interval):
                        updates = []
                        for t, iv in live_widgets_by_name.values():
                            is_active = t is _toggle
                            updates.append(gr.Checkbox(visible=is_active))
                            updates.append(gr.Slider(visible=is_active))
                        return updates

                    tab.select(_select_visibility, outputs=all_live_components)
        return demo

    def launch(self, **kwargs):
        self.build().launch(**kwargs)


def _build_live_source() -> tuple[LiveStereoSource | None, dai.Pipeline | None]:
    """Optional: construct a LiveStereoSource if depthai is available/configured.

    Does NOT start the pipeline: estimators that configure the StereoDepth node
    (e.g. DepthAiSgbmEstimator) must do so before pipeline.start().
    """
    try:
        import depthai as dai
    except Exception:
        return None, None
    pipeline = dai.Pipeline()
    source = LiveStereoSource.from_config(
        pipeline=pipeline, image_width=640, image_height=400, fps=30
    )
    return source, pipeline


if __name__ == "__main__":
    camera, pipeline = (
        _build_live_source()
    )  # None if depthai/camera unavailable -> upload-only mode
    app = DepthEstimationApp(camera_source=camera, pipeline=pipeline)
    if pipeline is not None:
        pipeline.start()
    app.launch()
