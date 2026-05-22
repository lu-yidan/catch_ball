#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense D435/D455 continuous tennis-ball tracker (HSV + AprilTag + centre frame).

Keeps the camera pipeline running in a background thread so grasp planning /
execution can read fresh ball positions for dynamic section-3 adjustment.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from detect_tennis_and_blue_disk_hsv import (
    BLUE_DIAMETER_M,
    BLUE_H_HIGH,
    BLUE_H_LOW,
    BLUE_S_MIN,
    BLUE_V_MIN,
    DEPTH_MAX,
    DEPTH_MIN,
    TENNIS_H_HIGH,
    TENNIS_H_LOW,
    TENNIS_RADIUS_M,
    TENNIS_S_MIN,
    TENNIS_V_MIN,
    format_point,
    update_ema,
)
from detect_tennis_blue_disk_apriltag import (
    body_to_optical,
    center_axis_points_in_tag,
    detect_apriltag_pose,
    draw_center_axes,
    draw_label_lines,
    intrinsics_dist_coeffs,
    intrinsics_matrix,
    make_apriltag_detector,
    point_optical_to_tag,
    point_tag_to_center_with_options,
    process_target,
    rpy_deg_to_matrix,
    write_target_json,
)

try:
    import cv2
    import pyrealsense2 as rs
except ImportError as exc:
    rs = None
    cv2 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class TrackerConfig:
    width: int = 1280
    height: int = 720
    tag_id: int = 0
    tag_size_m: float = 0.25
    tag_every: int = 5
    tag_max_age: int = 15
    center_origin_in_tag_cm: tuple[float, float, float] | None = (50.0, -30.0, -27.0)
    tag_origin_in_center: tuple[float, float, float] | None = None
    tag_rpy_in_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tennis_radius_m: float = TENNIS_RADIUS_M
    output_json: Path | None = None
    show_viz: bool = False
    no_ema: bool = False
    stationary_time_s: float = 2.0
    stationary_flicker_mm: float = 75.0
    blue_h_low: int = BLUE_H_LOW
    blue_h_high: int = BLUE_H_HIGH
    blue_s_min: int = BLUE_S_MIN
    blue_v_min: int = BLUE_V_MIN
    blue_diameter_m: float = BLUE_DIAMETER_M
    blue_circularity: float = 0.30
    center_axis_len_m: float = 0.10
    show_mask: bool = False


@dataclass
class BallObservation:
    valid: bool
    center_mm: np.ndarray | None
    center_m: np.ndarray | None
    tag_m: np.ndarray | None
    tag_status: str
    tag_age_frames: int
    depth_m: float
    depth_src: str
    timestamp_s: float
    reason: str | None = None
    blue_center_mm: np.ndarray | None = None
    blue_center_m: np.ndarray | None = None
    blue_tag_m: np.ndarray | None = None
    blue_depth_m: float = 0.0
    blue_depth_src: str = "—"
    blue_valid: bool = False

    def __post_init__(self) -> None:
        if self.center_mm is not None:
            self.center_mm = np.asarray(self.center_mm, dtype=float)
        if self.tag_m is not None:
            self.tag_m = np.asarray(self.tag_m, dtype=float)
        if self.blue_center_mm is not None:
            self.blue_center_mm = np.asarray(self.blue_center_mm, dtype=float)
        if self.blue_center_m is not None:
            self.blue_center_m = np.asarray(self.blue_center_m, dtype=float)
        if self.blue_tag_m is not None:
            self.blue_tag_m = np.asarray(self.blue_tag_m, dtype=float)


class TennisBallTracker:
    """Background D435 observer; call start() before grasp, stop() after."""

    def __init__(self, config: TrackerConfig):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "TennisBallTracker needs opencv-python and pyrealsense2"
            ) from _IMPORT_ERROR
        self.config = config
        self._lock = threading.Lock()
        self._latest = BallObservation(
            valid=False,
            center_mm=None,
            center_m=None,
            tag_m=None,
            tag_status="NO_TAG",
            tag_age_frames=999,
            depth_m=0.0,
            depth_src="—",
            timestamp_s=0.0,
            reason="not started",
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = None
        self._on_update: Callable[[BallObservation], None] | None = None
        self._stable_start: float | None = None
        self._stable_anchor_mm: np.ndarray | None = None
        self._last_center_mm: np.ndarray | None = None
        self._last_center_time: float | None = None
        self._stationary_elapsed_s = 0.0
        self._flicker_mm = 0.0
        self._grasp_ready = False
        self._last_json_write_s = 0.0
        self._last_grasp_ready_written: bool | None = None
        self._last_json_error_log_s = 0.0

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> TennisBallTracker:
        center_tag_cm = getattr(args, "center_origin_in_tag_cm", None)
        center_tag = getattr(args, "center_origin_in_tag", None)
        if center_tag_cm is not None:
            center_tag = tuple(float(x) * 0.01 for x in center_tag_cm)
        cfg = TrackerConfig(
            width=int(getattr(args, "width", 1280)),
            height=int(getattr(args, "height", 720)),
            tag_id=int(getattr(args, "tag_id", 0)),
            tag_size_m=float(getattr(args, "tag_size", 0.25)),
            tag_every=int(getattr(args, "tag_every", 5)),
            tag_max_age=int(getattr(args, "tag_max_age", 15)),
            center_origin_in_tag_cm=(
                tuple(float(x) for x in center_tag_cm)
                if center_tag_cm is not None
                else TrackerConfig.center_origin_in_tag_cm
            ),
            tag_origin_in_center=(
                tuple(float(x) for x in args.tag_origin_in_center)
                if getattr(args, "tag_origin_in_center", None) is not None
                else None
            ),
            tag_rpy_in_center=tuple(
                float(x) for x in getattr(args, "tag_rpy_in_center", (0.0, 0.0, 0.0))
            ),
            tennis_radius_m=float(getattr(args, "tennis_radius", TENNIS_RADIUS_M)),
            output_json=getattr(args, "output_json", None),
            show_viz=bool(getattr(args, "show_viz", False)),
            no_ema=bool(getattr(args, "no_ema", False)),
            stationary_time_s=float(getattr(args, "stationary_time", 2.0)),
            stationary_flicker_mm=float(
                getattr(args, "stationary_flicker", getattr(args, "stationary_speed", 75.0))
            ),
            blue_h_low=int(getattr(args, "blue_h_low", BLUE_H_LOW)),
            blue_h_high=int(getattr(args, "blue_h_high", BLUE_H_HIGH)),
            blue_s_min=int(getattr(args, "blue_s_min", BLUE_S_MIN)),
            blue_v_min=int(getattr(args, "blue_v_min", BLUE_V_MIN)),
            blue_diameter_m=float(getattr(args, "blue_diameter", BLUE_DIAMETER_M)),
            blue_circularity=float(getattr(args, "blue_circularity", 0.30)),
            center_axis_len_m=float(getattr(args, "center_axis_len", 0.10)),
            show_mask=bool(getattr(args, "show_mask", False)),
        )
        if center_tag is not None and cfg.center_origin_in_tag_cm is None:
            cfg.center_origin_in_tag_cm = tuple(x * 100.0 for x in center_tag)
        return cls(cfg)

    def set_update_callback(self, callback: Callable[[BallObservation], None] | None) -> None:
        self._on_update = callback

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="d435-tracker", daemon=True)
        self._thread.start()
        print("[D435] tracker thread started (camera stays open for whole grasp)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None
        if self.config.show_viz and cv2 is not None:
            cv2.destroyAllWindows()
        print("[D435] tracker stopped")

    def get_latest(self) -> BallObservation:
        with self._lock:
            return BallObservation(
                valid=self._latest.valid,
                center_mm=(
                    None if self._latest.center_mm is None else self._latest.center_mm.copy()
                ),
                center_m=(
                    None if self._latest.center_m is None else self._latest.center_m.copy()
                ),
                tag_status=self._latest.tag_status,
                tag_age_frames=self._latest.tag_age_frames,
                depth_m=self._latest.depth_m,
                depth_src=self._latest.depth_src,
                timestamp_s=self._latest.timestamp_s,
                reason=self._latest.reason,
            )

    def wait_valid(
        self,
        timeout_s: float = 30.0,
        poll_s: float = 0.05,
    ) -> BallObservation:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            obs = self.get_latest()
            if obs.valid:
                return obs
            time.sleep(poll_s)
        raise TimeoutError(
            f"No valid tennis ball observation within {timeout_s:.1f}s "
            f"(last: {self.get_latest().reason or self.get_latest().tag_status})"
        )

    def wait_stationary(
        self,
        duration_s: float = 2.0,
        max_flicker_mm: float = 75.0,
        poll_s: float = 0.05,
        timeout_s: float = 120.0,
    ) -> BallObservation:
        """
        Wait until the ball stays within max_flicker_mm of an anchor point
        for duration_s (stationary / low jitter).
        """
        deadline = time.time() + timeout_s
        stable_start: float | None = None
        anchor: np.ndarray | None = None

        print(
            f"[D435] waiting for ball stationary >= {duration_s:.1f}s "
            f"(flicker < {max_flicker_mm:.0f} mm)..."
        )
        while time.time() < deadline:
            obs = self.get_latest()
            now = time.time()
            if not obs.valid or obs.center_mm is None:
                if stable_start is not None:
                    print("[D435] ball lost while stabilizing; timer reset")
                stable_start = None
                anchor = None
                time.sleep(poll_s)
                continue

            center = obs.center_mm
            if anchor is None:
                anchor = center.copy()
                stable_start = now

            flicker_mm = float(np.linalg.norm(center - anchor))
            if flicker_mm <= max_flicker_mm:
                if stable_start is None:
                    stable_start = now
                    print(
                        f"[D435] flicker {flicker_mm:.0f} mm, stable timer started"
                    )
                elif now - stable_start >= duration_s:
                    print(
                        f"[D435] ball stationary for {duration_s:.1f}s "
                        f"(flicker={flicker_mm:.0f} mm), start grasp"
                    )
                    return obs
            else:
                if stable_start is not None:
                    print(
                        f"[D435] flicker {flicker_mm:.0f} mm > {max_flicker_mm:.0f} mm, "
                        "timer reset"
                    )
                anchor = center.copy()
                stable_start = now

            time.sleep(poll_s)

        raise TimeoutError(
            f"Ball did not stay stationary for {duration_s:.1f}s within {timeout_s:.1f}s"
        )

    def _update_stationary_state(self, valid: bool, center_mm: np.ndarray | None, now: float) -> None:
        cfg = self.config
        if not valid or center_mm is None:
            self._stable_start = None
            self._stable_anchor_mm = None
            self._last_center_mm = None
            self._last_center_time = None
            self._stationary_elapsed_s = 0.0
            self._flicker_mm = 0.0
            self._grasp_ready = False
            return

        if self._stable_anchor_mm is None:
            self._stable_anchor_mm = center_mm.copy()
            self._stable_start = now

        flicker_mm = float(np.linalg.norm(center_mm - self._stable_anchor_mm))
        self._flicker_mm = flicker_mm
        if flicker_mm <= cfg.stationary_flicker_mm:
            if self._stable_start is None:
                self._stable_start = now
            self._stationary_elapsed_s = now - self._stable_start
        else:
            self._stable_anchor_mm = center_mm.copy()
            self._stable_start = now
            self._stationary_elapsed_s = 0.0

        self._grasp_ready = self._stationary_elapsed_s >= cfg.stationary_time_s
        self._last_center_mm = center_mm.copy()
        self._last_center_time = now

    def _publish(self, obs: BallObservation) -> None:
        with self._lock:
            self._latest = obs
        if self.config.output_json is not None:
            now = time.time()
            state_changed = self._grasp_ready != self._last_grasp_ready_written
            if now - self._last_json_write_s >= 0.12 or state_changed:
                origin_tag_m = None
                if self.config.center_origin_in_tag_cm is not None:
                    origin_tag_m = tuple(
                        float(x) * 0.01 for x in self.config.center_origin_in_tag_cm
                    )
                try:
                    write_target_json(
                        self.config.output_json,
                        obs.center_m,
                        obs.tag_m,
                        obs.tag_status,
                        obs.tag_age_frames,
                        obs.depth_m,
                        obs.depth_src,
                        self.config.tennis_radius_m,
                        grasp_ready=self._grasp_ready,
                        stationary_elapsed_s=self._stationary_elapsed_s,
                        flicker_mm=self._flicker_mm,
                        center_origin_in_tag_m=origin_tag_m,
                        blue_center=obs.blue_center_m,
                        blue_tag=obs.blue_tag_m,
                        blue_depth=obs.blue_depth_m,
                        blue_src=obs.blue_depth_src,
                        blue_diameter_m=self.config.blue_diameter_m,
                    )
                    self._last_json_write_s = now
                    self._last_grasp_ready_written = self._grasp_ready
                except OSError as exc:
                    if now - self._last_json_error_log_s >= 2.0:
                        print(f"[D435] ball_target.json write retry: {exc}")
                        self._last_json_error_log_s = now
        if self._on_update is not None:
            self._on_update(obs)

    def _resolve_center_transform(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        cfg = self.config
        center_origin_in_tag = None
        if cfg.center_origin_in_tag_cm is not None:
            center_origin_in_tag = np.array(cfg.center_origin_in_tag_cm, dtype=float) * 0.01
        tag_origin_in_center = None
        r_center_tag = None
        if cfg.tag_origin_in_center is not None:
            tag_origin_in_center = np.array(cfg.tag_origin_in_center, dtype=float)
            r_center_tag = rpy_deg_to_matrix(*cfg.tag_rpy_in_center)
        return center_origin_in_tag, r_center_tag, tag_origin_in_center

    def _start_pipeline(self):
        pipeline = rs.pipeline()
        fps_tries = [(60, 60), (30, 30), (15, 15)]
        last_err = None
        for c_fps, d_fps in fps_tries:
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, c_fps
            )
            cfg.enable_stream(
                rs.stream.depth, self.config.width, self.config.height, rs.format.z16, d_fps
            )
            try:
                print(
                    f"[D435] opening {self.config.width}x{self.config.height} "
                    f"color {c_fps}Hz depth {d_fps}Hz"
                )
                profile = pipeline.start(cfg)
                pipeline.wait_for_frames(timeout_ms=5000)
                self._pipeline = pipeline
                return profile
            except RuntimeError as err:
                last_err = err
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
        raise RuntimeError(f"RealSense start failed: {last_err!r}")

    def _run_loop(self) -> None:
        cfg = self.config
        center_origin_in_tag, r_center_tag, tag_origin_in_center = self._resolve_center_transform()
        try:
            profile = self._start_pipeline()
        except Exception as exc:
            self._publish(
                BallObservation(
                    valid=False,
                    center_mm=None,
                    center_m=None,
                    tag_m=None,
                    tag_status="NO_CAM",
                    tag_age_frames=999,
                    depth_m=0.0,
                    depth_src="—",
                    timestamp_s=time.time(),
                    reason=str(exc),
                )
            )
            return

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intrin = color_profile.get_intrinsics()
        depth_intrin = depth_profile.get_intrinsics()
        color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        tag_detector = make_apriltag_detector()

        import cv2 as _cv2

        tennis_low = np.array([TENNIS_H_LOW, TENNIS_S_MIN, TENNIS_V_MIN], dtype=np.uint8)
        tennis_high = np.array([TENNIS_H_HIGH, 255, 255], dtype=np.uint8)
        blue_low = np.array([cfg.blue_h_low, cfg.blue_s_min, cfg.blue_v_min], dtype=np.uint8)
        blue_high = np.array([cfg.blue_h_high, 255, 255], dtype=np.uint8)

        tennis_ema = None
        blue_ema = None
        frame_idx = 0
        last_r_tag_to_opt = None
        last_t_tag_in_opt = None
        last_tag_rvec = None
        last_tag_corners = None
        last_tag_ids = None
        last_tag_frame = -10**9

        try:
            while not self._stop.is_set():
                frame_idx += 1
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth_arr = np.asanyarray(depth_frame.get_data())

                run_tag = last_r_tag_to_opt is None or frame_idx % max(cfg.tag_every, 1) == 0
                if run_tag:
                    detected = detect_apriltag_pose(
                        color, tag_detector, color_intrin, cfg.tag_size_m, cfg.tag_id
                    )
                    r_new, t_new, tag_rvec_new, tag_corners_new, tag_ids_new, _rejected = detected
                    if r_new is not None:
                        last_r_tag_to_opt = r_new
                        last_t_tag_in_opt = t_new
                        last_tag_rvec = tag_rvec_new
                        last_tag_corners = tag_corners_new
                        last_tag_ids = tag_ids_new
                        last_tag_frame = frame_idx
                    elif frame_idx - last_tag_frame > cfg.tag_max_age:
                        last_r_tag_to_opt = None
                        last_t_tag_in_opt = None
                        last_tag_rvec = None
                        last_tag_corners = None
                        last_tag_ids = None

                tennis_det, tennis_body_raw, tennis_opt, tennis_depth, tennis_src, tennis_mask = process_target(
                    color,
                    depth_arr,
                    depth_scale,
                    color_intrin,
                    depth_intrin,
                    color_to_depth_extr,
                    tennis_low,
                    tennis_high,
                    3.0,
                    220.0,
                    0.60,
                    cfg.tennis_radius_m,
                    5,
                    True,
                )
                blue_det, blue_body_raw, blue_opt, blue_depth, blue_src, blue_mask = process_target(
                    color,
                    depth_arr,
                    depth_scale,
                    color_intrin,
                    depth_intrin,
                    color_to_depth_extr,
                    blue_low,
                    blue_high,
                    3.0,
                    220.0,
                    cfg.blue_circularity,
                    cfg.blue_diameter_m * 0.5,
                    5,
                    False,
                )
                tennis_ema = update_ema(tennis_ema, tennis_body_raw, cfg.no_ema)
                blue_ema = update_ema(blue_ema, blue_body_raw, cfg.no_ema)
                tennis_opt = body_to_optical(tennis_ema)
                blue_opt = body_to_optical(blue_ema)
                tennis_tag = point_optical_to_tag(
                    tennis_opt, last_r_tag_to_opt, last_t_tag_in_opt
                )
                blue_tag = point_optical_to_tag(blue_opt, last_r_tag_to_opt, last_t_tag_in_opt)
                tennis_center = point_tag_to_center_with_options(
                    tennis_tag, r_center_tag, tag_origin_in_center, center_origin_in_tag
                )
                blue_center = point_tag_to_center_with_options(
                    blue_tag, r_center_tag, tag_origin_in_center, center_origin_in_tag
                )
                center_axes_draw = center_axis_points_in_tag(
                    center_origin_in_tag,
                    r_center_tag,
                    tag_origin_in_center,
                    cfg.center_axis_len_m,
                )

                tag_age = frame_idx - last_tag_frame
                tag_status = (
                    "TAG"
                    if tag_age == 0
                    else (f"TAG-{tag_age}" if last_r_tag_to_opt is not None else "NO_TAG")
                )
                blue_valid = (
                    blue_center is not None
                    and np.all(np.isfinite(blue_center))
                    and DEPTH_MIN < float(blue_depth) < DEPTH_MAX
                    and blue_src != "—"
                )
                valid = (
                    tennis_center is not None
                    and np.all(np.isfinite(tennis_center))
                    and DEPTH_MIN < float(tennis_depth) < DEPTH_MAX
                    and tennis_src != "—"
                    and last_r_tag_to_opt is not None
                )
                center_mm = (
                    np.asarray(tennis_center, dtype=float) * 1000.0
                    if tennis_center is not None
                    else None
                )
                blue_center_mm = (
                    np.asarray(blue_center, dtype=float) * 1000.0
                    if blue_center is not None
                    else None
                )
                now = time.time()
                self._update_stationary_state(valid, center_mm, now)
                obs = BallObservation(
                    valid=valid,
                    center_mm=center_mm,
                    center_m=(
                        np.asarray(tennis_center, dtype=float)
                        if tennis_center is not None
                        else None
                    ),
                    tag_m=(
                        np.asarray(tennis_tag, dtype=float)
                        if tennis_tag is not None
                        else None
                    ),
                    blue_center_mm=blue_center_mm,
                    blue_center_m=(
                        np.asarray(blue_center, dtype=float)
                        if blue_center is not None
                        else None
                    ),
                    blue_tag_m=(
                        np.asarray(blue_tag, dtype=float) if blue_tag is not None else None
                    ),
                    blue_depth_m=float(blue_depth),
                    blue_depth_src=blue_src,
                    blue_valid=bool(blue_valid),
                    tag_status=tag_status,
                    tag_age_frames=int(tag_age),
                    depth_m=float(tennis_depth),
                    depth_src=tennis_src,
                    timestamp_s=now,
                    reason=None if valid else "waiting for tag/depth",
                )
                self._publish(obs)

                if cfg.show_viz:
                    vis = color.copy()
                    if last_tag_ids is not None and last_tag_corners is not None:
                        _cv2.aruco.drawDetectedMarkers(vis, last_tag_corners, last_tag_ids)
                    if last_tag_rvec is not None and last_t_tag_in_opt is not None:
                        _cv2.drawFrameAxes(
                            vis,
                            intrinsics_matrix(color_intrin),
                            intrinsics_dist_coeffs(color_intrin),
                            last_tag_rvec,
                            last_t_tag_in_opt.reshape(3, 1),
                            cfg.tag_size_m * 0.4,
                        )
                    draw_center_axes(
                        vis,
                        center_axes_draw,
                        last_r_tag_to_opt,
                        last_t_tag_in_opt,
                        color_intrin,
                    )
                    if tennis_det is not None:
                        cx, cy, r_px = tennis_det
                        _cv2.circle(vis, (cx, cy), int(r_px), (0, 255, 0), 2)
                        draw_label_lines(
                            vis,
                            [
                                f"T tag {format_point(tennis_tag)}",
                                f"T center {format_point(tennis_center)}",
                            ],
                            (cx + 8, max(18, cy - 24)),
                            (0, 255, 0),
                        )
                    if blue_det is not None:
                        cx, cy, r_px = blue_det
                        _cv2.circle(vis, (cx, cy), int(r_px), (255, 0, 0), 2)
                        draw_label_lines(
                            vis,
                            [
                                f"B tag {format_point(blue_tag)}",
                                f"B center {format_point(blue_center)}",
                            ],
                            (cx + 8, max(36, cy + 18)),
                            (255, 0, 0),
                        )
                    ready_txt = (
                        "GRASP_READY"
                        if self._grasp_ready
                        else f"stable {self._stationary_elapsed_s:.1f}/{cfg.stationary_time_s:.0f}s"
                    )
                    ready_color = (0, 255, 0) if self._grasp_ready else (0, 220, 255)
                    blue_txt = "BLUE_OK" if blue_valid else "BLUE—"
                    _cv2.putText(
                        vis,
                        f"{tag_status}  {blue_txt}  {ready_txt}  "
                        f"flick={self._flicker_mm:.0f}/{cfg.stationary_flicker_mm:.0f}mm",
                        (10, 28),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        ready_color,
                        2,
                    )
                    if center_mm is not None:
                        _cv2.putText(
                            vis,
                            f"T_mm=[{center_mm[0]:.0f},{center_mm[1]:.0f},{center_mm[2]:.0f}]",
                            (10, 54),
                            _cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )
                    if blue_center_mm is not None:
                        bmm = blue_center_mm
                        _cv2.putText(
                            vis,
                            f"B_mm=[{bmm[0]:.0f},{bmm[1]:.0f},{bmm[2]:.0f}]",
                            (10, 76),
                            _cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 200, 0),
                            2,
                        )
                    if cfg.show_mask:
                        tennis_bgr = _cv2.cvtColor(tennis_mask, _cv2.COLOR_GRAY2BGR)
                        blue_bgr = _cv2.cvtColor(blue_mask, _cv2.COLOR_GRAY2BGR)
                        out = np.hstack([vis, tennis_bgr, blue_bgr])
                    else:
                        out = vis
                    _cv2.imshow("D435_tennis_blue_tag", out)
                    if _cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop.set()
        finally:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass
                self._pipeline = None
