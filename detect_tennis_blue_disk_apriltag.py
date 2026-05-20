"""
detect_tennis_blue_disk_apriltag.py

Detect three objects in one RealSense D455 stream:
  - tennis ball: HSV circle detection + visual/sensor depth fusion
  - blue end disk: HSV circle detection + RealSense depth at detected centre
  - AprilTag: tag36h11, id=0, 0.25 m square

Outputs:
  - tennis / blue positions in camera body frame
  - tennis / blue positions in AprilTag frame
  - optionally tennis / blue positions in a user-defined centre frame

AprilTag frame:
  origin at tag centre; +X to tag right, +Y to tag bottom in the image when
  viewed front-on; units are metres. This is the object frame used by solvePnP.

Centre frame:
  pass the known AprilTag pose in that centre frame:
    --tag-origin-in-center X Y Z
    --tag-rpy-in-center ROLL PITCH YAW   # degrees, optional; default 0 0 0
  Then p_center = R_center_tag @ p_tag + tag_origin_in_center.

  If you instead know the centre point in AprilTag coordinates, use:
    --center-origin-in-tag X Y Z          # metres
    --center-origin-in-tag-cm X Y Z       # centimetres
  This assumes centre-frame axes are parallel to tag-frame axes, so:
    p_center = p_tag - center_origin_in_tag

Usage:
    python detect_tennis_blue_disk_apriltag.py
    python detect_tennis_blue_disk_apriltag.py --show-mask
    python detect_tennis_blue_disk_apriltag.py --no-viz
    python detect_tennis_blue_disk_apriltag.py --tag-origin-in-center 0.2 0.0 0.5
"""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from detect_tennis_and_blue_disk_hsv import (
    BLUE_DIAMETER_M,
    BLUE_H_HIGH,
    BLUE_H_LOW,
    BLUE_S_MIN,
    BLUE_V_MIN,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_SAMPLE_RADIUS,
    TENNIS_H_HIGH,
    TENNIS_H_LOW,
    TENNIS_RADIUS_M,
    TENNIS_S_MIN,
    TENNIS_V_MIN,
    VIS_WEIGHT,
    _FPS,
    color_pixel_to_depth_pixel,
    deproject_body,
    detect_colored_circle,
    format_point,
    median_depth_at,
    update_ema,
)


TAG_SIZE_M = 0.25
TAG_ID = 0


def body_to_optical(p_body):
    """Camera body frame (X-fwd, Y-left, Z-up) -> optical frame (Z-fwd, X-right, Y-down)."""
    if p_body is None:
        return None
    x, y, z = p_body
    return np.array([-y, -z, x], dtype=np.float64)


def rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def intrinsics_matrix(intrin):
    return np.array([
        [intrin.fx, 0.0, intrin.ppx],
        [0.0, intrin.fy, intrin.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def intrinsics_dist_coeffs(intrin):
    coeffs = np.array(intrin.coeffs, dtype=np.float64).reshape(-1, 1)
    if coeffs.size == 0:
        return np.zeros((5, 1), dtype=np.float64)
    return coeffs


def make_apriltag_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect_apriltag_pose(frame_bgr, detector, color_intrin, tag_size_m, tag_id):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        return None, None, None, corners, ids, rejected

    ids_flat = ids.reshape(-1)
    matches = np.where(ids_flat == tag_id)[0]
    if len(matches) == 0:
        return None, None, None, corners, ids, rejected

    idx = int(matches[0])
    image_points = corners[idx].reshape(4, 2).astype(np.float64)
    half = tag_size_m * 0.5
    object_points = np.array([
        [-half, -half, 0.0],
        [ half, -half, 0.0],
        [ half,  half, 0.0],
        [-half,  half, 0.0],
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics_matrix(color_intrin),
        intrinsics_dist_coeffs(color_intrin),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None, corners, ids, rejected

    r_tag_to_opt, _ = cv2.Rodrigues(rvec)
    t_tag_in_opt = tvec.reshape(3)
    return r_tag_to_opt, t_tag_in_opt, rvec, corners, ids, rejected


def point_optical_to_tag(p_opt, r_tag_to_opt, t_tag_in_opt):
    if p_opt is None or r_tag_to_opt is None or t_tag_in_opt is None:
        return None
    return r_tag_to_opt.T @ (p_opt - t_tag_in_opt)


def point_tag_to_center(p_tag, r_center_tag, tag_origin_in_center):
    if p_tag is None or r_center_tag is None or tag_origin_in_center is None:
        return None
    return r_center_tag @ p_tag + tag_origin_in_center


def point_tag_to_center_with_options(p_tag, r_center_tag, tag_origin_in_center, center_origin_in_tag):
    if p_tag is None:
        return None
    if center_origin_in_tag is not None:
        return p_tag - center_origin_in_tag
    return point_tag_to_center(p_tag, r_center_tag, tag_origin_in_center)


def center_origin_tag_from_options(r_center_tag, tag_origin_in_center, center_origin_in_tag):
    if center_origin_in_tag is not None:
        return center_origin_in_tag
    if r_center_tag is not None and tag_origin_in_center is not None:
        return -(r_center_tag.T @ tag_origin_in_center)
    return None


def project_tag_point_to_pixel(p_tag, r_tag_to_opt, t_tag_in_opt, color_intrin):
    if p_tag is None or r_tag_to_opt is None or t_tag_in_opt is None:
        return None
    p_opt = r_tag_to_opt @ p_tag + t_tag_in_opt
    if p_opt[2] <= 1e-6:
        return None
    u = color_intrin.fx * p_opt[0] / p_opt[2] + color_intrin.ppx
    v = color_intrin.fy * p_opt[1] / p_opt[2] + color_intrin.ppy
    if not (0 <= u < color_intrin.width and 0 <= v < color_intrin.height):
        return None
    return int(u + 0.5), int(v + 0.5)


def center_axis_points_in_tag(center_origin_in_tag, r_center_tag, tag_origin_in_center, axis_len_m):
    """Return center origin and +X/+Y/+Z endpoints, all expressed in AprilTag frame."""
    origin_tag = center_origin_tag_from_options(r_center_tag, tag_origin_in_center, center_origin_in_tag)
    if origin_tag is None:
        return None

    if center_origin_in_tag is not None:
        # In this mode, the centre-frame axes are assumed parallel to tag-frame axes.
        r_tag_center = np.eye(3)
    else:
        # p_center = R_center_tag @ p_tag + t, so centre axes in tag frame are R^T columns.
        r_tag_center = r_center_tag.T

    return {
        "origin": origin_tag,
        "x": origin_tag + r_tag_center @ np.array([axis_len_m, 0.0, 0.0]),
        "y": origin_tag + r_tag_center @ np.array([0.0, axis_len_m, 0.0]),
        "z": origin_tag + r_tag_center @ np.array([0.0, 0.0, axis_len_m]),
    }


def draw_label_lines(image, lines, origin_px, color, line_height=16):
    x, y = origin_px
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def draw_center_axes(image, axis_points_tag, r_tag_to_opt, t_tag_in_opt, color_intrin):
    if axis_points_tag is None:
        return None

    origin_px = project_tag_point_to_pixel(
        axis_points_tag["origin"], r_tag_to_opt, t_tag_in_opt, color_intrin)
    if origin_px is None:
        return None

    cv2.circle(image, origin_px, 8, (0, 0, 255), -1)
    cv2.circle(image, origin_px, 13, (255, 255, 255), 2)
    cv2.putText(
        image,
        "CENTER",
        (origin_px[0] + 10, max(20, origin_px[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    axes = [
        ("X", "x", (0, 0, 255)),
        ("Y", "y", (0, 255, 0)),
        ("Z", "z", (255, 0, 0)),
    ]
    for label, key, color in axes:
        end_px = project_tag_point_to_pixel(
            axis_points_tag[key], r_tag_to_opt, t_tag_in_opt, color_intrin)
        if end_px is None:
            continue
        cv2.arrowedLine(image, origin_px, end_px, color, 3, tipLength=0.25)
        cv2.putText(
            image,
            label,
            (end_px[0] + 4, end_px[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return origin_px


def process_target(
    color,
    depth_arr,
    depth_scale,
    color_intrin,
    depth_intrin,
    color_to_depth_extr,
    hsv_low,
    hsv_high,
    min_radius_px,
    max_radius_px,
    circularity,
    physical_radius_m,
    depth_radius_px,
    use_visual_depth,
):
    cx, cy, r_px, mask = detect_colored_circle(
        color, hsv_low, hsv_high, min_radius_px, max_radius_px, circularity)
    p_body = None
    depth_m = 0.0
    depth_src = "—"

    if cx is None:
        return None, None, None, 0.0, depth_src, mask

    dx, dy = color_pixel_to_depth_pixel(
        cx, cy, depth_arr, depth_scale, color_intrin, depth_intrin, color_to_depth_extr)
    surface_depth = median_depth_at(dx, dy, depth_arr, depth_scale, depth_radius_px)

    if use_visual_depth:
        visual_depth = color_intrin.fx * physical_radius_m / r_px if r_px > 0 else 0.0
        sensor_depth = surface_depth + physical_radius_m if surface_depth > 0 else 0.0
        vis_ok = DEPTH_MIN < visual_depth < DEPTH_MAX
        sensor_ok = DEPTH_MIN < sensor_depth < DEPTH_MAX
        if vis_ok and sensor_ok:
            ratio = visual_depth / sensor_depth
            if 0.5 < ratio < 2.0:
                depth_m = VIS_WEIGHT * visual_depth + (1 - VIS_WEIGHT) * sensor_depth
                depth_src = "fused"
            else:
                depth_m = visual_depth
                depth_src = "visual"
        elif vis_ok:
            depth_m = visual_depth
            depth_src = "visual"
        elif sensor_ok:
            depth_m = sensor_depth
            depth_src = "sensor"
    else:
        depth_m = surface_depth
        depth_src = "sensor" if DEPTH_MIN < depth_m < DEPTH_MAX else "—"

    p_body = deproject_body(cx, cy, depth_m, color_intrin)
    return (cx, cy, r_px), p_body, body_to_optical(p_body), depth_m, depth_src, mask


def main():
    parser = argparse.ArgumentParser(
        description="Detect tennis, blue disk, and AprilTag; report target coordinates relative to tag/centre.")
    parser.add_argument("--no-viz", action="store_true", help="disable OpenCV window")
    parser.add_argument("--no-ema", action="store_true", help="disable EMA smoothing for tennis/blue camera positions")
    parser.add_argument("--show-mask", action="store_true", help="show tennis/blue HSV masks")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")

    parser.add_argument("--tag-id", type=int, default=TAG_ID, help="AprilTag id")
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_M, help="AprilTag side length in meters")
    parser.add_argument("--tag-origin-in-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="AprilTag origin coordinates in centre frame, metres")
    parser.add_argument("--tag-rpy-in-center", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="AprilTag orientation in centre frame, degrees; default 0 0 0")
    parser.add_argument("--center-origin-in-tag", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="centre origin coordinates in AprilTag frame, metres")
    parser.add_argument("--center-origin-in-tag-cm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="centre origin coordinates in AprilTag frame, centimetres")
    parser.add_argument("--tag-every", type=int, default=3,
                        help="run AprilTag detection every N frames and reuse the latest pose")
    parser.add_argument("--tag-max-age", type=int, default=15,
                        help="discard reused AprilTag pose after this many frames without detection")
    parser.add_argument("--center-axis-len", type=float, default=0.10,
                        help="length of visualized centre-frame XYZ axes in metres")

    parser.add_argument("--tennis-h-low", type=int, default=TENNIS_H_LOW)
    parser.add_argument("--tennis-h-high", type=int, default=TENNIS_H_HIGH)
    parser.add_argument("--tennis-s-min", type=int, default=TENNIS_S_MIN)
    parser.add_argument("--tennis-v-min", type=int, default=TENNIS_V_MIN)
    parser.add_argument("--tennis-radius", type=float, default=TENNIS_RADIUS_M, help="tennis ball radius in meters")

    parser.add_argument("--blue-h-low", type=int, default=BLUE_H_LOW)
    parser.add_argument("--blue-h-high", type=int, default=BLUE_H_HIGH)
    parser.add_argument("--blue-s-min", type=int, default=BLUE_S_MIN)
    parser.add_argument("--blue-v-min", type=int, default=BLUE_V_MIN)
    parser.add_argument("--blue-diameter", type=float, default=BLUE_DIAMETER_M, help="blue disk diameter in meters")

    parser.add_argument("--min-radius-px", type=float, default=3.0, help="minimum blob radius in pixels")
    parser.add_argument("--max-radius-px", type=float, default=220.0, help="maximum blob radius in pixels")
    parser.add_argument("--tennis-circularity", type=float, default=0.60)
    parser.add_argument("--blue-circularity", type=float, default=0.30)
    parser.add_argument("--depth-radius", type=int, default=DEPTH_SAMPLE_RADIUS, help="depth median patch radius in pixels")
    args = parser.parse_args()

    viz = not args.no_viz
    tennis_low = np.array([args.tennis_h_low, args.tennis_s_min, args.tennis_v_min], dtype=np.uint8)
    tennis_high = np.array([args.tennis_h_high, 255, 255], dtype=np.uint8)
    blue_low = np.array([args.blue_h_low, args.blue_s_min, args.blue_v_min], dtype=np.uint8)
    blue_high = np.array([args.blue_h_high, 255, 255], dtype=np.uint8)

    tag_origin_in_center = None
    r_center_tag = None
    center_origin_in_tag = None
    if args.center_origin_in_tag is not None and args.center_origin_in_tag_cm is not None:
        raise ValueError("Use only one of --center-origin-in-tag or --center-origin-in-tag-cm.")
    if args.center_origin_in_tag_cm is not None:
        center_origin_in_tag = np.array(args.center_origin_in_tag_cm, dtype=np.float64) * 0.01
    elif args.center_origin_in_tag is not None:
        center_origin_in_tag = np.array(args.center_origin_in_tag, dtype=np.float64)

    if args.tag_origin_in_center is not None:
        tag_origin_in_center = np.array(args.tag_origin_in_center, dtype=np.float64)
        r_center_tag = rpy_deg_to_matrix(*args.tag_rpy_in_center)
        print(f"[INFO] Centre transform enabled: tag_origin_in_center={tag_origin_in_center} "
              f"tag_rpy_in_center_deg={args.tag_rpy_in_center}")
    elif center_origin_in_tag is not None:
        print(f"[INFO] Centre transform enabled: center_origin_in_tag={center_origin_in_tag} "
              "(centre axes assumed parallel to tag axes)")
    else:
        print("[INFO] Centre transform disabled; pass --center-origin-in-tag-cm X Y Z "
              "or --tag-origin-in-center X Y Z to enable it.")

    print(f"[INFO] AprilTag: family=36h11 id={args.tag_id} size={args.tag_size:.3f}m")
    print(f"[INFO] AprilTag detection cadence: every {max(args.tag_every, 1)} frame(s), "
          f"max reuse age {args.tag_max_age} frames")
    print(f"[INFO] Tennis HSV: H=[{args.tennis_h_low},{args.tennis_h_high}] "
          f"S>={args.tennis_s_min} V>={args.tennis_v_min} radius={args.tennis_radius:.3f}m")
    print(f"[INFO] Blue HSV:   H=[{args.blue_h_low},{args.blue_h_high}] "
          f"S>={args.blue_s_min} V>={args.blue_v_min} diameter={args.blue_diameter:.3f}m")

    pipeline = rs.pipeline()
    fps_tries = [(60, 60), (30, 30), (15, 15)]

    def start_pipeline():
        last_err = None
        for c_fps, d_fps in fps_tries:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, d_fps)
            try:
                print(f"[INFO] Starting RealSense ({args.width}x{args.height} color {c_fps}Hz depth {d_fps}Hz)...")
                profile = pipeline.start(cfg)
                pipeline.wait_for_frames(timeout_ms=5000)
                print(f"[INFO] RealSense OK ({c_fps}/{d_fps} Hz)")
                return profile
            except RuntimeError as e:
                last_err = e
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                print(f"[WARN] Profile failed: {e}")
        raise RuntimeError(f"RealSense failed. Tried {fps_tries}. Last: {last_err!r}")

    profile = start_pipeline()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    depth_intrin = depth_profile.get_intrinsics()
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    tag_detector = make_apriltag_detector()

    tennis_ema = None
    blue_ema = None
    fps = _FPS()
    frame_idx = 0
    last_r_tag_to_opt = None
    last_t_tag_in_opt = None
    last_tag_rvec = None
    last_tag_corners = None
    last_tag_ids = None
    last_tag_frame = -10**9

    try:
        while True:
            frame_idx += 1
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_arr = np.asanyarray(depth_frame.get_data())

            run_tag_detection = (
                last_r_tag_to_opt is None
                or frame_idx % max(args.tag_every, 1) == 0
            )
            if run_tag_detection:
                detected = detect_apriltag_pose(
                    color, tag_detector, color_intrin, args.tag_size, args.tag_id)
                r_tag_to_opt_new, t_tag_in_opt_new, tag_rvec_new, tag_corners_new, tag_ids_new, _ = detected
                if r_tag_to_opt_new is not None:
                    last_r_tag_to_opt = r_tag_to_opt_new
                    last_t_tag_in_opt = t_tag_in_opt_new
                    last_tag_rvec = tag_rvec_new
                    last_tag_corners = tag_corners_new
                    last_tag_ids = tag_ids_new
                    last_tag_frame = frame_idx
                elif frame_idx - last_tag_frame > args.tag_max_age:
                    last_r_tag_to_opt = None
                    last_t_tag_in_opt = None
                    last_tag_rvec = None
                    last_tag_corners = tag_corners_new
                    last_tag_ids = tag_ids_new

            r_tag_to_opt = last_r_tag_to_opt
            t_tag_in_opt = last_t_tag_in_opt
            tag_rvec = last_tag_rvec
            tag_corners = last_tag_corners
            tag_ids = last_tag_ids

            tennis_det, tennis_body_raw, tennis_opt, tennis_depth, tennis_src, tennis_mask = process_target(
                color, depth_arr, depth_scale, color_intrin, depth_intrin, color_to_depth_extr,
                tennis_low, tennis_high, args.min_radius_px, args.max_radius_px,
                args.tennis_circularity, args.tennis_radius, args.depth_radius,
                use_visual_depth=True)
            blue_det, blue_body_raw, blue_opt, blue_depth, blue_src, blue_mask = process_target(
                color, depth_arr, depth_scale, color_intrin, depth_intrin, color_to_depth_extr,
                blue_low, blue_high, args.min_radius_px, args.max_radius_px,
                args.blue_circularity, args.blue_diameter * 0.5, args.depth_radius,
                use_visual_depth=False)

            tennis_ema = update_ema(tennis_ema, tennis_body_raw, args.no_ema)
            blue_ema = update_ema(blue_ema, blue_body_raw, args.no_ema)
            tennis_opt = body_to_optical(tennis_ema)
            blue_opt = body_to_optical(blue_ema)

            tennis_tag = point_optical_to_tag(tennis_opt, r_tag_to_opt, t_tag_in_opt)
            blue_tag = point_optical_to_tag(blue_opt, r_tag_to_opt, t_tag_in_opt)
            tennis_center = point_tag_to_center_with_options(
                tennis_tag, r_center_tag, tag_origin_in_center, center_origin_in_tag)
            blue_center = point_tag_to_center_with_options(
                blue_tag, r_center_tag, tag_origin_in_center, center_origin_in_tag)
            center_axes_for_draw = center_axis_points_in_tag(
                center_origin_in_tag, r_center_tag, tag_origin_in_center, args.center_axis_len)

            fps.tick()
            tag_age = frame_idx - last_tag_frame
            tag_status = "TAG" if tag_age == 0 else (f"TAG-{tag_age}" if r_tag_to_opt is not None else "NO_TAG")
            print(f"\r[{tag_status}] "
                  f"tennis_cam={format_point(tennis_ema)} tennis_tag={format_point(tennis_tag)} "
                  f"tennis_center={format_point(tennis_center)}  "
                  f"blue_cam={format_point(blue_ema)} blue_tag={format_point(blue_tag)} "
                  f"blue_center={format_point(blue_center)}  "
                  f"depths(T={tennis_depth:.2f}/{tennis_src}, B={blue_depth:.2f}/{blue_src}) "
                  f"fps={fps.fps:5.1f}", end="", flush=True)

            if viz or args.show_mask:
                vis = color.copy()
                if tag_ids is not None and tag_corners is not None and len(tag_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(vis, tag_corners, tag_ids)
                if tag_rvec is not None:
                    cv2.drawFrameAxes(
                        vis,
                        intrinsics_matrix(color_intrin),
                        intrinsics_dist_coeffs(color_intrin),
                        tag_rvec,
                        t_tag_in_opt.reshape(3, 1),
                        args.tag_size * 0.4,
                    )

                draw_center_axes(vis, center_axes_for_draw, r_tag_to_opt, t_tag_in_opt, color_intrin)

                if tennis_det is not None:
                    cx, cy, r_px = tennis_det
                    cv2.circle(vis, (cx, cy), int(r_px), (0, 255, 0), 2)
                    draw_label_lines(
                        vis,
                        [f"T tag {format_point(tennis_tag)}",
                         f"T center {format_point(tennis_center)}"],
                        (cx + 8, max(18, cy - 24)),
                        (0, 255, 0),
                    )
                if blue_det is not None:
                    cx, cy, r_px = blue_det
                    cv2.circle(vis, (cx, cy), int(r_px), (255, 0, 0), 2)
                    draw_label_lines(
                        vis,
                        [f"B tag {format_point(blue_tag)}",
                         f"B center {format_point(blue_center)}"],
                        (cx + 8, max(36, cy + 18)),
                        (255, 0, 0),
                    )

                cv2.putText(vis, f"{tag_status}  dual HSV + AprilTag  {fps.fps:.1f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if args.show_mask:
                    tennis_bgr = cv2.cvtColor(tennis_mask, cv2.COLOR_GRAY2BGR)
                    blue_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
                    out = np.hstack([vis, tennis_bgr, blue_bgr])
                else:
                    out = vis

                cv2.imshow("tennis_blue_disk_apriltag", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        if viz or args.show_mask:
            cv2.destroyAllWindows()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
