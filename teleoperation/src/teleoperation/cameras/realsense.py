"""
This file contains utilities for recording frames from Intel Realsense cameras.
"""

import argparse
import concurrent.futures
import logging
import math
import os
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from teleoperation.cameras.configs import IntelRealSenseCameraConfig
from teleoperation.utils.robot_devices import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from teleoperation.utils.common import capture_timestamp_utc

SERIAL_NUMBER_INDEX = 1
DEFAULT_READY_TIMEOUT_S = float(os.getenv("LEROBOT_RS_READY_TIMEOUT_S", "20"))


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return image[..., ::-1].copy()


def rotate_image(image: np.ndarray, rotation: int | None) -> np.ndarray:
    if rotation is None:
        return image
    if rotation == -90:
        return np.rot90(image, 1).copy()
    if rotation == 90:
        return np.rot90(image, -1).copy()
    if rotation == 180:
        return np.rot90(image, 2).copy()
    raise ValueError(f"Unsupported rotation value: {rotation}")


def find_cameras(raise_when_empty=True, mock=False) -> list[dict]:
    """
    Find the names and the serial numbers of the Intel RealSense cameras
    connected to the computer.
    """
    if mock:
        raise NotImplementedError("RealSense mock mode is not included in this standalone collector.")
    import pyrealsense2 as rs

    cameras = []
    try:
        for device in rs.context().query_devices():
            serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
            name = device.get_info(rs.camera_info.name)
            cameras.append(
                {
                    "serial_number": serial_number,
                    "name": name,
                }
            )
    except Exception as exc:
        if mock:
            raise
        logging.warning("pyrealsense2 device enumeration failed, falling back to rs-enumerate-devices: %r", exc)
        cameras = _find_cameras_via_rs_enumerate_devices()

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
        )

    return cameras


def _find_cameras_via_rs_enumerate_devices() -> list[dict]:
    result = subprocess.run(
        ["rs-enumerate-devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rs-enumerate-devices failed while trying to enumerate RealSense cameras. "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )

    cameras = []
    current_name = None
    current_serial = None

    def maybe_commit():
        if current_name is not None and current_serial is not None:
            cameras.append(
                {
                    "serial_number": current_serial,
                    "name": current_name,
                }
            )

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Device info"):
            maybe_commit()
            current_name = None
            current_serial = None
            continue
        if line.startswith("Name"):
            current_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Serial Number"):
            serial_text = line.split(":", 1)[1].strip()
            if serial_text.isdigit():
                current_serial = int(serial_text)

    maybe_commit()
    return cameras


def save_image(img_array, serial_number, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{serial_number}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save image for camera {serial_number} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    serial_numbers: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given serial number.
    """
    if serial_numbers is None or len(serial_numbers) == 0:
        camera_infos = find_cameras(mock=mock)
        serial_numbers = [cam["serial_number"] for cam in camera_infos]

    print("Connecting cameras")
    cameras = []
    for cam_sn in serial_numbers:
        print(f"{cam_sn=}")
        config = IntelRealSenseCameraConfig(
            serial_number=cam_sn, fps=fps, width=width, height=height, mock=mock
        )
        camera = IntelRealSenseCamera(config)
        camera.connect()
        print(
            f"IntelRealSenseCamera({camera.serial_number}, fps={camera.fps}, width={camera.width}, height={camera.height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed, and we will end up
                    # saving the same images from the cameras multiple times until the RAM/disk is full.
                    image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")

                    executor.submit(
                        save_image,
                        image,
                        camera.serial_number,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    busy_wait(1 / fps - dt_s)

                if time.perf_counter() - start_time > record_time_s:
                    break

                print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

                frame_index += 1
    finally:
        print(f"Images have been saved to {images_dir}")
        for camera in cameras:
            camera.disconnect()


class IntelRealSenseCamera:
    """
    The IntelRealSenseCamera class is similar to OpenCVCamera class but adds additional features for Intel Real Sense cameras:
    - is instantiated with the serial number of the camera - won't randomly change as it can be the case of OpenCVCamera for Linux,
    - can also be instantiated with the camera's name — if it's unique — using IntelRealSenseCamera.init_from_name(),
    - depth map can be returned.

    To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/intelrealsense.py --images-dir outputs/images_from_intelrealsense_cameras
    ```

    When an IntelRealSenseCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of instantiating with a serial number:
    ```python
    from teleoperation.cameras.configs import IntelRealSenseCameraConfig

    config = IntelRealSenseCameraConfig(serial_number=128422271347)
    camera = IntelRealSenseCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of instantiating with a name if it's unique:
    ```
    config = IntelRealSenseCameraConfig(name="Intel RealSense D405")
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=30, width=1280, height=720)
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=90, width=640, height=480)
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=90, width=640, height=480, color_mode="bgr")
    # Note: might error out upon `camera.connect()` if these settings are not compatible with the camera
    ```

    Example of returning depth:
    ```python
    config = IntelRealSenseCameraConfig(serial_number=128422271347, use_depth=True)
    camera = IntelRealSenseCamera(config)
    camera.connect()
    color_image, depth_map = camera.read()
    ```
    """

    def __init__(
        self,
        config: IntelRealSenseCameraConfig,
    ):
        self.config = config
        if config.name is not None:
            self.serial_number = self.find_serial_number_from_name(config.name)
        else:
            self.serial_number = config.serial_number
        self.fps = int(config.fps) if config.fps is not None else None
        self.width = int(config.width) if config.width is not None else None
        self.height = int(config.height) if config.height is not None else None
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.force_hardware_reset = config.force_hardware_reset
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.last_read_error = None
        self.logs = {}

        # TODO(alibets): Do we keep original width/height or do we define them after rotation?
        self.rotation = config.rotation

    def find_serial_number_from_name(self, name):
        camera_infos = find_cameras()
        camera_names = [cam["name"] for cam in camera_infos]
        this_name_count = Counter(camera_names)[name]
        if this_name_count > 1:
            # TODO(aliberts): Test this with multiple identical cameras (Aloha)
            raise ValueError(
                f"Multiple {name} cameras have been detected. Please use their serial number to instantiate them."
            )

        name_to_serial_dict = {cam["name"]: cam["serial_number"] for cam in camera_infos}
        cam_sn = name_to_serial_dict[name]

        return cam_sn

    def _make_rs_config(self, rs):
        config = rs.config()
        config.enable_device(str(self.serial_number))

        if self.fps and self.width and self.height:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.width and self.height:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            else:
                config.enable_stream(rs.stream.depth)

        return config

    def _hardware_reset_device(self, rs, wait_timeout_s: float = 15.0):
        if self.mock or not self.force_hardware_reset:
            return

        ctx = rs.context()
        target = None
        try:
            devices = list(ctx.query_devices())
        except Exception:
            return

        for device in devices:
            serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
            if serial_number == self.serial_number:
                target = device
                break

        if target is None or not hasattr(target, "hardware_reset"):
            return

        target.hardware_reset()

        deadline = time.perf_counter() + wait_timeout_s
        while time.perf_counter() < deadline:
            time.sleep(1.0)
            try:
                camera_infos = find_cameras(raise_when_empty=False, mock=self.mock)
            except Exception:
                continue

            if any(cam["serial_number"] == self.serial_number for cam in camera_infos):
                return

        raise RuntimeError(
            f"IntelRealSenseCamera({self.serial_number}) did not reappear after hardware reset."
        )

    def start_async_worker(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

    def wait_until_ready(self, timeout_s: float = DEFAULT_READY_TIMEOUT_S):
        deadline = time.perf_counter() + timeout_s
        while self.color_image is None:
            if time.perf_counter() >= deadline:
                if self.last_read_error is not None:
                    raise RuntimeError(
                        f"IntelRealSenseCamera({self.serial_number}) failed to produce frames. "
                        f"Last read error: {self.last_read_error!r}"
                    )
                raise RuntimeError(
                    f"IntelRealSenseCamera({self.serial_number}) did not produce the first frame "
                    f"within {timeout_s} seconds."
                )
            time.sleep(max(1 / self.fps, 0.05))

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is already connected."
            )

        if self.mock:
            raise NotImplementedError("RealSense mock mode is not included in this standalone collector.")
        import pyrealsense2 as rs

        last_error = None
        for attempt in range(3):
            profile = None
            pipeline = rs.pipeline()
            try:
                if attempt > 0:
                    self._hardware_reset_device(rs)
                    time.sleep(2.0)

                config = self._make_rs_config(rs)
                profile = pipeline.start(config)

                color_stream = profile.get_stream(rs.stream.color)
                color_profile = color_stream.as_video_stream_profile()
                actual_fps = color_profile.fps()
                actual_width = color_profile.width()
                actual_height = color_profile.height()

                # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
                if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
                    raise OSError(
                        f"Can't set {self.fps=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_fps}."
                    )
                if self.width is not None and self.width != actual_width:
                    raise OSError(
                        f"Can't set {self.width=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_width}."
                    )
                if self.height is not None and self.height != actual_height:
                    raise OSError(
                        f"Can't set {self.height=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_height}."
                    )

                self.camera = pipeline
                self.fps = round(actual_fps)
                self.width = round(actual_width)
                self.height = round(actual_height)
                self.color_image = None
                self.depth_map = None
                self.last_read_error = None
                self.is_connected = True
                return
            except Exception as exc:
                last_error = exc
                self.last_read_error = exc
                self.color_image = None
                self.depth_map = None
                self.is_connected = False
                self.camera = None
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(1.0)

        camera_infos = find_cameras(raise_when_empty=False, mock=self.mock)
        serial_numbers = [cam["serial_number"] for cam in camera_infos]
        if self.serial_number not in serial_numbers:
            raise ValueError(
                f"`serial_number` is expected to be one of these available cameras {serial_numbers}, but {self.serial_number} is provided instead."
            )

        raise RuntimeError(
            f"Can't access IntelRealSenseCamera({self.serial_number}) after retries. Last error: {last_error!r}"
        )

    def read(
        self, temporary_color: str | None = None, timeout_ms: int = 10000
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 480 x 640 x 3)
        of type `np.uint8`, contrarily to the pytorch format which is float channel first.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 480 x 640) of type np.uint16.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        frame = self.camera.wait_for_frames(timeout_ms=timeout_ms)

        color_frame = frame.get_color_frame()

        if not color_frame:
            raise OSError(f"Can't capture color image from IntelRealSenseCamera({self.serial_number}).")

        color_image = np.asanyarray(color_frame.get_data())

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # IntelRealSense uses RGB format as default (red, green, blue).
        if requested_color_mode == "bgr":
            color_image = rgb_to_bgr(color_image)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = rotate_image(color_image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(f"Can't capture depth image from IntelRealSenseCamera({self.serial_number}).")

            depth_map = np.asanyarray(depth_frame.get_data())

            h, w = depth_map.shape
            if h != self.height or w != self.width:
                raise OSError(
                    f"Can't capture depth map with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
                )

            if self.rotation is not None:
                depth_map = rotate_image(depth_map, self.rotation)

            return color_image, depth_map
        else:
            return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.use_depth:
                    self.color_image, self.depth_map = self.read()
                else:
                    self.color_image = self.read()
                self.last_read_error = None
            except Exception as exc:
                self.last_read_error = exc
                self.logs["last_read_error"] = repr(exc)
                time.sleep(max(1 / self.fps, 0.1))

    def async_read(self):
        """Access the latest color image"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        self.start_async_worker()

        deadline = time.perf_counter() + DEFAULT_READY_TIMEOUT_S
        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise RuntimeError(
                    "The thread responsible for `self.async_read()` took too much time to start. "
                    "There might be an issue. Verify that `self.thread.start()` has been called."
                )
            if time.perf_counter() >= deadline:
                detail = ""
                if self.last_read_error is not None:
                    detail = f" Last read error: {self.last_read_error!r}"
                raise RuntimeError(
                    f"IntelRealSenseCamera({self.serial_number}) did not produce the first frame "
                    f"within {DEFAULT_READY_TIMEOUT_S} seconds.{detail}"
                )

        if self.use_depth:
            return self.color_image, self.depth_map
        else:
            return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.stop()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `IntelRealSenseCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--serial-numbers",
        type=int,
        nargs="*",
        default=None,
        help="List of serial numbers used to instantiate the `IntelRealSenseCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_intelrealsense_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
