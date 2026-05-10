from typing import Protocol

import numpy as np

from teleoperation.cameras.configs import CameraConfig, IntelRealSenseCameraConfig


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> list[Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "intelrealsense":
            from teleoperation.cameras.realsense import IntelRealSenseCamera

            cameras[key] = IntelRealSenseCamera(cfg)
        else:
            raise ValueError(f"The camera type '{cfg.type}' is not supported by this project.")

    return cameras


def make_camera(camera_type, **kwargs) -> Camera:
    if camera_type == "intelrealsense":
        from teleoperation.cameras.realsense import IntelRealSenseCamera

        config = IntelRealSenseCameraConfig(**kwargs)
        return IntelRealSenseCamera(config)

    else:
        raise ValueError(f"The camera type '{camera_type}' is not supported by this project.")
