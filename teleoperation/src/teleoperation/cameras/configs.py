from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntelRealSenseCameraConfig:
    name: str | None = None
    serial_number: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = False
    rotation: int | None = None
    mock: bool = False

    @property
    def type(self) -> str:
        return "intelrealsense"

    def __post_init__(self) -> None:
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                "Only one of name or serial_number may be set for an Intel RealSense camera."
            )
        if self.color_mode not in {"rgb", "bgr"}:
            raise ValueError(f"color_mode must be 'rgb' or 'bgr' (got {self.color_mode!r}).")
        if self.rotation not in {-90, None, 90, 180}:
            raise ValueError(f"rotation must be one of -90, None, 90, 180 (got {self.rotation!r}).")

        self.channels = 3
        any_stream_value = self.fps is not None or self.width is not None or self.height is not None
        all_stream_values = self.fps is not None and self.width is not None and self.height is not None
        if any_stream_value and not all_stream_values:
            raise ValueError("fps, width, and height must be set together or left unset together.")


CameraConfig = IntelRealSenseCameraConfig

