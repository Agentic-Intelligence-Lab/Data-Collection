from __future__ import annotations

from typing import Protocol


class Robot(Protocol):
    robot_type: str
    cameras: dict

    @property
    def camera_features(self) -> dict: ...

    @property
    def motor_features(self) -> dict: ...

