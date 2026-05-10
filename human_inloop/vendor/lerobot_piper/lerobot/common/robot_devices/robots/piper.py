"""
Teleoperation support for Agilex Piper.

Supports three modes:
1. Legacy single-arm + gamepad control.
2. Software leader/follower recording where manual leader arms drive follower arms.
3. Passive follower recording for hardware-linked master/slave setups.
"""

import time
from dataclasses import replace

import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

EE_POSE_COMPONENTS = (
    "x_m",
    "y_m",
    "z_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
)

PASSIVE_TIMING_FEATURES = {
    "real_observation_timestamp_s": {"dtype": "float64", "shape": (1,), "names": None},
    "real_action_timestamp_s": {"dtype": "float64", "shape": (1,), "names": None},
    "real_observation_wall_time_ns": {"dtype": "int64", "shape": (1,), "names": None},
    "real_action_wall_time_ns": {"dtype": "int64", "shape": (1,), "names": None},
    "real_transition_delta_s": {"dtype": "float64", "shape": (1,), "names": None},
    "real_transition_fps": {"dtype": "float64", "shape": (1,), "names": None},
}


class PiperRobot:
    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig()

        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.configured_camera_names = list(self.cameras.keys())
        self.unavailable_cameras = {}

        follower_cfg = self.config.follower_arms or self.config.follower_arm
        self.follower_arms = make_motors_buses_from_configs(follower_cfg)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)

        if not self.follower_arms:
            raise ValueError("PiperRobot requires at least one follower arm.")

        self.passive_recording_mode = self.config.passive_recording
        self.multi_arm_mode = bool(self.leader_arms)
        if self.passive_recording_mode and self.multi_arm_mode:
            raise ValueError("Passive recording mode cannot be combined with leader_arms.")
        if self.multi_arm_mode and set(self.leader_arms) != set(self.follower_arms):
            raise ValueError(
                "In leader/follower mode, leader_arms and follower_arms must use the same keys."
            )

        # Backward compatibility with the original single-arm implementation.
        self.piper_motors = self.follower_arms
        self.arm = self.follower_arms.get("main")
        if self.arm is None and len(self.follower_arms) == 1:
            self.arm = next(iter(self.follower_arms.values()))

        if self.multi_arm_mode or self.passive_recording_mode:
            self.teleop = None
        elif not self.inference_time:
            from lerobot.common.robot_devices.teleop.gamepad import SixAxisArmController

            self.teleop = SixAxisArmController()
        else:
            self.teleop = None

        self.logs = {}
        self.is_connected = False

    def _is_missing_realsense_error(self, exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "`serial_number` is expected to be one of these available cameras" in str(exc)

    def _format_missing_camera_summary(self) -> str:
        parts = []
        for name, info in self.unavailable_cameras.items():
            serial_number = info.get("serial_number", "unknown")
            reason = info.get("reason")
            if reason:
                parts.append(f"{name}({serial_number}): {reason}")
            else:
                parts.append(f"{name}({serial_number})")
        return ", ".join(parts)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = get_motor_names(self.follower_arms)
        state_names = get_motor_names(self.follower_arms)
        ee_pose_names = self._get_ee_pose_names(self.follower_arms)
        features = {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": (len(ee_pose_names),),
                "names": ee_pose_names,
            },
        }
        if self.passive_recording_mode:
            features["action.ee_pose"] = {
                "dtype": "float32",
                "shape": (len(ee_pose_names),),
                "names": ee_pose_names,
            }
            features.update(PASSIVE_TIMING_FEATURES)
        return features

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def _state_dict_to_tensor(self, arm, state_dict: dict[str, float]) -> torch.Tensor:
        return torch.as_tensor([state_dict[name] for name in arm.motors], dtype=torch.float32)

    def _ee_pose_dict_to_tensor(self, ee_pose_dict: dict[str, float]) -> torch.Tensor:
        return torch.as_tensor(
            [ee_pose_dict[axis] for axis in ("x", "y", "z", "rx", "ry", "rz")],
            dtype=torch.float32,
        )

    def _get_ee_pose_names(self, arms) -> list[str]:
        return [f"{arm_name}_{component}" for arm_name in arms for component in EE_POSE_COMPONENTS]

    def _capture_images(self) -> dict[str, torch.Tensor]:
        images = {}
        for camera in self.cameras.values():
            camera.start_async_worker()
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        return images

    def _read_arm_observation(
        self, arm_role: str, name: str, arm
    ) -> tuple[torch.Tensor, torch.Tensor]:
        before_read_t = time.perf_counter()
        if hasattr(arm, "read_observation"):
            observation = arm.read_observation()
            state_dict = observation["state"]
            ee_pose_dict = observation["ee_pose"]
            self.logs[f"read_{arm_role}_{name}_joint_feedback_hz"] = observation.get("joint_hz", 0.0)
            self.logs[f"read_{arm_role}_{name}_ee_pose_feedback_hz"] = observation.get("ee_pose_hz", 0.0)
            self.logs[f"read_{arm_role}_{name}_joint_timestamp_s"] = observation.get("joint_timestamp_s", 0.0)
            self.logs[f"read_{arm_role}_{name}_ee_pose_timestamp_s"] = observation.get("ee_pose_timestamp_s", 0.0)
        else:
            state_dict = arm.read()
            ee_pose_dict = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}

        state = self._state_dict_to_tensor(arm, state_dict)
        ee_pose = self._ee_pose_dict_to_tensor(ee_pose_dict)
        read_dt_s = time.perf_counter() - before_read_t
        self.logs[f"read_{arm_role}_{name}_dt_s"] = read_dt_s
        self.logs[f"read_{arm_role}_{name}_pos_dt_s"] = read_dt_s
        return state, ee_pose

    def _collect_arm_observation(self, arms, arm_role: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        arm_state = {}
        arm_ee_pose = {}
        for name, arm in arms.items():
            state, ee_pose = self._read_arm_observation(arm_role, name, arm)
            arm_state[name] = state
            arm_ee_pose[name] = ee_pose

        stacked_state = torch.cat([arm_state[name] for name in arms])
        stacked_ee_pose = torch.cat([arm_ee_pose[name] for name in arms])
        return arm_state, arm_ee_pose, stacked_state, stacked_ee_pose

    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )

        if self.multi_arm_mode:
            for name, arm in self.follower_arms.items():
                print(f"Connecting {name} follower arm.")
                arm.connect(enable=True)
            for name, arm in self.leader_arms.items():
                print(f"Connecting {name} leader arm.")
                arm.connect(enable=False)
        elif self.passive_recording_mode:
            for name, arm in self.follower_arms.items():
                can_name = getattr(arm, "can_name", "unknown_can")
                print(f"Preparing {name} follower arm for passive recording on {can_name}.")
                state = arm.wait_for_feedback(timeout_s=3.0)
                print(f"{name} ({can_name}) status_hz={arm.get_status_hz():.1f}")
                print(f"{name} initial_state={state}")
                if arm.get_status_hz() <= 0.0:
                    raise RuntimeError(
                        f"{name} on {can_name} did not receive valid CAN feedback during initialization. "
                        f"Check CAN wiring, power, and interface mapping."
                    )
        else:
            self.arm.connect(enable=True)
            print("piper connected")

        connected_cameras = {}
        available_cameras = {}
        self.unavailable_cameras = {}
        for name, camera in self.cameras.items():
            try:
                camera.connect()
                print(f"camera {name} connected")
                connected_cameras[name] = camera
                available_cameras[name] = camera
            except Exception as exc:
                if self._is_missing_realsense_error(exc):
                    serial_number = getattr(camera, "serial_number", None)
                    self.unavailable_cameras[name] = {
                        "serial_number": serial_number,
                        "reason": "not detected by the system",
                        "error": repr(exc),
                    }
                    print(
                        f"Warning: camera {name} ({serial_number}) is missing and will be skipped for this run."
                    )
                    continue

                for connected_camera in connected_cameras.values():
                    try:
                        connected_camera.disconnect()
                    except Exception:
                        pass
                raise

        # Start camera workers as soon as the pipelines are up so the first
        # recorded step can reuse frames that are already in flight.
        self.cameras = available_cameras
        if not self.cameras and self.configured_camera_names:
            raise RuntimeError(
                "None of the configured cameras are currently available. "
                f"Missing cameras: {self._format_missing_camera_summary()}."
            )

        for camera in self.cameras.values():
            camera.start_async_worker()

        if self.unavailable_cameras:
            print(
                "Warning: continuing without configured camera(s): "
                f"{self._format_missing_camera_summary()}."
            )

        self.is_connected = True
        print("All connected")

        # Legacy single-arm flow keeps the old reset-on-connect behavior.
        if not self.multi_arm_mode and not self.passive_recording_mode:
            self.run_calibration()

    def disconnect(self) -> None:
        if self.teleop is not None:
            self.teleop.stop()

        if self.multi_arm_mode:
            for name, arm in self.follower_arms.items():
                print(f"Moving {name} follower arm to safe position.")
                arm.safe_disconnect()

            if self.follower_arms:
                print("Disabling follower arms after 5 seconds")
                time.sleep(5)
                for arm in self.follower_arms.values():
                    arm.connect(enable=False)

            for arm in self.leader_arms.values():
                arm.connect(enable=False)
        elif self.passive_recording_mode:
            print("Passive recording mode: leaving follower arms untouched on disconnect.")
        else:
            self.arm.safe_disconnect()
            print("piper disable after 5 seconds")
            time.sleep(5)
            self.arm.connect(enable=False)

        for cam in self.cameras.values():
            if not getattr(cam, "is_connected", False):
                continue
            cam.disconnect()

        self.is_connected = False

    def run_calibration(self):
        if not self.is_connected:
            raise ConnectionError()

        for arm in self.follower_arms.values():
            arm.apply_calibration()

        if self.teleop is not None:
            self.teleop.reset()

    def _leader_follower_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        leader_pos, leader_ee_pose, _, _ = self._collect_arm_observation(self.leader_arms, "leader")

        follower_goal_pos = {}
        for name, arm in self.follower_arms.items():
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]
            follower_goal_pos[name] = goal_pos
            arm.write(goal_pos.tolist())
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        if not record_data:
            return

        _, _, state, ee_pose = self._collect_arm_observation(self.follower_arms, "follower")
        action = torch.cat([follower_goal_pos[name] for name in self.follower_arms])
        images = self._capture_images()

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        obs_dict["observation.ee_pose"] = ee_pose
        action_dict["action"] = action
        if self.passive_recording_mode:
            action_dict["action.ee_pose"] = torch.cat([leader_ee_pose[name] for name in self.leader_arms])
        for name, image in images.items():
            obs_dict[f"observation.images.{name}"] = image

        return obs_dict, action_dict

    def _read_follower_state(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        follower_pos, _, state, _ = self._collect_arm_observation(self.follower_arms, "follower")
        return follower_pos, state

    def capture_passive_record_observation(self) -> dict[str, torch.Tensor]:
        if not self.passive_recording_mode:
            raise RuntimeError("capture_passive_record_observation is only available in passive recording mode.")

        if not self.is_connected:
            raise ConnectionError()

        _, _, state, ee_pose = self._collect_arm_observation(self.follower_arms, "follower")
        images = self._capture_images()

        observation = {
            "observation.state": state,
            "observation.ee_pose": ee_pose,
        }
        for name, image in images.items():
            observation[f"observation.images.{name}"] = image
        return observation

    def _passive_record_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        observation = self.capture_passive_record_observation()
        state = observation["observation.state"]
        ee_pose = observation["observation.ee_pose"]

        if not record_data:
            return

        # No software action is sent in passive mode. Record the executed follower state
        # so the dataset still contains a valid action tensor.
        action = state.clone()
        obs_dict = dict(observation)
        action_dict = {
            "action": action,
            "action.ee_pose": ee_pose.clone(),
        }

        return obs_dict, action_dict

    def _single_arm_gamepad_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            self.teleop = SixAxisArmController()

        before_read_t = time.perf_counter()
        if hasattr(self.arm, "read_observation"):
            arm_observation = self.arm.read_observation()
            state_dict = arm_observation["state"]
            ee_pose = self._ee_pose_dict_to_tensor(arm_observation["ee_pose"])
        else:
            state_dict = self.arm.read()
            ee_pose = torch.zeros(6, dtype=torch.float32)
        action_dict = self.teleop.get_action()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        target_joints = list(action_dict.values())
        self.arm.write(target_joints)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return

        state = self._state_dict_to_tensor(self.arm, state_dict)
        action = torch.as_tensor(list(action_dict.values()), dtype=torch.float32)
        images = self._capture_images()

        obs_dict, action_out = {}, {}
        obs_dict["observation.state"] = state
        obs_dict["observation.ee_pose"] = ee_pose
        action_out["action"] = action
        for name, image in images.items():
            obs_dict[f"observation.images.{name}"] = image

        return obs_dict, action_out

    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.multi_arm_mode:
            return self._leader_follower_step(record_data=record_data)
        if self.passive_recording_mode:
            return self._passive_record_step(record_data=record_data)

        return self._single_arm_gamepad_step(record_data=record_data)

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        if self.multi_arm_mode or self.passive_recording_mode:
            action = action.to("cpu")
            from_idx = 0
            for arm in self.follower_arms.values():
                to_idx = from_idx + len(arm.motors)
                arm.write(action[from_idx:to_idx].tolist())
                from_idx = to_idx
            return action

        target_joints = action.tolist()
        self.arm.write(target_joints)
        return action

    def capture_observation(self) -> dict:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        if self.multi_arm_mode or self.passive_recording_mode:
            _, _, state, ee_pose = self._collect_arm_observation(self.follower_arms, "follower")
        else:
            before_read_t = time.perf_counter()
            if hasattr(self.arm, "read_observation"):
                arm_observation = self.arm.read_observation()
                state_dict = arm_observation["state"]
                ee_pose = self._ee_pose_dict_to_tensor(arm_observation["ee_pose"])
            else:
                state_dict = self.arm.read()
                ee_pose = torch.zeros(6, dtype=torch.float32)
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            state = self._state_dict_to_tensor(self.arm, state_dict)

        images = self._capture_images()

        obs_dict = {
            "observation.state": state,
            "observation.ee_pose": ee_pose,
        }
        for name, image in images.items():
            obs_dict[f"observation.images.{name}"] = image
        return obs_dict

    def teleop_safety_stop(self):
        if not self.multi_arm_mode and not self.passive_recording_mode:
            self.run_calibration()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
