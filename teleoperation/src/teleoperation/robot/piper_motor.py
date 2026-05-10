import math
import time
from typing import Any, Dict

from piper_sdk import *

from teleoperation.robot.motor_config import PiperMotorsBusConfig

class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.can_name = config.can_name
        self.piper = C_PiperInterface_V2(config.can_name)
        # Start the CAN read threads, but skip the eager PiperInit query burst.
        # On freshly activated CAN interfaces these startup queries often fail
        # before feedback traffic becomes stable.
        self.piper.ConnectPort(piper_init=False)
        self.motors = config.motors
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）
        self.pose_position_scale_m = 1e-6  # SDK pose position is reported in 0.001 mm
        self.pose_rotation_scale_rad = math.pi / 180_000.0  # SDK pose rotation is reported in 0.001 degrees
        self.move_spd_rate_ctrl = 50
        self._joint_mode_configured = False

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable: bool = True) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 (-92000~92000) / 57324.840764
                joint_2 (float): 关节2角度 -1300 ~ 90000 / 57324.840764
                joint_3 (float): 关节3角度 2400 ~ -80000 / 57324.840764
                joint_4 (float): 关节4角度 -90000~90000 / 57324.840764
                joint_5 (float): 关节5角度 19000~-77000 / 57324.840764
                joint_6 (float): 关节6角度 -90000~90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.joint_factor)
        joint_1 = round(target_joint[1]*self.joint_factor)
        joint_2 = round(target_joint[2]*self.joint_factor)
        joint_3 = round(target_joint[3]*self.joint_factor)
        joint_4 = round(target_joint[4]*self.joint_factor)
        joint_5 = round(target_joint[5]*self.joint_factor)
        gripper_range = round(target_joint[6]*1000*1000)

        if not self._joint_mode_configured:
            self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate_ctrl, 0x00) # joint control
            self._joint_mode_configured = True
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°
    
    def _joint_state_from_msgs(self, joint_msg, gripper_msg) -> Dict[str, float]:
        joint_state = joint_msg.joint_state
        gripper_state = gripper_msg.gripper_state
        return {
            "joint_1": joint_state.joint_1 / self.joint_factor,
            "joint_2": joint_state.joint_2 / self.joint_factor,
            "joint_3": joint_state.joint_3 / self.joint_factor,
            "joint_4": joint_state.joint_4 / self.joint_factor,
            "joint_5": joint_state.joint_5 / self.joint_factor,
            "joint_6": joint_state.joint_6 / self.joint_factor,
            "gripper": gripper_state.grippers_angle / 1_000_000,
        }

    def _ee_pose_from_msg(self, end_pose_msg) -> Dict[str, float]:
        end_pose = end_pose_msg.end_pose
        return {
            "x": float(end_pose.X_axis) * self.pose_position_scale_m,
            "y": float(end_pose.Y_axis) * self.pose_position_scale_m,
            "z": float(end_pose.Z_axis) * self.pose_position_scale_m,
            "rx": float(end_pose.RX_axis) * self.pose_rotation_scale_rad,
            "ry": float(end_pose.RY_axis) * self.pose_rotation_scale_rad,
            "rz": float(end_pose.RZ_axis) * self.pose_rotation_scale_rad,
        }

    def read(self) -> Dict:
        """
            返回与 write 对齐的控制量:
            - 6 个关节: 弧度
            - gripper: 0~0.08 的张开量
        """
        joint_msg = self.piper.GetArmJointMsgs()
        gripper_msg = self.piper.GetArmGripperMsgs()
        return self._joint_state_from_msgs(joint_msg, gripper_msg)

    def read_ee_pose(self) -> Dict[str, float]:
        """Read the current end-effector pose reported by the official Piper SDK."""
        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        return self._ee_pose_from_msg(end_pose_msg)

    def read_observation(self) -> Dict[str, Any]:
        """
            Read the current joint state, end-effector pose and SDK timing metadata.
        """
        joint_msg = self.piper.GetArmJointMsgs()
        gripper_msg = self.piper.GetArmGripperMsgs()
        end_pose_msg = self.piper.GetArmEndPoseMsgs()
        return {
            "state": self._joint_state_from_msgs(joint_msg, gripper_msg),
            "ee_pose": self._ee_pose_from_msg(end_pose_msg),
            "joint_timestamp_s": float(getattr(joint_msg, "time_stamp", 0.0)),
            "joint_hz": float(getattr(joint_msg, "Hz", 0.0)),
            "gripper_timestamp_s": float(getattr(gripper_msg, "time_stamp", 0.0)),
            "gripper_hz": float(getattr(gripper_msg, "Hz", 0.0)),
            "ee_pose_timestamp_s": float(getattr(end_pose_msg, "time_stamp", 0.0)),
            "ee_pose_hz": float(getattr(end_pose_msg, "Hz", 0.0)),
        }

    def get_status_hz(self) -> float:
        try:
            return float(self.piper.GetArmStatus().Hz)
        except Exception:
            return 0.0

    def wait_for_feedback(self, timeout_s: float = 3.0, min_hz: float = 1.0) -> Dict:
        deadline = time.time() + timeout_s
        last_state = self.read()
        while time.time() < deadline:
            last_state = self.read()
            if self.get_status_hz() >= min_hz:
                return last_state
            time.sleep(0.1)
        return last_state
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)
