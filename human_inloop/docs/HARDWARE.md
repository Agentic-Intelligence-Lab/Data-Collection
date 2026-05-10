# Hardware Guide

This workflow controls a real Piper arm. Validate software first, then collect only with a clear workspace and an operator ready to stop the robot.

## Piper CAN Interface

The active arm is selected with `PIPER_CAN_NAME`, usually `can1` for this local setup.

```bash
export PIPER_CAN_NAME=can1
ip link show "$PIPER_CAN_NAME"
```

Bring up the CAN interface according to your robot interface documentation before running collection. `scripts/validate_environment.py` checks that the network interface exists but does not send robot commands.

## Camera Layout

Default camera keys:

- `head`
- `left_wrist`
- `right_wrist`
- `front_view`

Default local serials are kept as environment-configurable values in `.env.example`. They are examples for this rig, not universal values. Update these before collecting on another machine:

- `HEAD_CAMERA_SERIAL`
- `LEFT_WRIST_CAMERA_SERIAL`
- `RIGHT_WRIST_CAMERA_SERIAL`
- `FRONT_VIEW_CAMERA_SERIAL`

You can also provide a full JSON camera layout through `ROBOT_CAMERAS_JSON`.

## Safety Checklist Before Collection

- Robot is powered and mechanically unobstructed.
- Emergency stop is reachable and tested.
- The follower arm can move through the task area without collision.
- The master arm/follower connection procedure is understood by the operator.
- Cables do not limit or pull the arm during correction.
- All cameras stream reliably at the configured resolution and FPS.
- CAN interface name matches `PIPER_CAN_NAME`.
- `ACTION_SCALE` and `MAX_ABS_DELTA` are conservative for the first run.
- Start with `ENABLE_ARM=false` or `DRY_RUN=true` when validating a new setup.

## Human Correction Flow

1. Let the model run in `MODEL ROLLOUT`.
2. Press or click `i Intervention` to pause policy control.
3. Connect the master arm according to your hardware procedure.
4. Click `Start Human Correction`.
5. Perform the correction and complete the task.
6. Save success or failure.
7. Follow the prompt to disconnect the master arm before policy control resumes if your setup requires it.
8. Reset the scene and start the next rollout.

If your gripper teleoperation path is stable without disconnecting the master/follower link, the software does not enforce disconnection. Treat the prompt as a safety guardrail for policy-control handoff.

## Non-Motion Modes

- `DRY_RUN=true`: runs the loop without sending policy actions to the robot after observations are captured.
- `VALIDATE_ONLY=true`: loads the policy and runs a dummy inference, then exits.
- `ENABLE_ARM=false`: connects without explicitly enabling the arm from this script.

These modes are useful for setup checks, but they do not replace physical safety checks.
