"""Physical constants, workspace bounds and the semantic vocabulary.

Every number here was measured off the vendored menagerie MJCF
(`assets/so101/so101.xml`) rather than assumed -- see docs/DECISIONS.md for the
probe results that produced them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "so101"
ROBOT_XML = ASSETS_DIR / "so101.xml"

# --------------------------------------------------------------------------
# Robot description
# --------------------------------------------------------------------------
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)

N_ARM = len(ARM_JOINTS)          # 5 -- the SO-101 arm is 5-DoF, not 6
N_DOF = len(ALL_JOINTS)          # 6 including the gripper

TCP_SITE = "tcp_site"
SCENE_CAM = "scene_cam"
WRIST_CAM = "wrist_cam"

# Gripper joint angle -> jaw separation, measured on the model:
#   -0.17 -> 10.8 mm | 0.00 -> 21.7 mm | 0.50 -> 55.1 mm | 1.74 -> 120.5 mm
GRIPPER_OPEN = 0.9        # wide enough to clear a 25 mm cube with margin
GRIPPER_CLOSED = -0.10    # commands a squeeze against a 25 mm cube

# TCP site placement inside the `gripper` body.
# local +x is the jaw opening direction, local -z points out along the fingers.
# The site is rotated 180 deg about x so that the TCP frame reads as
#   z_tcp = approach direction (out of the fingers)
#   x_tcp = jaw opening direction
# The x offset is critical and was calibrated empirically, not guessed: the
# fixed jaw's contact pads sit at local x ~ -0.010, so the grasp centre must be
# at least CUBE_HALF *plus clearance* out in +x or the pad (and the jaw-tip
# capsule) spawns inside the cube, jams the descent and saturates shoulder_lift.
# Measured over 8 seeds: x=0.002 -> 3/8 success, x>=0.004 -> 8/8. See ISSUE-001.
TCP_LOCAL_POS = (0.005, 0.0, -0.082)
TCP_LOCAL_QUAT = (0.0, 1.0, 0.0, 0.0)

# --------------------------------------------------------------------------
# Scene layout. The table top surface is the z = 0 plane and the arm base is
# bolted to it, which keeps every workspace number in base coordinates.
# --------------------------------------------------------------------------
TABLE_Z = 0.0
TABLE_HALF = (0.45, 0.45, 0.02)
TABLE_LEG_Z = -0.75

# Objects and zones live in an annulus in front of the arm.
#
# These bounds are NOT the raw kinematic reach (0.478 m radially). The binding
# constraint is that a *straight-down* approach is only reachable over a narrow
# radius band, and that band shrinks as the TCP rises: the 5-DoF arm runs out of
# in-plane pitch authority. Measured convergence of the top-down IK over
# {grasp, mid, hover} heights and +/-45 deg of jaw yaw:
#     r 0.16-0.30, hover 0.070 ->  89.9 %
#     r 0.16-0.26, hover 0.070 -> 100.0 %
# so 0.16-0.26 is the honest usable workspace. See docs/DECISIONS.md D-004.
WORKSPACE_R_MIN = 0.16
WORKSPACE_R_MAX = 0.26
WORKSPACE_YAW = np.deg2rad(60.0)   # +/- about the +x axis
MIN_ENTITY_SEPARATION = 0.085      # centre-to-centre, objects and zones alike

CUBE_HALF = 0.0125                 # 25 mm cubes

# Trajectory heights, all in table coordinates (table top = z = 0).
#
# GRASP_Z is set by the *jaw tip*, not by the cube centre. The fixed jaw's tip
# geoms sit ~22 mm below the TCP, so aiming the TCP at the cube's mid-height
# (12.5 mm) buries the tip 5.7 mm inside the tabletop: the actuators saturate,
# the arm is shoved 6 mm back up, and the dragging tip knocks the cube away
# during the descent. Measured lowest-tip height vs GRASP_Z:
#     0.0165 -> -5.7 mm (buried, 2.94/2.94 N.m saturated, +6 mm sag)
#     0.0220 -> -0.2 mm (grazing)
#     0.0260 -> +3.7 mm (clear, no actuator load)
# 0.0235 clears the table by ~1.5 mm while the jaw pads still straddle the
# cube's upper half. See ISSUE-002.
GRASP_Z = 0.0235                   # TCP height when closing on a cube
HOVER_Z = 0.070                    # top of the reachable straight-down envelope
PLACE_Z = 0.0245                   # release height above the target zone
ZONE_RADIUS = 0.042                # flat target-zone discs
ZONE_HEIGHT = 0.0015

# --------------------------------------------------------------------------
# Control / recording
# --------------------------------------------------------------------------
SIM_TIMESTEP = 0.005               # from the MJCF: 200 Hz physics
CONTROL_HZ = 50.0                  # policy/actuator update rate
# Recording runs 1:1 with control (50 Hz, comfortably above the 30 Hz floor).
# Decoupling the two would force a resample of either the observation or the
# action stream, which is exactly how an off-by-one action lag gets introduced.
RECORD_HZ = CONTROL_HZ
PHYSICS_SUBSTEPS = 4               # 200 Hz physics / 50 Hz control, exact
IMAGE_SIZE = (240, 320)            # (H, W) for both cameras

# --------------------------------------------------------------------------
# Semantic vocabulary for instruction generation
# --------------------------------------------------------------------------
OBJECT_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.12, 0.12, 1.0),
    "blue": (0.12, 0.25, 0.85, 1.0),
    "yellow": (0.92, 0.82, 0.10, 1.0),
    "orange": (0.95, 0.48, 0.06, 1.0),
}
ZONE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "green": (0.15, 0.70, 0.25, 0.85),
    "purple": (0.55, 0.20, 0.75, 0.85),
    "cyan": (0.10, 0.75, 0.80, 0.85),
    "pink": (0.95, 0.45, 0.70, 0.85),
}
OBJECT_SHAPES = ("cube",)
ZONE_SHAPES = ("circle",)

# Home must park the TCP well ABOVE the tabletop. The obvious-looking
# [0, -0.35, 0.9, 1.0, 0] rest pose puts the TCP at z = 0.023 -- the gripper
# literally sits on the table inside the object annulus, so the very first
# motion of every episode dragged the jaws through the cubes. This pose parks
# the TCP at (0.198, 0, 0.233), clear of everything. See ISSUE-003.
HOME_QPOS = np.array([0.0, -1.2, 0.4, 1.2, 0.0, GRIPPER_OPEN])
