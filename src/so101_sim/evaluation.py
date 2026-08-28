"""Binary success validation.

The directive's rule is strict: the object passes only if its *entire* base
footprint lies inside the target zone perimeter. We evaluate that exactly, on
the cube's four rotated base corners, rather than approximating with a centre
distance -- and we additionally require the cube to be resting on the table and
at rest, so a cube still pinched in the jaws or mid-bounce cannot score a pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CUBE_HALF, TABLE_Z, ZONE_RADIUS


@dataclass
class SuccessReport:
    success: bool
    reason: str
    center_distance: float          # zone centre -> cube centre, in xy
    max_corner_distance: float      # zone centre -> furthest base corner, in xy
    cube_height: float
    cube_speed: float
    center_in_zone: bool            # the lenient metric, reported alongside

    def to_dict(self) -> dict:
        return {
            "success": bool(self.success),
            "reason": self.reason,
            "center_distance": float(self.center_distance),
            "max_corner_distance": float(self.max_corner_distance),
            "cube_height": float(self.cube_height),
            "cube_speed": float(self.cube_speed),
            "center_in_zone": bool(self.center_in_zone),
        }


def base_corners_xy(center_xy: np.ndarray, yaw: float, half: float = CUBE_HALF) -> np.ndarray:
    """The four xy corners of the cube's base square, rotated by `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    local = np.array([[+half, +half], [+half, -half], [-half, -half], [-half, +half]])
    return center_xy[None, :] + local @ rot.T


def evaluate_placement(
    cube_pos: np.ndarray,
    cube_yaw: float,
    cube_vel: np.ndarray,
    zone_xy: np.ndarray,
    *,
    zone_radius: float = ZONE_RADIUS,
    height_tol: float = 0.008,
    speed_tol: float = 0.03,
) -> SuccessReport:
    cube_pos = np.asarray(cube_pos, float)
    zone_xy = np.asarray(zone_xy, float)

    center_d = float(np.linalg.norm(cube_pos[:2] - zone_xy))
    corners = base_corners_xy(cube_pos[:2], cube_yaw)
    max_corner_d = float(np.linalg.norm(corners - zone_xy[None, :], axis=1).max())
    speed = float(np.linalg.norm(cube_vel))
    height = float(cube_pos[2])

    resting_z = TABLE_Z + CUBE_HALF
    center_in_zone = center_d <= zone_radius

    if abs(height - resting_z) > height_tol:
        return SuccessReport(False, "cube not resting on the table", center_d, max_corner_d,
                             height, speed, center_in_zone)
    if speed > speed_tol:
        return SuccessReport(False, "cube still moving", center_d, max_corner_d,
                             height, speed, center_in_zone)
    if max_corner_d > zone_radius:
        return SuccessReport(False, "footprint crosses the zone perimeter", center_d,
                             max_corner_d, height, speed, center_in_zone)
    return SuccessReport(True, "ok", center_d, max_corner_d, height, speed, center_in_zone)
