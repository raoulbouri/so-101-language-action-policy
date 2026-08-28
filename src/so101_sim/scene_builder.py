"""Programmatic MJCF scene construction.

The vendored menagerie MJCF is parsed as XML and *modified in memory* for every
episode. This gives us two things a static `scene.xml` cannot:

1. the TCP site can be injected into the `gripper` body, which an `<include>`
   is not able to do;
2. object/zone count, colour and pose are properties of the seed rather than of
   a file on disk.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco

from .constants import (
    ASSETS_DIR,
    CUBE_HALF,
    IMAGE_SIZE,
    ROBOT_XML,
    SCENE_CAM,
    TABLE_HALF,
    TABLE_LEG_Z,
    TCP_LOCAL_POS,
    TCP_LOCAL_QUAT,
    TCP_SITE,
    ZONE_HEIGHT,
    ZONE_RADIUS,
)
from .randomization import EpisodeSpec


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _add_tcp_site(root: ET.Element) -> None:
    gripper = root.find(".//body[@name='gripper']")
    if gripper is None:
        raise RuntimeError("could not find the 'gripper' body in the SO-101 MJCF")
    ET.SubElement(
        gripper,
        "site",
        {
            "name": TCP_SITE,
            "pos": _fmt(TCP_LOCAL_POS),
            "quat": _fmt(TCP_LOCAL_QUAT),
            "size": "0.004",
            "rgba": "1 0 1 0.6",
            "group": "3",
        },
    )


def _add_table_and_lighting(worldbody: ET.Element, asset: ET.Element) -> None:
    ET.SubElement(
        asset,
        "texture",
        {"type": "skybox", "builtin": "gradient", "rgb1": "0.35 0.45 0.6",
         "rgb2": "0.05 0.06 0.1", "width": "512", "height": "512"},
    )
    ET.SubElement(
        asset,
        "texture",
        {"type": "2d", "name": "tabletex", "builtin": "checker", "rgb1": "0.62 0.60 0.56",
         "rgb2": "0.55 0.53 0.50", "width": "300", "height": "300"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "tablemat", "texture": "tabletex", "texuniform": "true",
         "texrepeat": "8 8", "reflectance": "0.05"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "floormat", "rgba": "0.18 0.19 0.22 1", "reflectance": "0.0"},
    )

    # castshadow is off on both lights: the default shadow map is far too coarse
    # for a 25 mm cube and paints blotchy shadow acne across the tabletop, which
    # would otherwise be baked straight into the training images.
    ET.SubElement(worldbody, "light", {"pos": "0.3 0 1.4", "dir": "0 0 -1",
                                       "directional": "true", "castshadow": "false",
                                       "diffuse": "0.6 0.6 0.6"})
    ET.SubElement(worldbody, "light", {"pos": "0.15 -0.5 0.9", "dir": "0.05 0.5 -1",
                                       "directional": "false", "castshadow": "false",
                                       "diffuse": "0.35 0.35 0.35"})
    # Floor well below the table so it never participates in manipulation contacts.
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "pos": f"0 0 {TABLE_LEG_Z}",
                                      "size": "3 3 0.05", "material": "floormat",
                                      "contype": "0", "conaffinity": "0"})
    # Table slab: top face sits exactly on z = 0.
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "table", "type": "box", "pos": f"0.12 0 {-TABLE_HALF[2]:.6g}",
         "size": _fmt(TABLE_HALF), "material": "tablemat",
         "condim": "3", "friction": "1 0.005 0.0001"},
    )


def _add_scene_camera(worldbody: ET.Element) -> None:
    """Wide workspace view, framed on the reachable annulus rather than the base.

    Aiming at the robot base spent most of the frame on empty table with the arm
    and cubes in a small central patch. The camera targets a massless body at the
    centre of the workspace instead, close enough that the annulus fills the shot.
    """
    target = ET.SubElement(worldbody, "body", {"name": "workspace_center",
                                               "pos": "0.21 0 0.03", "mocap": "true"})
    ET.SubElement(target, "site", {"name": "workspace_center_site", "size": "0.001",
                                   "rgba": "0 0 0 0", "group": "4"})
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": SCENE_CAM,
            "mode": "targetbody",
            "target": "workspace_center",
            "pos": "0.44 -0.34 0.33",
            "fovy": "52",
            "resolution": f"{IMAGE_SIZE[1]} {IMAGE_SIZE[0]}",
        },
    )


def _add_entities(worldbody: ET.Element, spec: EpisodeSpec) -> None:
    for zone in spec.zones:
        # Flat, non-colliding target discs: they mark a region, they are not
        # something the arm or the cubes can push against.
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": zone.name,
                "type": "cylinder",
                "pos": f"{zone.pos[0]:.6g} {zone.pos[1]:.6g} {ZONE_HEIGHT / 2:.6g}",
                "size": f"{ZONE_RADIUS:.6g} {ZONE_HEIGHT / 2:.6g}",
                "rgba": _fmt(zone.rgba),
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

    for obj in spec.objects:
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": obj.name,
                "pos": f"{obj.pos[0]:.6g} {obj.pos[1]:.6g} {CUBE_HALF:.6g}",
                "euler": f"0 0 {obj.yaw:.6g}",
            },
        )
        ET.SubElement(body, "freejoint", {"name": f"{obj.name}_free"})
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.name}_geom",
                "type": "box",
                "size": _fmt((CUBE_HALF,) * 3),
                "rgba": _fmt(obj.rgba),
                "mass": "0.035",
                "condim": "6",
                "friction": "1.2 0.02 0.002",
                "solref": "0.008 1",
                "priority": "0",
            },
        )
        ET.SubElement(body, "site", {"name": f"{obj.name}_center", "size": "0.002",
                                     "rgba": "1 1 1 0.0", "group": "4"})


def build_scene_xml(spec: EpisodeSpec) -> str:
    """Return a complete MJCF string for this episode."""
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    # meshdir is relative to the source XML; we compile from a string, so make
    # it absolute.
    compiler.set("meshdir", str(ASSETS_DIR / "assets"))

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("unexpected SO-101 MJCF layout")

    _add_tcp_site(root)
    _add_table_and_lighting(worldbody, asset)
    _add_scene_camera(worldbody)
    _add_entities(worldbody, spec)

    return ET.tostring(root, encoding="unicode")


def build_model(spec: EpisodeSpec) -> mujoco.MjModel:
    """Compile the episode scene into an `MjModel`."""
    return mujoco.MjModel.from_xml_string(build_scene_xml(spec))
