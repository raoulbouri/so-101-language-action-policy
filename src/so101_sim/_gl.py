"""Pick a MuJoCo OpenGL backend that works on the current machine.

MuJoCo defaults to GLFW, which needs an X11 display. On a headless GPU server
there is none, and every render call dies with

    mujoco.FatalError: an OpenGL platform library has not been loaded

The fix is EGL, which renders on the GPU without a display server. This module
selects it automatically and must run *before* `import mujoco` anywhere in the
process -- hence it is imported at the top of `so101_sim/__init__.py`.

An explicit MUJOCO_GL in the environment always wins, so anyone who knows better
(osmesa on a CPU-only box, for example) can override.
"""

from __future__ import annotations

import os
import platform


def select_backend() -> str:
    """Set MUJOCO_GL if unset, and return the backend in force."""
    existing = os.environ.get("MUJOCO_GL", "").strip()
    if existing:
        return existing

    # macOS renders through its own CGL backend; leave it alone.
    if platform.system() == "Darwin":
        return "default (macOS CGL)"

    # Linux: always prefer EGL.
    #
    # An earlier version only chose EGL when DISPLAY was unset. That is wrong:
    # sshing with X11 forwarding SETS DISPLAY without providing a usable X
    # server, so GLFW was selected and every render failed -- and whether it
    # failed depended on how you happened to connect, which made it flaky
    # rather than merely broken. EGL renders on the GPU and works both headless
    # and with a display, so it is the right default on any Linux GPU box.
    # An explicit MUJOCO_GL still wins (see above) for the CPU-only osmesa case.
    if platform.system() == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
        return "egl"

    return "default (glfw)"


def rendering_available() -> bool:
    """True if an offscreen MuJoCo context can actually be created.

    Used to skip render-dependent tests rather than fail them: a headless box
    without EGL can still train perfectly well, because training reads the HDF5
    and never renders.
    """
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>"
        )
        renderer = mujoco.Renderer(model, height=16, width=16)
        renderer.close()
        return True
    except Exception:  # noqa: BLE001 - any GL failure means unavailable
        return False


BACKEND = select_backend()
