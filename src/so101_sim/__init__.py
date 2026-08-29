"""Language-conditioned ACT data generation for the SO-101 arm in MuJoCo."""

# Must precede any `import mujoco` in this process: on a headless Linux box the
# default GLFW backend has no display and every render call fails. See _gl.py.
from . import _gl as _gl

__version__ = "0.1.0"
