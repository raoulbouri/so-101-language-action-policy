"""Language-conditioned ACT for the SO-101 dataset.

Kept separate from `so101_sim` (the data-generation pipeline) on purpose: this
package only ever *reads* the HDF5 produced there, so the two can be developed
and tested independently.
"""

# Select the MuJoCo GL backend BEFORE anything can `import mujoco`.
#
# This has to live in the package __init__ rather than in rollout.py: import
# sorting puts `import mujoco` (third-party) above `from so101_sim...` (first-
# party), so by the time so101_sim/__init__.py ran its own selector, mujoco had
# already resolved its backend and the setting was ignored. A package __init__
# always executes before any of its submodules, so this is the earliest hook we
# control. See ISSUE-015.
from so101_sim import _gl as _gl

__version__ = "0.1.0"
