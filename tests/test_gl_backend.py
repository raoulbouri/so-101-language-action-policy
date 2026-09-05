"""The MuJoCo GL backend must be selected before anything imports mujoco.

This ordering has broken twice (ISSUE-012, ISSUE-015) and both times the symptom
was a silent rendering failure at *evaluation* time -- after hours of training --
rather than an error at import. These tests run in subprocesses so each starts
from a clean module table.
"""

import subprocess
import sys
import textwrap


def _run(code: str) -> str:
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_gl_selector_runs_before_mujoco_is_imported():
    """Importing so101_act must set the backend while mujoco is still unloaded."""
    out = _run("""
        import sys
        import so101_act
        print('gl' , 'so101_sim._gl' in sys.modules)
        print('mj' , 'mujoco' in sys.modules)
    """)
    assert "gl True" in out, "GL selector did not run on `import so101_act`"
    assert "mj False" in out, (
        "mujoco was imported before the GL backend was chosen -- the backend "
        "setting is then ignored. See ISSUE-015."
    )


def test_gl_selector_runs_before_mujoco_via_so101_sim():
    out = _run("""
        import sys
        import so101_sim
        print('gl' , 'so101_sim._gl' in sys.modules)
        print('mj' , 'mujoco' in sys.modules)
    """)
    assert "gl True" in out
    assert "mj False" in out


def test_rollout_import_still_yields_a_usable_backend():
    """After the full rollout import chain, a backend must be in force."""
    out = _run("""
        import so101_act.rollout  # noqa
        from so101_sim._gl import BACKEND
        print('backend', BACKEND)
    """)
    assert "backend" in out
    assert "None" not in out
