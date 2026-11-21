"""Collection rules shared by the geometry test directories."""
from pathlib import Path

import pytest


def mark_direct_blender(items, directory):
    """Direct imports are automatic; helper/module dependencies need explicit marks."""
    directory = Path(directory)
    for item in items:
        if not item.path.is_relative_to(directory):
            continue
        code = item.obj.__code__
        if (set(code.co_names) & {'bpy', 'bmesh', 'mathutils'}
                or ('importorskip' in code.co_names and 'bpy' in code.co_consts)):
            item.add_marker(pytest.mark.blender)
