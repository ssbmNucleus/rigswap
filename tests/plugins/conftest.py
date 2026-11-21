from pathlib import Path

import pytest


_BLENDER_MODULES = {
    'melee/prep/test_join.py',
    'melee/prep/test_layers.py',
    'melee/prep/test_normals.py',
    'melee/prep/test_prep.py',
    'melee/prep/test_triangles.py',
    'melee/prep/test_twins.py',
    'brawl/prep/test_prep.py',
    'brawl/prep/test_textures.py',
    'vrm/prep/test_prep.py',
}


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(items):
    root = Path(__file__).parent
    for item in items:
        if not item.path.is_relative_to(root):
            continue
        if item.path.relative_to(root).as_posix() in _BLENDER_MODULES:
            item.add_marker(pytest.mark.blender)
        # These modules already declare their external prerequisites as skipif.
        # Mark regardless of the condition: collection must agree on every worker,
        # and absent assets must not silently put an integration test in fast.
        for mark in item.iter_markers('skipif'):
            reason = mark.kwargs.get('reason', '')
            if 'Blender 4' in reason:
                item.add_marker(pytest.mark.blender)
            if any(word in reason for word in ('RIGSWAP_', 'files', 'targets', 'worker', 'add-on')):
                item.add_marker(pytest.mark.plugin)
    yield
