import tempfile

import pytest

from rigswap import ai


# These exercise real waits/process scheduling. They belong before merge even when
# a particular run happens to squeeze below one second.
_REAL_WAITS = {
    'test_the_turns_to_send_hold_across_processes',
    'test_only_some_asks_send_at_once_and_the_waits_overlap',
    'test_a_second_deadline_fails',
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in {'test_describe_step.py', 'test_swap.py'}:
            item.add_marker(pytest.mark.blender)
        if (not any(item.get_closest_marker(m) for m in ('blender', 'plugin', 'slow'))
                and item.originalname not in _REAL_WAITS):
            item.add_marker(pytest.mark.fast)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    """Set native/Python scratch roots before collection can import Blender.

    xdist gives each worker a different basetemp, including across concurrent runs.
    Subprocesses inherit these variables, so Blender 4 and the plugin tools use it too.
    """
    config = session.config
    if getattr(config.option, 'numprocesses', 0) and not hasattr(config, 'workerinput'):
        return
    folder = config._tmp_path_factory.getbasetemp() / 'native'
    folder.mkdir()
    patch = pytest.MonkeyPatch()
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        patch.setenv(name, str(folder))
    patch.setattr(tempfile, 'tempdir', str(folder))
    config.add_cleanup(patch.undo)


@pytest.fixture(autouse=True)
def own_turns_to_send(tmp_path_factory, monkeypatch):
    """A test's asks send nothing real, so they take their turns to send apart from the machine's (ai.Turns): never
    waiting on the turns real runs on this machine hold while they send their pictures."""
    # tmp_path_factory is local to an xdist worker, never the workers' shared parent.
    folder = tmp_path_factory.mktemp('turns')
    monkeypatch.setattr(ai, 'TURNS_DIR', folder)
    monkeypatch.setattr(ai, '_sending', ai.Turns(ai.SENDING_AT_ONCE, folder))
