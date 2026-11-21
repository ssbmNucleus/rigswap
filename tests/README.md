# Running tests

Install the development dependencies with `python -m pip install -e .[dev]`.
Use the Python environment containing `bpy` (Python 3.13, Blender 5.2.2).
When using a shared environment from a worktree, set `PYTHONPATH` to that worktree
so subprocesses test the same checkout.

- Per commit: `pytest -m fast`.
- Before merge: `pytest -m 'not fast' -n 4` with the environments below.
- The full set is both commands. `pytest -n 4` runs both together.

## Markers

| Marker | Meaning |
| --- | --- |
| `fast` | No bpy, Blender subprocess, or real game files; measured below one second per test. |
| `blender` | Uses bpy, including through helpers or the module under test, or starts Blender. |
| `plugin` | Needs real plugin files, prepared targets, or an external plugin tool. Synthetic plugin tests do not need this marker. |
| `slow` | Measured above five seconds, counting setup, call and teardown together. |

Resource and duration markers overlap. `not fast` includes **all** other tests,
including tests taking one to five seconds; it is broader than `slow`.
Collection rules live in `conftest.py` files; mixed modules mark individual tests.
Mark Blender calls through helpers explicitly with `@pytest.mark.blender`. Mark inline
plugin-file/tool prerequisites with `@pytest.mark.plugin`, and measured long tests
with `@pytest.mark.slow`. Other tests enter `fast` by default, apart from the three
AI tests that exercise real waits and process scheduling.
Keep new resource-dependent tests marked even when their dependencies are absent.
Markers select tests; existing missing-dependency skips still apply.
Use `pytest --durations=0 -q tests` to refresh timings when tests change.

## Plugin environments

Set paths to your own legally obtained files and prepared targets. In PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:RIGSWAP_MELEE_FILES = '<folder of vanilla v1.02 fighter files>'
$env:RIGSWAP_MELEE_TARGETS = '<folder of prepared targets>'
pytest -m 'not fast' -n 4
```

Also configure `RIGSWAP_BLENDER4` and `RIGSWAP_HSDRAW` if their default installations
are unavailable. Keep generated files on a disk with room with `--basetemp <scratch folder>/test-temp/<run>`.
An editable installation from another checkout can resolve removed modules there;
use an isolated environment with this checkout and the same installed dependencies when testing deletions.

## Parallel isolation

Each xdist worker has its own Python process and in-process bpy instance.
Before collection, `tests/conftest.py` points `TMP`, `TEMP`, `TMPDIR` and Python's
cached temp directory at that worker's unique pytest base directory. Child tools
inherit those paths. Keep generated files under `tmp_path` or `tmp_path_factory`;
never use the workers' shared parent directory for mutable test data.

AI tests replace both the default sending directory and the active sending locks
for every test. Module-level caches remain process-local; tests that monkeypatch
cache inputs must restore or replace those caches as well. The deadline test
advances a fake clock between keep-alives, so CPU contention cannot change its result.
