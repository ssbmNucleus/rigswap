import json

import pytest

from rigswap import ai
from rigswap.core import run
from rigswap.load import open as load
from rigswap.plugins import Failed
from rigswap.plugins import melee
from rigswap.plugins.melee import costume
from rigswap.plugins.melee.facts.fighters import in_scope

READY = costume.ready


@pytest.fixture(autouse=True)
def targets_made(monkeypatch):
    """The targets are there: make's own steps are what these tests are about (``ready`` has tests of its own)."""
    monkeypatch.setattr(costume, 'ready', lambda fighter: None)


def _made(folder, name):
    (folder / 'frames' / name).mkdir(parents=True)
    for end in costume.TARGET_FILES:
        (folder / f'{name}{end}').write_text('{}', encoding='utf-8')


def _game_files(folder, but=()):
    """Every in-scope fighter's game files in ``folder`` (empty files), but those named."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in in_scope():
        for name in (f'Pl{f.code}Nr.dat', f'Pl{f.code}.dat', f'Pl{f.code}AJ.dat'):
            if name not in but:
                (folder / name).write_text('', encoding='utf-8')
    return folder


def test_ready_makes_only_the_targets_not_made_yet_the_fighter_or_every_one_for_auto(monkeypatch, tmp_path):
    monkeypatch.setenv(melee.TARGETS_VAR, str(tmp_path / 'targets'))
    monkeypatch.setenv(melee.FILES_VAR, str(_game_files(tmp_path / 'files')))
    made = []

    def prepare(names=None):
        made.extend(names)
        for name in names:
            _made(tmp_path / 'targets', name)
        return {name: 'made' for name in names}
    monkeypatch.setattr(melee, 'prepare', prepare)
    monkeypatch.setattr(melee, 'describe_from_tables', lambda name, folder: made.append(f'described {name}'))
    READY('Mario')
    assert made == ['Mario', 'described Mario']
    READY('Mario')
    assert len(made) == 2, 'made once per machine'
    READY(costume.AUTO)
    others = [f.name for f in in_scope() if f.name != 'Mario']
    assert [m for m in made[2:] if not m.startswith('described')] == others


def test_an_import_naming_no_fighter_is_offered_those_whose_game_files_are_there(monkeypatch, tmp_path):
    """An app that saves no costume for the Ice Climbers gives no animations of theirs. Getting every fighter in scope
    ready, an import naming no fighter failed there on Popo (2026-11-10: PlPpAJ.dat is not in the app's files), in
    every build since Popo and Nana became targets. Nana plays Popo's animations, so she goes with him."""
    files = _game_files(tmp_path / 'files', but=('PlPpAJ.dat',))
    names = [f.name for f in costume.offered(files)]
    assert 'Popo' not in names and 'Nana' not in names
    assert names == [f.name for f in in_scope() if f.name not in ('Popo', 'Nana')] and 'Mario' in names
    monkeypatch.setenv(melee.TARGETS_VAR, str(tmp_path / 'targets'))
    monkeypatch.setenv(melee.FILES_VAR, str(files))
    made = []

    def prepare(names=None):
        made.extend(names)
        for name in names:
            _made(tmp_path / 'targets', name)
        return {name: 'made' for name in names}
    monkeypatch.setattr(melee, 'prepare', prepare)
    monkeypatch.setattr(melee, 'describe_from_tables', lambda name, folder: None)
    READY(costume.AUTO)
    assert made == names
    monkeypatch.setenv(melee.FILES_VAR, str(tmp_path / 'empty'))
    (tmp_path / 'empty').mkdir()
    with pytest.raises(Failed, match='no fighter has its game files'):
        READY(costume.AUTO)


def test_ready_fails_naming_the_fighter_it_could_not_make(monkeypatch, tmp_path):
    monkeypatch.setenv(melee.TARGETS_VAR, str(tmp_path))
    monkeypatch.setattr(melee, 'prepare', lambda names=None: {names[0]: 'failed: PlMrNr.dat is not in files'})
    with pytest.raises(Failed, match='Mario could not be made ready from the game files: failed: PlMrNr.dat'):
        READY('Mario')


def test_a_target_that_cannot_be_made_is_said_as_its_own_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(load, 'run', lambda download, out: None)

    def cannot(fighter):
        raise Failed('Mario could not be made ready from the game files')
    monkeypatch.setattr(costume, 'ready', cannot)
    result = costume.make('model.zip', 'Mario', tmp_path)
    assert result['status'] == 'failed' and result['failed']['stage'] == 'ready'


def test_a_costume_is_loaded_swapped_and_exported_and_its_asks_cost_is_totalled(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(load, 'run', lambda download, out: asked.append(('load', download, out)))
    monkeypatch.setattr(ai, 'ask', lambda q, p, s, d, cache_dir=None, route=None: (d, {'cost': .01}))

    def swap(load_dir, fighter, out, ask, name=None):
        ask('q', [], {}, 'a', None)
        ask('q', [], {}, 'b', None)
        return {'fighter': fighter, 'export': str(out / '4-export' / f'{fighter}.dat')}
    monkeypatch.setattr(run, 'swap', swap)
    result = costume.make('model.zip', 'Mario', tmp_path)
    assert asked == [('load', 'model.zip', tmp_path / '1-load')]
    assert result['status'] == 'done' and result['dat'].endswith('Mario.dat')
    assert result['asks'] == 2 and abs(result['cost_usd'] - .02) < 1e-9
    assert json.loads((tmp_path / 'result.json').read_text(encoding='utf-8')) == result


def test_through_the_relay_an_import_and_its_changes_ask_in_its_own_cache_whatever_cache_is_named(monkeypatch,
                                                                                               tmp_path):
    """The app names one cache for every import; through the relay each import's answers are its own: a second
    import of the same file by another assistant is asked afresh, and a change to an import reads that import's."""
    caches = []
    monkeypatch.setattr(ai, 'ask', lambda q, p, s, d, cache_dir=None, route=None: (caches.append(cache_dir) or d,
                                                                                 {'cost': 0}))

    def swap(load_dir, fighter, out, ask, name=None):
        ask('q', [], {}, 'a', None)
        return _swapper([])(load_dir, fighter, out, ask, name)
    monkeypatch.setattr(run, 'swap', swap)
    monkeypatch.setenv(ai.RELAY_ENV, str(tmp_path / 'asks'))
    for name in ('a', 'b'):
        monkeypatch.setattr(load, 'run', _loaded(tmp_path / name))
        costume.make('model.zip', 'Mario', tmp_path / name, cache=tmp_path / 'shared')
    costume.fix(tmp_path / 'a', sizes={'head': 0.8}, cache=tmp_path / 'shared')
    assert caches == [tmp_path / 'a' / 'cache', tmp_path / 'b' / 'cache', tmp_path / 'a' / 'cache']
    # no relay: the cache named, shared
    caches.clear()
    monkeypatch.delenv(ai.RELAY_ENV)
    monkeypatch.setattr(load, 'run', _loaded(tmp_path / 'c'))
    costume.make('model.zip', 'Mario', tmp_path / 'c', cache=tmp_path / 'shared')
    assert caches == [tmp_path / 'shared']


def test_a_failure_names_its_stage_and_cause(monkeypatch, tmp_path):
    def nothing(download, out):
        raise RuntimeError('nothing to open in model.zip')
    monkeypatch.setattr(load, 'run', nothing)
    result = costume.make('model.zip', 'Mario', tmp_path)
    assert result['status'] == 'failed' and result['failed']['stage'] == 'load'
    assert 'nothing to open' in result['failed']['cause'] and result['cost_usd'] == 0


def test_auto_swaps_onto_the_fighter_the_swap_picks_and_says_which(monkeypatch, tmp_path):
    monkeypatch.setattr(load, 'run', lambda download, out: None)
    told = []

    def swap(load_dir, fighter, out, ask, name=None):
        told.append((fighter, name))
        return {'fighter': 'Kirby', 'fighter_pick': {'fighter': 'Kirby', 'why': 'round', 'picked': True},
                'export': 'x.dat'}
    monkeypatch.setattr(run, 'swap', swap)
    result = costume.make('Waddle Dee.zip', costume.AUTO, tmp_path)
    assert told == [('auto', 'Waddle Dee.zip')]
    assert result['fighter'] == 'Kirby' and result['fighter_pick']['why'] == 'round'


def test_a_pick_that_fails_is_said_as_its_own_stage(monkeypatch, tmp_path):
    from rigswap.plugins.melee import pick
    monkeypatch.setattr(load, 'run', lambda download, out: None)

    def swap(load_dir, fighter, out, ask, name=None):
        raise pick.Failed('ValueError: no fighter matched')
    monkeypatch.setattr(run, 'swap', swap)
    result = costume.make('Waddle Dee.zip', costume.AUTO, tmp_path)
    assert result['status'] == 'failed' and result['failed']['stage'] == 'pick'


def _swapper(built, alarms=(), deaf=False):
    """run.swap's stand-in: notes each build (its folder, the record it was given, RIGSWAP_FILL) and writes the
    costume file and the record a real one leaves; the builds named in ``alarms`` trip the tripwire. The costume file
    is made from the record it was given, so a changed record is a changed file; ``deaf``: the build takes no notice
    of the record and writes the same file every time."""
    import os
    from pathlib import Path

    def swap(load_dir, fighter, out, ask, name=None):
        given = Path(load_dir) / 'record.json'
        given = json.loads(given.read_text(encoding='utf-8')) if given.is_file() else None
        built.append((Path(out).name, given, os.environ.get('RIGSWAP_FILL')))
        (Path(out) / '4-export').mkdir(parents=True)
        (Path(out) / 'record.json').write_text(json.dumps(given or {'head': 1.0, 'hips': 'hips'}), encoding='utf-8')
        dat = Path(out) / '4-export' / f'{fighter}.dat'
        dat.write_bytes(b'dat' if deaf else json.dumps(given, sort_keys=True).encode())
        alarm = Path(out).name in alarms
        return {'fighter': fighter, 'export': str(dat), 'record': {'source': 'hand' if given else 'look'},
                'tripwire': {'alarm': alarm, 'what': 'the legs are folded under the body' if alarm else ''}}
    return swap


def _loaded(folder):
    def load_run(download, out):
        (folder / '1-load').mkdir(parents=True, exist_ok=True)
    return load_run


def test_a_draft_the_tripwire_calls_a_wreck_is_tried_once_more_and_no_further(monkeypatch, tmp_path):
    monkeypatch.setattr(load, 'run', _loaded(tmp_path / 'a'))
    built = []
    monkeypatch.setattr(run, 'swap', _swapper(built, alarms=('2-swap',)))
    result = costume.make('model.zip', 'Mario', tmp_path / 'a')
    assert [(name, fill) for name, _, fill in built] == [('2-swap', None), ('2-swap-again', 'again')]
    assert result['status'] == 'done' and 'warning' not in result, 'the second try stands where the tripwire passes it'
    assert '2-swap-again' in result['dat'] and '2-swap-again' in result['record']
    # both wrecks: the first stands, said plainly, and nothing is tried a third time
    built.clear()
    monkeypatch.setattr(load, 'run', _loaded(tmp_path / 'b'))
    monkeypatch.setattr(run, 'swap', _swapper(built, alarms=('2-swap', '2-swap-again')))
    result = costume.make('model.zip', 'Mario', tmp_path / 'b')
    assert len(built) == 2 and '2-swap-again' not in result['dat']
    assert result['warning'] == 'this probably did not work: the legs are folded under the body'
    assert json.loads((tmp_path / 'b' / 'done.json').read_text(encoding='utf-8')) == result


def test_a_change_builds_again_from_the_last_costume_made_and_a_failed_one_leaves_it(monkeypatch, tmp_path):
    from rigswap.core import complaint
    monkeypatch.setattr(load, 'run', _loaded(tmp_path))
    built = []
    monkeypatch.setattr(run, 'swap', _swapper(built))
    first = costume.make('model.zip', 'Mario', tmp_path)
    # a size named outright: multiplied into the record, which goes beside the load; built into its own folder
    changed = costume.fix(tmp_path, sizes={'head': 0.8})
    assert changed['status'] == 'done' and changed['changed'] == {'head': [1.0, 0.8]} and changed['why'] == ''
    assert built[-1][:2] == ('2-swap-fix1', {'head': 0.8, 'hips': 'hips'})
    assert changed['dat'] != first['dat'] and '2-swap-fix1' in changed['record']
    assert json.loads((tmp_path / 'done.json').read_text(encoding='utf-8')) == changed

    # a complaint the record cannot hold fails at 'fix' in the look's words, and the last costume made still stands
    def cannot(swap_dir, words, ask, heard=None):
        raise complaint.Unfixable('the texture is wrong, not the body')
    monkeypatch.setattr(complaint, 'fix', cannot)
    failed = costume.fix(tmp_path, words='the face texture is blurry')
    assert failed['status'] == 'failed'
    assert failed['failed'] == {'stage': 'fix', 'cause': 'Unfixable: the texture is wrong, not the body'}
    assert json.loads((tmp_path / 'result.json').read_text(encoding='utf-8')) == failed
    assert json.loads((tmp_path / 'done.json').read_text(encoding='utf-8')) == changed

    # what the person said: the look names the change (stood in for here), read from the last costume's folder
    seen = []

    def look(swap_dir, words, ask, heard=None):
        seen.append((swap_dir.name, words, heard))
        return {'head': 0.8, 'hips': 'hips', 'front': [0, 1, 0]}, {'front': [[0, -1, 0], [0, 1, 0]]}, 'he faces away'
    monkeypatch.setattr(complaint, 'fix', look)
    again = costume.fix(tmp_path, words='he faces backwards', heard='turn him round, by 180')
    assert seen == [('2-swap-fix1', 'he faces backwards', 'turn him round, by 180')], \
        "the person's own words, and what an assistant made of them, apart"
    assert again['why'] == 'he faces away'
    assert built[-1][:2] == ('2-swap-fix2', {'head': 0.8, 'hips': 'hips', 'front': [0, 1, 0]})
    # nothing made yet: said, not crashed
    (tmp_path / 'none').mkdir()
    empty = costume.fix(tmp_path / 'none', sizes={'head': 0.8})
    assert empty['failed'] == {'stage': 'fix', 'cause': 'Unfixable: there is no finished costume here to change'}


def test_a_change_that_builds_the_same_file_is_a_failure_said_plainly(monkeypatch, tmp_path):
    """The record changed and the build took no notice (a size the fit did not apply): the costume is the same file as
    before, so nothing was changed, and it is never handed on as a second costume."""
    monkeypatch.setattr(load, 'run', _loaded(tmp_path))
    built = []
    monkeypatch.setattr(run, 'swap', _swapper(built, deaf=True))
    first = costume.make('model.zip', 'Mario', tmp_path)
    failed = costume.fix(tmp_path, sizes={'head': 0.8})
    assert failed['status'] == 'failed' and failed['failed']['stage'] == 'fix'
    assert failed['failed']['cause'] == ('Unfixable: the change (head: 1.0 to 0.8) made no difference: the costume '
                                         'came out the same as before')
    assert json.loads((tmp_path / 'done.json').read_text(encoding='utf-8')) == first, 'the first costume still stands'


@pytest.mark.parametrize('platform,loaded,dropped', [('linux', False, ['DISPLAY', 'WAYLAND_DISPLAY']),
                                                     ('linux', True, []), ('darwin', False, []),
                                                     ('win32', False, [])])
def test_headless_drops_the_display_on_linux_before_bpy(platform, loaded, dropped):
    """bpy draws through EGL without DISPLAY; with it, a broken GLX crashes the process (costume.headless)."""
    from rigswap.plugins.melee import costume
    environ = {'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-0', 'HOME': '/home/u'}
    modules = {'bpy': object()} if loaded else {}
    assert costume.headless(environ, platform, modules) == dropped
    assert all(name not in environ for name in dropped) and environ['HOME'] == '/home/u'
    assert len(environ) == 3 - len(dropped)
