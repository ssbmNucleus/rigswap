"""export/: the costume written and read back (run), with HSDRawViewer, Blender and the DAT writers faked."""
from importlib import import_module
import json
from pathlib import Path

import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee import dae as melee_dae
from rigswap.plugins.melee.export import checks as melee_checks, dobjs as melee_dobjs, eyes as melee_eyes
from rigswap.plugins.melee.export.dobjs import FighterTables
from rigswap.plugins.melee.facts import fighters
from tests.plugins.melee.export.test_checks import EYES_READ_BACK, KEVIN, KEVIN_EYES, _kevin_read_back
from tests.plugins.melee.test_dae import _objects_model, _triangle_model
from tests.plugins.melee.facts.test_parts import LINK_SWORD
# melee.export is the plugin's function; this is its folder of that name
melee_export = import_module('rigswap.plugins.melee.export')


@pytest.fixture(autouse=True)
def fake_physics_skin(monkeypatch):
    monkeypatch.setattr(melee_export, 'preserve_samplers', lambda *a: {})
    monkeypatch.setattr(melee_export, 'preserve_sidedness', lambda *a: [])
    monkeypatch.setattr(melee_export, 'fill_forms', lambda *a: {})
    monkeypatch.setattr(melee_export, 'forms_check', lambda *a: {'pass': True})
    # These tests fake Blender and all DAT readers; physics now reads the skin
    # even when the model has no accessory candidates.
    monkeypatch.setattr(melee_export.physics, 'split', lambda build, work, f: (build / 'build.blend', [], []))
    monkeypatch.delenv(melee.TARGETS_VAR, raising=False)


def test_export_rejects_actual_file_over_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(melee_export, 'plan_alpha_modes', lambda *a: (None, {}))
    for name in ('PlMrNr.dat', 'PlMr.dat'):
        (tmp_path / name).write_bytes(b'vanilla')
    model = melee_dae.Model()
    monkeypatch.setattr(melee_export, 'export_dae', lambda *a: None)
    monkeypatch.setattr(melee_export, 'read_dae', lambda *a: model)
    monkeypatch.setattr(melee_export, 'blend_to_glb', lambda *a: None)
    monkeypatch.setattr(melee_export, 'glb_to_dae', lambda *a: None)
    monkeypatch.setattr(melee_export, 'image_count_check', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_export, 'material_count_check', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_export, 'import_dae', lambda *a: None)
    monkeypatch.setattr(melee_checks, 'costume_budget', lambda *a: 64)
    monkeypatch.setattr(melee_dobjs, 'fighter_tables', lambda *a: FighterTables())
    def fill(*args, **kwargs):
        Path(args[4]).write_bytes(bytes(65))
        return {}
    monkeypatch.setattr(melee_export, 'fill_vanilla_dobjs', fill)
    with pytest.raises(Failed, match=r'65 file bytes \(96 in allocation units.*budget 64.*32 over'):
        melee.export(tmp_path, tmp_path / 'out.dat', 'Mario', tmp_path)


def _fake_export(tmp_path, monkeypatch, model, back, low, formats, buffers, fighter='Mario', eyes=None):
    """melee.export(tmp_path, ...) runs with HSDRawViewer, Blender and the DAT writers faked: ``model`` is the .dae
    as read_dae reads it, ``back`` and ``low`` the read-backs, ``formats`` prep's texture formats by material,
    ``buffers`` the written DObjs' image buffers. ``eyes``, when given, is what melee_eyes.apply returns (an eyes.json
    beside the build makes the export call it)."""
    f = fighters.fighter(fighter)
    for name in (f'Pl{f.code}Nr.dat', f'Pl{f.code}.dat'):
        (tmp_path / name).write_bytes(b'vanilla')
    asked = tmp_path / 'formats.json'
    asked.write_text(json.dumps(formats), encoding='utf-8')
    monkeypatch.setattr(melee_export, '_formats_by_dae_id', lambda *a: asked)
    monkeypatch.setattr(melee_export, 'export_dae', lambda *a: a[2])
    monkeypatch.setattr(melee_export, 'blend_to_glb', lambda *a: a[1])
    monkeypatch.setattr(melee_export, 'glb_to_dae', lambda *a: a[1])
    monkeypatch.setattr(melee_export, 'image_count_check', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_export, 'material_count_check', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_export, 'import_dae', lambda *a: None)
    monkeypatch.setattr(melee_export, 'dat_joints', lambda *a, **kw: {})
    monkeypatch.setattr(melee_export, 'hsdraw', lambda *a: None)
    monkeypatch.setattr(melee_export, 'plan_alpha_modes', lambda *a: (None, {}))
    monkeypatch.setattr(melee_export, 'alpha_mode_check', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_export, 'written_alike', lambda *a: {})
    def read(path):
        if path.parent.name == 'readback-low':
            return low
        return back if path.parent.name == 'readback' else model
    monkeypatch.setattr(melee_export, 'read_dae', read)
    def fill(*args, **kwargs):
        Path(args[4]).write_bytes(bytes(64))
        return {'low_detail': {'left_out': [0]}}
    monkeypatch.setattr(melee_export, 'fill_vanilla_dobjs', fill)
    monkeypatch.setattr(melee_dobjs, 'fighter_tables', lambda *a: FighterTables())
    monkeypatch.setattr(melee_dobjs, 'layout', lambda *a: [])
    monkeypatch.setattr(melee_export, 'check_palettes', lambda *a: {'pass': True})
    monkeypatch.setattr(melee_checks, 'image_buffers', lambda *a: buffers)
    if eyes is not None:
        (tmp_path / 'eyes.json').write_text('{}', encoding='utf-8')
        monkeypatch.setattr(melee_eyes, 'apply', lambda *a: eyes)


def test_export_passes_the_joined_builds_eye_plan_to_placement_and_animation(tmp_path, monkeypatch):
    import copy
    model = _objects_model(KEVIN)
    back = _kevin_read_back(next(iter(EYES_READ_BACK)))
    eye_result = {'moved': KEVIN_EYES, 'added_by_eyes': {'materials': 1}}
    _fake_export(tmp_path, monkeypatch, model, back, copy.deepcopy(model), {}, [], 'Zelda', eyes=eye_result)
    planned = {'destinations': {'left': 26, 'right': 25}}
    sampling = {'materials': [{'place': 0, 'written': [0, 0]}]}
    sampler_calls = []

    def samplers(glb, dae, imported, symbol):
        sampler_calls.append((glb, dae))
        return sampling

    def blend(*args):
        Path(args[1]).with_name('joins.json').write_text(json.dumps({'eye_plan': planned}), encoding='utf-8')
        return args[1]

    fill = melee_export.fill_vanilla_dobjs
    def placed(*args, **kwargs):
        assert len(sampler_calls) == 1
        assert kwargs['deferred'] == [1, 2]
        return fill(*args, **kwargs)

    def apply(*args):
        assert args[5]['eye_plan'] == planned
        return eye_result

    monkeypatch.setattr(melee_export, 'blend_to_glb', blend)
    monkeypatch.setattr(melee_export, 'preserve_samplers', samplers)
    monkeypatch.setattr(melee_export, 'fill_vanilla_dobjs', placed)
    monkeypatch.setattr(melee_eyes, 'dae_dobjs', lambda _: [('body', 'skin'), ('left', 'eyes'), ('right', 'eyes')])
    monkeypatch.setattr(melee_eyes, 'apply', apply)
    melee.export(tmp_path, tmp_path / 'out.dat', 'Zelda', tmp_path)
    report = json.loads((tmp_path / 'export/readback.json').read_text())
    assert report['joins']['texture_samplers'] == sampling


@pytest.mark.parametrize('eyes', sorted(EYES_READ_BACK))
def test_export_passes_the_material_check_whatever_the_eye_copies_read_back_as(tmp_path, monkeypatch, eyes):
    """runs/pilot3/kevin-Zelda replayed: the eyes step moved Kevin's two eyes (one material) into Zelda's animated
    places 26 and 25, each with a copy of the eye material, and the read-back named both copies one material. The
    check expected the copies as two (by address) and failed an export that lost nothing."""
    import copy
    model, back = _objects_model(KEVIN), _kevin_read_back(eyes)
    record = {'moved': KEVIN_EYES, 'added_by_eyes': {'materials': 1}}     # as melee_eyes.apply returns it
    _fake_export(tmp_path, monkeypatch, model, back, copy.deepcopy(model), {}, [], 'Zelda', record)
    melee.export(tmp_path, tmp_path / 'out.dat', 'Zelda', tmp_path)
    report = json.loads((tmp_path / 'export/readback.json').read_text())
    assert report['checks']['materials']['pass'] and (report['checks']['materials']['before'],
                                                      report['checks']['materials']['after']) == (2, 2)
    # The same export losing the hair's material fails.
    _fake_export(tmp_path, monkeypatch, model, _kevin_read_back(eyes, hair='Material_334B1CC4'),
                 copy.deepcopy(model), {}, [], 'Zelda', record)
    with pytest.raises(Failed, match='materials'):
        melee.export(tmp_path, tmp_path / 'out.dat', 'Zelda', tmp_path)


@pytest.mark.parametrize('failure', [None, 'low_positions', 'materials', 'weights', 'winding', 'texture_formats',
                                    'dobj_images'])
def test_export_enforces_named_checks_and_uses_a_fresh_folder(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(melee_export, 'plan_alpha_modes', lambda *a: (None, {}))
    monkeypatch.setattr(melee_export, 'alpha_mode_check', lambda *a: {'pass': True})
    import copy
    model = _triangle_model()
    model.face_materials = ['skin']
    model.material_images = {'skin': ['source.png']}
    model.face_objects = ['Joint_0_Object_0']
    back, low = copy.deepcopy(model), copy.deepcopy(model)
    image = tmp_path / 'Texture_0_CMP.png'
    image.write_bytes(b'pixels')
    back.material_images = {'skin': [str(image)]}
    if failure == 'low_positions':
        low.points += [0, 0, 1]
    elif failure == 'materials':
        back.materials = 2
    elif failure == 'weights':
        back.weights[0] = {'JOBJ_1': 1.}
    elif failure == 'winding':
        back.faces = [[2, 1, 0]]
    elif failure == 'texture_formats':
        back.material_images = {}
    _fake_export(tmp_path, monkeypatch, model, back, low, {'skin': 'CMPR'},
                 [{'drawn': True, 'textures': [] if failure == 'dobj_images' else [{'pass': True}]}])
    if failure:
        with pytest.raises(Failed, match=failure):
            melee.export(tmp_path, tmp_path / 'out.dat', 'Mario', tmp_path)
        report = json.loads((tmp_path / 'export/readback.json').read_text())
        assert report['checks'][failure]['pass'] is False
    else:
        folders = []
        for _ in range(2):
            melee.export(tmp_path, tmp_path / 'out.dat', 'Mario', tmp_path)
            report = json.loads((tmp_path / 'export/readback.json').read_text())
            assert all(c['pass'] for c in report['checks'].values())
            folders.append(report['export_directory'])
        assert folders[0] != folders[1]
        assert 'verdict' not in report and 'retry' not in report
        assert not (tmp_path / 'export/gate.json').exists()


@pytest.mark.parametrize('change', [0., 1e-7])
def test_skeleton_check_uses_the_written_game_transforms_exactly(tmp_path, monkeypatch, change):
    import copy
    import numpy as np
    model = _triangle_model()
    model.face_materials, model.face_objects = ['plain'], ['Joint_0_Object_0']
    model.material_images = {}
    back = copy.deepcopy(model)
    vanilla = copy.deepcopy(model)
    # A local frame changed during target conversion. The importer discards these DAE transforms.
    model.joints = {'JOBJ_0': np.eye(4)}
    model.parents = {'JOBJ_0': None}
    vanilla.joints, vanilla.parents = copy.deepcopy(model.joints), model.parents.copy()
    model.joints['JOBJ_0'][0, 3] = .00031
    _fake_export(tmp_path, monkeypatch, model, back, copy.deepcopy(back), {}, [])
    read = melee_export.read_dae
    monkeypatch.setattr(melee_export, 'read_dae', lambda path: vanilla if path.parent.name == 'vanilla' else read(path))
    def game_joints(path, *args, **kwargs):
        matrix = np.eye(4)
        if path.name == 'out.dat':
            matrix[0, 3] += change
        return {'JOBJ_0': matrix}
    monkeypatch.setattr(melee_export, 'dat_joints', game_joints)
    if change:
        with pytest.raises(Failed, match='skeleton'):
            melee.export(tmp_path, tmp_path / 'out.dat', 'Mario', tmp_path)
    else:
        melee.export(tmp_path, tmp_path / 'out.dat', 'Mario', tmp_path)
    check = json.loads((tmp_path / 'export/readback.json').read_text())['checks']['skeleton']
    assert check == {'matrix_change': change, 'tolerance': 0, 'pass': not bool(change)}


def test_two_exports_of_one_fighter_at_once_read_their_own_vanilla_files(tmp_path, monkeypatch):
    """HSDRawViewer opens a file it reads for writing too and lets others only read it, so a second run on the same
    vanilla Pl<code>Nr.dat fails while the first is in it (two builds of one fighter exported side by side). Each
    export reads its own copies; faked here as an HSDRawViewer that holds every .dat it is given until both exports
    are inside it."""
    import copy
    import threading
    model = _triangle_model()
    model.face_materials, model.material_images = ['skin'], {'skin': ['source.png']}
    model.face_objects = ['Joint_0_Object_0']
    image = tmp_path / 'Texture_0_CMP.png'
    image.write_bytes(b'pixels')
    back = copy.deepcopy(model)
    back.material_images = {'skin': [str(image)]}
    _fake_export(tmp_path, monkeypatch, model, back, copy.deepcopy(model), {'skin': 'CMPR'},
                 [{'drawn': True, 'textures': [{'pass': True}]}])
    vanilla = {tmp_path / 'PlMrNr.dat', tmp_path / 'PlMr.dat'}
    held, lock, given = {}, threading.Lock(), []

    def hsdraw_reads(paths):
        both_inside = threading.Barrier(2, timeout=10)

        def run(*args):
            dats = [Path(a).resolve() for a in args if str(a).endswith('.dat') and Path(a).is_file()]
            with lock:
                given.extend(dats)
                if any(p in held for p in dats):
                    both_inside.abort()
                    raise Failed(f'HSDRawViewer: the process cannot access {dats[0].name}: another process uses it')
                held.update(dict.fromkeys(dats))
            try:
                both_inside.wait()
            finally:
                with lock:
                    for p in dats:
                        held.pop(p, None)
            return paths(args)
        return run

    def both(make):
        monkeypatch.setattr(melee_export, 'export_dae', hsdraw_reads(lambda a: a[2]))
        errors = []

        def one(name):
            try:
                melee.export(tmp_path / name, tmp_path / name / 'out.dat', 'Mario', tmp_path)
            except Exception as e:               # noqa: BLE001 - every failure is the test's finding
                errors.append(e)
        for name in 'ab':
            (tmp_path / name).mkdir(exist_ok=True)
        with monkeypatch.context() as m:
            make(m)
            threads = [threading.Thread(target=one, args=(n,)) for n in 'ab']
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        return errors

    assert both(lambda m: None) == []
    assert given and not vanilla & set(given)                  # HSDRawViewer read copies, never the vanilla files
    for name in 'ab':
        copies = list((tmp_path / name / 'export').glob('run-*/vanilla-files/*.dat'))
        assert sorted(p.name for p in copies) == ['PlMr.dat', 'PlMrNr.dat']
        assert all(p.read_bytes() == b'vanilla' for p in copies)
    # Without the copies the second export of the pair fails as HSDRawViewer did.
    errors = both(lambda m: m.setattr(melee_export, 'vanilla_copies', lambda folder, *paths: list(paths)))
    assert any('another process uses it' in str(e) for e in errors)


def test_export_blanks_a_replaced_weapons_places_and_says_so(tmp_path, monkeypatch):
    import copy
    model = _triangle_model()
    model.face_materials, model.material_images = ['skin'], {'skin': ['source.png']}
    model.face_objects = ['Joint_0_Object_0']
    image = tmp_path / 'Texture_0_CMP.png'
    image.write_bytes(b'pixels')
    back = copy.deepcopy(model)
    back.material_images = {'skin': [str(image)]}
    _fake_export(tmp_path, monkeypatch, model, back, copy.deepcopy(model), {'skin': 'CMPR'},
                 [{'drawn': True, 'textures': [{'pass': True}]}], fighter='Link')
    given = {}
    monkeypatch.setattr(melee_export, 'blend_to_glb', lambda *a: given.setdefault('reserved', a[5]) and a[1])
    fill = melee_export.fill_vanilla_dobjs

    def filled(*args, **kwargs):
        given['kept'] = kwargs['keep']
        return fill(*args, **kwargs)
    monkeypatch.setattr(melee_export, 'fill_vanilla_dobjs', filled)
    (tmp_path / 'weapons_replaced.json').write_text(json.dumps([LINK_SWORD]), encoding='utf-8')
    melee.export(tmp_path, tmp_path / 'out.dat', 'Link', tmp_path)
    keep = tuple(p for p in fighters.fighter('Link').keep if p not in (*range(71, 76), 83, 84, 85, 93, 94))
    assert given == {'reserved': keep, 'kept': keep}
    report = json.loads((tmp_path / 'export/readback.json').read_text())
    assert report['weapons_replaced'] == [LINK_SWORD] and report['weapon_forms_hidden'] == ['sword_stowed', 'scabbard']
