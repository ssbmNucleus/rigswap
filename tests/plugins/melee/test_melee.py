"""The plugin's interface (rigswap.plugins.melee): the fighters as targets."""
import json

import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee.facts import fighters
from rigswap.plugins.melee.targets import camera as targets_camera
from rigswap.vocabulary import Description


def _memory(name='Mario'):
    return {'schema': 1, 'fighter': name, 'triangles': 10, 'kept': 96,
            'bytes': {'geometry': 120, 'retained': 60, 'placement': 20},
            'method': 'test importer record', 'importer': {'version': 'test'}, 'vanilla_sha256': 'test'}


@pytest.mark.parametrize('name', ['Mario', 'Bowser', 'Kirby'])
def test_target_uses_its_fighters_cap_and_camera_scale(tmp_path, name):
    f = fighters.fighter(name)
    _prepared(tmp_path, name)
    cam = targets_camera.camera(f.scale, 14.4)
    (tmp_path / f'{name}.camera.json').write_text(json.dumps(cam))
    Description(body_type='biped', up='+y', front='+z').save(tmp_path / f'{name}.description.json')
    target = melee.target(name, tmp_path)
    assert target.max_influences == f.max_influences
    assert target.camera['model_scale'] == f.scale
    assert target.camera['eye'][1] == pytest.approx(targets_camera.CAMERA_HEIGHT / f.scale)
    assert target.memory == _memory(name)


def _prepared(folder, name='Mario'):
    (folder / f'{name}.glb').write_bytes(b'glTF')
    (folder / f'{name}.camera.json').write_text(json.dumps(targets_camera.camera(1.1, 14.4)), encoding='utf-8')
    (folder / f'{name}.memory.json').write_text(json.dumps(_memory(name)), encoding='utf-8')


def test_target_requires_a_memory_record_from_prepare(tmp_path):
    _prepared(tmp_path)
    Description(body_type='biped', up='+y', front='+z').save(tmp_path / 'Mario.description.json')
    (tmp_path / 'Mario.memory.json').unlink()
    with pytest.raises(Failed, match=r'Mario.memory.json.*melee.prepare'):
        melee.target('Mario', tmp_path)


def test_target_fails_until_the_fighter_is_described(tmp_path):
    with pytest.raises(Failed, match='not been prepared'):
        melee.target('Mario', tmp_path)
    _prepared(tmp_path)
    with pytest.raises(Failed, match='must be described first'):
        melee.target('Mario', tmp_path)
    with pytest.raises(Failed, match='Mario'):
        melee.targets(tmp_path)
    Description(body_type='biped', up='+y', front='+z').save(tmp_path / 'Mario.description.json')
    (tmp_path / 'Mario.pose-run.json').write_text('{}', encoding='utf-8')
    t = melee.target('mario', tmp_path)
    assert t.name == 'Mario' and t.skeleton == t.original == tmp_path / 'Mario.glb'
    assert t.poses == [tmp_path / 'Mario.pose-run.json']
    assert t.camera['projection'] == 'perspective' and t.camera['fov_y'] == 35.0
    assert [x.name for x in melee.targets(tmp_path)] == ['Mario']
