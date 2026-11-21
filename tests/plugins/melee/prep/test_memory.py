"""prep/memory.py: the costume's share of preload memory, the kept parts' budget, the triangle palettes."""
import os
from pathlib import Path

import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee.facts import fighters
from rigswap.plugins.melee.prep import memory as prep_memory, textures as prep_textures
from tests.plugins.melee.facts.test_parts import LINK_SWORD
from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _grid, _material, _mesh


def test_gx_sizes_and_each_fighters_share_of_memory():
    assert prep_textures.gx_bytes('CMPR', 256, 256) == 32_768
    assert prep_textures.gx_bytes('CMPR', 190, 190) == 18_432              # stored in whole 8x8 blocks
    assert prep_textures.gx_bytes('CI8', 64, 64) == 4096 + 512             # and its palette
    assert prep_textures.gx_bytes('RGB5A3', 32, 32) == 2048
    # the old importer's arithmetic, as its preload_heap.py works it through
    for name, budget in (('Peach', 2_675_392), ('Zelda', 2_128_736), ('Jigglypuff', 2_808_960), ('Ness', 2_493_088),
                         ('DonkeyKong', 2_713_760)):
        assert prep_memory.costume_budget(fighters.fighter(name).preloaded) == budget, name


def test_measured_costs_control_the_decision_and_are_recorded():
    measurement = {'triangles': 3, 'bytes': {'geometry': 10, 'retained': 4, 'placement': 1},
                   'method': 'test archive blocks'}
    record = prep_memory.hold_memory_budget([], triangles=10, budget=96, kept=32, measurement=measurement)
    assert record['before'] == {'bytes': 96, 'textures': 0, 'geometry': 34, 'retained': 14, 'placement': 4, 'kept': 32}
    assert record['measurement'] == measurement and record['status'] == 'fits'
    with pytest.raises(Failed, match='over with every texture'):
        prep_memory.hold_memory_budget([], triangles=10, budget=64, kept=32, measurement=measurement)


@pytest.mark.blender
def test_a_costume_over_its_memory_halves_its_costliest_textures_or_fails_with_the_numbers():
    measurement = {'triangles': 10, 'bytes': {'geometry': 120, 'retained': 60, 'placement': 20}}
    _empty_scene()
    materials = [_material(f'sheet{i}', _flat(256, 256, (40 * i, 90, 200, 255))) for i in range(4)]
    textures, _ = prep_textures.limit_textures([_grid('body', 2, materials)])
    assert [t.bytes() for t in textures] == [32_768] * 4                # 131,072 of textures

    record = prep_memory.hold_memory_budget(textures, triangles=100, budget=120_000, kept=1000,
                                          measurement=measurement)
    assert record['before']['bytes'] == 134_080  # textures + measured overhead + kept, rounded to 32 bytes
    assert record['halved'] == [{'image': 'sheet3', 'from': [256, 256], 'to': [128, 128]}]
    assert record['after']['bytes'] <= 120_000
    assert tuple(textures[3].image.size) == (128, 128)

    _empty_scene()
    materials = [_material(f'sheet{i}', _flat(256, 256, (40 * i, 90, 200, 255))) for i in range(4)]
    textures, _ = prep_textures.limit_textures([_grid('body', 2, materials)])
    with pytest.raises(Failed, match=r'32,768 of textures.*20,000: 12,768 over'):
        prep_memory.hold_memory_budget(textures, triangles=0, budget=20_000, kept=0, measurement=measurement)


def test_no_legacy_rates_or_optional_measurement():
    for name in ('GEOMETRY', 'RETAINED', 'PLACEMENT'):
        assert not hasattr(prep_memory, name + '_BYTES_PER_TRIANGLE')
    with pytest.raises(TypeError, match='measurement'):
        prep_memory.costume_estimate([], 1, 0)


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlLkNr.dat').is_file(),
                    reason=f'needs vanilla files in {melee.FILES_VAR}')
def test_prep_budgets_only_the_parts_a_build_keeps():
    link = fighters.fighter('Link')
    whole, less = prep_memory.budget(link), prep_memory.budget(link, replaced=[LINK_SWORD])
    # the sword, the sword put away (places 93, 94) and the scabbard (83 to 85) are not kept, their triangles given back
    assert less['triangles'] == whole['triangles'] - 287 - 217 - 53 == 917 - 287 - 217 - 53
    assert less['bytes'] < whole['bytes'] and not set(less['places']) & {*range(71, 76), 83, 84, 85, 93, 94}
    assert less['not_kept'] == {'sword': {'places': [71, 72, 73, 74, 75], 'triangles': 287},
                                'sword_stowed': {'places': [93, 94], 'triangles': 217},
                                'scabbard': {'places': [83, 84, 85], 'triangles': 53}}


@pytest.mark.blender
def test_prep_counts_influence_sets_per_palette_and_per_material():
    _empty_scene()
    verts = []
    for i in range(12):
        verts.extend([(i, 0, 0), (i, 1, 0), (i, 0, 1)])
    ob = _mesh('body', verts, [(3*i, 3*i+1, 3*i+2) for i in range(12)])
    a, b = ob.vertex_groups.new(name='JOBJ_0'), ob.vertex_groups.new(name='JOBJ_1')
    for i in range(12):
        a.add([3*i, 3*i+1, 3*i+2], (i+1)/20, 'REPLACE')
        b.add([3*i, 3*i+1, 3*i+2], 1-(i+1)/20, 'REPLACE')
    record = prep_memory.check_mesh_palettes([ob])
    assert record['meshes'] == [{'mesh': 'body', 'material': 0, 'sizes': [10, 2]}]
    import bpy
    for n in ('skin', 'eye'):
        ob.data.materials.append(bpy.data.materials.new(n))
    for face in ob.data.polygons:
        face.material_index = face.index % 2
    record = prep_memory.check_mesh_palettes([ob])
    assert [r['sizes'] for r in record['meshes']] == [[6], [6]]
    with pytest.raises(Failed, match='11 envelopes; limit 10'):
        prep_memory.check_palette_size(11, 'body, material 0, palette 1')
