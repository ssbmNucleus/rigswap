"""prep/join.py: a joint's looks fit to the vanilla places."""
import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee.export.dobjs import FighterTables, Slot
from rigswap.plugins.melee.prep import join as prep_join
from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _material, _mesh, melee_prep


@pytest.mark.parametrize('copied', [False, True])
def test_meshes_of_one_material_take_one_place_and_stay_as_they_are(copied):
    """A place is one material and a chain of PObjs, so meshes that look alike need one place however many they are;
    the import still makes a DObj for each, which is what ``dobjs`` counts."""
    _empty_scene()
    mat = _material('skin', _flat(4, 4))
    for i, name in enumerate(('a', 'b', 'c')):
        ob = _mesh(name, [(i, 0, 0), (i, 1, 0), (i, 0, 1)], [(0, 1, 2)], [mat.copy() if copied else mat])
        ob.vertex_groups.new(name=f'JOBJ_{i}').add([0, 1, 2], 1, 'REPLACE')
    record = prep_join.join_for_places(melee_prep.drawn_meshes(), [Slot(0, 0)], FighterTables(body=(0,)))
    assert record['looks'] == {'before': 1, 'after': 1} and record['dobjs'] == 3
    assert bool(record['material_aliases']) == copied
    assert sorted(o.name for o in melee_prep.drawn_meshes()) == ['a', 'b', 'c']


def test_different_settings_are_different_looks():
    _empty_scene()
    mat = _material('skin', _flat(4, 4))
    for i in range(3):
        copy = mat.copy()
        bsdf = next(n for n in copy.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        bsdf.inputs['Roughness'].default_value = i / 3
        _mesh(str(i), [(i, 0, 0), (i, 1, 0), (i, 0, 1)], [(0, 1, 2)], [copy])
    with pytest.raises(Failed, match='3 DObjs.*2 places'):
        prep_join.join_for_places(melee_prep.drawn_meshes(), [Slot(0, i) for i in range(2)], FighterTables())


def test_the_eyes_take_places_too():
    """golduck-zip on Bowser: the plan counted the build's DObjs without the eyes the blink keeps apart (21) and the
    import made 22. The eyes' look takes a place of its own, and their DObjs are counted."""
    _empty_scene()
    skin, eye = _material('skin', _flat(4, 4)), _material('eye', _flat(4, 4, (255, 255, 255, 255)))
    for i, (name, mat) in enumerate((('a', skin), ('b', skin), ('eyes', eye))):
        ob = _mesh(name, [(i, 0, 0), (i, 1, 0), (i, 0, 1)], [(0, 1, 2)], [mat])
        ob.vertex_groups.new(name='JOBJ_0').add([0, 1, 2], 1, 'REPLACE')
    tables = FighterTables(body=(0, 1), low=(2, 3))
    record = prep_join.join_for_places(melee_prep.drawn_meshes(), [Slot(0, i, (0,)) for i in range(4)], tables,
                                       keep=['eyes'])
    assert record['looks'] == {'before': 2, 'after': 2}
    assert record['dobjs'] == 3 and record['kept_apart'] == ['eyes']
    assert sorted(o.name for o in melee_prep.drawn_meshes()) == ['a', 'b', 'eyes']
    # with only the eyes and two looks that cannot share an image over the places, it fails, here, with their count
    _empty_scene()
    for i, (name, mat) in enumerate((('a', _material('one', _flat(4, 4))), ('b', _material('two', _flat(2, 2))),
                                     ('eyes', _material('eye', _flat(4, 4, (255, 255, 255, 255)))))):
        _mesh(name, [(i, 0, 0), (i, 1, 0), (i, 0, 1)], [(0, 1, 2)], [mat])
    with pytest.raises(Failed, match='3 DObjs.*2 places'):
        prep_join.join_for_places(melee_prep.drawn_meshes(), [Slot(0, i, (0,)) for i in range(4)], tables,
                                  keep=['eyes'])


@pytest.mark.parametrize('shared', [False, True])
def test_a_material_short_of_its_places_tobjs_is_padded_never_atlased(shared):
    """Pikachu's places before its eyes all have a TObj, and the quadrupeds' white parts have none: with every place
    there drawn, no blank DObj carries the TObj they lack, and the untextured material is padded
    (dobjs.texture_counts). Where two meshes look alike, a place is left over and carries it."""
    import bpy
    slots = [Slot(0, i, (0,)) for i in range(3)] + [Slot(0, 3, (0,), animated=(0,))]
    tables = FighterTables(animated=(3,), body=(0, 1, 2, 3))
    _empty_scene()
    white = bpy.data.materials.new('white')
    skin = _material('skin', _flat(4, 4))
    fur = skin if shared else _material('fur', _flat(4, 4, (90, 60, 30, 255)))
    for i, (name, mat) in enumerate((('claws', white), ('body', skin), ('tail', fur))):
        ob = _mesh(name, [(i, 0, 0), (i, 1, 0), (i, 0, 1)], [(0, 1, 2)], [mat])
        ob.vertex_groups.new(name='JOBJ_0').add([0, 1, 2], 1, 'REPLACE')
        ob.data.uv_layers.new().data.foreach_set('uv', [0, 0, 1, 0, 0, 1])
    record = prep_join.join_for_places(melee_prep.drawn_meshes(), slots, tables,
                                       formats={'skin': 'CMPR', 'fur': 'CMPR'})
    assert record['looks']['after'] == (2 if shared else 3) and record['dobjs'] == 3 and not record.get('atlases')
