from types import SimpleNamespace

import numpy as np
import pytest

from rigswap.core import variants


def test_a_name_without_the_additions_names_the_switch_or_part():
    plain = variants.plain
    assert plain('Mario_Blink_VIS_O_OBJShape.001') == 'Mario_Blink'
    assert plain('Roy_Blink_VIS_O_OBJ_O_NOSORT_FAR') == 'Roy_Blink'
    assert plain('Mario_Eye_VIS_O_OBJ__Layer_2') == 'Mario_Eye'
    assert plain('donkey_Rarm_O_OBJ_O_SORTEACHNODEShape1') == 'donkey_Rarm'
    assert plain('mario_cap_on') == 'mario_cap_on' and plain('mesh15_001') == 'mesh15_001'


def test_a_level_of_detail_is_read_from_its_word_or_number():
    level = variants.level
    assert level('body_c00_high') == ('body c', 0) and level('body_c00_low') == ('body c', 2)
    assert level('hairLow') == ('hair', 2) and level('koopa_Shell_low') == ('koopa shell', 2)
    assert level('Body_LOD1') == ('body', 1)
    for name in ('lower_arm', 'Hollow', 'Below', 'Highlight', 'midriff', 'Body'):
        assert level(name) is None


def _unit(key, lo, hi, **kw):
    u = SimpleNamespace(key=key, lo=np.array(lo, float), hi=np.array(hi, float), meshes=[key], switched=False,
                        bone=0, bones={0}, triangles=100, extents=np.array([1., 2., 3.]))
    for k, v in kw.items():
        setattr(u, k, v)
    return u


def test_only_a_coarser_copy_that_fills_the_finer_ones_box_goes():
    us = [_unit('body_c00_highShape', [0, 0, 0], [1, 1, 2]), _unit('body_c00_lowShape', [0, 0, 0], [1, 1, 2]),
          _unit('koopa_Shell', [2, 0, 0], [3, 1, 1]), _unit('koopa_Shell_lowShape', [2, 0, 0], [3, 1, 1]),
          _unit('arm_high', [0, 0, 0], [1, 1, 1]), _unit('arm_low', [0, 0, -2], [1, 1, -1])]
    assert variants.coarser(us, 3.0) == {'body_c00_lowShape': 'body_c00_highShape',
                                         'koopa_Shell_lowShape': 'koopa_Shell'}


def test_the_look_is_told_what_the_fighter_holds():
    # Ocarina of Time's Link idles with his sword and shield on his back; Link holds them through play
    assert variants.holding([]) == ''
    both = variants.holding(['sword', 'shield'])
    assert 'its sword and its shield, one in each hand' in both and 'stowed on its back or in a sheath' in both
    assert 'its sword in its hand' in variants.holding(['sword']) and 'each' not in variants.holding(['sword'])


class _Piece(SimpleNamespace):
    __hash__, __eq__ = object.__hash__, object.__eq__      # one piece is itself, as variants.Unit is


@pytest.mark.blender
def test_one_name_on_bones_apart_is_a_piece_in_each_place_and_on_bones_in_common_one_piece():
    import bpy
    from rigswap.core import rig as rigs
    bpy.ops.wm.read_homefile(use_empty=True)
    arm = bpy.data.objects.new('skeleton', bpy.data.armatures.new('skeleton'))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    spine = arm.data.edit_bones.new('spine')
    spine.head, spine.tail = (0, 0, 0), (0, 0, 1)
    for name, x in (('l_hand', 1), ('r_hand', -1)):
        b = arm.data.edit_bones.new(name)
        b.head, b.tail, b.parent = (x, 0, 1), (2 * x, 0, 1), spine
    bpy.ops.object.mode_set(mode='OBJECT')

    def mesh(name, bone, x):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(x, 0, 0), (x + 0.5, 0, 0), (x, 0, 0.5)], [], [(0, 1, 2)])
        ob = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(ob)
        ob.vertex_groups.new(name=bone).add([0, 1, 2], 1.0, 'REPLACE')
        ob.modifiers.new('Armature', 'ARMATURE').object = arm
        return ob
    obs = [mesh('body', 'spine', 0), mesh('body.001', 'spine', 0.2),       # one piece cut by material
           mesh('gu.001', 'r_hand', -1.5), mesh('gu', 'l_hand', 1.5)]      # Frieza's fists, one in each hand
    us, named = variants.units(rigs.extract('synthetic', arm, obs))
    by_key = {u.key: u for u in us}
    assert not named and sorted(by_key) == ['body', 'gu', 'gu.001']
    assert by_key['body'].meshes == ['body', 'body.001']
    assert by_key['gu'].meshes == ['gu'] and by_key['gu.001'].meshes == ['gu.001'], 'the bare name keeps its key'
    assert by_key['gu'].bones != by_key['gu.001'].bones


@pytest.mark.blender
def test_a_file_of_one_mesh_is_offered_the_surfaces_it_is_welded_from_and_names_them_the_same_each_time():
    """Taz's OBJ is one mesh with his go-kart, its wheels and a floor shadow: the surfaces welded apart, those painted
    alike together, are pieces of their own, largest first, the largest keeping the mesh's name, weights and
    materials going with each; a seam the file split (copies of a corner in one place) parts nothing, and a crumb too
    small to show goes with the seen surface nearest it."""
    import bpy
    from rigswap.core import look, rig as rigs
    bpy.ops.wm.read_homefile(use_empty=True)
    arm = bpy.data.objects.new('skeleton', bpy.data.armatures.new('skeleton'))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    arm.data.edit_bones.new('hips').tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    crumb = 10. / look.VIEW / 4
    body = [(0, 0, 0), (1, 0, 0), (0, 0, 2), (1, 0, 0), (0, 0, 2), (1, 0, 2)]    # two faces, their seam's corners copied
    kart = [(3, 0, 0), (9, 0, 0), (3, 4, 0), (9, 4, 0), (6, -2, 0)]             # four faces across the floor
    wheels = [(3, 5, 0), (4, 5, 0), (3, 5, 1), (8, 5, 0), (9, 5, 0), (8, 5, 1)]  # two apart, painted alike
    speck = [(0.5, 0.5, 1), (0.5 + crumb, 0.5, 1), (0.5, 0.5 + crumb, 1)]
    faces = [(0, 1, 2), (3, 5, 4), (6, 7, 8), (7, 9, 8), (6, 10, 7), (7, 10, 9), (11, 12, 13), (14, 15, 16),
             (17, 18, 19)]
    paints = [0, 0, 1, 1, 1, 1, 2, 2, 1]
    me = bpy.data.meshes.new('taz')
    me.from_pydata(body + kart + wheels + speck, [], faces)
    for name in ('fur', 'paint', 'rubber'):
        me.materials.append(bpy.data.materials.new(name))
    me.polygons.foreach_set('material_index', paints)
    ob = bpy.data.objects.new('taz', me)
    bpy.context.collection.objects.link(ob)
    ob.vertex_groups.new(name='hips').add(list(range(20)), 1.0, 'REPLACE')
    ob.modifiers.new('Armature', 'ARMATURE').object = arm
    first = rigs.extract('synthetic', arm, [ob])
    corners = [tuple(v.co) for v in me.vertices]
    parted, made = variants.surfaces(first)
    assert sorted(parted.meshes) == ['taz', 'taz 2', 'taz 3']
    count = {n: len(bpy.data.objects[n].data.polygons) for n in parted.meshes}
    assert count == {'taz': 4, 'taz 2': 3, 'taz 3': 2}, 'the kart; the body, its seam whole, and the speck by it; the wheels'
    painted = {n: {bpy.data.objects[n].data.materials[p.material_index].name
                   for p in bpy.data.objects[n].data.polygons} for n in parted.meshes}
    assert painted == {'taz': {'paint'}, 'taz 2': {'fur', 'paint'}, 'taz 3': {'rubber'}}
    assert len(parted.points) == 20
    for n in parted.meshes:
        part = bpy.data.objects[n]
        assert part.vertex_groups[0].name == 'hips' and part.modifiers[0].object == arm
    assert {u.key for u in variants.units(parted)[0]} == {'taz', 'taz 2', 'taz 3'}
    same, none = variants.surfaces(parted)
    assert same is parted and none is None, 'a file of more meshes is as it was'
    # the look over, the pieces kept are the file's one mesh again: without the kart and its wheels, him and his speck
    left = variants.whole(variants.drop(parted, ['taz', 'taz 3']), made)
    assert left.meshes == ['taz'] and [o.name for o in bpy.data.objects if o.type == 'MESH'] == ['taz']
    him = bpy.data.objects['taz']
    assert him.data.name == 'taz' and len(him.data.polygons) == 3 and len(left.points) == 9
    assert {him.data.materials[p.material_index].name for p in him.data.polygons} == {'fur', 'paint'}
    assert him.vertex_groups[0].name == 'hips' and him.modifiers[0].object == arm
    assert variants.whole(left, None) is left
    # nothing left out: the mesh as the file has it, corner for corner
    bpy.data.objects.remove(him)
    me = bpy.data.meshes.new('taz')
    me.from_pydata(body + kart + wheels + speck, [], faces)
    for name in ('fur', 'paint', 'rubber'):
        me.materials.append(bpy.data.materials[name])
    me.polygons.foreach_set('material_index', paints)
    ob = bpy.data.objects.new('taz', me)
    bpy.context.collection.objects.link(ob)
    ob.vertex_groups.new(name='hips').add(list(range(20)), 1.0, 'REPLACE')
    ob.modifiers.new('Armature', 'ARMATURE').object = arm
    parted, made = variants.surfaces(rigs.scene('synthetic'))
    same = variants.whole(parted, made)
    assert same.meshes == ['taz'] and [tuple(v.co) for v in bpy.data.objects['taz'].data.vertices] == corners
    assert [tuple(q.vertices) for q in bpy.data.objects['taz'].data.polygons] == faces
    # one mesh painted alike all over (a head of hair cards) is one piece
    bpy.data.objects['taz'].data.polygons.foreach_set('material_index', [0] * len(faces))
    alike = rigs.scene('synthetic')
    same, none = variants.surfaces(alike)
    assert same is alike and none is None
