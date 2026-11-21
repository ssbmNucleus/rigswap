import json
import numpy as np
import pytest

from rigswap.core import rig


def _load(tmp_path, entries):
    (tmp_path / 'models.json').write_text(json.dumps(
        [dict(e, armatures=1, skinned=True, bones=e.get('bones', 27)) for e in entries]), encoding='utf-8')
    return tmp_path


def test_a_variant_extending_the_base_name_is_passed_over_for_the_base(tmp_path):
    load = _load(tmp_path, [{'file': 'Rosetta.dae', 'triangles': 4741}, {'file': 'Rosetta_bake.dae', 'triangles': 9482},
                            {'file': 'RosettaReading.dae', 'triangles': 5410}])
    assert rig.pick(load)['file'] == 'Rosetta.dae'


def test_a_3ds_rip_split_by_uv_layer_is_the_model_itself(tmp_path):
    load = _load(tmp_path, [{'file': 'pc03_m4.fbx', 'triangles': 820}, {'file': 'pc03_m4_uvsplit.fbx', 'triangles': 1640},
                            {'file': 'pc03_m4.dae', 'triangles': 820}])
    assert rig.pick(load)['file'] == 'pc03_m4_uvsplit.fbx'


@pytest.mark.blender
def test_a_mesh_bound_to_nothing_rides_the_root():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    arm = bpy.data.objects.new('skeleton', bpy.data.armatures.new('skeleton'))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    spine = arm.data.edit_bones.new('spine')
    spine.head, spine.tail = (0, 0, 0), (0, 0, 1)
    hand = arm.data.edit_bones.new('hand')
    hand.head, hand.tail, hand.parent = (1, 0, 1), (2, 0, 1), spine
    bpy.ops.object.mode_set(mode='OBJECT')

    def mesh(name):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 0, 1)], [], [(0, 1, 2)])
        ob = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(ob)
        return ob
    body = mesh('body')
    body.vertex_groups.new(name='hand').add([0, 1, 2], 1.0, 'REPLACE')
    body.modifiers.new('Armature', 'ARMATURE').object = arm
    mesh('staff')                                       # Pink's: no skin, no parent
    read = rig.extract('synthetic', arm, [bpy.data.objects['body'], bpy.data.objects['staff']])
    assert read.meshes == ['body', 'staff']
    assert np.allclose(read.weights[:3], [[0, 1]] * 3) and np.allclose(read.weights[3:], [[1, 0]] * 3)


@pytest.mark.blender
@pytest.mark.parametrize('builder', ['onepiece', 'segments'])
def test_builder_twins_and_axis_survive_scene_rereading(builder):
    import bpy
    from mathutils import Matrix
    from rigswap.core import onepiece, segments
    heads = np.array([[0., 0, 0], [1, 0, 0], [-1, 0, 0]])
    points = np.vstack([h + [[0, 0, 0], [.1, 0, 0], [0, 0, .1]] for h in heads])
    data = bpy.data.meshes.new('statements-test')
    data.from_pydata(points.tolist(), [], [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    ob = bpy.data.objects.new('statements-test', data)
    bpy.context.collection.objects.link(ob)
    parents, tails = np.array([-1, 0, 0]), heads + [0, 0, .2]
    twin, axis = {0: 0, 1: 2, 2: 1}, np.array([1., 0, 0])
    if builder == 'onepiece':
        skel = onepiece.Skeleton(parents, heads, tails, None, {}, np.ones(3), nodes=[0, 1, 2], twins=twin)
        arm = onepiece.rig_up([ob], [9], np.repeat(np.eye(3), 3, axis=0), skel, axis)
    else:
        arm = segments._rig_up([ob], [9], np.repeat(np.arange(3), 3), parents, heads, tails,
                               np.array([0, 2, 1]), axis)
    try:
        names = list(json.loads(arm['rigswap_twins']))
        first = rig.scene('synthetic')
        # A later scene read sees the same statements, in transformed world space.
        arm.matrix_world = Matrix.Rotation(np.pi / 2, 4, 'Z')
        again = rig.scene('synthetic')
        for read in (first, again):
            stated = {read.names[a]: read.names[b] for a, b in read.twins.items()}
            assert stated == {names[0]: names[0], names[1]: names[2], names[2]: names[1]}
        assert np.allclose(first.mirror, [1, 0, 0])
        assert np.allclose(again.mirror, [0, 1, 0], atol=1e-6)
    finally:
        bpy.data.objects.remove(arm, do_unlink=True)
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.context.view_layer.update()     # else the next case walks the view layer's stale list (None)
