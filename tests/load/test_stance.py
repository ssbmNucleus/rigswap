import numpy as np
import pytest

from rigswap.load import stance, texture

pytestmark = pytest.mark.blender


def scene(child=False):
    """Two bones side by side, or with ``child`` the second hanging off the first."""
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    data = bpy.data.armatures.new('rig')
    arm = bpy.data.objects.new('rig', data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bones = []
    for name, x in [('body', 0), ('hidden', 2)]:
        bone = data.edit_bones.new(name)
        bone.head, bone.tail = (x, 0, 0), (x, 1, 0)
        bones.append(bone)
    if child:
        bones[1].parent = bones[0]
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def triangle(name, bone, arm, x=0):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x, 0, 0), (x + 1, 0, 0), (x, 1, 0)], [], [(0, 1, 2)])
    me.normals_split_custom_set([(0, 0, 1)] * 3)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if arm:
        ob.vertex_groups.new(name=bone).add([0, 1, 2], 1, 'REPLACE')
        ob.modifiers.new('skin', 'ARMATURE').object = arm
    return ob


def test_idle_uses_mean_frame_bakes_normals_and_keeps_zero_scale_joint():
    import bpy
    arm = scene()
    body = triangle('body', 'body', arm)
    triangle('prop', 'hidden', arm, 2)
    pb = arm.pose.bones['body']
    for frame, z in [(0, 0), (1, 1), (2, 5)]:
        pb.location.z = z
        pb.keyframe_insert('location', frame=frame)
        arm.pose.bones['hidden'].scale = (0, 0, 0)
        arm.pose.bones['hidden'].keyframe_insert('scale', frame=frame)
    arm.animation_data.action.name = 'actor_Idle'
    record = stance.settle()
    assert record['frame'] == 1 and record['hidden'] == ['prop']
    assert set(arm.data.bones.keys()) == {'body', 'hidden'}
    assert arm.data.bones['hidden'].length == pytest.approx(1)
    assert arm.animation_data is None
    assert np.allclose(arm.data.bones['body'].head_local, (0, 0, 1))
    assert body.data.has_custom_normals
    assert np.allclose([v.co.z for v in body.data.vertices], 1)
    graph = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(graph)
    assert np.allclose([v.co[:] for v in ev.data.vertices], [v.co[:] for v in body.data.vertices])
    assert np.allclose([n.vector[:] for n in body.data.corner_normals], [(0, 0, 1)] * 3)


def test_no_named_idle_leaves_pose_animation_and_skin_untouched():
    arm = scene()
    ob = triangle('body', 'body', arm)
    arm.pose.bones['body'].location.z = 2
    arm.pose.bones['body'].keyframe_insert('location', frame=1)
    arm.animation_data.action.name = 'Run'
    action = arm.animation_data.action
    assert stance.settle() is None
    assert arm.animation_data.action == action
    assert arm.pose.bones['body'].location.z == 2 and ob.data.vertices[0].co.z == 0


def test_surface_shown_during_idle_is_not_dropped():
    arm = scene()
    triangle('body', 'body', arm)
    triangle('blink', 'hidden', arm, 2)
    for frame, scale in [(0, 0), (1, 1), (2, 0)]:
        arm.pose.bones['hidden'].scale = (scale,) * 3
        arm.pose.bones['hidden'].keyframe_insert('scale', frame=frame)
    arm.animation_data.action.name = 'Idle'
    assert stance.settle()['hidden'] == []


def test_copies_a_rounding_apart_share_skin_before_pose():
    arm = scene()
    a, b = triangle('a', 'body', arm), triangle('b', 'hidden', arm)
    b.data.vertices[0].co.x += 1e-7
    arm.pose.bones['hidden'].location.z = 1
    arm.pose.bones['hidden'].keyframe_insert('location', frame=0)
    arm.animation_data.action.name = 'Idle'
    stance.settle()
    assert np.allclose([v.co[:] for v in a.data.vertices], [v.co[:] for v in b.data.vertices], atol=1e-6)
    assert np.allclose([v.co.z for v in a.data.vertices], .5)
    assert a.data.has_custom_normals and b.data.has_custom_normals


def test_points_that_are_not_copies_keep_their_skin():
    arm = scene()
    a, b = triangle('a', 'body', arm), triangle('b', 'hidden', arm, 1e-3)
    arm.pose.bones['hidden'].location.z = 1
    arm.pose.bones['hidden'].keyframe_insert('location', frame=0)
    arm.animation_data.action.name = 'Idle'
    stance.settle()
    assert np.allclose([v.co.z for v in a.data.vertices], 0) and np.allclose([v.co.z for v in b.data.vertices], 1)


def test_uv_mapping_bakes_on_only_its_material_faces_without_idle():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    ob = triangle('mesh', None, None)
    me = ob.data
    me.clear_geometry()
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [], [(0, 1, 2), (1, 3, 2)])
    me.uv_layers.new(name='UVMap')
    before = np.array([(0, 0), (1, 0), (0, 1)] * 2, float)
    me.uv_layers.active.data.foreach_set('uv', before.ravel())
    for name in ('transformed', 'plain'):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        me.materials.append(mat)
    me.polygons[1].material_index = 1
    tree = me.materials[0].node_tree
    uv = tree.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'UVMap'
    mapping = tree.nodes.new('ShaderNodeMapping')
    mapping.inputs['Location'].default_value = (.2, .3, 0)
    mapping.inputs['Scale'].default_value = (2, 3, 1)
    mapping.inputs['Rotation'].default_value.z = np.pi / 2
    image = tree.nodes.new('ShaderNodeTexImage')
    tree.links.new(uv.outputs['UV'], mapping.inputs['Vector'])
    tree.links.new(mapping.outputs['Vector'], image.inputs['Vector'])
    tree.links.new(image.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    assert stance.settle() is None
    assert len(texture.bake()) == 1
    after = np.array([d.uv[:] for d in me.uv_layers.active.data])
    assert np.allclose(after[:3], [[.2, .3], [.2, 2.3], [-2.8, .3]], atol=1e-6)
    assert np.array_equal(after[3:], before[3:])
    assert not any(n.type == 'MAPPING' for n in tree.nodes)
    assert texture.bake() == []


def test_a_transform_on_a_normal_map_moves_no_uvs():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    me = triangle('mesh', None, None).data
    me.uv_layers.new(name='UVMap')
    mat = bpy.data.materials.new('m')
    mat.use_nodes = True
    me.materials.append(mat)
    tree = mat.node_tree
    uv, mapping = tree.nodes.new('ShaderNodeUVMap'), tree.nodes.new('ShaderNodeMapping')
    mapping.inputs['Location'].default_value = (.5, 0, 0)
    bump = tree.nodes.new('ShaderNodeTexImage')
    normal = tree.nodes.new('ShaderNodeNormalMap')
    tree.links.new(uv.outputs['UV'], mapping.inputs['Vector'])
    tree.links.new(mapping.outputs['Vector'], bump.inputs['Vector'])
    tree.links.new(bump.outputs['Color'], normal.inputs['Color'])
    tree.links.new(normal.outputs['Normal'], tree.nodes['Principled BSDF'].inputs['Normal'])
    assert texture.bake() == []


def test_a_rotated_idle_preserves_the_evaluated_custom_normals():
    arm = scene()
    ob = triangle('body', 'body', arm)
    arm.pose.bones['body'].rotation_mode = 'XYZ'
    arm.pose.bones['body'].rotation_euler.x = np.pi / 2
    arm.pose.bones['body'].keyframe_insert('rotation_euler', frame=0)
    arm.animation_data.action.name = 'Idle'
    stance.settle()
    assert ob.data.has_custom_normals
    assert np.allclose([n.vector[:] for n in ob.data.corner_normals], [(0, -1, 0)] * 3, atol=1e-6)


def turned_child(parent_scale, child_scale, turn=np.radians(37)):
    """A child turned about z under a scaled parent, keyed as a TRS file stores it; its triangle sits at its head."""
    arm = scene(child=True)
    triangle('body', 'body', arm)
    ob = triangle('hand', 'hidden', arm, 2)
    body, hand = arm.pose.bones['body'], arm.pose.bones['hidden']
    body.scale = parent_scale
    hand.rotation_mode = 'XYZ'
    hand.rotation_euler.z = turn
    hand.scale = child_scale
    body.keyframe_insert('scale', frame=0)
    hand.keyframe_insert('rotation_euler', frame=0)
    hand.keyframe_insert('scale', frame=0)
    arm.animation_data.action.name = 'Idle'
    return arm, ob


def test_a_child_of_a_parent_folded_flat_stays_flat_where_the_product_would_spike():
    # the shape of a game's hidden joint in a TRS file: the parent folded on one axis, the child's channel what its
    # axes measure in that parent's, hundreds of times its length
    arm, ob = turned_child((1e-3, 1, 1), (.8, 600, 1))
    stance.settle()
    points = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
    assert np.ptp(points, axis=0).max() < 1.5
    assert arm.data.bones['hidden'].inherit_scale == 'FULL'


def test_a_child_of_an_even_parent_is_the_matrix_product():
    arm, ob = turned_child((2, 2, 2), (1, 3, 1), np.radians(30))
    rest = [v.co.copy() for v in ob.data.vertices]
    stance.settle()
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    head = np.array([4., 0, 0])                 # the child's head, 2 out from the parent's, doubled
    want = [head + 2 * turn @ (np.array([1, 3, 1]) * (np.array(v) - (2, 0, 0))) for v in rest]
    assert np.allclose([v.co[:] for v in ob.data.vertices], want, atol=1e-5)


def test_a_turned_child_of_an_uneven_parent_has_no_shear():
    arm, ob = turned_child((1, 2, 1), (1, 1, 1))
    stance.settle()
    a, b, c = (np.array(v.co) for v in ob.data.vertices)
    assert np.dot(b - a, c - a) == pytest.approx(0, abs=1e-5)
