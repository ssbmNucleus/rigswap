import numpy as np
import pytest

from rigswap.core import refine


@pytest.mark.parametrize('source,target,expected', [(700, 4995, 1), (1000, 9000, 1), (100, 1400, 2),
                                                   (8000, 4995, 0), (100, 250, 0)])
def test_cuts_are_nearest_in_triangle_count(source, target, expected):
    assert refine.cuts(source, target) == expected


def triangle(name, normals, points=((0, 0, 0), (2, 0, 0), (0, 2, 0))):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata(points, [], [(0, 1, 2)])
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    me.polygons[0].use_smooth = True
    me.normals_split_custom_set(normals)
    return ob


@pytest.mark.blender
def test_new_points_keep_uvs_materials_attributes_and_interpolate_skin():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    ob = triangle('curved', [(-.6, 0, .8), (.6, 0, .8), (0, .6, .8)])
    me = ob.data
    uv = me.uv_layers.new(name='paint')
    uv.data.foreach_set('uv', [0, 0, 1, 0, 0, 1])
    me.attributes.new('region', 'INT', 'FACE').data[0].value = 7
    mat = bpy.data.materials.new('paint')
    me.materials.append(mat)
    ob.vertex_groups.new(name='parent').add([0, 2], 1., 'REPLACE')
    ob.vertex_groups[0].add([1], .5, 'REPLACE')
    ob.vertex_groups.new(name='child').add([1], .5, 'REPLACE')
    ob.shape_key_add(name='Basis')
    key = ob.shape_key_add(name='blink')
    for v in key.data:
        v.co.z += .2
    report = refine.resolution([ob], [many(4)])
    assert report['cuts'] == 1 and len(me.polygons) == 4 and len(me.vertices) == 6
    assert me.materials[0] == mat
    assert {p.value for p in me.attributes['region'].data} == {7}
    mid = next(v for v in me.vertices if abs(v.co.x - 1) < 1e-3 and abs(v.co.y) < 1e-3)
    assert mid.co.z > .1  # the PN curve leaves the flat chord, keeping its original endpoints
    assert sorted(g.weight for g in mid.groups) == pytest.approx([.25, .75])
    for poly in me.polygons:
        for index in poly.loop_indices:
            if me.loops[index].vertex_index == mid.index:
                assert tuple(me.uv_layers['paint'].data[index].uv) == pytest.approx((.5, 0))
    for basis, key in zip(me.shape_keys.key_blocks[0].data, me.shape_keys.key_blocks[1].data):
        assert np.array(key.co) - np.array(basis.co) == pytest.approx([0, 0, .2], abs=1e-6)
    assert all(np.isfinite(n.vector).all() for n in me.corner_normals)


@pytest.mark.blender
@pytest.mark.parametrize('crease', [False, True])
def test_split_copies_share_midpoints_but_authored_creases_stay_straight(crease):
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    a = triangle('upper', [(-.6, 0, .8), (.6, 0, .8), (0, .6, .8)])
    b = triangle('lower', [(0, 0, -1)] * 3 if crease else [(.6, 0, .8), (-.6, 0, .8), (0, -.6, .8)],
                 [(2, 0, 0), (0, 0, 0), (0, -2, 0)])
    # Eight triangles request exactly one cut for this two-triangle surface.
    target = triangle('target', [(0, 0, 1)] * 3)
    target.data.clear_geometry()
    target.data.from_pydata([(i, y, 0) for i in range(8) for y in (0, 1, 2)], [],
                            [(3*i, 3*i+1, 3*i+2) for i in range(8)])
    refine.resolution([a, b], [target])
    mids = [next(v.co[:] for v in ob.data.vertices if abs(v.co.x - 1) < 1e-3 and abs(v.co.y) < 1e-3)
            for ob in (a, b)]
    assert mids[0] == pytest.approx(mids[1], abs=1e-6)
    assert abs(mids[0][2]) < 1e-6 if crease else mids[0][2] > .1


@pytest.mark.blender
def test_copies_of_a_seam_stay_exactly_one_place():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    a, b = (.1234567, .7654321, .3333333), (1.2345678, .1111111, .9876543)
    up, side = (.2345678, 1.3456789, .5555555), (.8765432, -1.2222222, .4444444)
    normals = [(-.6, 0, .8), (.6, 0, .8), (0, .6, .8)]
    upper = triangle('upper', normals, (a, b, up))
    lower = triangle('lower', [normals[1], normals[0], (0, -.6, .8)], (b, a, side))
    target = triangle('target', [(0, 0, 1)] * 3)
    target.data.clear_geometry()
    target.data.from_pydata([(i, y, 0) for i in range(8) for y in (0, 1, 2)], [],
                            [(3*i, 3*i+1, 3*i+2) for i in range(8)])
    refine.resolution([upper, lower], [target])
    points = [{v.co[:] for v in ob.data.vertices} for ob in (upper, lower)]
    assert len(points[0] & points[1]) == 3     # the seam's two ends and its new point, to the last bit


def fold(crease):
    """Two faces in two meshes, meeting square along an edge."""
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    a = triangle('flat', [(0, 0, 1), (0, 0, 1), (0, 0, 1)] if crease else [(.6, 0, .8), (.6, 0, .8), (0, 0, 1)],
                 [(0, 0, 0), (0, 2, 0), (-2, 1, 0)])
    b = triangle('wall', [(1, 0, 0), (1, 0, 0), (1, 0, 0)] if crease else [(.6, 0, .8), (.6, 0, .8), (1, 0, 0)],
                 [(0, 2, 0), (0, 0, 0), (0, 1, -2)])
    return [a, b]


@pytest.mark.blender
def test_shading_reads_the_surface_and_every_corner_at_a_place_shares_it():
    meshes = fold(crease=False)
    files = [refine.stored(ob) for ob in meshes]
    refine.shade(meshes, files)
    half = np.sqrt(.5)
    for ob in meshes:
        at = {tuple(ob.data.vertices[ob.data.loops[i].vertex_index].co): n.vector[:]
              for i, n in enumerate(ob.data.corner_normals)}
        assert at[0, 0, 0] == pytest.approx((half, 0, half), abs=1e-3)
        assert at[0, 2, 0] == pytest.approx((half, 0, half), abs=1e-3)
    assert meshes[0].data.corner_normals[2].vector[:] == pytest.approx((0, 0, 1), abs=1e-3)


@pytest.mark.blender
def test_corners_the_file_kept_apart_stay_apart():
    meshes = fold(crease=True)
    refine.shade(meshes, [refine.stored(ob) for ob in meshes])
    assert all(n.vector[:] == pytest.approx((0, 0, 1), abs=1e-3) for n in meshes[0].data.corner_normals)
    assert all(n.vector[:] == pytest.approx((1, 0, 0), abs=1e-3) for n in meshes[1].data.corner_normals)


@pytest.mark.blender
def test_dense_mesh_is_untouched_and_multiple_cuts_are_four_way():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    ob = triangle('model', [(0, 0, 1)] * 3)
    assert refine.resolution([ob], [ob])['cuts'] == 0
    assert len(ob.data.polygons) == 1
    target = triangle('target', [(0, 0, 1)] * 3)
    target.data.clear_geometry()
    target.data.from_pydata([(i, y, 0) for i in range(16) for y in (0, 1, 2)], [],
                            [(3*i, 3*i+1, 3*i+2) for i in range(16)])
    assert refine.resolution([ob], [target])['triangles'] == 16
    assert all(len(p.vertices) == 3 for p in ob.data.polygons)


def many(count):
    """A mesh of ``count`` disjoint triangles, only its count matters."""
    target = triangle('target', [(0, 0, 1)] * 3)
    target.data.clear_geometry()
    target.data.from_pydata([(i, y, 0) for i in range(count) for y in (0, 1, 2)], [],
                            [(3*i, 3*i+1, 3*i+2) for i in range(count)])
    return target


@pytest.mark.blender
def test_a_cut_marks_where_it_crossed_a_hard_seam_and_stays_on_the_surface():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    ob = triangle('ring', [(0, 0, 1)] * 3)
    ob.vertex_groups.new(name='shin').add([0], 1., 'REPLACE')
    ob.vertex_groups.new(name='thigh').add([1, 2], 1., 'REPLACE')
    refine.resolution([ob], [many(4)])
    me = ob.data
    crossed = refine.seam([ob])
    assert sorted(tuple(np.round(me.vertices[i].co[:2], 3)) for i in crossed) == [(0, 1), (1, 0)]
    for i in crossed:
        assert sorted(g.weight for g in me.vertices[i].groups) == pytest.approx([.5, .5])
    assert sum(len(v.groups) == 1 for v in me.vertices) == 4      # the three corners, and the thigh's own edge


@pytest.mark.blender
def test_further_cuts_keep_the_seam_a_line_and_the_rest_each_ends_own():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    ob = triangle('ring', [(0, 0, 1)] * 3)
    ob.vertex_groups.new(name='shin').add([0], 1., 'REPLACE')
    ob.vertex_groups.new(name='thigh').add([1, 2], 1., 'REPLACE')
    assert refine.resolution([ob], [many(14)])['cuts'] == 2
    me = ob.data
    crossed = set(refine.seam([ob]).tolist())
    assert len(crossed) == 3                                      # the two crossings and the point between them
    for v in me.vertices:
        assert (v.index in crossed) == (len(v.groups) == 2)


def test_blend_starts_its_band_at_a_point_a_cut_marked():
    from types import SimpleNamespace
    from rigswap.core import repose
    # A strip across a joint at x=2, the cut's point in the middle. Interpolated, it reads as an authored blend.
    points = np.array([[x, 0., 0.] for x in range(5)])
    w = np.array([[1, 0], [1, 0], [.5, .5], [0, 1], [0, 1.]])
    rig = SimpleNamespace(points=points, edges=[(i, i + 1) for i in range(4)])
    embedded = {'a': (np.zeros(3), np.array([8., 0, 0])), 'b': (np.zeros(3), np.array([8., 0, 0]))}
    unmarked, _ = repose.Surface(rig).blend(w, ['a', 'b'], embedded)
    assert unmarked == pytest.approx(w)
    marked, report = repose.Surface(rig, [2]).blend(w, ['a', 'b'], embedded)
    assert marked[:, 1] == pytest.approx([0, .25, .5, .75, 1.])
    assert report['pairs'][0]['bones'] == ['a', 'b']
