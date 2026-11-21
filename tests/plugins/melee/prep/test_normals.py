"""prep/normals.py: no normal without a direction."""
import numpy as np
import pytest

from rigswap.plugins.melee.prep import normals as prep_normals
from tests.plugins.melee.prep.test_prep import _empty_scene, _mesh


def test_a_sliver_is_removed():
    """A face with no width draws nothing; the .dae's float text rounds its short edge to a point, and the import
    rejects the zero normal there. Three corners on one line, or two a hair apart, both go; a real face stays."""
    _empty_scene()
    ob = _mesh('sliver', [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 1 + 5e-7, 0), (1, 2, 0), (5, 5, 0),
                          (6, 5, 0), (5, 6, 0)], [(0, 1, 2), (3, 4, 5), (6, 7, 8)], smooth=True)
    record = prep_normals.repair_normals([ob])
    assert record['collapsed_faces_removed'] == 2
    assert len(ob.data.polygons) == 1
    assert np.allclose(np.linalg.norm(prep_normals.corner_normals(ob.data), axis=1), 1, atol=1e-4)


def test_a_speck_the_written_file_cannot_hold_is_removed():
    """A face no wider on any axis than twice the written position error may be written as a point, or turned over
    (Taz's source crumbs, a thousandth of a unit across, came back collapsed): it goes. A face that wide on one axis
    stays, however thin, and so does one as small but in a scene scaled up past the step."""
    bpy = _empty_scene()
    step = 2 * prep_normals.POSITION_ERROR
    ob = _mesh('specks', [(0, 0, 0), (0.6 * step, 0, 0), (0, 0.9 * step, 0.3 * step),                # a speck
                          (5, 5, 5), (5 + 1.5 * step, 5, 5), (5, 5 + 0.2 * step, 5),                  # wider on x
                          (9, 9, 9), (10, 9, 9), (9, 10, 9)], [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    big = _mesh('scaled', [(0, 0, 0), (0.6 * step, 0, 0), (0, 0.9 * step, 0)], [(0, 1, 2)])
    big.scale = (4, 4, 4)
    bpy.context.view_layer.update()
    record = prep_normals.repair_normals([ob, big])
    assert record['meshes']['specks']['collapsed_faces_removed'] == 1
    assert len(ob.data.polygons) == 2 and len(ob.data.vertices) == 6
    assert len(big.data.polygons) == 1
    assert prep_normals.speck([(0, 0, 0), (step, 0, 0), (0, step, step)])
    assert not prep_normals.speck([(0, 0, 0), (1.01 * step, 0, 0), (0, 0, 0)])


def test_a_card_folded_back_to_back_is_shaded_flat_where_its_corners_have_no_direction():
    """Two faces on one edge facing opposite ways, smooth: the fan at the edge's ends sums to nothing, so no custom
    normal can be held there; those faces are shaded flat and each corner takes its face's normal."""
    _empty_scene()
    # an edge along x, and a tip above its far end on each side, a hair apart: two faces nearly back to back
    ob = _mesh('card', [(1, 0.04, 0.47), (0, 0, 0), (1, 0, 0), (1, 0.038, 0.472)], [(0, 1, 2), (3, 2, 1)],
               smooth=True)
    ob.data.normals_split_custom_set([(1, 0, 0)] * len(ob.data.loops))     # as the file's own normals came
    before = prep_normals.corner_normals(ob.data)
    bad = ~prep_normals.usable_normals(before)
    assert bad.any()
    record = prep_normals.repair_normals([ob])
    assert record['meshes']['card']['shaded_flat'] == 2
    normals = prep_normals.corner_normals(ob.data)
    assert prep_normals.usable_normals(normals).all()
    for p in ob.data.polygons:
        for i in range(p.loop_start, p.loop_start + p.loop_total):
            # a corner that had no direction takes its face's normal; the others keep theirs
            assert np.allclose(normals[i], p.normal if bad[i] else before[i], atol=1e-3)


def test_collapsed_corners_and_loose_vertices_are_removed_without_losing_surface_data(tmp_path):
    bpy = _empty_scene()
    ob = _mesh('collapsed', [(0, 0, 0), (0, 0, 0), (0, 1, 0), (1, 0, 0), (9, 9, 9)],
               [(0, 1, 2), (1, 3, 2)], smooth=True)
    group = ob.vertex_groups.new(name='JOBJ_0')
    group.add(list(range(5)), 1., 'REPLACE')
    uv = ob.data.uv_layers.new(name='UVMap')
    for loop in ob.data.loops:
        uv.data[loop.index].uv = ob.data.vertices[loop.vertex_index].co[:2]
    record = prep_normals.repair_normals([ob])
    assert record['collapsed_faces_removed'] == 1
    assert record['loose_vertices_removed'] == 2
    assert len(ob.data.polygons) == 1 and len(ob.data.vertices) == 3
    assert all(v.groups[0].weight == 1 for v in ob.data.vertices)
    assert all(np.allclose(ob.data.uv_layers[0].data[l.index].uv, ob.data.vertices[l.vertex_index].co[:2])
               for l in ob.data.loops)
    path = str(tmp_path / 'repaired.blend')
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    assert prep_normals.usable_normals(prep_normals.corner_normals(bpy.data.objects['collapsed'].data)).all()
    assert not any(prep_normals.repair_normals([bpy.data.objects['collapsed']])[k]
                   for k in ('repaired', 'collapsed_faces_removed', 'loose_vertices_removed'))


@pytest.mark.parametrize('invalid', [(0., 0., 0.), (float('nan'), 0., 0.), (float('inf'), 0., 0.)])
def test_invalid_normals_use_the_average_of_incident_faces(monkeypatch, invalid):
    _empty_scene()
    ob = _mesh('corner', [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [(0, 1, 2), (0, 3, 1)])
    real = prep_normals.corner_normals
    original = real(ob.data)
    broken = original.copy()
    broken[0] = invalid
    reads = iter([broken])
    monkeypatch.setattr(prep_normals, 'corner_normals', lambda me: next(reads, real(me)))
    record = prep_normals.repair_normals([ob])
    assert record['meshes']['corner']['faces'] == 1
    after = real(ob.data)
    assert np.allclose(after[0], [0., 2 ** -.5, 2 ** -.5], atol=1e-3)
    assert np.allclose(after[1:], original[1:], atol=1e-3)


def test_normal_fallbacks_use_neighbours_then_outward_including_the_centre():
    _empty_scene()
    ob = _mesh('line', [(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, 2)], smooth=True)
    fixes, sources = prep_normals._normal_fallbacks(ob.data, np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]))
    assert sources[0] == sources[2] == 'neighbours'
    assert np.allclose([fixes[0], fixes[2]], [[0, 1, 0], [0, 1, 0]])
    fixes, sources = prep_normals._normal_fallbacks(ob.data, np.full((3, 3), np.nan))
    assert sources == ['outward'] * 3
    assert np.allclose(fixes, [[-1, 0, 0], [-1, 0, 0], [1, 0, 0]])


def test_a_material_whose_every_face_is_taken_out_gets_no_texture_format():
    """A material drawn only on specks has no face once step 5 removes them: textures.json no longer names it, or the
    export fails on a format named for a material the file lacks (Taz's eyes cut from his crumbs, 2026-11-19)."""
    from types import SimpleNamespace
    from rigswap.plugins.melee.prep.textures import texture_formats
    bpy = _empty_scene()
    step = 2 * prep_normals.POSITION_ERROR
    ob = _mesh('eyes', [(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 5), (5 + 0.5 * step, 5, 5), (5, 5 + 0.5 * step, 5)],
               [(0, 1, 2), (3, 4, 5)])
    for name in ('skin', 'eyes'):
        ob.data.materials.append(bpy.data.materials.new(name))
    ob.data.polygons[1].material_index = 1
    textures = [SimpleNamespace(materials=['skin'], code=lambda: 'CMPR'),
                SimpleNamespace(materials=['eyes'], code=lambda: 'RGB5A3')]
    assert texture_formats(textures, [ob]) == {'skin': 'CMPR', 'eyes': 'RGB5A3'}
    prep_normals.repair_normals([ob])
    assert texture_formats(textures, [ob]) == {'skin': 'CMPR'}
    assert texture_formats(textures) == {'skin': 'CMPR', 'eyes': 'RGB5A3'}
