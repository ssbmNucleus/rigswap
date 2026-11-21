"""prep/triangles.py: the triangle budget."""
import numpy as np

from rigswap.plugins.melee.prep import triangles as prep_triangles
from tests.plugins.melee.prep.test_prep import _empty_scene, _grid, _hidden_collection, _keyed


def test_a_mesh_over_the_triangle_budget_is_decimated_to_it_keeping_uvs_and_weights():
    _empty_scene()
    big, small = _grid('big', 30), _grid('small', 10)                   # 1800 and 200 triangles
    record = prep_triangles.limit_triangles([big, small], 1000)
    assert record['before'] == 2000 and record['decimated']
    assert 900 <= record['after'] <= 1000
    assert record['meshes']['big'][1] > 3 * record['meshes']['small'][1]     # each by the same share
    for ob in (big, small):
        assert ob.data.uv_layers and ob.vertex_groups.keys() == ['JOBJ_1'] and not ob.modifiers
        weights = [v.groups[0].weight for v in ob.data.vertices if len(v.groups) == 1]
        assert len(weights) == len(ob.data.vertices) and 0.49 < min(weights) and max(weights) < 1.01
        uv = np.empty(len(ob.data.loops) * 2)
        ob.data.uv_layers[0].data.foreach_get('uv', uv)
        co = np.array([ob.data.vertices[loop.vertex_index].co[:2] for loop in ob.data.loops])
        assert np.allclose(uv.reshape(-1, 2), co, atol=1e-4)            # the UVs still follow the surface

    under = prep_triangles.limit_triangles([big, small], 5000)
    assert not under['decimated'] and under['after'] == under['before']


def test_decimation_that_lands_over_the_budget_takes_the_rest_again(monkeypatch):
    """The Decimate modifier lands a few triangles off its share, over as often as under: Spider-Man on Zelda came to
    9,004 of 9,000 after the second pass and failed. Each pass here keeps four faces more than asked."""
    _empty_scene()
    big, small = _grid('big', 30), _grid('small', 10)                   # 1800 and 200 triangles
    real = prep_triangles._decimate
    monkeypatch.setattr(prep_triangles, '_decimate',
                        lambda o, ratio: real(o, min(1.0, ratio + 4 / len(o.data.polygons))))
    record = prep_triangles.limit_triangles([big, small], 1000)
    assert 990 <= record['after'] <= 1000 and record['passes'] > 2


def test_a_mesh_with_a_shape_key_is_left_its_shape_and_can_then_be_decimated():
    _empty_scene()
    ob = _grid('face', 30)                                              # 1800 triangles
    ob.shape_key_add(name='Basis')
    smile = ob.shape_key_add(name='smile')
    for i, v in enumerate(ob.data.vertices):
        smile.data[i].co = (v.co.x, v.co.y, 1.0)
    smile.value = 0.5                                                   # the build shows it half way: z 0.5
    record = prep_triangles.apply_shape_keys([ob])
    assert record == {'face': ['Basis', 'smile']} and ob.data.shape_keys is None
    co = np.array([v.co[:] for v in ob.data.vertices])
    assert np.allclose(co[:, 2], 0.5)
    assert prep_triangles.apply_shape_keys([ob]) == {}                      # nothing left to apply
    decimated = prep_triangles.limit_triangles([ob], 1000)
    assert decimated['decimated'] and decimated['after'] <= 1000


def test_the_shape_keys_of_a_hidden_mesh_with_no_active_key_are_applied_through_the_data():
    """Blender's glTF importer leaves an object with no active shape key (majorfirepower-Zelda, runs/pilot6), and
    bpy.ops.object.shape_key_remove's poll refuses one: "context is incorrect". Not the active object, in a hidden
    collection, its keys at rest: it comes out with none, in the basis's shape."""
    bpy = _empty_scene()
    face = _keyed('face', 4, collection=_hidden_collection())
    bpy.context.view_layer.objects.active = _grid('body', 2)
    rest = np.array([v.co[:] for v in face.data.vertices])
    assert prep_triangles.apply_shape_keys([face]) == {'face': ['Basis', 'pain', 'blink']}
    assert face.data.shape_keys is None
    assert np.array_equal(np.array([v.co[:] for v in face.data.vertices]), rest)


def test_a_decimated_mesh_is_held_to_the_fighters_influence_cap_again():
    """A collapse mixes the weights of the vertices it merges: on a grid whose every column of vertices is on its own
    bone, the decimated vertices come out on several bones, past the cap weights.solve held them to (HSDRawViewer
    refused Spider-Man on Mario: 7 weights on a vertex). The decimated meshes are held to the cap again, each vertex's
    weights summing to 1; the record says how many there were."""
    _empty_scene()
    n = 30
    ob = _grid('striped', n)
    ob.vertex_groups.remove(ob.vertex_groups['JOBJ_1'])
    columns = [ob.vertex_groups.new(name=f'JOBJ_{x}') for x in range(n + 1)]
    for v in ob.data.vertices:
        columns[round(v.co.x * n)].add([v.index], 1.0, 'REPLACE')
    record = prep_triangles.limit_triangles([ob], 150, max_influences=2)
    held = record['influences']
    assert held['cap'] == 2 and held['most_found'] > 2 and held['vertices_over'] > 0
    for v in ob.data.vertices:
        weights = [g.weight for g in v.groups if g.weight > 0]
        assert 1 <= len(weights) <= 2 and abs(sum(weights) - 1) < 1e-5
