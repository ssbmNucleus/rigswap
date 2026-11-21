"""prep/layers.py: the thin layers' back faces."""
import numpy as np

from rigswap.plugins.melee.prep import layers as prep_layers, normals as prep_normals, triangles as prep_triangles
from tests.plugins.melee.prep.test_prep import _empty_scene, _grid


def test_a_layer_counts_twice_and_gets_a_back_layer_facing_the_other_way():
    _empty_scene()
    cape = _grid('cape', 10)                                            # 200 triangles
    body = _grid('body', 10)
    record = prep_triangles.limit_triangles([cape, body], 500, doubled=['cape'])
    assert record['before'] == 600 and record['after'] <= 500
    front = prep_triangles.count_triangles([cape])
    normals = prep_normals.corner_normals(cape.data)

    backs = prep_layers.two_sided_layers([cape, body], ['cape'])
    assert backs['triangles_added'] == front and prep_triangles.count_triangles([cape]) == 2 * front
    after = prep_normals.corner_normals(cape.data)
    assert np.allclose(after[:len(normals)], normals, atol=1e-3)
    assert np.allclose(after[len(normals):].mean(axis=0), -normals.mean(axis=0), atol=1e-3)
    assert all(len(v.groups) == 1 for v in cape.data.vertices)          # the back layer keeps the weights
    assert prep_layers.read_layers('nowhere') == {}


def _box(name, size=1.):
    """A closed cube of six quads, its normals outward, centred on the origin."""
    import bpy
    s = size / 2
    verts = [(x, y, z) for z in (-s, s) for y in (-s, s) for x in (-s, s)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(bpy.data.materials.new(name))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def test_the_faces_drawn_from_behind_are_an_open_sheet_s_and_they_count_twice_and_get_a_back():
    """A brim over a closed head: every face of the open brim shows its back from below, no face of the closed box
    does; a material that culls its back is drawn from one side, so none of its faces is marked; a detail's objects
    are left as the detail made them."""
    _empty_scene()
    head = _box('head')
    brim = _grid('brim', 4)                                              # 32 triangles, facing up, above the head
    brim.location = (-.5, -.5, 1.)
    brim.data.materials.append(head.data.materials[0].copy())
    culled = _grid('culled', 2)                                          # 8 triangles, drawn from one side only
    culled.location = (-.5, -.5, 2.)
    culled.data.materials.append(head.data.materials[0].copy())
    culled.data.materials[0].use_backface_culling = True
    eyes = _grid('eyes', 1)
    eyes.location = (-.5, -.5, -2.)
    eyes.data.materials.append(head.data.materials[0].copy())
    import bpy
    bpy.context.view_layer.update()
    meshes = [head, brim, culled, eyes]
    assert prep_layers.mark_drawn_from_behind(meshes, keep=['eyes']) == {'brim': 16}
    assert not prep_layers.back_faces(head).any() and prep_layers.back_faces(brim).all()
    assert prep_triangles.limit_triangles(meshes, 1000)['before'] == 12 + 2 * 32 + 8 + 2
    record = prep_layers.two_sided_layers(meshes, {})
    assert record['triangles_added'] == 32 and record['meshes'] == {'brim': 32}
    assert prep_triangles.count_triangles([brim]) == 64
    assert all(prep_layers.BACK not in o.data.attributes for o in meshes)  # the mark is taken off once used


def test_a_face_is_seen_from_behind_where_a_ray_from_its_back_side_leaves_the_model():
    looks = prep_layers.directions()
    assert len(looks) == 42 and np.allclose(np.linalg.norm(looks, axis=1), 1)
    normals, centres = np.array([[0, 0, 1.], [0, 0, 1.]]), np.zeros((2, 3))
    # nothing in the way: the back is seen; everything in the way: it is not
    assert prep_layers.seen_from_behind(normals, centres, lambda o, d: False, 1.).tolist() == [True, True]
    assert prep_layers.seen_from_behind(normals, centres, lambda o, d: True, 1.).tolist() == [False, False]
    # only the rays straight down are blocked (a floor right under it): still seen from the sides below
    assert prep_layers.seen_from_behind(normals[:1], centres[:1], lambda o, d: d[2] < -.99, 1.).tolist() == [True]
