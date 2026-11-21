"""prep/twins.py: one surface drawn twice, and the copies of it the game cannot draw."""
import numpy as np

from rigswap.plugins.melee.prep import twins as prep_twins
from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _grid, _material


def _striped(w, h, alpha=255):
    """A colourful opaque image (red and blue stripes), alpha ``alpha``."""
    pixels = _flat(w, h, (200, 40, 40, alpha))
    pixels[::2] = (40, 40, 200, alpha)
    return pixels


def _names(meshes):
    return sorted(o.name for o in meshes if len(o.data.polygons))


def test_of_two_opaque_copies_of_a_surface_the_one_in_colour_stays_and_a_grey_pass_goes():
    """Sparks of Hope's Mario: msh_body and msh_body_UV1, the same vertices, the copy drawing a grey reflection image.
    The game shows whichever it draws last; the body in its own colours stays."""
    _empty_scene()
    body = _grid('msh_body', 4, [_material('body', _striped(8, 8))])
    shine = _grid('msh_body_UV1', 4, [_material('shine', _flat(8, 8, (90, 90, 90, 255)))])
    record = prep_twins.drop_twins([body, shine])
    assert [r['dropped'] for r in record['twins']] == [['msh_body_UV1/shine']]
    assert record['triangles_dropped'] == {'msh_body_UV1/shine': 32}
    import bpy
    assert 'msh_body_UV1' not in bpy.data.objects and len(body.data.polygons) == 16


def test_a_copy_named_as_another_s_pass_goes_and_copies_nothing_tells_apart_stay():
    """Genshin's Body.UV1: Body's surface again, drawing its image through the second UV set. Copies of one image
    named apart only by their numbers (meshnode_11, meshnode_12) are left as the build has them."""
    _empty_scene()
    image = _striped(8, 8)
    copies = [_grid(name, 2, [_material(name, image)]) for name in ('Body.UV1', 'Body')]
    record = prep_twins.drop_twins(copies)
    assert record['twins'][0]['dropped'] == ['Body.UV1/Body.UV1']
    _empty_scene()
    copies = [_grid(name, 2, [_material(name, image)]) for name in ('meshnode_1', 'meshnode_12')]
    record = prep_twins.drop_twins(copies)
    assert record['twins'][0]['dropped'] == [] and not record['triangles_dropped']
    assert prep_twins.extends('msh_body_UV1', 'msh_body') and prep_twins.extends('Body.UV1.001', 'Body')
    assert not prep_twins.extends('meshnode_12', 'meshnode_1') and not prep_twins.extends('Body', 'Body')


def test_a_decal_over_a_surface_stays_and_a_detail_is_never_read():
    """Brawl's eyes: an iris cutout over the eye white, the same faces. A pass that leaves texels out draws over the
    copy under it in the game too. A detail's objects (the eyes the export makes blink) are left as they are."""
    _empty_scene()
    white = _grid('eye_white', 4, [_material('white', _flat(8, 8, (250, 250, 250, 255)))])
    iris = _striped(8, 8)
    iris[:, :4, 3] = 0
    over = _grid('eye_iris', 4, [_material('iris', iris)])
    record = prep_twins.drop_twins([white, over])
    assert record['twins'][0]['dropped'] == [] and not record['triangles_dropped']
    detail = _grid('eyes', 4, [_material('grey', _flat(8, 8, (90, 90, 90, 255)))])
    assert prep_twins.drop_twins([white, detail], keep=['eyes']) == {'twins': [], 'linings': [],
                                                                     'triangles_dropped': {}}


def _flipped(ob, by):
    """``ob`` with its faces turned round, moved ``by`` along z: the surface's back a step behind it."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.location.z = by
    import bpy
    bpy.context.view_layer.update()
    return ob


def test_an_opaque_lining_behind_glass_goes_and_one_behind_an_opaque_surface_stays():
    """Olimar's helmet: the glass (alpha 53 to 158 everywhere, its source OPAQUE) with a white lining just inside it,
    facing in. Drawn opaque, the lining is a solid wall seen through the glass; the glass draws its own back. A gown's
    lining behind its opaque cloth is the gown's back, and stays."""
    _empty_scene()
    glass = _flat(8, 8, (3, 3, 3, 64))
    glass[::2, :, 3] = 158
    outer = _grid('soto', 4, [_material('soto', glass)])
    inner = _flipped(_grid('naka', 4, [_material('naka', _flat(1, 1, (255, 255, 255, 255)))]), -0.005)
    record = prep_twins.drop_twins([outer, inner])
    (row,) = record['linings']
    assert row['kept'] == 'soto/soto' and row['dropped'] == ['naka/naka']
    assert row['copies']['soto/soto']['mode'] == 'Blend' and row['copies']['naka/naka']['mode'] == 'Test'

    _empty_scene()
    cloth = _grid('gown', 4, [_material('gown', _striped(8, 8))])
    back = _flipped(_grid('gown_lining', 4, [_material('lining', _flat(8, 8, (240, 230, 220, 255)))]), -0.005)
    record = prep_twins.drop_twins([cloth, back])
    assert record['linings'][0]['dropped'] == [] and _names([cloth, back]) == ['gown', 'gown_lining']


def test_surfaces_apart_or_only_partly_on_each_other_are_not_one_surface():
    _empty_scene()
    a = _grid('a', 4, [_material('a', _striped(8, 8))])
    b = _grid('b', 4, [_material('b', _flat(8, 8, (90, 90, 90, 255)))])
    b.location.z = 0.2                                      # a twentieth of the model away: another surface
    c = _grid('c', 2, [_material('c', _flat(8, 8, (90, 90, 90, 255)))])
    c.scale = (0.5, 0.5, 1)                                 # on a quarter of a: a patch, not a copy of it
    import bpy
    bpy.context.view_layer.update()
    assert prep_twins.drop_twins([a, b, c]) == {'twins': [], 'linings': [], 'triangles_dropped': {}}
