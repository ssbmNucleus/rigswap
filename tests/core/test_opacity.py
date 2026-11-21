import numpy as np
import pytest

from rigswap.core import opacity


def _scene(alpha, wired=False, rgb=(1, 1, 1), flat=False):
    """One quad drawing a 4x4 image across its whole UV square, its alpha ``alpha`` (4x4, the top row first)."""
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    return _card('quad', 'skin', 0, alpha, rgb, wired, flat)


def _card(name, image_name, y, alpha, rgb=(1, 1, 1), wired=False, flat=False):
    """A unit quad at depth ``y``, facing -y, drawing a 4x4 image of ``rgb`` with alpha ``alpha`` across its UVs;
    ``flat``: the image is that one colour all over, else a picture (one texel another shade)."""
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0, y, 0), (1, y, 0), (1, y, 1), (0, y, 1)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name='UVMap')
    uv.data.foreach_set('uv', [0, 0, 1, 0, 1, 1, 0, 1])
    image = bpy.data.images.new(image_name, 4, 4, alpha=True)
    pixels = np.ones((4, 4, 4), np.float32)
    pixels[..., :3] = rgb
    if not flat:                                        # a picture: its colours differ (one texel another shade)
        shade = np.asarray(rgb, np.float32)
        pixels[3, 3, :3] = np.where(shade > 0.5, shade - 0.1, shade + 0.1)
    pixels[..., 3] = np.asarray(alpha, np.float32)[::-1] / 255
    image.pixels.foreach_set(pixels.ravel())
    image.alpha_mode = 'CHANNEL_PACKED'                 # as the load's images are: Workbench draws them whole
    mat = bpy.data.materials.new(image_name)
    mat.use_nodes = True
    tree = mat.node_tree
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image = image
    bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    if wired:
        tree.links.new(tex.outputs['Alpha'], bsdf.inputs['Alpha'])
    me.materials.append(mat)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new(name, me))
    return image


@pytest.mark.blender
def test_an_image_its_alpha_cuts_where_a_material_wiring_none_draws_it_is_asked_about():
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0                                    # its left half clear
    _scene(alpha)
    (row,) = opacity.cutting()
    assert row['image'] == 'skin' and row['materials'] == ['skin'] and 0.3 < row['cut'] < 0.7
    _scene(alpha, wired=True)                           # its source says its alpha is opacity
    assert opacity.cutting() == []
    _scene(np.full((4, 4), 255))                        # nothing clear: nothing to cut
    assert opacity.cutting() == []


@pytest.mark.blender
def test_an_image_of_one_colour_whose_alpha_varies_is_not_asked_about():
    """Its alpha is all the picture it has (brows, a shade under a cap's brim): drawn whole it is a flat patch. Diddy
    Kong's cap shadow, black all over, was answered whole on 2026-11-10 and lay as black strips over his eyes."""
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0
    _scene(alpha, rgb=(0, 0, 0), flat=True)
    alone = []
    assert opacity.cutting(alone) == [] and alone == ['skin']
    _scene(np.zeros((4, 4)), rgb=(0, 0, 0), flat=True)  # one colour and one alpha: it says nothing, and is asked
    (row,) = opacity.cutting()
    assert row['image'] == 'skin' and row['cut'] == 1.0


@pytest.mark.blender
def test_the_cut_picture_shows_what_lies_behind_the_cut(tmp_path):
    import bpy
    from PIL import Image
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0
    _scene(alpha)
    me = bpy.data.meshes.new('behind')                  # a blue quad behind the clear half
    me.from_pydata([(-.5, .5, -.5), (1.5, .5, -.5), (1.5, .5, 1.5), (-.5, .5, 1.5)], [], [(0, 1, 2, 3)])
    blue = bpy.data.materials.new('blue')
    blue.diffuse_color = (0, 0, 1, 1)
    me.materials.append(blue)
    bpy.context.scene.collection.objects.link(bpy.data.objects.new('behind', me))
    rows = [dict(r, letter='A') for r in opacity.cutting()]
    (path,) = opacity.cut_or_whole(rows, tmp_path / 'opacity.png')
    sheet = np.asarray(Image.open(path).convert('RGB')).astype(int)
    blueish = (sheet[..., 2] - sheet[..., 0] > 80).sum(axis=0)
    half = sheet.shape[1] // 2
    # cut (1, the left) shows it through the clear half from the front; from the back both see it in front
    assert blueish[:half].sum() - blueish[half:].sum() > .1 * opacity.TILE_ON_MODEL ** 2
    assert bpy.data.objects['quad'].data.name == 'quad' and len(bpy.data.meshes) == 2   # the meshes given back


@pytest.mark.blender
def test_the_rest_of_the_model_is_drawn_as_the_answers_draw_it(tmp_path):
    import bpy
    from PIL import Image
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0
    _scene(alpha, rgb=(1, 0, 0))                        # red, its left half clear
    _card('brow', 'brow', -.25, alpha, (0, 0, 0))       # in front, black, clear over the same half (Toon Link's brow)
    rows = [dict(r, letter=chr(ord('A') + k)) for k, r in enumerate(opacity.cutting())]
    skin = next(k for k, r in enumerate(rows) if r['image'] == 'skin')

    def red_whole(whole):
        sheet = np.asarray(Image.open(opacity.cut_or_whole(rows, tmp_path / 'opacity.png', whole)[skin])
                           .convert('RGB')).astype(int)
        red = (sheet[..., 0] - sheet[..., 1] > 80) & (sheet[..., 0] - sheet[..., 2] > 80)
        return red[:, sheet.shape[1] // 2:].sum()       # in 2, where the red is drawn whole
    # the brow cut, as the export draws it unanswered, shows the red through its clear half from the front; whole,
    # its black hides it
    assert red_whole(()) - red_whole({'brow'}) > .1 * opacity.TILE_ON_MODEL ** 2
    drawn = {n.image.name for m in bpy.data.materials if m.node_tree for n in m.node_tree.nodes if n.type == 'TEX_IMAGE'}
    assert drawn == {'skin', 'brow'} and {'skin', 'brow'} <= set(bpy.data.images.keys())   # the images given back
    assert not [i for i in bpy.data.images if '-cut' in i.name or '-whole' in i.name]


@pytest.mark.blender
def test_the_answers_drawing_an_image_whole_are_asked_again_over_the_model_drawn_so(tmp_path):
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0
    _scene(alpha)
    _card('brow', 'brow', -.25, alpha, (0, 0, 0))
    (tmp_path / 'build').mkdir()
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append(pictures)
        return {'rows': [{'row': 'A', 'way': 2}, {'row': 'B', 'way': 1}]}, {}
    rec = opacity.ready(tmp_path / 'build', tmp_path, ask, 'Toon Link')
    assert len(asked) == 2 * opacity.VOTES and 'opacity-again' in asked[-1][0]
    assert [a['others_whole'] for a in rec['asks']] == [[], [rec['asked'][0]['image']]]
    assert [o['image'] for o in rec['opaque']] == [rec['asked'][0]['image']]


@pytest.mark.blender
def test_an_image_the_answers_draw_whole_is_drawn_opaque(tmp_path):
    import bpy
    alpha = np.full((4, 4), 255)
    alpha[:, :2] = 0
    _scene(alpha)
    (tmp_path / 'build').mkdir()
    asked = []

    def ask(question, pictures, schema, default, route=None):
        asked.append((route, pictures))
        return {'rows': [{'row': 'A', 'way': 2}]}, {}
    rec = opacity.ready(tmp_path / 'build', tmp_path, ask, 'Bean')
    assert len(asked) == opacity.VOTES and asked[0][0] == 'core opacity' and len(asked[0][1]) == 1
    assert rec['opaque'] == [{'image': 'skin', 'as': 'skin-opaque'}]
    drawn = next(n for n in bpy.data.materials['skin'].node_tree.nodes if n.type == 'TEX_IMAGE').image
    assert drawn.name == 'skin-opaque'
    pixels = np.empty(64, np.float32)
    drawn.pixels.foreach_get(pixels)
    assert np.allclose(pixels[3::4], 1)
    assert (tmp_path / 'build' / 'build.blend').is_file() and (tmp_path / 'build' / 'opacity.json').is_file()
