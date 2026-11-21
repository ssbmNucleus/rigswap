import numpy as np
import pytest

from rigswap.core import render


@pytest.mark.blender
def test_a_still_draws_a_colour_set_where_a_material_draws_it_and_nowhere_else(tmp_path):
    """Taz's fur is a colour set on a pale material: Workbench drew him pale grey in every ask's picture. A face under
    a material that draws the set shows it, times the material's colour; a face under one that doesn't (an image, a
    piece greyed for the question) is as Workbench draws it, set or no set; a scene drawing no set renders once."""
    import bpy
    from PIL import Image
    bpy.ops.wm.read_homefile(use_empty=True)
    me = bpy.data.meshes.new('m')                                       # two squares side by side, facing -y
    me.from_pydata([(-2, 0, -1), (0, 0, -1), (0, 0, 1), (-2, 0, 1), (0, 0, -1), (2, 0, -1), (2, 0, 1), (0, 0, 1)],
                   [], [(0, 1, 2, 3), (4, 5, 6, 7)])
    colour = me.color_attributes.new('Col', 'BYTE_COLOR', 'CORNER')
    colour.data.foreach_set('color', np.tile([1., 0., 0., 1.], 8))      # red over both
    fur, plain = (bpy.data.materials.new(n) for n in ('fur', 'plain'))
    for mat in (fur, plain):
        mat.use_nodes = True
        mat.diffuse_color = (1., 1., 1., 1.)
        me.materials.append(mat)
    tree = fur.node_tree
    shader = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    tree.links.new(tree.nodes.new('ShaderNodeVertexColor').outputs['Color'], shader.inputs['Base Color'])
    me.polygons.foreach_set('material_index', [0, 1])
    ob = bpy.data.objects.new('m', me)
    bpy.context.collection.objects.link(ob)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    cam.data.type, cam.data.ortho_scale = 'ORTHO', 4.
    cam.location, cam.rotation_euler = (0, -5, 0), (np.pi / 2, 0, 0)
    bpy.context.collection.objects.link(cam)
    scene = bpy.context.scene
    scene.camera = cam
    render._prepare_scene()
    scene.render.resolution_x = scene.render.resolution_y = 64
    assert render._draws_set(fur) and not render._draws_set(plain)

    def drawn(name):
        scene.render.filepath = str(tmp_path / name)
        render.still()
        with Image.open(tmp_path / name) as img:
            px = np.asarray(img.convert('RGB'), dtype=int)
        return px[32, 16], px[32, 48]
    left, right = drawn('set.png')
    assert left[0] > 150 and left[1] < 40 and left[2] < 40, 'the set, where its material draws it'
    assert right.min() > 150 and np.ptp(right) < 12, 'no set under a material that draws none'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['set.png'], 'the passes are cleaned up'
    assert tuple(fur.diffuse_color) == (1., 1., 1., 1.) and scene.display.shading.color_type == 'TEXTURE'
    colour.data.foreach_set('color', np.ones(32))                       # a white set says nothing: the picture is
    me.update()                                                         # the plain render's, pixel for pixel
    drawn('white.png')
    tree.links.clear()                                                  # nothing draws the set: one plain render
    left, right = drawn('plain.png')
    assert left.min() > 150 and np.ptp(left) < 12 and abs(left - right).max() < 12
    with Image.open(tmp_path / 'white.png') as a, Image.open(tmp_path / 'plain.png') as b:
        assert a.mode == b.mode and a.tobytes() == b.tobytes()


@pytest.mark.parametrize('bold,found,expect', [
    (True, {'arialbd.ttf'}, 'arialbd.ttf'),                       # Windows: as before
    (True, {'Arial Bold.ttf', 'DejaVuSans-Bold.ttf'}, 'Arial Bold.ttf'),   # macOS names its Arial so
    (True, {'DejaVuSans-Bold.ttf'}, 'DejaVuSans-Bold.ttf'),       # a Linux desktop
    (False, {'Arial.ttf'}, 'Arial.ttf'),
    (True, set(), 'built-in'),                                    # none of them: Pillow's own
])
def test_the_label_font_is_found_by_each_system_name(monkeypatch, bold, found, expect):
    """A Mac has Arial as 'Arial Bold.ttf', not 'arialbd.ttf': without that name its labels were Pillow's regular face."""
    tried = []

    def truetype(name, size):
        tried.append(name)
        if name in found:
            return name
        raise OSError(name)
    monkeypatch.setattr(render.ImageFont, 'truetype', truetype)
    monkeypatch.setattr(render.ImageFont, 'load_default', lambda size: 'built-in')
    assert render._font(20, bold=bold) == expect
    assert tried[0] == ('arialbd.ttf' if bold else 'arial.ttf')
