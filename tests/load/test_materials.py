import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from rigswap.load import materials


class Pixels:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).ravel()

    def __len__(self):
        return len(self.values)

    def foreach_get(self, values):
        values[:] = self.values

    def foreach_set(self, values):
        self.values[:] = values


class Image:
    def __init__(self, name, pixels):
        self.name, self.pixels = name, Pixels(pixels)
        self.colorspace_settings = NS(name='sRGB')
        self.packed = False

    def update(self):
        pass

    def pack(self):
        self.packed = True


class Socket:
    def __init__(self, name, value=None):
        self.name, self.default_value, self.links = name, value, []

    @property
    def is_linked(self):
        return bool(self.links)


class Nodes(list):
    def new(self, kind):
        assert kind == 'ShaderNodeTexImage'
        colour = Socket('Color')
        node = NS(type='TEX_IMAGE', image=None, outputs={'Color': colour})
        colour.node = node
        self.append(node)
        return node


class Links:
    def new(self, source, target):
        target.links[:] = [NS(from_node=source.node, from_socket=source)]


def material(name, image, colour, alpha_linked=False):
    nodes, links = Nodes(), Links()
    source = nodes.new('ShaderNodeTexImage')
    source.image = image
    base, alpha = Socket('Base Color', colour), Socket('Alpha', 1.)
    links.new(source.outputs['Color'], base)
    if alpha_linked:
        alpha.links.append(NS(from_node=source, from_socket=Socket('Alpha')))
    nodes.append(NS(type='BSDF_PRINCIPLED', inputs={'Base Color': base, 'Alpha': alpha}))
    return NS(name=name, node_tree=NS(nodes=nodes, links=links)), base, alpha


def scene(monkeypatch, *mats):
    made = []

    def new(name, width, height, alpha):
        assert (width, height, alpha) == (1, 1, True)
        image = Image(name, [0, 0, 0, 0])
        made.append(image)
        return image

    monkeypatch.setitem(sys.modules, 'bpy', NS(data=NS(materials=mats, images=NS(new=new))))
    return made


def test_an_empty_image_gives_way_to_each_materials_colour_without_mutating_shared_pixels(monkeypatch):
    original = Image('empty', [[0, 0, 0, 0], [1, .3, .7, 0]])
    a, base_a, alpha_a = material('cloth', original, (.18, .04, .7, 1.))
    b, base_b, _ = material('trim', original, (.5, .5, .5, 1.))
    c, base_c, alpha_c = material('veil', original, (1., 1., 1., 1.), alpha_linked=True)
    before = original.pixels.values.copy()
    made = scene(monkeypatch, a, b, c)
    record = materials.convert()
    assert json.loads(json.dumps(record)) == record
    assert [r['material'] for r in record] == ['cloth', 'trim']
    assert record[0]['colour'] == [.18, .04, .7, 1.]
    assert record[0]['image'] == 'empty'
    assert record[0]['texel'] == [118, 56, 218, 255]
    assert record[1]['texel'] == [188, 188, 188, 255]
    assert len(made) == 2 and all(i.packed for i in made)
    assert base_a.links[0].from_node.image is made[0]
    assert base_b.links[0].from_node.image is made[1]
    assert base_c.links[0].from_node.image is original and alpha_c.is_linked
    assert not alpha_a.is_linked and alpha_a.default_value == 1.
    np.testing.assert_array_equal(original.pixels.values, before)
    np.testing.assert_allclose(made[0].pixels.values, np.array([118, 56, 218, 255]) / 255)
    assert materials.convert() == []                         # already converted, including after a DAE import


@pytest.mark.parametrize('alpha', [1., 1 / 255, 1e-8])
def test_any_nonzero_alpha_keeps_the_texture_even_when_its_rgb_is_black(monkeypatch, alpha):
    original = Image('dark', [[0, 0, 0, 0], [0, 0, 0, alpha]])
    mat, base, _ = material('surface', original, (.8, .8, .8, 1.))
    scene(monkeypatch, mat)
    assert materials.convert() == []
    assert base.links[0].from_node.image is original


@pytest.mark.parametrize('kind', ['missing_pixels', 'missing_image', 'untextured', 'composed', 'alpha_as_colour'])
def test_only_a_direct_empty_colour_image_is_converted(monkeypatch, kind):
    original = Image('empty', [] if kind == 'missing_pixels' else [[0, 0, 0, 0]])
    mat, base, _ = material('surface', original, (.8, .8, .8, 1.))
    link = base.links[0]
    if kind == 'missing_image':
        link.from_node.image = None
    elif kind == 'untextured':
        base.links.clear()
    elif kind == 'composed':
        link.from_node.type = 'MIX'
    elif kind == 'alpha_as_colour':
        link.from_socket.name = 'Alpha'
    made = scene(monkeypatch, mat)
    assert materials.convert() == [] and made == []


def test_material_black_stays_black_and_opacity_is_not_a_colour_channel(monkeypatch):
    mat, _, alpha = material('surface', Image('empty', [[1, 1, 1, 0]]), (0., 0., 0., .3))
    alpha.default_value = .3
    made = scene(monkeypatch, mat)
    assert materials.convert()[0]['texel'] == [0, 0, 0, 255]
    assert alpha.default_value == .3 and len(made) == 1
    assert materials.colour_texel((0.001, 1., 0., .3)) == (3, 255, 0, 255)


@pytest.mark.parametrize('already_converted', [False, True])
def test_load_converts_before_packing_and_records_the_colour_once(tmp_path, monkeypatch, already_converted):
    from rigswap.load import open as load
    calls = []
    record = [{'material': 'surface', 'colour': [.2, .4, .6, 1.]}]

    def imported(path):
        calls.append('import')
        return {'material_colours': record} if already_converted else {}

    def converted():
        calls.append('convert')
        return record

    monkeypatch.setitem(load.IMPORTERS, '.obj', imported)
    monkeypatch.setattr(materials, 'convert', converted)
    monkeypatch.setattr(load.stance, 'settle', lambda: None)
    monkeypatch.setattr(load.texture, 'bake', lambda: [])
    monkeypatch.setattr(load, 'settle_order', lambda: calls.append('settle'))
    monkeypatch.setattr(load, '_pack_images', lambda: calls.append('pack') or [])
    monkeypatch.setitem(sys.modules, 'bpy', NS(
        data=NS(objects=[], materials=[], images=[]),
        ops=NS(wm=NS(read_homefile=lambda **kw: None, save_as_mainfile=lambda **kw: calls.append('save')))))
    source = tmp_path / 'source.obj'
    source.write_text('')
    listing = load.run(source, tmp_path / 'load')
    assert calls == ['import'] + ([] if already_converted else ['convert']) + ['settle', 'pack', 'save']
    assert listing[0]['material_colours'] == record


@pytest.mark.blender
def test_obj_material_colour_reaches_the_glb_as_a_packed_srgb_texture(tmp_path):
    import io
    import bpy
    from PIL import Image as PILImage
    from rigswap.load import open as load
    import json
    import struct

    PILImage.new('RGBA', (8, 8), (0, 0, 0, 0)).save(tmp_path / 'empty.png')
    (tmp_path / 'surface.mtl').write_text('newmtl surface\nKd 0.18 0.04 0.7\nd 1\nmap_Kd empty.png\n')
    obj = tmp_path / 'surface.obj'
    obj.write_text('mtllib surface.mtl\no surface\nv 0 0 0\nv 1 0 0\nv 0 1 0\n'
                   'vt 0 0\nvt 1 0\nvt 0 1\nusemtl surface\nf 1/1 2/2 3/3\n')
    listing = load.run(obj, tmp_path / 'load')
    assert listing[0]['material_colours'][0]['texel'] == [118, 56, 218, 255]
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / 'load/models' / listing[0]['blend']))
    output = tmp_path / 'surface.glb'
    bpy.ops.export_scene.gltf(filepath=str(output), export_format='GLB')
    raw = output.read_bytes()
    json_size = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20 + json_size])
    binary = raw[28 + json_size:]
    pbr = doc['materials'][0]['pbrMetallicRoughness']
    assert pbr.get('baseColorFactor', [1, 1, 1, 1]) == [1, 1, 1, 1]
    index = doc['textures'][pbr['baseColorTexture']['index']]['source']
    view = doc['bufferViews'][doc['images'][index]['bufferView']]
    start = view.get('byteOffset', 0)
    with PILImage.open(io.BytesIO(binary[start:start + view['byteLength']])) as image:
        assert image.convert('RGBA').getpixel((0, 0)) == (118, 56, 218, 255)
