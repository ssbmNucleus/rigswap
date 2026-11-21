"""Capacity atlases preserve texture sampling, not just the count of materials."""
from types import SimpleNamespace as Obj

import numpy as np
import pytest

from rigswap.plugins.melee.prep.atlas import pack


def _sample(pixels, uv, extension):
    """Bilinear sample in normalized UV coordinates, with the image's original edge rule."""
    size = np.array(pixels.shape[1::-1])
    xy = np.asarray(uv) * size - 0.5
    base = np.floor(xy).astype(int)
    frac = xy - base
    out = np.zeros(4)
    for dx in (0, 1):
        for dy in (0, 1):
            p = base + (dx, dy)
            if extension == 'REPEAT':
                p = p % size
            elif extension == 'MIRROR':
                p = np.minimum(p % (2 * size), 2 * size - 1 - p % (2 * size))
            else:
                p = np.clip(p, 0, size - 1)
            weight = (frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1])
            out += pixels[::-1][p[1], p[0]] * weight
    return out


@pytest.mark.parametrize('extension', ['REPEAT', 'MIRROR', 'EXTEND'])
@pytest.mark.parametrize('fmt', ['CMPR', 'RGBA8', 'CI8:RGB565', 'RGB565'])
def test_atlas_preserves_samples_inside_and_across_uv_edges(extension, fmt):
    rng = np.random.default_rng(12)
    images = [rng.integers(0, 256, (h, w, 4), dtype=np.uint8) for h, w in ((4, 8), (8, 4))]
    for image in images:
        image[..., 3] = 255
    bounds = [((-1, -0.5), (2, 1.5)), ((0, 0), (1, 1))]
    atlas, transforms = pack(images, bounds, [extension] * 2, fmt)
    assert max(atlas.shape[:2]) <= 1024
    for source, (scale, offset), (lo, hi) in zip(images, transforms, bounds):
        points = [lo, hi, (0, 0), (1, 1), (0.5, 0.5), *rng.uniform(lo, hi, (30, 2))]
        for uv in points:
            assert np.allclose(_sample(source, uv, extension),
                               _sample(atlas, np.asarray(uv) * scale + offset, 'EXTEND'))


def test_each_image_is_laid_out_by_its_own_wrap():
    rng = np.random.default_rng(5)
    images = [rng.integers(0, 256, (8, 8, 4), dtype=np.uint8) for _ in range(2)]
    for image in images:
        image[..., 3] = 255
    bounds = [((-0.5, -0.5), (1.5, 1.5))] * 2
    atlas, transforms = pack(images, bounds, ['REPEAT', 'MIRROR'], 'RGBA8')
    for source, extension, (scale, offset) in zip(images, ('REPEAT', 'MIRROR'), transforms):
        for uv in rng.uniform(-0.5, 1.5, (30, 2)):
            assert np.allclose(_sample(source, uv, extension), _sample(atlas, uv * scale + offset, 'EXTEND'))


def test_atlas_refuses_resizing_or_alpha_and_palette_loss():
    small = np.full((8, 8, 4), 255, np.uint8)
    bounds = [((0, 0), (1, 1))] * 2
    repeat = ['REPEAT'] * 2
    # GX's longest side binds an atlas, not prep's limit on one image
    assert pack([np.full((256, 256, 4), 255, np.uint8), small], bounds, repeat, 'CMPR') is not None
    assert pack([np.full((1024, 1024, 4), 255, np.uint8), small], bounds, repeat, 'CMPR') is None
    assert pack([small, small], [((0, 0), (1000, 1)), bounds[1]], repeat, 'CMPR') is None
    assert pack([small, small], [((float('nan'), 0), (1, 1)), bounds[1]], repeat, 'CMPR') is None
    cutout = small.copy()
    cutout[0, 0, 3] = 0
    assert pack([small, cutout], bounds, repeat, 'CMPR') is None
    many = np.random.default_rng(7).integers(0, 256, (16, 16, 4), dtype=np.uint8)
    many[..., 3] = 255
    assert pack([many, small], bounds, repeat, 'CI8:RGB565') is None


def test_two_formats_share_one_holding_each_as_its_own_did():
    from rigswap.plugins.melee.prep.atlas import shared_formats
    few = np.full((8, 8, 4), 255, np.uint8)
    few[:4, :, :3] = (40, 80, 200)
    many = np.random.default_rng(3).integers(0, 256, (16, 16, 4), dtype=np.uint8)
    many[..., 3] = 255
    assert shared_formats(['CMPR', 'CMPR'], [many, many]) == ['CMPR']
    # Waddle Dee's CI8 face and CMPR bandana: their colours together in CI8's palette where they fit, else RGB565
    assert shared_formats(['CI8:RGB565', 'CMPR'], [few, few]) == ['CI8:RGB565', 'RGB565']
    assert shared_formats(['CI8:RGB565', 'CMPR'], [few, many]) == ['RGB565']
    assert shared_formats(['RGBA8', 'CMPR'], [few, many]) == ['RGBA8']
    cutout = few.copy()
    cutout[0, 0, 3] = 0
    holed = many.copy()
    holed[0, 0, 3] = 0
    assert shared_formats(['CI8:RGB5A3', 'CMPR'], [cutout, holed]) == ['RGB5A3']
    assert shared_formats(['CMPR', 'CI8:RGB565'], [cutout, few]) == []


def test_atlas_preserves_existing_alpha_plan_and_transparent_repair():
    opaque = np.full((16, 16, 4), 255, dtype=np.uint8)
    mostly_opaque = opaque.copy()
    mostly_opaque[0, 0, 3] = 64
    soft = opaque.copy()
    soft[..., 3] = 64
    bounds = [((0, 0), (1, 1))] * 2
    # Both images are partial alpha, but only one asks the existing alpha rules to blend.
    assert pack([mostly_opaque, soft], bounds, ['REPEAT'] * 2, 'RGB5A3') is None
    clear = opaque.copy()
    clear[..., 3] = 0
    cutout = opaque.copy()
    cutout[:8, :, 3] = 0
    # Both are cutouts; one needs the export's fully-transparent repair, the other does not.
    assert pack([clear, cutout], bounds, ['REPEAT'] * 2, 'CMPR') is None
    assert pack([cutout, cutout], bounds, ['REPEAT'] * 2, 'CMPR') is not None


def test_image_eligibility_keeps_vertex_colour_multiply_graphs():
    from rigswap.plugins.melee.prep.atlas import _texture

    class Node:
        def __init__(self, kind):
            self.type = kind

    image, mix, colour, shader = [Node(t) for t in ('TEX_IMAGE', 'MIX', 'VERTEX_COLOR', 'BSDF_PRINCIPLED')]
    image.image = Obj(source='FILE', size=(8, 8))
    image.projection, image.interpolation = 'FLAT', 'Linear'
    image.inputs = {'Vector': Obj(is_linked=False)}
    destination = Obj(name='Base Color')
    tree = Obj(animation_data=None, nodes=[image, mix, colour, shader], links=[
        Obj(from_node=image, to_node=mix), Obj(from_node=colour, to_node=mix),
        Obj(from_node=mix, to_node=shader, to_socket=destination)])
    material = Obj(node_tree=tree)
    assert _texture(material) is image
    destination.name = 'Roughness'
    assert _texture(material) is None
    destination.name = 'Base Color'
    image.inputs['Vector'].is_linked = True
    assert _texture(material) is None


@pytest.mark.parametrize('places', [1, 3])
def test_capacity_join_retries_planner_until_the_looks_fit(monkeypatch, places):
    """Exercise the capacity loop without Blender: three looks on one mesh share images until the places hold them,
    and every material stays on its faces."""
    import sys
    from rigswap.plugins.melee.prep import atlas, join
    from rigswap.plugins.melee.export.dobjs import FighterTables, Slot

    class Material:
        def __init__(self, name, roughness):
            self.name, self.roughness = name, roughness
            self.node_tree = Obj(nodes=[], links=[], animation_data=None)
            self.use_backface_culling = True

        def get(self, key):
            return None

    one, two, three = Material('one', 0.1), Material('two', 0.2), Material('three', 0.3)
    slots = [Obj(material=m) for m in (one, two, three)]
    mesh = Obj(name='body', material_slots=slots, data=Obj(polygons=[Obj(material_index=i) for i in range(3)]))
    calls = []

    def merge(meshes, protected, materials, joint, formats, settings):
        a, b = sorted({m.roughness for m in materials(mesh)})[:2]
        drawn = [m for m in materials(mesh) if m.roughness in (a, b)]
        for m in drawn:
            m.roughness = a
        calls.append({'materials': sorted(m.name for m in drawn), 'format': 'CMPR'})
        return calls[-1]

    monkeypatch.setitem(sys.modules, 'bpy', Obj())  # image writes must not occur in this test
    monkeypatch.setattr(atlas, 'merge_images', merge)
    report = join.join_for_places([mesh], [Slot(0, i) for i in range(places)], FighterTables(body=tuple(range(places))),
                                  formats={m.name: 'CMPR' for m in (one, two, three)})
    assert report['dobjs'] == 3 and report['looks']['before'] == 3 and report['looks']['after'] == places
    assert [s.material for s in slots] == [one, two, three] and not report['material_aliases']
    assert len(calls) == 3 - places and report.get('atlases', []) == calls
    assert report['one_place'] == ({'three': 'one', 'two': 'one'} if places == 1 else {})


@pytest.mark.parametrize('case', [None, 'settings', 'joint', 'kept', 'shared_mesh', 'format', 'culling', 'wrap'])
def test_two_looks_share_an_image_only_when_compatible(monkeypatch, case):
    """Blocked: other settings, no joint in common, kept, a shared mesh. Shared: two formats (written in one holding
    both), two cullings, two wraps. Each material stays, keeps its culling and draws the atlas clamped."""
    import sys
    from rigswap.plugins.melee.prep import atlas
    blocked = case in ('settings', 'joint', 'kept', 'shared_mesh')

    class Image:
        source, colorspace_settings, alpha_mode = 'MOVIE', Obj(name='sRGB'), 'STRAIGHT'

    class Material:
        def __init__(self, name, culled=True, extension='REPEAT'):
            self.name, self.use_backface_culling, self.node_tree = name, culled, Obj(nodes=[])
            self.texture = Obj(image=Image(), extension=extension)

    a, b = Material('a'), Material('b', case != 'culling', 'MIRROR' if case == 'wrap' else 'REPEAT')
    pixels = np.full((8, 8, 4), 255, dtype=np.uint8)
    meshes = []
    for i, material in enumerate((a, b)):
        layer = Obj(active_render=True, data=[Obj(uv=(0, 0)), Obj(uv=(1, 0)), Obj(uv=(0, 1))])
        meshes.append(Obj(name=f'mesh{i}', material_slots=[Obj(material=material)],
                          joint=i if case == 'joint' else 0,
                          data=Obj(users=2 if case == 'shared_mesh' else 1, uv_layers=[layer],
                                   polygons=[Obj(index=0, material_index=0, loop_indices=(0, 1, 2))])))
    written = []
    monkeypatch.setitem(sys.modules, 'bpy', Obj(data=Obj(images=Obj(new=lambda *a, **kw: Obj(
        colorspace_settings=Obj(name='sRGB'), alpha_mode='STRAIGHT')))))
    monkeypatch.setattr(atlas, '_texture', lambda m: m.texture)
    monkeypatch.setattr(atlas, 'read_pixels', lambda image: pixels.copy())
    monkeypatch.setattr(atlas, 'write_pixels', lambda image, p: written.append(p))
    record = atlas.merge_images(meshes, {a} if case == 'kept' else set(),
                                lambda o: {s.material for s in o.material_slots}, lambda o: o.joint,
                                {'a': 'CMPR', 'b': 'CI8:RGB565' if case == 'format' else 'CMPR'},
                                lambda m, key, ignore=(): m.name if case == 'settings' else 'same')
    if blocked:
        assert record is None and not written
        assert a.texture.image is not b.texture.image and a.texture.extension == 'REPEAT'
        return
    # one colour fits CI8's palette: the CMPR image goes into it with the CI8 one
    assert record['materials'] == ['a', 'b'] and record['format'] == ('CI8:RGB565' if case == 'format' else 'CMPR')
    assert record.get('source_formats') == (['CMPR', 'CI8:RGB565'] if case == 'format' else None)
    assert meshes[0].material_slots[0].material is a and meshes[1].material_slots[0].material is b
    assert a.use_backface_culling and b.use_backface_culling is (case != 'culling')
    assert len(written) == 1 and a.texture.image is b.texture.image
    assert a.texture.extension == b.texture.extension == 'EXTEND'
    for mesh in meshes:
        for corner in mesh.data.uv_layers[0].data:
            assert np.allclose(_sample(written[0], corner.uv, 'EXTEND'), (255, 255, 255, 255))


@pytest.mark.blender
def test_real_atlas_reduces_dobjs_preserves_skin_and_keeps_eyes():
    from rigswap.plugins.melee.prep.join import join_for_places
    from rigswap.plugins.melee.prep.textures import read_pixels
    from rigswap.plugins.melee.export.dobjs import FighterTables, Slot
    from tests.plugins.melee.prep.test_prep import _empty_scene, _material, _mesh

    _empty_scene()
    pixels = [np.random.default_rng(n).integers(0, 256, (8, 8, 4), dtype=np.uint8) for n in (1, 2)]
    for p in pixels:
        p[..., 3] = 255
    a, b = [_material(name, p) for name, p in zip(('a', 'b'), pixels)]
    mesh = _mesh('body', [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 1, 2), (1, 3, 2)], [a, b])
    mesh.data.polygons[1].material_index = 1
    mesh.vertex_groups.new(name='JOBJ_0').add([0, 1, 2, 3], 1, 'REPLACE')
    uv = mesh.data.uv_layers.new()
    original = [(0, 0), (1, 0), (0, 1), (0, 0), (1, 1), (0, 1)]
    uv.data.foreach_set('uv', np.asarray(original).ravel())
    eye = _material('eye', pixels[0])
    eyes = _mesh('eyes', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], [eye])
    report = join_for_places([mesh, eyes], [Slot(0, 0)], FighterTables(), keep=['eyes'], deferred=['eyes'],
                             formats={'a': 'CMPR', 'b': 'CMPR', 'eye': 'CMPR'})
    assert report['dobjs'] == 3 and len(report['atlases']) == 1 and report['one_place'] == {'b': 'a'}
    assert eyes.material_slots[0].material == eye
    assert len(mesh.data.polygons) == 2 and len(mesh.data.vertices) == 4
    assert all(v.groups[0].weight == 1 for v in mesh.data.vertices)
    assert [s.material for s in mesh.material_slots] == [a, b]
    image, = {n.image for m in (a, b) for n in m.node_tree.nodes if n.type == 'TEX_IMAGE'}
    for i, before in enumerate(original):
        assert np.allclose(_sample(pixels[i // 3], before, 'REPEAT'),
                           _sample(read_pixels(image), uv.data[i].uv, 'EXTEND'), atol=0.001)


@pytest.mark.blender
def test_duplicate_image_datablocks_share_material_without_atlas():
    from rigswap.plugins.melee.prep.join import join_for_places
    from rigswap.plugins.melee.export.dobjs import FighterTables, Slot
    from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _material, _mesh

    _empty_scene()
    a, b = [_material(n, _flat(4, 4)) for n in ('a', 'b')]
    mesh = _mesh('body', [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 1, 2), (1, 3, 2)], [a, b])
    mesh.data.polygons[1].material_index = 1
    report = join_for_places([mesh], [Slot(0, 0)], FighterTables())
    assert report['dobjs'] == 1 and 'atlases' not in report
    assert report['material_aliases'] == {'b': 'a'}
