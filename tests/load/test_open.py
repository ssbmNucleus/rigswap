import io
import json
import os
from pathlib import Path
import posixpath
import zipfile

import pytest

from rigswap.load import open as load


def _zip(entries):
    """A zip's bytes from {name: bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_nested_zips_open_three_levels_down_and_stay_inside(tmp_path):
    level4 = _zip({'deep4.obj': b'v 0 0 0'})
    level3 = _zip({'deep3.obj': b'v 0 0 0', 'more.zip': level4})
    level2 = _zip({'deep2.obj': b'v 0 0 0', 'more.zip': level3})
    level1 = _zip({'char.dae': b'<x/>', '../escape.obj': b'v 0 0 0', 'more.zip': level2})
    outer = tmp_path / 'download.zip'
    outer.write_bytes(_zip({'top.fbx': b'x', 'inner.zip': level1, 'inner/already.txt': b'a folder by that name',
                            'broken.zip': b'not a zip'}))
    src = load.unpack(outer, tmp_path / 'source')
    found = {p.relative_to(src).as_posix() for p in load.model_files(src)}
    assert found == {'top.fbx', 'inner (2)/char.dae', 'inner (2)/more/deep2.obj', 'inner (2)/more/more/deep3.obj'}
    assert (src / 'inner (2)' / 'more' / 'more' / 'more.zip').is_file()     # level 4 stays a zip
    assert not any(p.name == 'escape.obj' for p in tmp_path.rglob('*'))
    assert (src / 'broken.zip').is_file() and not (src / 'broken').exists()


def _raw_zip(entries):
    """A zip's bytes from {name: bytes}, each name written as given: a backslash stays one, as in a zip that
    Windows PowerShell's Compress-Archive made (zipfile itself would turn it into '/' on Windows)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, data in entries.items():
            info = zipfile.ZipInfo('entry')
            info.filename = name
            z.writestr(info, data)
    return buf.getvalue()


class _PosixOs:
    """The os module as zipfile sees it on Linux and macOS: '/' its only separator."""
    sep, altsep, path = '/', None, posixpath

    def __getattr__(self, name):
        return getattr(os, name)


@pytest.fixture
def posix_zipfile(monkeypatch):
    """zipfile reading names and extracting as it does on Linux and macOS, on any platform."""
    monkeypatch.setattr(zipfile, 'os', _PosixOs())


def test_a_zip_with_backslash_folders_extracts_into_those_folders(tmp_path, posix_zipfile):
    """SSBM Nucleus 0.9.0 build 3, Linux: a zip from Compress-Archive (Clank\\Clank.obj) was extracted as files named
    'Clank\\Clank.obj' and 'Clank\\Clank.mtl', so the OBJ's mtllib found no Clank.mtl beside it."""
    download = tmp_path / 'Clank.zip'
    download.write_bytes(_raw_zip({'Clank\\': b'', 'Clank\\Clank.obj': b'mtllib Clank.mtl\nv 0 0 0\n',
                                   'Clank\\Clank.mtl': b'newmtl a\nmap_Kd tex\\body.png\n',
                                   'Clank\\tex\\body.png': b'png', 'Clank\\..\\..\\escape.obj': b'v 0 0 0',
                                   'C:\\escape2.obj': b'v 0 0 0', '\\escape3.obj': b'v 0 0 0'}))
    with zipfile.ZipFile(download) as z:
        assert 'Clank\\Clank.obj' in z.namelist(), 'zipfile kept the backslash, as on Linux'
    src = load.unpack(download, tmp_path / 'source')
    assert sorted(p.relative_to(src).as_posix() for p in src.rglob('*')) == [
        'Clank', 'Clank/Clank.mtl', 'Clank/Clank.obj', 'Clank/tex', 'Clank/tex/body.png']
    assert [p.relative_to(src).as_posix() for p in load.model_files(src)] == ['Clank/Clank.obj']
    assert not any(p.name.startswith('escape') for p in tmp_path.rglob('*'))


def test_an_entry_name_keeps_inside_the_folder():
    assert load.entry_name('Clank\\tex\\body.png') == 'Clank/tex/body.png'
    assert load.entry_name('Clank\\') == 'Clank/' and load.entry_name('./a//b') == 'a/b'
    for outside in ('..\\x.obj', 'a/../../x.obj', '/etc/x', '\\x', 'C:\\x.obj', 'c:x.obj', '', '.', './'):
        assert load.entry_name(outside) is None, outside


def test_a_bare_model_brings_the_files_it_refers_to(tmp_path):
    folder = tmp_path / 'downloads'
    folder.mkdir()
    for name in ('char.gltf', 'char.bin', 'skin.png', 'other.obj', 'other.mtl', 'notes.txt', 'big.zip'):
        (folder / name).write_bytes(b'x')
    (folder / 'textures').mkdir()
    src = load.unpack(folder / 'char.gltf', tmp_path / 'source')
    assert sorted(p.name for p in src.iterdir()) == ['char.bin', 'char.gltf', 'other.mtl', 'skin.png']


@pytest.mark.blender
def test_triangles_are_counted_after_triangulating_each_face(tmp_path):
    obj = tmp_path / 'quad_and_tri.obj'
    obj.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 3 0 0\nv 2 1 0\nf 1 2 3 4\nf 5 6 7\n',
                   encoding='utf-8')
    listing = load.run(obj, tmp_path / 'load')
    assert listing[0]['triangles'] == 3 and listing[0]['vertices'] == 7


@pytest.mark.blender
def test_smd_is_recorded_but_is_not_a_model_offered_in_round1(tmp_path):
    download = tmp_path / 'download.zip'
    download.write_bytes(_zip({'a.SMD': b'version 1', 'nested.zip': _zip({'other.smd': b'version 1'}),
                              'body.obj': b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n'}))
    out = tmp_path / 'load'
    listing = load.run(download, out)
    assert load.MODEL_SUFFIXES == set(load.IMPORTERS)
    assert [p.name for p in load.model_files(out / 'source')] == ['body.obj']
    assert [(m['n'], m['file'], m['opened']) for m in listing] == [(1, 'body.obj', True)]
    assert json.loads((out / 'models.json').read_text()) == listing
    assert json.loads((out / 'load.json').read_text()) == {'unsupported': [
        {'file': 'a.SMD', 'why': 'no importer for .smd in Blender 5.2'},
        {'file': str(Path('nested') / 'other.smd'), 'why': 'no importer for .smd in Blender 5.2'}]}


@pytest.mark.blender
def test_a_bare_smd_records_why_it_cannot_open_before_failing(tmp_path):
    download = tmp_path / 'only.smd'
    download.write_text('version 1', encoding='utf-8')
    out = tmp_path / 'load'
    with pytest.raises(RuntimeError, match=r'only.smd: no importer for .smd in Blender 5.2'):
        load.run(download, out)
    assert json.loads((out / 'load.json').read_text()) == {'unsupported': [
        {'file': 'only.smd', 'why': 'no importer for .smd in Blender 5.2'}]}
    assert not load.model_files(out / 'source')


# Son Goku (JUMP All-Star Brawl): four objects with the same triangles at the same place, and two with one quad each.
TWINS_OBJ = ''.join(
    f'o {name}\nv 0 0 {z}\nv 1 0 {z}\nv 1 1 {z}\nv 0 1 {z}\nf {4 * k + 1} {4 * k + 2} {4 * k + 3} {4 * k + 4}\n'
    for k, (name, z) in enumerate([('body', 0), ('body_UV1', 0), ('body_outline', 0), ('body_outline_UV1', 0),
                                   ('hair', 2), ('face', 3)]))


def _shuffled_import(order):
    """An OBJ import that links the objects again in ``order`` (indices into the imported order), as Blender's FBX
    importer links a skinned file's objects in an order of its own each run. Returns the importer and the order it
    left, before load settles it."""
    left = []

    def importer(p):
        import bpy
        load.IMPORTERS['.obj.real'](p)
        coll = bpy.context.scene.collection
        objects = list(coll.objects)
        for ob in objects:
            coll.objects.unlink(ob)
        for i in order:
            coll.objects.link(objects[i])
        left.extend(ob.name for ob in coll.objects)
    return importer, left


@pytest.mark.blender
def test_the_same_file_linked_in_two_orders_loads_in_one_and_keeps_its_mesh_order(tmp_path, monkeypatch):
    import bpy
    (tmp_path / 'goku.obj').write_text(TWINS_OBJ, encoding='utf-8')
    monkeypatch.setitem(load.IMPORTERS, '.obj.real', load.IMPORTERS['.obj'])
    numbered, linked = [], []
    for k, order in enumerate([(3, 5, 0, 4, 2, 1), (1, 0, 2, 3, 4, 5)]):
        importer, left = _shuffled_import(order)
        monkeypatch.setitem(load.IMPORTERS, '.obj', importer)
        listing = load.run(tmp_path / 'goku.obj', tmp_path / f'load{k}')
        linked.append(left)
        bpy.ops.wm.open_mainfile(filepath=str(tmp_path / f'load{k}' / 'models' / listing[0]['blend']))
        numbered.append([o.name for o in bpy.context.scene.objects if o.type == 'MESH'])
    assert linked[0] != linked[1]                       # the importer left them in two orders
    assert numbered[0] == numbered[1]                   # the pieces are numbered alike all the same
    assert numbered[0][:4] == ['body', 'body_UV1', 'body_outline', 'body_outline_UV1']


@pytest.mark.blender
def test_objects_are_settled_by_name_then_mesh_then_triangles_whatever_suffix_blender_added(tmp_path):
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    coll = bpy.context.scene.collection
    for name, n_faces in (('zeta', 1), ('hand.001', 1), ('hand', 2), ('arm', 3)):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(i, 0, 0) for i in range(n_faces + 2)], [], [(0, i + 1, i + 2) for i in range(n_faces)])
        coll.objects.link(bpy.data.objects.new(name, me))
    coll.objects.link(bpy.data.objects.new('Armature', None))
    assert load.settle_order() == ['Armature', 'arm', 'hand.001', 'hand', 'zeta']   # 'hand' twice: fewer triangles first


def test_a_level_of_detail_is_read_from_the_name_in_its_common_spellings():
    lod = load.level_of_detail
    assert lod('Body_LOD0') == ('body', 0) and lod('body_lod_1') == ('body', 1) and lod('BodyLOD2') == ('body', 2)
    assert lod('Body LOD-3') == ('body', 3) and lod('BodyLod1') == ('body', 1) and lod('Body_LOD1.001') == ('body', 1)
    # an exporter's running numbers are not part of the base name
    assert lod('_000_Body_LOD0') == lod('_003_Body_LOD0') == ('body', 0) and lod('_004_Body_LOD1') == ('body', 1)
    for name in ('Body', 'Melody1', 'Lodge2', 'Body_LOD', 'Clod_3', 'hand.001'):
        assert lod(name) is None


def test_only_the_finest_level_of_each_base_stays():
    names = ['_000_Body_LOD0', '_001_Body_LOD1', '_002_Body_LOD2', '_003_Body_LOD0', '_004_Body_LOD1', 'Hat']
    assert load.finest_levels(names) == {'_001_Body_LOD1': ('body', 1, 0), '_002_Body_LOD2': ('body', 2, 0),
                                         '_004_Body_LOD1': ('body', 1, 0)}
    # no LOD0: the lowest number stays
    assert load.finest_levels(['arm_lod1', 'arm_lod3']) == {'arm_lod3': ('arm', 3, 1)}
    # a base at one level only, however many meshes, is left alone; so is a mesh with no level
    assert load.finest_levels(['main_lod1', 'main_lod1.001', 'eyes.LOD1', 'cape', 'Hat_LOD0']) == {}


@pytest.mark.blender
def test_the_coarser_levels_a_file_names_are_removed_with_their_meshes():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    coll = bpy.context.scene.collection
    for name in ('Body_LOD0', 'Body_LOD1', 'Body_LOD2', 'Hat'):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        coll.objects.link(bpy.data.objects.new(name, me))
    removed = load.drop_coarser_levels()
    assert [r['mesh'] for r in removed] == ['Body_LOD1', 'Body_LOD2'] and removed[0]['triangles'] == 1
    assert sorted(o.name for o in bpy.data.objects) == ['Body_LOD0', 'Hat']
    assert sorted(m.name for m in bpy.data.meshes) == ['Body_LOD0', 'Hat']


@pytest.mark.blender
def test_a_colour_set_colours_only_the_faces_that_draw_no_image():
    # Cloud: our conversion's white layer first, then his own set, his colour where no image is drawn and white over
    # his face's. Smash Ultimate: its set half (its neutral) over faces drawing an image.
    import bpy
    import numpy as np
    bpy.ops.wm.read_homefile(use_empty=True)
    face = bpy.data.materials.new('face')
    face.use_nodes = True
    tex = face.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.new('face', 2, 2)
    face.node_tree.links.new(tex.outputs['Color'], face.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    body = bpy.data.materials.new('body')
    body.use_nodes = True
    half, own = (.216, .216, .216, 1), (.2, .16, .14, 1)
    meshes = {}
    for name, faces, layers in (('cloud', [(0, 1, 2), (1, 3, 2)], {'Color': [(1, 1, 1, 1)] * 6,
                                                                  'Color.001': [half] * 3 + [own] * 3}),
                                ('ultimate', [(0, 1, 2)], {'Color': [half] * 3})):
        me = meshes[name] = bpy.data.meshes.new(name)
        me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], [], faces)
        me.materials.append(face)
        me.materials.append(body)
        me.polygons[-1].material_index = len(faces) - 1
        for layer, values in layers.items():
            me.color_attributes.new(layer, 'FLOAT_COLOR', 'CORNER').data.foreach_set('color', np.ravel(values))
        me.color_attributes.render_color_index = 0
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(name, me))
    assert load.colours() == {'cloud': {'white_over_images': ['Color.001'], 'dropped': ['Color'], 'drawn_by': ['body']},
                              'ultimate': {'white_over_images': ['Color'], 'dropped': ['Color']}}
    layers = meshes['cloud'].color_attributes
    assert [a.name for a in layers] == ['Color.001'] and layers.render_color_index == 0
    drawn = np.empty(24)
    layers['Color.001'].data.foreach_get('color', drawn)
    assert np.allclose(drawn.reshape(-1, 4), [(1, 1, 1, 1)] * 3 + [own] * 3)
    assert not len(meshes['ultimate'].color_attributes)
    assert load.colours() == {}


@pytest.mark.blender
def test_the_faces_drawing_no_image_draw_their_set_times_their_own_colour():
    # Looney Tunes Racing's Taz (an OBJ): his fur is a set on a material drawing no image (Kd 0.8, Blender's default),
    # his eyes and mouth images with a grey shading set over them. Blender's OBJ importer keeps the set on the mesh
    # only, and the costume's export carries only a set a material draws: he came out the material's grey.
    import bpy
    import numpy as np
    bpy.ops.wm.read_homefile(use_empty=True)
    face = bpy.data.materials.new('face')
    face.use_nodes = True
    tex = face.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.new('face', 2, 2)
    face.node_tree.links.new(tex.outputs['Color'], face.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    fur = bpy.data.materials.new('fur')
    fur.use_nodes = True
    fur.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.8, .8, .8, 1)
    unused = bpy.data.materials.new('unused')
    unused.use_nodes = True
    me = bpy.data.meshes.new('taz')
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0)], [], [(0, 1, 2), (2, 3, 4)])
    for m in (face, fur, unused):
        me.materials.append(m)
    me.polygons[1].material_index = 1
    brown, grey = (.38, .24, .19, 1), (.32, .32, .32, 1)
    me.color_attributes.new('Color', 'FLOAT_COLOR', 'POINT').data.foreach_set(
        'color', np.ravel([grey, grey, brown, brown, brown]))
    bpy.context.scene.collection.objects.link(bpy.data.objects.new('taz', me))
    assert load.colours() == {'taz': {'white_over_images': ['Color'], 'dropped': [], 'drawn_by': ['fur']}}
    shader = fur.node_tree.nodes['Principled BSDF']
    mix = shader.inputs['Base Color'].links[0].from_node
    assert (mix.type, mix.data_type, mix.blend_type) == ('MIX', 'RGBA', 'MULTIPLY')
    a, b = (next(s for s in mix.inputs if s.identifier == k) for k in ('A_Color', 'B_Color'))
    assert a.links[0].from_node.type == 'VERTEX_COLOR' and a.links[0].from_node.layer_name == ''
    assert np.allclose(b.default_value, (.8, .8, .8, 1))
    assert face.node_tree.nodes['Principled BSDF'].inputs['Base Color'].links[0].from_node.type == 'TEX_IMAGE'
    assert not unused.node_tree.nodes['Principled BSDF'].inputs['Base Color'].is_linked     # on no face of it
    drawn = np.empty(20)
    me.color_attributes['Color'].data.foreach_get('color', drawn)
    # vertices 0 and 1, only the image's face's, go white; 2, which the fur's face shares, keeps his brown
    assert np.allclose(drawn.reshape(-1, 4), [(1, 1, 1, 1), (1, 1, 1, 1), brown, brown, brown])
    assert load.colours() == {}                                                    # wired once


def _fbx(objects, connections):
    """A parsed FBX's element tree (parse_fbx's shape) from {uid: (kind, name, file)} and [(src, dst, property)]."""
    from collections import namedtuple
    elem = namedtuple('FBXElem', ('id', 'props', 'props_type', 'elems'))
    nodes = [elem(kind, [uid, name.encode() + b'\x00\x01' + kind], b'LSS',
                  [elem(b'RelativeFilename', [file.encode()], b'S', [])] if file else [])
             for uid, (kind, name, file) in objects.items()]
    links = [elem(b'C', [b'OP', s, d, p], b'SLLS', []) if p else elem(b'C', [b'OO', s, d], b'SLL', [])
             for s, d, p in connections]
    return elem(b'', [], b'', [elem(b'Objects', [], b'', nodes), elem(b'Connections', [], b'', links)])


def test_of_several_textures_on_one_property_the_first_is_the_material_s_texture():
    # LittleBigPlanet's Sackboy (written by Blender 2.79): his knit tiled three times, then the dirt map laid over it,
    # both on DiffuseColor and SpecularColor. Blender's importer took the last.
    root = _fbx({1: (b'Material', 'weave', ''), 2: (b'Texture', 'knit_2_diffuse', r'Textures\knit_2_diffuse.png'),
                 3: (b'Texture', 'pinched_dirt', 'pinched_dirt.png'), 4: (b'Texture', 'knit_2_bump', 'knit_2_bump.png'),
                 5: (b'Material', 'tongue', ''), 6: (b'Texture', 'pink_stars', 'pink_stars.png'),
                 9: (b'Model', 'body', '')},
                [(9, 0, None), (1, 9, None), (2, 1, b'DiffuseColor'), (2, 1, b'SpecularColor'), (4, 1, b'NormalMap'),
                 (3, 1, b'DiffuseColor'), (3, 1, b'SpecularColor'), (6, 5, b'DiffuseColor')])
    assert load.base_layers(root) == [
        {'material': 'weave', 'property': 'DiffuseColor', 'kept': 'knit_2_diffuse (knit_2_diffuse.png)',
         'dropped': ['pinched_dirt (pinched_dirt.png)']},
        {'material': 'weave', 'property': 'SpecularColor', 'kept': 'knit_2_diffuse (knit_2_diffuse.png)',
         'dropped': ['pinched_dirt (pinched_dirt.png)']}]
    kept = [(c.props[1], c.props[2], c.props[3] if len(c.props) > 3 else None) for c in root.elems[1].elems]
    assert kept == [(9, 0, None), (1, 9, None), (2, 1, b'DiffuseColor'), (2, 1, b'SpecularColor'),
                    (4, 1, b'NormalMap'), (6, 5, b'DiffuseColor')]
    assert load.base_layers(root) == []


def test_a_texture_found_under_several_folders_is_the_one_in_the_folder_the_file_recorded():
    # Splatoon's hair FBX names M_TeamColor_Alb.png bare, and Blender's search takes the body's in images/ first
    found = 'Female/images/m_teamcolor_alb.png'
    same = [found] + [f'Female/images/Hair00_0{k}/m_teamcolor_alb.png' for k in range(6)]
    recorded = [r'C:\Users\G\Splatoon 2\Model\Hair\Hair00_00_Nin_NX_NVN_szs\M_TeamColor_Alb.png', 'M_TeamColor_Alb.png']
    assert load.meant(found, recorded, same) == 'Female/images/Hair00_00/m_teamcolor_alb.png'


def test_the_search_s_pick_stands_where_no_folder_or_two_alike_are_named():
    found = 'Female/images/m_body_alb.png'
    same = [found, 'Female/images/00_Hlf/m_body_alb.png']
    assert load.meant(found, [r'C:\G\Player00_Nin_NX_NVN_szs\M_Body_Alb.png'], same) == found      # none named
    assert load.meant(found, ['M_Body_Alb.png'], same) == found                                   # no folder recorded
    two = ['a/Hair00_00/x.png', 'a/00_Hair00/x.png']
    assert load.meant('b/x.png', [r'C:\Hair00_00_szs\x.png'], two) == 'b/x.png'                   # both named alike


def test_of_folders_named_the_one_with_most_words_is_meant():
    # Inkling's tank on her back records ...\Tnk_Simple_Shot_First_Nin_NX_NVN_szs\; images/Tnk/ holds the plain tank's
    found = 'Clothes/images/Tnk/m_glass_alb.png'
    same = [found, 'Clothes/images/Tnk_Shot_First/m_glass_alb.png']
    recorded = [r'C:\G\ModelUpd\Tnk\Tnk_Simple_Shot_First_Nin_NX_NVN_szs\M_Glass_Alb.png', 'M_Glass_Alb.png']
    assert load.meant(found, recorded, same) == 'Clothes/images/Tnk_Shot_First/m_glass_alb.png'
    assert load.meant(found, [r'C:\G\Tnk_Simple_Nin_NX_NVN_szs\M_Glass_Alb.png'], same) == found
