"""prep/: the game's limits applied to a build, in order (run); and the Blender scenes the Melee tests build."""
from importlib import import_module
import json

import numpy as np
import pytest

from rigswap.plugins import melee
from rigswap.plugins.melee.facts import fighters
from rigswap.plugins.melee.prep import layers as prep_layers, memory as prep_memory, triangles as prep_triangles
# melee.prep is the plugin's function; this is its folder of that name
melee_prep = import_module('rigswap.plugins.melee.prep')


@pytest.fixture
def memory_targets(tmp_path, monkeypatch):
    from tests.plugins.melee.test_melee import _memory
    folder = tmp_path / 'targets'
    folder.mkdir()
    row = _memory()
    (folder / 'Mario.memory.json').write_text(json.dumps(row), encoding='utf-8')
    monkeypatch.setenv(melee.TARGETS_VAR, str(folder))
    return row


def _empty_scene():
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    return bpy


def _material(name, pixels):
    """A material drawing an image of ``pixels`` ((h, w, 4) uint8, top row first) through Base Color, from the front
    only (use_backface_culling), so prep gives no face of it a back for being drawn from behind (test_layers.py has
    the faces a two-sided material shows from behind)."""
    import bpy
    h, w = pixels.shape[:2]
    image = bpy.data.images.new(name, w, h, alpha=True)
    image.pixels.foreach_set((pixels[::-1].astype(np.float32) / 255).ravel())
    material = bpy.data.materials.new(name)
    material.use_backface_culling = True
    tree = material.node_tree
    bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    texture = tree.nodes.new('ShaderNodeTexImage')
    texture.image = image
    tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
    return material


def _mesh(name, verts, faces, materials=(), smooth=False):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    for m in materials:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _grid(name, n, materials=()):
    """An n x n grid of quads (2 n^2 triangles) with UVs and a vertex group weighting each vertex 0.5 to 1 by its x."""
    verts = [(x / n, y / n, 0.) for y in range(n + 1) for x in range(n + 1)]
    faces = [(y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x)
             for y in range(n) for x in range(n)]
    ob = _mesh(name, verts, faces, materials)
    uv = ob.data.uv_layers.new(name='UVMap')
    for loop in ob.data.loops:
        uv.data[loop.index].uv = verts[loop.vertex_index][:2]
    group = ob.vertex_groups.new(name='JOBJ_1')
    for v in ob.data.vertices:
        group.add([v.index], 0.5 + v.co.x / 2, 'REPLACE')
    for i, m in enumerate(ob.data.polygons):
        m.material_index = i % max(1, len(materials))
    return ob


def _flat(w, h, rgba=(200, 120, 40, 255)):
    return np.tile(np.array(rgba, np.uint8), (h, w, 1))


def _hidden_collection(name='hidden'):
    """A collection in the scene, hidden in the viewport and the view layer."""
    import bpy
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    collection.hide_viewport = True
    bpy.context.view_layer.layer_collection.children[name].hide_viewport = True
    return collection


def _keyed(name, n, materials=(), collection=None):
    """An n x n grid (_grid) with a Basis and two shape keys at rest, its object made from the mesh afterwards, as
    Blender's glTF importer makes one: it has no active shape key (Major Firepower's face_UV, majorfirepower-Zelda in
    runs/pilot6). In ``collection``, or the scene's."""
    import bpy
    made = _grid(name, n, materials)
    made.shape_key_add(name='Basis')
    for key, z in (('pain', 0.25), ('blink', -0.5)):
        moved = made.shape_key_add(name=key)
        moved.value = 0                                             # at rest (Blender adds one at 1)
        for point in moved.data:
            point.co.z = z
    groups, me = [g.name for g in made.vertex_groups], made.data
    bpy.data.objects.remove(made)
    ob = bpy.data.objects.new(name, me)
    for group in groups:
        ob.vertex_groups.new(name=group)
    (collection or bpy.context.scene.collection).objects.link(ob)
    assert ob.active_shape_key is None and [k.value for k in me.shape_keys.key_blocks[1:]] == [0, 0]
    return ob


def test_prep_writes_the_build_its_texture_formats_and_what_it_did(tmp_path, memory_targets):
    bpy = _empty_scene()
    _grid('body', 4, [_material('skin', _flat(1024, 512))])
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / 'build.blend'))
    (tmp_path / 'target.json').write_text(json.dumps({'name': 'Mario'}), encoding='utf-8')

    report = melee.prep(tmp_path)
    assert report['fighter'] == 'Mario' and report['triangles']['budget'] == 9000
    assert report['textures']['images'][0]['size'] == [256, 128]
    assert report['memory']['budget_bytes'] == prep_memory.costume_budget(fighters.fighter('Mario').preloaded)
    assert report['memory']['measurement'] == memory_targets
    assert report['memory']['before'] == {'textures': 16384, 'geometry': 384, 'retained': 192,
                                          'placement': 64, 'kept': 96, 'bytes': 17120}
    assert json.loads((tmp_path / 'textures.json').read_text(encoding='utf-8')) == {'skin': 'CMPR'}
    assert json.loads((tmp_path / 'prep.json').read_text(encoding='utf-8')) == report
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / 'build.blend'))
    assert tuple(bpy.data.images['skin'].size) == (256, 128)

    # Sidedness comes from topology, independent of a stale thin-layer reading.
    (tmp_path / 'layers.json').write_text(json.dumps(['cape']), encoding='utf-8')
    report = melee.prep(tmp_path)
    assert report['sidedness']['pieces'] == [dict(kind='object', id='body', boundary_edges=16, two_sided=True)]


def test_prep_applies_the_shape_keys_of_a_hidden_mesh_with_no_active_key(tmp_path, memory_targets):
    """majorfirepower-Zelda (runs/pilot6) failed in prep: "Operator bpy.ops.object.shape_key_remove.poll() failed,
    context is incorrect". Blender's glTF importer had left the face's object with no active shape key, which that
    operator's poll needs. A mesh not active, in a hidden collection, with no active key, comes out with its keys gone
    and the basis's shape (they were at rest)."""
    bpy = _empty_scene()
    skin = _material('skin', _flat(4, 4))
    face = _keyed('face', 4, [skin], collection=_hidden_collection())
    rest = np.array([v.co[:] for v in face.data.vertices])
    bpy.context.view_layer.objects.active = _grid('body', 2, [skin])
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / 'build.blend'))
    report = melee_prep.run(tmp_path, fighters.fighter('Mario'))
    assert report['shape_keys'] == {'face': ['Basis', 'pain', 'blink']}
    assert json.loads((tmp_path / 'prep.json').read_text(encoding='utf-8'))['shape_keys'] == report['shape_keys']
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / 'build.blend'))
    face = bpy.data.objects['face']
    assert face.data.shape_keys is None and face.users_collection[0].name == 'hidden'
    assert np.array_equal(np.array([v.co[:] for v in face.data.vertices]), rest)


def test_prep_drops_rendering_layers_but_keeps_physical_layers(tmp_path, memory_targets):
    bpy = _empty_scene()
    build = tmp_path / '3-build'
    build.mkdir()
    describe = tmp_path / '2-describe'
    describe.mkdir()
    pieces = [{'id': 1, 'belongs': 'character', 'moves_as': 'body'},
              {'id': 2, 'belongs': 'layer_of', 'layer_of': 1, 'moves_as': 'layer'},
              {'id': 3, 'belongs': 'character', 'moves_as': 'layer'}]
    (describe / 'description.json').write_text(json.dumps({'pieces': pieces}), encoding='utf-8')
    mats = [_material(n, _flat(4, 4)) for n in ('skin', 'outline')]
    ob = _mesh('body', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2), (2, 1, 0)], mats)
    ob.data.polygons[1].material_index = 1
    ob.data.attributes.new(prep_layers.PIECE, 'INT', 'FACE').data.foreach_set('value', [1, 2])
    sleeve = _mesh('sleeve', [(0, 0, 1), (1, 0, 1), (0, 1, 1)], [(0, 1, 2)], [mats[0]])
    sleeve.data.attributes.new(prep_layers.PIECE, 'INT', 'FACE').data[0].value = 3
    for mesh in (ob, sleeve):
        mesh.vertex_groups.new(name='JOBJ_0').add([0, 1, 2], 1, 'REPLACE')
    bpy.ops.wm.save_as_mainfile(filepath=str(build / 'build.blend'))
    report = melee_prep.run(build, fighters.fighter('Mario'))
    assert report['dropped_pieces'] == [{'piece': 2, 'meshes': ['body'], 'faces': 1, 'triangles': 1}]
    assert set(json.loads((build / 'textures.json').read_text())) == {'skin'}
    bpy.ops.wm.open_mainfile(filepath=str(build / 'build.blend'))
    assert prep_triangles.count_triangles(melee_prep.drawn_meshes()) == 2  # no back geometry: the open sleeve
    assert len(bpy.data.objects['body'].data.polygons) == 1                # draws both sides by its culling
    assert len(bpy.data.objects['sleeve'].data.polygons) == 1
    assert not bpy.data.objects['sleeve'].data.materials[0].use_backface_culling
    assert {v.value for o in melee_prep.drawn_meshes() for v in o.data.attributes[prep_layers.PIECE].data} == {1, 3}


def test_prep_keeps_the_faces_a_detail_draws_where_the_description_calls_them_a_rendering_layer(tmp_path,
                                                                                                memory_targets):
    # spiderman-Zelda (runs/pilot4): the description called Spider-Man's eye lenses (pieces 6 and 7) a layer over his
    # mask (belongs layer_of 1), the eye round chose the same two pieces as his eyes, and prep dropped both eye objects
    # whole: the export found no DObj for eyes_left.
    bpy = _empty_scene()
    build, describe = tmp_path / '3-build', tmp_path / '2-describe'
    build.mkdir()
    describe.mkdir()
    pieces = [{'id': 1, 'belongs': 'character', 'moves_as': 'body'},
              {'id': 5, 'belongs': 'layer_of', 'layer_of': 1, 'moves_as': 'layer'},
              {'id': 6, 'belongs': 'layer_of', 'layer_of': 1, 'moves_as': 'layer'}]
    (describe / 'description.json').write_text(json.dumps({'pieces': pieces}), encoding='utf-8')
    skin, eyes = (_material(n, _flat(4, 4)) for n in ('skin', 'eyes'))
    body = _mesh('body', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2), (2, 1, 0)], [skin])
    body.data.attributes.new(prep_layers.PIECE, 'INT', 'FACE').data.foreach_set('value', [1, 5])
    eye = _mesh('eyes_left', [(0, 0, 1), (1, 0, 1), (0, 1, 1)], [(0, 1, 2)], [eyes])
    eye.data.attributes.new(prep_layers.PIECE, 'INT', 'FACE').data[0].value = 6
    for mesh in (body, eye):
        mesh.vertex_groups.new(name='JOBJ_0').add([0, 1, 2], 1, 'REPLACE')
    (build / 'eyes.json').write_text(json.dumps({'objects': [{'name': 'eyes_left', 'side': 'left', 'pieces': [6]}]}),
                                     encoding='utf-8')
    bpy.ops.wm.save_as_mainfile(filepath=str(build / 'build.blend'))
    report = melee_prep.run(build, fighters.fighter('Mario'))
    assert report['dropped_pieces'] == [{'piece': 5, 'meshes': ['body'], 'faces': 1, 'triangles': 1}]
    assert report['rendering_layers_kept'] == [{'piece': 6, 'meshes': ['eyes_left'], 'faces': 1, 'triangles': 1}]
    assert set(json.loads((build / 'textures.json').read_text())) == {'skin', 'eyes'}
    bpy.ops.wm.open_mainfile(filepath=str(build / 'build.blend'))
    assert len(bpy.data.objects['eyes_left'].data.polygons) == 1 and len(bpy.data.objects['body'].data.polygons) == 1
    assert melee_prep.detail_objects(build) == ['eyes_left']


@pytest.mark.parametrize('alpha_link', ['none', 'direct', 'through_math'])
def test_a_missing_alpha_image_is_not_drawn_and_the_build_says_so(tmp_path, memory_targets, alpha_link):
    """An absent alpha image cannot define a cutout; a missing colour alone keeps the material's own colour."""
    bpy = _empty_scene()
    skin = _material('skin', _flat(4, 4))
    lost = _material('lost', _flat(4, 4))
    node = next(n for n in lost.node_tree.nodes if n.type == 'TEX_IMAGE')
    node.image.source = 'FILE'
    node.image.filepath = str(tmp_path / 'not-shipped.png')
    node.image.reload()
    bsdf = next(n for n in lost.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    if alpha_link == 'direct':
        lost.node_tree.links.new(node.outputs['Alpha'], bsdf.inputs['Alpha'])
    elif alpha_link == 'through_math':
        math = lost.node_tree.nodes.new('ShaderNodeMath')
        math.operation = 'MULTIPLY'
        math.inputs[1].default_value = 1.
        lost.node_tree.links.new(node.outputs['Alpha'], math.inputs[0])
        lost.node_tree.links.new(math.outputs[0], bsdf.inputs['Alpha'])
    bsdf.inputs['Base Color'].default_value = (0.5, 0.25, 0.25, 1.0)
    _grid('body', 2, [skin, lost])
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp_path / 'build.blend'))
    (tmp_path / 'target.json').write_text(json.dumps({'name': 'Mario'}), encoding='utf-8')
    report = melee.prep(tmp_path)
    assert len(report['missing_images']) == 1
    row = report['missing_images'][0]
    assert row['material'] == 'lost' and 'Principled BSDF: Base Color' in row['unlinked']
    formats = json.loads((tmp_path / 'textures.json').read_text(encoding='utf-8'))
    bpy.ops.wm.open_mainfile(filepath=str(tmp_path / 'build.blend'))
    faces = bpy.data.objects['body'].data.polygons
    if alpha_link == 'none':
        assert 'drawn' not in row and set(formats) == {'skin', 'lost'}
        assert [r['material'] for r in report['colours']] == ['lost']
        assert len(faces) == 4
    else:
        assert row['drawn'] is False and 'alpha image lost is missing from the source' in row['said']
        assert set(formats) == {'skin'} and report['colours'] == []
        assert len(faces) == 2 and all(p.material_index == 0 for p in faces)
        saved = json.loads((tmp_path / 'prep.json').read_text())
        assert saved['missing_images'] == report['missing_images']


def test_a_present_alpha_image_and_an_unconnected_missing_image_keep_the_faces(tmp_path):
    from rigswap.plugins.melee.prep.textures import drop_missing_images
    bpy = _empty_scene()
    cutout = _material('cutout', _flat(4, 4, (200, 120, 40, 128)))
    tree = cutout.node_tree
    image = next(n for n in tree.nodes if n.type == 'TEX_IMAGE')
    shader = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    tree.links.new(image.outputs['Alpha'], shader.inputs['Alpha'])
    unused = tree.nodes.new('ShaderNodeTexImage')
    unused.image = bpy.data.images.new('unused', 1, 1)
    unused.image.source = 'FILE'
    unused.image.filepath = str(tmp_path / 'absent.png')
    unused.image.reload()
    mesh = _grid('surface', 2, [cutout])
    assert drop_missing_images([mesh]) == [] and len(mesh.data.polygons) == 4
