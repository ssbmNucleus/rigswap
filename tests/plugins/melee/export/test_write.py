"""export/write.py: HSDRawViewer and Blender 4 both ways, and the vanilla DObjs put back around the build's."""
import json
import math
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rigswap.plugins import Failed, melee
from rigswap.plugins.melee import dae as melee_dae, dat as melee_dat
from rigswap.plugins.melee.export import checks as melee_checks, dobjs as melee_dobjs, write as export_write
from rigswap.plugins.melee.facts import fighters, parts as facts_parts
from rigswap.plugins.melee.prep import memory as prep_memory
from rigswap.plugins.melee.targets import prepare as targets_prepare
from tests.plugins.melee.test_dae import _dae, _dae_with_materials, _joint, _matrix
from tests.plugins.melee.test_dat import _DatBuilder, _costume, _fighter_file
from tests.plugins.melee.prep.test_prep import _empty_scene, _flat, _grid, _material, _mesh


def _fake_hsdraw(log):
    calls = []

    def run(*args, timeout=600):
        calls.append([str(a) for a in args])
        Path(args[4]).write_text('<COLLADA/>', encoding='utf-8')
        return log
    return calls, run


def test_export_dae_asks_for_the_dobjs_drawn_at_spawn(tmp_path, monkeypatch):
    calls, run = _fake_hsdraw('DObjs: 85 in the model; exporting 37 (high, options 0:0,1:0,2:0,4:0,6:0)')
    monkeypatch.setattr(export_write, 'hsdraw', run)
    export_write.export_dae(tmp_path / 'PlPeNr.dat', 'PlyPeach5K_Share_joint', tmp_path / 'a.dae')
    export_write.export_dae(tmp_path / 'PlPeNr.dat', 'PlyPeach5K_Share_joint', tmp_path / 'b.dae',
                     drawn_by=fighters.fighter('Peach'))
    assert calls[0][5:] == []
    assert calls[1][5:] == ['--names', 'place', '--dobjs', 'high', '--fighter', str(tmp_path / 'PlPe.dat'),
                            '--show', '0:0,1:0,2:0,4:0,6:0']


def test_export_dae_refuses_a_build_that_ignores_dobjs(tmp_path, monkeypatch):
    monkeypatch.setattr(export_write, 'hsdraw', _fake_hsdraw('SUCCESS: Exported model')[1])
    with pytest.raises(Failed, match='does not know --dobjs'):
        export_write.export_dae(tmp_path / 'PlMrNr.dat', 'PlyMario5K_Share_joint', tmp_path / 'a.dae',
                         drawn_by=fighters.fighter('Mario'))


def test_import_always_requests_alpha_test_and_passes_material_overrides(tmp_path, monkeypatch):
    calls = []
    def run(*args):
        calls.append(args)
        Path(args[5]).write_bytes(b'written')
        return 'preserving original bones'
    monkeypatch.setattr(export_write, 'hsdraw', run)
    export_write.import_dae('vanilla.dat', 'root', 'model.dae', tmp_path / 'a.dat')
    export_write.import_dae('vanilla.dat', 'root', 'model.dae', tmp_path / 'b.dat', 'formats.json', 'alpha.json')
    assert calls[0][6:] == ('--alpha-mode', 'Test')
    assert calls[1][6:] == ('--alpha-mode', 'Test', '--texture-formats', 'formats.json', '--alpha-modes', 'alpha.json')


def test_the_alpha_plan_reads_the_import_gate_in_the_format_the_import_is_told(tmp_path):
    from PIL import Image
    dae = _dae_with_materials(tmp_path / 'model.dae', ('skin', 'eye'))
    skin = np.full((16, 16, 4), 200, np.uint8)
    skin[..., 3] = 255
    skin[0, :4, 3] = 242                    # cammy's Tex_0012: just under opaque, which RGB5A3 writes opaque
    eye = skin.copy()
    eye[5, 5, 3] = 0
    for name, pixels in (('skin', skin), ('eye', eye)):
        Image.fromarray(pixels, 'RGBA').save(tmp_path / f'{name}.png')
    doc = json.dumps({'materials': [{'name': 'skin'}, {'name': 'eye', 'alphaMode': 'MASK'}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    model = melee_dae.Model(face_materials=['skin-material', 'eye-material'],
                            material_images={m: [str(tmp_path / f'{m[:-9]}.png')] for m in ('skin-material',
                                                                                         'eye-material')})
    formats = tmp_path / 'texture-formats.json'
    formats.write_text(json.dumps({'skin-material': 'RGB5A3', 'eye-material': 'RGB5A3'}), encoding='utf-8')
    path, rows = export_write.plan_alpha_modes(glb, dae, model, formats)
    # what the import is told is the same either way; the gate is what the check holds the file to
    assert json.loads(path.read_text(encoding='utf-8')) == {'skin-material': 'Test', 'eye-material': 'Test'}
    assert rows['skin-material']['transparent'] is False and rows['skin-material']['below_255'] == 4 / 256
    assert rows['eye-material']['transparent'] is True and rows['eye-material']['transparent_texels'] == 1 / 256
    assert rows['eye-material']['format'] == 'RGB5A3' and rows['eye-material']['source'] == 'eye'
    # with no formats the import writes its own, CMP, turned to RGB5A3 here: 242 is partial alpha
    _, rows = export_write.plan_alpha_modes(glb, dae, model)
    assert rows['skin-material']['format'] == 'RGB5A3' and rows['skin-material']['transparent'] is False


@pytest.mark.parametrize('source_alpha,code,mode,soft,basis', [
    (242, None, 'Test', 0., 'encoded'),
    (192, None, 'Blend', 1., 'encoded'),
    (192, 'RGB565', 'Test', 0., 'encoded'),
    (192, 'CI8:RGB5A3', 'Blend', 1., 'source for unmirrored encodings'),
    (192, 'CI8:IA8', 'Blend', 1., 'source for unmirrored encodings'),
])
@pytest.mark.parametrize('with_uvs', [False, True])
def test_mode_reads_encoded_alpha_at_drawn_samples(tmp_path, source_alpha, code, mode, soft, basis, with_uvs):
    from PIL import Image
    dae = tmp_path / 'model.dae'
    dae.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>'
                   '<material id="surface-material" name="surface"/></library_materials></COLLADA>')
    pixels = np.full((16, 16, 4), 255, np.uint8)
    pixels[..., 3] = source_alpha
    if with_uvs:
        # Most of the atlas is unused and would vote for the opposite mode.
        pixels[..., 3] = 192 if source_alpha == 242 else 255
        pixels[4:8, 4:8, 3] = source_alpha
    path = tmp_path / 'surface.png'
    Image.fromarray(pixels).save(path)
    doc = json.dumps({'materials': [{'name': 'surface', 'alphaMode': 'BLEND'}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    model = melee_dae.Model(face_materials=['surface-material'],
                           face_uvs=[[(.28, .70), (.42, .70), (.35, .58)]] if with_uvs else [],
                           material_images={'surface-material': [str(path)]})
    formats = tmp_path / 'formats.json'
    formats.write_text(json.dumps({'surface-material': code}))
    modes, rows = export_write.plan_alpha_modes(glb, dae, model, formats)
    row = rows['surface-material']
    assert row['mode'] == mode and row['soft_alpha'] == soft and row['alpha_basis'] == basis
    assert json.loads(modes.read_text()) == {'surface-material': mode}
    if basis != 'encoded':
        assert row['unmirrored_formats'] == [code] and row['transparent'] is None
    else:
        assert 'unmirrored_formats' not in row


def test_a_sidedness_variant_is_planned_with_its_source(tmp_path):
    # zerof-Marth's face mask (ns-7b): its culled copy kept only triangles drawing hard alpha, planned Test alone
    # against the mask's Blend, on the mask's own corners. A variant is its source drawn culled: one plan for both
    from PIL import Image
    dae = tmp_path / 'model.dae'
    dae.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_materials>'
                   '<material id="mask-material" name="mask"/><material id="mask_001-material" name="mask.001"/>'
                   '</library_materials></COLLADA>')
    pixels = np.full((16, 16, 4), 255, np.uint8)
    pixels[4:8, 4:8, 3] = 128                       # soft where the mask draws; the copy draws the solid corner
    path = tmp_path / 'mask.png'
    Image.fromarray(pixels).save(path)
    doc = json.dumps({'materials': [{'name': 'mask', 'alphaMode': 'BLEND'},
                                    {'name': 'mask.001', 'alphaMode': 'BLEND'}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    model = melee_dae.Model(face_materials=['mask-material', 'mask_001-material'],
                            face_uvs=[[(.28, .70), (.42, .70), (.35, .58)], [(.78, .20), (.92, .20), (.85, .08)]],
                            material_images={'mask-material': [str(path)], 'mask_001-material': [str(path)]})
    _, alone = export_write.plan_alpha_modes(glb, dae, model)
    assert (alone['mask-material']['mode'], alone['mask_001-material']['mode']) == ('Blend', 'Test')
    _, rows = export_write.plan_alpha_modes(glb, dae, model, None, {'mask.001': 'mask'})
    assert rows['mask-material']['mode'] == rows['mask_001-material']['mode'] == 'Blend'
    assert rows['mask-material']['soft_alpha'] == rows['mask_001-material']['soft_alpha'] == .5


@pytest.mark.parametrize('shared', [False, True])
def test_a_texture_an_opaque_material_draws_with_no_texel_through_is_written_opaque_and_said(tmp_path, shared):
    """goku-budokai-Zelda (runs/pilot6): the download's hair.png is black, every texel's alpha 0, and its material is
    opaque (.mtl d 1.0); the import's gate gave it an alpha test no texel passed, and Goku came out bald. The build
    draws it black. Now it is written opaque, as the build draws it, and the plan says so; a material drawing the same
    image with its alpha keeps it as it is, and that is said too."""
    from PIL import Image
    names = ('hair', 'face', 'veil') if shared else ('hair', 'face')
    dae = tmp_path / 'model.dae'
    dae.write_text('<?xml version="1.0" encoding="utf-8"?>'
                   '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"><library_materials>'
                   + ''.join(
                       f'<material id="{n}-material" name="{n}"/>' for n in names) + '</library_materials></COLLADA>',
                   encoding='utf-8')
    hair = np.zeros((16, 16, 4), np.uint8)
    face = np.full((16, 16, 4), 200, np.uint8)
    face[3, 3, 3] = 0                                   # a cutout, which keeps its alpha
    Image.fromarray(hair, 'RGBA').save(tmp_path / 'texture_000.png')
    Image.fromarray(face, 'RGBA').save(tmp_path / 'texture_001.png')
    files = {'hair': 'texture_000.png', 'face': 'texture_001.png', 'veil': 'texture_000.png'}
    doc = json.dumps({'materials': [{'name': 'hair', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}},
                                    {'name': 'face'}, {'name': 'veil', 'alphaMode': 'BLEND'}],
                      'textures': [{'source': 0}], 'images': [{'name': 'hair.png'}]}).encode()
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    model = melee_dae.Model(face_materials=[f'{n}-material' for n in names],
                            material_images={f'{n}-material': [str(tmp_path / files[n])] for n in names})
    _, rows = export_write.plan_alpha_modes(glb, dae, model)
    said = rows['hair-material']['fully_transparent']
    assert said['texture'] == 'hair.png' and said['alpha_max'] == 0
    with Image.open(tmp_path / 'texture_000.png') as image:
        written = np.asarray(image.convert('RGBA'))
    if shared:
        assert said['written'] == 'as it is' and 'veil' in said['said'] and 'draws nothing of hair' in said['said']
        assert (written[..., 3] == 0).all() and rows['hair-material']['transparent'] is True
    else:
        assert said['written'] == 'opaque' and said['said'].startswith('texture hair.png is fully transparent')
        assert (written[..., 3] == 255).all() and (written[..., :3] == 0).all()
        assert rows['hair-material']['transparent'] is False and rows['hair-material']['below_255'] == 0
    # the cutout is no fault: some of its texels pass the test, and its alpha is kept
    assert 'fully_transparent' not in rows['face-material'] and rows['face-material']['transparent'] is True
    with Image.open(tmp_path / 'texture_001.png') as image:
        assert np.asarray(image)[3, 3, 3] == 0


def test_a_texture_draws_nothing_when_no_texel_passes_its_alpha_test():
    from rigswap.plugins.melee.facts import alpha as facts_alpha
    clear = np.zeros((8, 8, 4), np.uint8)
    faint = clear.copy()
    faint[..., 3] = 100                                 # RGB5A3 writes it as 3 of 7: under the test's 0x80
    some = clear.copy()
    some[0, 0, 3] = 255
    assert facts_alpha.draws_nothing([clear], None, 'Test') is True
    assert facts_alpha.draws_nothing([faint], 'RGB5A3', 'Test') is True
    assert facts_alpha.draws_nothing([faint], 'RGB5A3', 'Blend') is False
    assert facts_alpha.draws_nothing([some], None, 'Test') is False
    assert facts_alpha.draws_nothing([], None, 'Test') is False
    assert facts_alpha.draws_nothing([clear], 'CI8:RGB5A3', 'Test') is None     # only the file can tell


def test_materials_are_written_alike_when_the_import_makes_the_same_bytes_of_them(tmp_path):
    from PIL import Image
    pixels = np.full((8, 8, 4), 90, np.uint8)
    Image.fromarray(pixels, 'RGBA').save(tmp_path / 'texture_002.png')
    Image.fromarray(pixels, 'RGBA').save(tmp_path / 'copy.png', compress_level=1)    # other file, same pixels
    pixels[0, 0, 0] = 91
    Image.fromarray(pixels, 'RGBA').save(tmp_path / 'other.png')
    images = {'body': 'texture_002.png', 'outline': 'texture_002.png', 'copy': 'copy.png', 'other': 'other.png',
              'ci8': 'texture_002.png', 'blend': 'texture_002.png', 'cutout': 'texture_002.png'}
    model = melee_dae.Model(face_materials=[*images, 'plain', 'plain2'],
                            material_images={m: [str(tmp_path / p)] for m, p in images.items()})
    opaque = {'mode': 'Test', 'transparent': False, 'format': 'CMP'}
    plan = {m: dict(opaque) for m in images}
    plan['ci8']['format'] = 'CI8:RGB565'
    plan['blend']['mode'] = 'Blend'                         # an opaque texture gets no alpha op whatever its mode
    plan['cutout']['transparent'] = True                    # ... and a transparent one gets its mode's
    alike = export_write.written_alike(model, plan)
    groups = {}
    for m, key in alike.items():
        groups.setdefault(key, set()).add(m)
    assert sorted(map(sorted, groups.values())) == [['blend', 'body', 'copy', 'outline'], ['ci8'], ['cutout'],
                                                    ['other'], ['plain', 'plain2']]
    # an eye decal clamped and the same image repeated: two TObjs, two materials (preserve_samplers' wrap modes)
    alike = export_write.written_alike(model, plan, {'outline': (0, 0), 'body': export_write.IMPORT_WRAP})
    assert alike['outline'] != alike['body'] == alike['copy']


def test_blenders_armature_node_is_taken_out_and_the_file_says_y_up(tmp_path):
    mesh = '<node id="body" name="body" type="NODE"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix></node>'
    dae = _dae(tmp_path / 'blender.dae', '<node id="Armature" name="Armature" type="NODE">'
               '<matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
               + _joint('JOBJ_0', _matrix(), _joint('JOBJ_1', _matrix(y=2))) + mesh + '</node>')
    before = melee_dae.read_dae(dae).positions()
    export_write._lay_out_as_hsdraw(dae)
    root = ET.parse(dae).getroot()
    ns = melee_dae.NS
    assert [n.get('name') for n in root.find(f'{ns}library_visual_scenes/{ns}visual_scene')] == ['JOBJ_0', 'body']
    assert root.find(f'{ns}asset/{ns}up_axis').text == 'Y_UP'
    after = melee_dae.read_dae(dae).positions()
    assert not melee_checks.compare_joints(before, after).moved()


@pytest.mark.blender
def test_six_influences_survive_the_plugin_glb_and_collada_paths(tmp_path):
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    arm = bpy.data.objects.new('arm', bpy.data.armatures.new('arm'))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    previous = None
    for i in range(6):
        bone = arm.data.edit_bones.new(f'JOBJ_{i}')
        bone.head, bone.tail, bone.parent = (0, 0, i), (0, 0, i + 1), previous
        previous = bone
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = _mesh('skin', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    mesh.modifiers.new('skin', 'ARMATURE').object = arm
    mesh.parent = arm
    for i in range(6):
        mesh.vertex_groups.new(name=f'JOBJ_{i}').add([0, 1, 2], 1 / 6, 'REPLACE')
    blend = tmp_path / 'build.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    glb = export_write.blend_to_glb(blend, tmp_path / 'model.glb')
    dae = export_write.glb_to_dae(glb, tmp_path / 'model.dae')
    assert all(len(w) == 6 for w in melee_dae.read_dae(dae).weights)
    target = targets_prepare.dae_to_glb(dae, tmp_path / 'target.glb')
    raw = target.read_bytes()
    doc = json.loads(raw[20:20 + struct.unpack_from('<I', raw, 12)[0]])
    assert all('JOINTS_1' in p['attributes'] for m in doc['meshes'] for p in m['primitives'])


def test_export_keys_the_texture_formats_by_the_dae_material_id(tmp_path):
    dae = _dae_with_materials(tmp_path / 'model.dae', ('skin', 'eye'))
    (tmp_path / 'textures.json').write_text(json.dumps({'skin': 'CMPR', 'eye': 'CI8:RGB565', 'unused': 'CMPR'}),
                                            encoding='utf-8')
    with pytest.raises(Failed, match='missing materials.*unused'):
        export_write._formats_by_dae_id(tmp_path / 'textures.json', dae)
    (tmp_path / 'textures.json').write_text(json.dumps({'skin': 'CMPR', 'eye': 'CI8:RGB565'}), encoding='utf-8')
    formats = export_write._formats_by_dae_id(tmp_path / 'textures.json', dae)
    assert json.loads(formats.read_text(encoding='utf-8')) == {'skin-material': 'CMPR', 'eye-material': 'CI8:RGB565'}
    assert export_write._formats_by_dae_id(tmp_path / 'none.json', dae) is None


@pytest.mark.blender
@pytest.mark.parametrize('mirrored', [False, True])
def test_export_bakes_empty_parents_and_preserves_mirrored_winding(mirrored):
    bpy = _empty_scene()
    from mathutils import Matrix, Vector
    arm = bpy.data.objects.new('Armature', bpy.data.armatures.new('Armature'))
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bone = arm.data.edit_bones.new('JOBJ_0')
    bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    parent = bpy.data.objects.new('translated empty', None)
    bpy.context.scene.collection.objects.link(parent)
    parent.location.x = 5
    arm.parent = parent
    ob = _mesh('triangle', [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    ob.parent = arm
    ob.scale.x = -1 if mirrored else 1
    bpy.context.view_layer.update()
    y_up = Matrix.Rotation(-math.pi / 2, 4, 'X')
    expected = [y_up @ ob.matrix_world @ v.co for v in ob.data.vertices]
    expected_normal = (y_up @ ob.matrix_world).to_3x3().inverted().transposed() @ Vector((0, 0, 1))
    namespace = {}
    exec(export_write.GLB_TO_DAE.split('src, dst =', 1)[0], namespace)
    namespace['bake_worlds'](arm, [ob])
    assert np.allclose(arm.data.bones[0].head_local, [5, 0, 0], atol=1e-6)
    assert np.allclose([v.co for v in ob.data.vertices], expected, atol=1e-6)
    assert ob.data.polygons[0].normal.dot(expected_normal) > .999
    assert np.allclose(arm.matrix_world, np.eye(4), atol=1e-6)
    assert np.allclose(ob.matrix_world, np.eye(4), atol=1e-6)


def test_a_texture_the_glb_draws_and_the_dae_lost_is_named(tmp_path):
    """Goku Jump's materials multiply their image by the vertex colours; Blender's Collada export then wrote no
    texture, and the costume drew every material untextured. The .glb is the reference: what it draws with a texture,
    the .dae must too."""
    dae = _dae_with_materials(tmp_path / 'model.dae', ('skin',))
    doc = json.dumps({'asset': {'version': '2.0'}, 'materials': [
        {'name': 'skin', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}},
        {'name': 'eye', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 1}}},
        {'name': 'plain', 'pbrMetallicRoughness': {'baseColorFactor': [1, 0, 0, 1]}}]}).encode()
    doc += b' ' * (-len(doc) % 4)
    glb = tmp_path / 'model.glb'
    glb.write_bytes(struct.pack('<3I', 0x46546C67, 2, 20 + len(doc)) + struct.pack('<2I', len(doc), 0x4E4F534A) + doc)
    assert export_write.textures_lost(glb, dae) == ['eye']
    _dae_with_materials(tmp_path / 'model.dae', ('skin', 'eye'))
    assert export_write.textures_lost(glb, dae) == []


def _blender4_or_none():
    try:
        from rigswap.load import convert
        return convert.blender4()
    except RuntimeError:
        return None


def _emissive_glb(path, **changes):
    material = {'name': 'skin', 'emissiveTexture': {'index': 0, 'texCoord': 1,
                'extensions': {'KHR_texture_transform': {'offset': [0.25, 0], 'scale': [0.5, 1]}}},
                'emissiveFactor': [1, 0.5, 0.25], 'pbrMetallicRoughness': {'baseColorFactor': [0, 0, 0, 0.75]},
                'extensions': {'KHR_materials_emissive_strength': {'emissiveStrength': 1.9}}, **changes}
    doc = json.dumps({'asset': {'version': '2.0'}, 'materials': [material],
                      'textures': [{'source': 0, 'sampler': 0}], 'samplers': [{'wrapS': 33648, 'wrapT': 33071}],
                      'images': [{'bufferView': 0, 'mimeType': 'image/png'}]}).encode()
    doc += b' ' * (-len(doc) % 4)
    binary = struct.pack('<2I', 8, 0x004E4942) + b'payload!'
    path.write_bytes(struct.pack('<3I', 0x46546C67, 2, 20 + len(doc) + len(binary))
                     + struct.pack('<2I', len(doc), 0x4E4F534A) + doc + binary)
    return material, binary


def test_emissive_only_texture_reaches_conversion_and_its_existing_checks(tmp_path, monkeypatch):
    glb, dae = tmp_path / 'model.glb', tmp_path / 'model.dae'
    original, binary = _emissive_glb(glb)

    def convert(script, src, dst):
        material = export_write._gltf(src)['materials'][0]
        assert material['pbrMetallicRoughness']['baseColorTexture'] == original['emissiveTexture']
        assert material['pbrMetallicRoughness']['baseColorFactor'] == [1, 0.5, 0.25, 0.75]
        assert material['emissiveTexture'] == original['emissiveTexture']
        assert material['extensions'] == original['extensions']
        _dae_with_materials(dst, ('skin',))
        tree = ET.parse(dst)
        ET.SubElement(ET.SubElement(tree.getroot(), melee_dae.NS + 'asset'), melee_dae.NS + 'up_axis').text = 'Y_UP'
        tree.write(dst)
        return 0, ''

    monkeypatch.setattr(export_write, '_blender4', convert)
    export_write.glb_to_dae(glb, dae)
    raw = glb.read_bytes()
    assert len(raw) == struct.unpack_from('<I', raw, 8)[0]
    assert struct.unpack_from('<I', raw, 12)[0] % 4 == 0
    assert raw.endswith(binary)
    assert export_write._gltf(glb)['samplers'] == [{'wrapS': 33648, 'wrapT': 33071}]
    assert export_write.glb_textured(glb) == {'skin'}
    assert export_write.textures_lost(glb, dae) == []
    _dae_with_materials(dae, ('eye',))
    assert export_write.textures_lost(glb, dae) == ['skin']
    export_write._emission_to_diffuse(glb)
    assert glb.read_bytes() == raw


@pytest.mark.parametrize('changes', [
    {'pbrMetallicRoughness': {'baseColorTexture': {'index': 1}, 'baseColorFactor': [0, 0, 0, 1]}},
    {'pbrMetallicRoughness': {'baseColorFactor': [0.1, 0, 0, 1]}},
    {'pbrMetallicRoughness': {}},
    {'emissiveTexture': None}, {'emissiveFactor': [0, 0, 0]},
    {'extensions': {'KHR_materials_emissive_strength': {'emissiveStrength': 0}}},
])
def test_emission_conversion_keeps_other_material_channels_byte_for_byte(tmp_path, changes):
    glb = tmp_path / 'model.glb'
    _emissive_glb(glb, **changes)
    original = glb.read_bytes()
    export_write._emission_to_diffuse(glb)
    assert glb.read_bytes() == original


@pytest.mark.blender
@pytest.mark.skipif(_blender4_or_none() is None, reason='needs a Blender 4.x (rigswap.load.convert)')
@pytest.mark.parametrize('emissive', [False, True])
def test_glb_to_dae_writes_the_colour_image(tmp_path, emissive):
    """Vertex-colour multiplication and emission-only graphs both retain their colour image in Collada."""
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    arm = bpy.data.objects.new('Armature', bpy.data.armatures.new('Armature'))
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bone = arm.data.edit_bones.new('JOBJ_0')
    bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    ob = _grid('body', 2, [_material('skin', _flat(8, 8))])
    ob.vertex_groups.new(name='JOBJ_0').add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
    ob.modifiers.new('rig', 'ARMATURE').object = arm
    ob.parent = arm
    colours = ob.data.color_attributes.new('Color', 'BYTE_COLOR', 'CORNER')
    for c in colours.data:
        c.color = (0.5, 0.5, 0.5, 1.0)
    tree = ob.data.materials[0].node_tree
    bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    image = next(n for n in tree.nodes if n.type == 'TEX_IMAGE')
    mix = tree.nodes.new('ShaderNodeMix')
    mix.data_type, mix.blend_type, mix.inputs['Factor'].default_value = 'RGBA', 'MULTIPLY', 1.0
    attribute = tree.nodes.new('ShaderNodeVertexColor')
    attribute.layer_name = 'Color'
    tree.links.new(image.outputs['Color'], mix.inputs['A'])
    tree.links.new(attribute.outputs['Color'], mix.inputs['B'])
    tree.links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
    if emissive:
        tree.links.remove(bsdf.inputs['Base Color'].links[0])
        bsdf.inputs['Base Color'].default_value = (0, 0, 0, 1)
        tree.links.new(image.outputs['Color'], bsdf.inputs['Emission Color'])
        bsdf.inputs['Emission Strength'].default_value = 1.9
    image.image.pack()
    glb = tmp_path / 'model.glb'
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', export_animations=False)
    bpy.ops.wm.read_homefile(use_empty=True)

    dae = export_write.glb_to_dae(glb, tmp_path / 'dae' / 'model.dae')
    model = melee_dae.read_dae(dae)
    assert model.materials == 1 and len(model.textures) == 1 and None not in model.textures.values()
    assert export_write.textures_lost(glb, dae) == []


def test_kept_resources_survive_and_their_envelopes_use_the_written_skeleton(tmp_path, monkeypatch):
    symbol = 'PlyTest_Share_joint'
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True)] * 3), (0, [((0,), True)])], {})
    vanilla = b.save(tmp_path / 'vanilla.dat', [(symbol, root)])
    d = melee_dat.Dat.read(vanilla)
    tree = melee_dat.joints(d, symbol)
    obj = melee_dat.joint_dobjs(d, tree[1])[0]
    pobj = d.ptr(obj + 0x0c)
    # A real bone pointer inside a retained envelope must not pull in the old skeleton.
    envelope = d.add(bytes(16))
    d.point(envelope, tree[1])
    struct.pack_into('>f', d.data, envelope + 4, 1.)
    d.point(pobj + 0x14, envelope)
    d.save(vanilla)
    b = _DatBuilder()
    b.add(bytes(100))  # move the import's joints so accidental old offsets fail
    root, _ = _costume(b, [(None, [((0,), True)]), (0, [])], {})
    imported = b.save(tmp_path / 'imported.dat', [(symbol, root)])
    fighter = _fighter_file(tmp_path / 'fighter.dat', [], {0: [0, 1, 3], 1: [2]})
    monkeypatch.setattr(facts_parts, 'for_symbol', lambda s: (3,))
    before = vanilla.read_bytes()
    report = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat')
    assert report['kept_parts']['pass'] and report['kept_parts']['bytes'] > 0
    assert report['kept_parts']['places'] == [3] and report['blank'] == 1
    out = melee_dat.Dat.read(tmp_path / 'out.dat')
    tree = melee_dat.joints(out, symbol)
    kept = melee_dat.joint_dobjs(out, tree[1])[0]
    assert out.ptr(out.ptr(out.ptr(kept + 0x0c) + 0x14)) == tree[1]
    assert vanilla.read_bytes() == before


@pytest.mark.plugin
@pytest.mark.skipif(not (Path(os.environ.get(melee.FILES_VAR, '.')) / 'PlLkNr.dat').is_file(),
                    reason=f'needs vanilla files in {melee.FILES_VAR}')
def test_every_fighters_kept_resources_copy_and_corruption_is_detected(tmp_path):
    files = Path(os.environ[melee.FILES_VAR])
    for f in fighters.in_scope():
        if not f.keep:
            continue
        path = files / f'Pl{f.code}Nr.dat'
        source, dest = melee_dat.Dat.read(path), melee_dat.Dat.read(path)
        kept, record = export_write.copy_parts(source, dest, f.symbol, f.keep)
        objects = [o for j in melee_dat.joints(dest, f.symbol) for o in melee_dat.joint_dobjs(dest, j)]
        for place, (material, geometry) in kept.items():
            dest.point(objects[place] + 8, material)
            dest.point(objects[place] + 0x0c, geometry)
        out = tmp_path / f'{f.name}.dat'
        dest.save(out)
        assert melee_checks.check_parts(path, out, f.symbol, record)['pass'], f.name
        assert prep_memory.budget(f)['bytes'] == record['bytes'] + 31  # first copied block's alignment
        assert prep_memory.budget(f)['triangles'] > 0
        dest.data[record['blocks'][0][1]] ^= 1
        dest.save(out)
        assert not melee_checks.check_parts(path, out, f.symbol, record)['pass'], f.name


def test_the_build_is_laid_into_the_vanilla_dobjs(tmp_path):
    symbol, matanim = 'PlyTest_Share_joint', 'PlyTest_Share_matanim_joint'
    animations = {0: [(), (0,), (), ()], 1: [()]}
    # vanilla: a root with four DObjs (the second one's TObj animated, the third with two TObjs) and a child joint with
    # one more; the game animates TObj 1 and draws DObjs 0-4 at spawn
    b = _DatBuilder()
    root, mroot = _costume(b, [(None, [((0,), True), ((0,), True), ((0, 1), True), ((0,), True)]),
                               (0, [((0,), True)])], animations)
    vanilla = b.save(tmp_path / 'PlTsNr.dat', [(symbol, root), (matanim, mroot)])
    fighter = _fighter_file(tmp_path / 'PlTs.dat', [1], {0: [0, 1, 2, 3, 4]})
    # the import: the build's two DObjs on the root (one textured, one not), the vanilla matanim tree kept
    b = _DatBuilder()
    root, mroot = _costume(b, [(None, [((0,), True), ((), True)]), (0, [])], animations)
    imported = b.save(tmp_path / 'imported.dat', [(symbol, root), (matanim, mroot)])

    tables = melee_dobjs.fighter_tables(fighter)
    assert tables.animated == (1,) and tables.body == (0, 1, 2, 3, 4) and not tables.elsewhere
    before = melee_dobjs.layout(vanilla, symbol)
    assert [(s.joint, s.local, s.textures, s.animated) for s in before] == [
        (0, 0, (0,), ()), (0, 1, (0,), (0,)), (0, 2, (0, 1), ()), (0, 3, (0,), ()), (1, 0, (0,), ())]

    record = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat')
    assert record['places'] == [3, 2] and record['blank'] == 3        # the untextured one chose first: after the eye
    after = melee_dobjs.layout(tmp_path / 'out.dat', symbol)
    assert melee_dobjs.layout_differs(before, after, tables.animated) is None
    assert [s.drawn for s in after] == [False, False, True, True, False]
    assert [s.textures for s in after] == [(0,), (0,), (), (0,), (0,)]
    assert melee_dobjs.texture_places(after, tables.animated) == [(1, 0)] and after[1].animated == (0,)
    out = melee_dat.Dat.read(tmp_path / 'out.dat')
    assert all(at + 4 <= len(out.data) and out.u32(at) < len(out.data) for at in out.relocations)
    assert record['low_detail']['places'] == {} and record['low_detail']['left_out'] == [0, 1]


def test_untextured_materials_fit_before_animated_textures_in_both_detail_sets(tmp_path):
    symbol, matanim = 'PlyTest_Share_joint', 'PlyTest_Share_matanim_joint'
    animations = {0: [()] * 9 + [(0,)]}
    b = _DatBuilder()
    root, mroot = _costume(b, [(None, [((0,), True)] * 10)], animations)
    vanilla = b.save(tmp_path / 'vanilla.dat', [(symbol, root), (matanim, mroot)])
    fighter = _fighter_file(tmp_path / 'fighter.dat', [9], {0: [0, 1, 2], 1: [3, 4, 5]})
    b = _DatBuilder()
    root, mroot = _costume(b, [(None, [((0,), True, 1), ((), True, 2), ((0,), True, 3)])], animations)
    imported = b.save(tmp_path / 'imported.dat', [(symbol, root), (matanim, mroot)])
    record = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat')
    assert record['low_detail']['left_out'] == []
    assert record['blank_texture_slots'] == {6: 3}
    before, after = melee_dobjs.layout(vanilla, symbol), melee_dobjs.layout(tmp_path / 'out.dat', symbol)
    assert melee_dobjs.layout_differs(before, after, (9,)) is None
    assert melee_dobjs.texture_places(after, (9,)) == [(9, 0)]
    source, out = melee_dat.Dat.read(imported), melee_dat.Dat.read(tmp_path / 'out.dat')
    source_objects = melee_dat.joint_dobjs(source, melee_dat.joints(source, symbol)[0])
    objects = melee_dat.joint_dobjs(out, melee_dat.joints(out, symbol)[0])
    for n, place in enumerate(record['places']):
        # No padding or copies of a drawn material: its original MObj and PObjs still draw it.
        assert out.ptr(objects[place] + 8) == source.ptr(source_objects[n] + 8)
        assert out.ptr(objects[place] + 12) == source.ptr(source_objects[n] + 12)
    assert after[record['places'][1]].textures == ()


def test_the_low_detail_dobjs_point_at_the_builds_own_geometry(tmp_path, monkeypatch):
    symbol = 'PlyTest_Share_joint'
    # vanilla: seven DObjs on the root, set 0 draws 0-3, set 1 (low detail) 4-6; nothing animated
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True)] * 7)], {})
    vanilla = b.save(tmp_path / 'PlTsNr.dat', [(symbol, root)])
    fighter = _fighter_file(tmp_path / 'PlTs.dat', [], {0: [0, 1, 2, 3], 1: [4, 5, 6]})
    assert melee_dobjs.fighter_tables(fighter).low == (4, 5, 6)
    # the import: three DObjs, the first and third from one material (render mode 1), the second from another
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True, 1), ((0,), True, 2), ((0,), True, 1)])], {})
    imported = b.save(tmp_path / 'imported.dat', [(symbol, root)])

    record = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat')
    assert record['places'] == [0, 1, 0] and record['blank'] == 3
    assert record['low_detail']['places'] == {4: [0, 2], 5: [1]} and record['low_detail']['left_out'] == []
    after = melee_dobjs.layout(tmp_path / 'out.dat', symbol)
    assert melee_dobjs.layout_differs(melee_dobjs.layout(vanilla, symbol), after, ()) is None
    assert [s.drawn for s in after] == [True, True, False, False, True, True, False]

    d = melee_dat.Dat.read(tmp_path / 'out.dat')
    dobjs = d.chain(d.ptr(melee_dat.joints(d, symbol)[0] + 0x10), 0x04)
    pobjs = [d.chain(d.ptr(x + 0x0C), 0x04) for x in dobjs]
    # place 5 is the second DObj again: the same material, the same PObjs
    assert d.ptr(dobjs[5] + 0x08) == d.ptr(dobjs[1] + 0x08) and pobjs[5] == pobjs[1]
    # places 0 and 4 draw the first and third in the first one's material: each a copy of the first's PObj, flags and
    # all (culling is a PObj's own), then the third's own
    assert d.ptr(dobjs[4] + 0x08) == d.ptr(dobjs[0] + 0x08)
    assert len(pobjs[0]) == len(pobjs[4]) == 2 and pobjs[4][0] != pobjs[0][0] and pobjs[4][1] == pobjs[0][1]
    assert bytes(d.data[pobjs[4][0] + 0x08:pobjs[4][0] + 0x18]) == bytes(d.data[pobjs[0][0] + 0x08:pobjs[0][0] + 0x18])
    assert all(at + 4 <= len(d.data) and d.u32(at) < len(d.data) for at in d.relocations)

    # What the low-detail set costs the file, so prep's memory budget has no geometry to hold for it: one copied PObj
    # (0x18 bytes and its next pointer's relocation entry) and two pointers to PObjs; the same costume with the
    # low-detail places left blank is no smaller by more.
    assert record['low_detail']['bytes_added'] == 0x18 + 4 + 2 * 4
    monkeypatch.setattr(melee_dobjs, 'choose_low_places', lambda vanilla, tables, build: ({}, [0, 1, 2]))
    export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'blank.dat')
    grown = (tmp_path / 'out.dat').stat().st_size - (tmp_path / 'blank.dat').stat().st_size
    assert grown <= record['low_detail']['bytes_added']


def test_dobjs_look_alike_only_with_the_same_wraps_and_pixel_engine(tmp_path):
    symbol = 'PlyTest_Share_joint'
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True, 1), ((0,), True, 1)])], {})
    d = melee_dat.Dat.read(b.save(tmp_path / 'imported.dat', [(symbol, root)]))
    first, second = melee_dat.joint_dobjs(d, melee_dat.joints(d, symbol)[0])
    assert export_write._look(d, first) == export_write._look(d, second)
    tobj = d.ptr(d.ptr(second + 0x08) + 0x08)
    struct.pack_into('>I', d.data, tobj + 0x34, 1)
    assert export_write._look(d, first) != export_write._look(d, second)
    struct.pack_into('>I', d.data, tobj + 0x34, 0)
    d.point(d.ptr(second + 0x08) + 0x14, d.add(bytes(0x0C)))
    assert export_write._look(d, first) != export_write._look(d, second)


def test_a_form_draws_the_object_it_is_a_form_of_and_a_form_of_nothing_stays_blank(tmp_path):
    """Kirby's jump: the game hides option 0 of model 0 and shows option 1. Place 0 is a form of 2, which draws the
    build: it draws the same MObj and PObjs, so no geometry is added. Place 1 is a form of 3, which draws nothing here.
    The check holds the costume to it: a form left empty is the fighter vanishing while the game shows it."""
    symbol = 'PlyTest_Share_joint'
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), False), ((0,), False), ((0,), True, 1), ((0,), False)])], {})
    out = b.save(tmp_path / 'out.dat', [(symbol, root)])
    tables = melee_dobjs.FighterTables(options={(0, 0, 0): (2, 3), (0, 0, 1): (0, 1), (1, 0, 0): (2,), (1, 0, 1): (0,)})
    assert melee_checks.forms_check(out, symbol, tables) == {'options': 4, 'empty': [(0, 0, 1), (1, 0, 1)],
                                                             'pass': False}
    assert export_write.fill_forms(out, tables, symbol) == {0: 2}
    d = melee_dat.Dat.read(out)
    dobjs = melee_dat.joint_dobjs(d, melee_dat.joints(d, symbol)[0])
    assert d.ptr(dobjs[0] + 0x08) == d.ptr(dobjs[2] + 0x08) and d.ptr(dobjs[0] + 0x0C) == d.ptr(dobjs[2] + 0x0C)
    assert d.ptr(dobjs[1] + 0x0C) is None
    assert melee_checks.forms_check(out, symbol, tables) == {'options': 4, 'empty': [], 'pass': True}


def test_a_dobj_with_a_chain_longer_than_the_game_loads_is_cut_over_places_of_its_own(tmp_path, monkeypatch):
    symbol = 'PlyTest_Share_joint'
    monkeypatch.setattr(melee_dobjs, 'MAX_POBJS', 3)
    # vanilla: eight DObjs on the root, set 0 draws 0-4, set 1 (low detail) 5-7
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True)] * 8)], {})
    vanilla = b.save(tmp_path / 'PlTsNr.dat', [(symbol, root)])
    fighter = _fighter_file(tmp_path / 'PlTs.dat', [], {0: [0, 1, 2, 3, 4], 1: [5, 6, 7]})
    # the import: one DObj with a chain of seven PObjs
    b = _DatBuilder()
    root = b.add(bytes(0x40))
    dobj, mobj = b.add(bytes(0x10)), b.add(bytes(0x18))
    b.point(root + 0x10, dobj)
    b.point(dobj + 0x08, mobj)
    link = dobj + 0x0C
    for _ in range(7):
        pobj = b.add(bytes(0x18))
        b.point(link, pobj)
        link = pobj + 0x04
    imported = b.save(tmp_path / 'imported.dat', [(symbol, root)])

    record = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat',)
    assert record['chains_cut'] == {0: {'pobjs': [3, 2, 2], 'places': [0, 1, 2]}} and record['places'] == [0]
    assert record['longest_chain'] == 3
    d = melee_dat.Dat.read(tmp_path / 'out.dat')
    dobjs = d.chain(d.ptr(melee_dat.joints(d, symbol)[0] + 0x10), 0x04)
    chains = [d.chain(d.ptr(x + 0x0C), 0x04) for x in dobjs]
    assert [len(c) for c in chains[:3]] == [3, 2, 2] and len({p for c in chains[:3] for p in c}) == 7
    assert {d.ptr(x + 0x08) for x in dobjs[:3]} == {mobj}                  # one material, three places
    # the low-detail set draws all seven again, three at most to a place
    assert record['low_detail']['places'] == {5: [0], 6: [0], 7: [0]} and record['low_detail']['left_out'] == []
    assert [len(c) for c in chains[5:]] == [3, 2, 2]


def test_a_corner_normal_with_no_direction_is_given_one_before_the_import():
    """A zero (or NaN) normal stops HSDRawViewer's import. The .dae's other normals stay as Blender wrote them."""
    import xml.etree.ElementTree as ET
    from rigswap.plugins.melee.dae import NS
    from rigswap.plugins.melee.export.write import repair_normals
    xml = (f'<COLLADA xmlns="{NS[1:-1]}"><library_geometries><geometry id="g"><mesh>'
           '<source id="g-normals"><float_array id="g-n" count="9">0 0 1 0 0 0 0 nan 0</float_array></source>'
           '<source id="g-positions"><float_array id="g-p" count="3">0 0 0</float_array></source>'
           '<triangles count="1"><input semantic="NORMAL" source="#g-normals" offset="1"/></triangles>'
           '</mesh></geometry></library_geometries></COLLADA>')
    root = ET.fromstring(xml)
    assert repair_normals(root) == {'g-normals': 2}
    values = [float(v) for v in root.find(f'.//{NS}source[@id="g-normals"]/{NS}float_array').text.split()]
    assert values == [0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert repair_normals(root) == {}


def test_a_mesh_with_no_texture_map_is_given_one_at_the_images_origin():
    """Kanade on Peach: one eight-point lock of hair with no map among 66 with one. Blender's glTF export wrote its
    material a second time under the same name, Blender 4 read that as mtl_chr_hair_00.002, and the export failed:
    "DAE material 'mtl_chr_hair_00.002' has no GLB source". A mesh with faces and no map is given one, every corner
    at the origin, where Blender draws a face that has none; any other mesh is left as it is."""
    from types import SimpleNamespace as NS
    from rigswap.plugins.melee.export.write import mapped

    class Layers(list):
        def new(self, name, do_init):
            layer = NS(name=name, set=None)
            layer.uv = NS(foreach_set=lambda key, values: setattr(layer, 'set', (key, list(values))))
            self.append(layer)
            return layer

    def mesh(name, faces, maps, corners=6):
        return NS(type='MESH', name=name, data=NS(polygons=[0] * faces, loops=[0] * corners, uv_layers=Layers(maps)))
    bare, has, empty = mesh('lock', 2, []), mesh('hair', 2, ['UVMap']), mesh('none', 0, [])
    assert mapped([bare, has, empty, NS(type='ARMATURE', name='rig')]) == ['lock']
    assert bare.data.uv_layers[0].set == ('vector', [0.0] * 12)
    assert has.data.uv_layers == ['UVMap'] and empty.data.uv_layers == []
