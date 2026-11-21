"""targets/prepare.py: a fighter's vanilla costume made a target .glb."""
import json
from pathlib import Path

import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee import dae as melee_dae
from rigswap.plugins.melee.export import write as export_write
from rigswap.plugins.melee.targets import prepare as targets_prepare
from tests.plugins.melee.export.test_write import _blender4_or_none
from tests.plugins.melee.prep.test_prep import _flat, _grid, _material


def test_prepare_adds_missing_memory_to_existing_targets_and_can_refresh(tmp_path, monkeypatch):
    from rigswap.plugins import melee
    from tests.plugins.melee.test_melee import _memory, _prepared
    files, out = tmp_path / 'files', tmp_path / 'targets'
    files.mkdir()
    out.mkdir()
    for name in ('PlMrNr.dat', 'PlMr.dat'):
        (files / name).touch()
    from rigswap.plugins.melee.targets import physics
    monkeypatch.setattr(physics, 'record', lambda *a: {'existing_chains': 0})
    framed = []
    monkeypatch.setattr(targets_prepare, 'write_frames', lambda f, files_dir, targets: framed.append(f.name))
    _prepared(out)
    (out / 'Mario.memory.json').unlink()
    calls = []

    def write(f, files_dir, targets):
        calls.append(f.name)
        (targets / f'{f.name}.memory.json').write_text(json.dumps(_memory(f.name)), encoding='utf-8')

    monkeypatch.setattr(targets_prepare, 'write_record', write)
    assert melee.prepare(files, out, ['Mario']) == {'Mario': 'made'}
    assert calls == ['Mario'] and (out / 'Mario.glb').read_bytes() == b'glTF'
    assert json.loads((out / 'Mario.physics.json').read_text()) == {'existing_chains': 0}
    assert framed == ['Mario']      # the game frames are target data too, written beside the rest
    assert melee.prepare(files, out, ['Mario']) == {'Mario': 'already there'}
    assert calls == ['Mario']
    assert melee.prepare(files, out, ['Mario'], refresh_memory=True) == {'Mario': 'made'}
    assert calls == ['Mario', 'Mario']
    (out / 'Mario.memory.json').write_text('{}')
    assert melee.prepare(files, out, ['Mario']) == {'Mario': 'made'}
    assert calls == ['Mario', 'Mario', 'Mario']


def test_prepare_records_importer_failure(tmp_path, monkeypatch):
    from tests.plugins.melee.test_melee import _prepared
    _prepared(tmp_path)
    for name in ('PlMrNr.dat', 'PlMr.dat'):
        (tmp_path / name).touch()
    from rigswap.plugins.melee.targets import physics
    monkeypatch.setattr(physics, 'record', lambda *a: {'existing_chains': 0})
    monkeypatch.setattr(targets_prepare, 'write_frames', lambda *a: None)
    (tmp_path / 'Mario.memory.json').unlink()

    def fail(*args):
        raise Failed('importer unavailable')

    monkeypatch.setattr(targets_prepare, 'write_record', fail)
    report = targets_prepare.prepare(tmp_path, tmp_path, ['Mario'])
    assert report == {'Mario': 'failed: importer unavailable'}
    assert json.loads((tmp_path / 'prepare.json').read_text()) == report
    assert not (tmp_path / 'Mario.memory.json').exists()


@pytest.mark.blender
@pytest.mark.skipif(_blender4_or_none() is None, reason='needs a Blender 4.x (rigswap.load.convert)')
def test_a_target_glb_has_the_texture_hsdrawviewer_left_out_of_the_dae(tmp_path, monkeypatch):
    """The prepared Mario.glb drew Material_94F89F52 untextured (Link.glb Material_DD6AD37D, Bowser.glb
    Material_C63B3773): HSDRawViewer's exporter samples a texture applied after lighting nowhere in the .dae, only
    naming it in <dae>.textures.json, and Blender's Collada importer then makes the material with no image. The .glb
    takes it from that file, as read_dae does; a .glb that still lacks one fails."""
    import bpy
    import re as regex
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
    next(n for n in ob.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE').image.pack()
    glb = tmp_path / 'model.glb'
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', export_animations=False)
    bpy.ops.wm.read_homefile(use_empty=True)
    dae = export_write.glb_to_dae(glb, tmp_path / 'vanilla' / 'model.dae')
    image = Path(next(iter(melee_dae.read_dae(dae).textures)))
    # as HSDRawViewer writes a texture applied after lighting: the effect samples nothing, the texture map names it
    text = regex.sub(r'<diffuse>\s*<texture[^>]*/>\s*</diffuse>', '<diffuse><color sid="diffuse">1 1 1 1</color>'
                     '</diffuse>', dae.read_text(encoding='utf-8'))
    dae.write_text(regex.sub(r'<newparam.*?</newparam>', '', text, flags=regex.S), encoding='utf-8')
    assert melee_dae.read_dae(dae).textures == {}
    dae.with_name(dae.name + '.textures.json').write_text(json.dumps({'skin': image.name}), encoding='utf-8')
    assert targets_prepare.drawn_after_lighting(dae) == {'skin': str(image)}
    target = targets_prepare.dae_to_glb(dae, tmp_path / 'target.glb')
    assert export_write.glb_textured(target) == {'skin'}
    monkeypatch.setattr(targets_prepare, 'drawn_after_lighting', lambda dae: {})
    with pytest.raises(Failed, match=r"no base colour texture on the materials \['skin'\]"):
        targets_prepare.dae_to_glb(dae, tmp_path / 'lost.glb')


def test_a_texture_sampled_only_for_its_highlight_is_not_wanted_on_base_colour(tmp_path):
    """Preparing Ganondorf failed on Material_DCDB4560: HSDRawViewer writes its specular stage's Texture_20 to
    <specular> over a plain diffuse, Blender keeps that off base colour, and the check wanted it there."""
    dae = tmp_path / 'model.dae'
    dae.write_text(
        '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_effects>'
        '<effect id="shine-fx"><profile_COMMON><technique sid="common"><phong>'
        '<diffuse><color>1 1 1 1</color></diffuse><specular><texture texture="a" texcoord="UV0"/></specular>'
        '</phong></technique></profile_COMMON></effect>'
        '<effect id="skin-fx"><profile_COMMON><technique sid="common"><phong>'
        '<diffuse><texture texture="b" texcoord="UV0"/></diffuse><specular><texture texture="a" texcoord="UV0"/>'
        '</specular></phong></technique></profile_COMMON></effect>'
        '<effect id="plain-fx"><profile_COMMON><technique sid="common"><phong>'
        '<diffuse><color>1 1 1 1</color></diffuse></phong></technique></profile_COMMON></effect>'
        '</library_effects><library_materials>'
        '<material id="m1" name="shine"><instance_effect url="#shine-fx"/></material>'
        '<material id="m2" name="skin"><instance_effect url="#skin-fx"/></material>'
        '<material id="m3" name="plain"><instance_effect url="#plain-fx"/></material>'
        '</library_materials></COLLADA>', encoding='utf-8')
    assert targets_prepare.uncoloured(dae) == {'shine'}
