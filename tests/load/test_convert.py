import numpy as np
import pytest

from rigswap.load import convert
from rigswap.load import open as load


def test_dae_conversion_keeps_material_and_normal_records_before_the_intermediate_glb(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    glb = tmp_path / 'converted.glb'
    normals = {'frame': 'mesh', 'meshes': {}}
    colours = [{'material': 'surface', 'colour': [.2, .4, .6, 1.]}]

    def run(args, **kwargs):
        assert args[-1] == str(glb)
        glb.write_bytes(b'converted')
        return SimpleNamespace(stdout=convert.RECORD + json.dumps(normals) + '\n' +
                               convert.MATERIAL_RECORD + json.dumps(colours) + '\n', stderr='')

    monkeypatch.setattr(convert, 'blender4', lambda: 'blender')
    monkeypatch.setattr(convert.subprocess, 'run', run)
    (tmp_path / 'source.dae').write_text('<COLLADA/>')
    facts = convert.dae_to_glb(tmp_path / 'source.dae', glb)
    assert facts == {'normals': normals, 'material_colours': colours}
    monkeypatch.setattr(convert, 'dae_to_glb', lambda *args: facts)
    imported = []
    monkeypatch.setattr(load, '_import_gltf', imported.append)
    assert load._import_dae(tmp_path / 'source.dae') == facts
    assert imported == [tmp_path / 'source.converted.glb']


def test_a_texture_without_its_texcoord_draws_with_the_file_s_one_set_of_them(tmp_path):
    # Deltarune's Pink: Blender 4 refuses the whole document over the attribute missing
    effect = '<diffuse><texture texture="pallete-sampler"/></diffuse>{}'
    one = ('<input semantic="TEXCOORD" source="#a-map" offset="2" set="0"/>'
           '<input semantic="TEXCOORD" source="#b-map" offset="2" set="0"/>')
    dae = tmp_path / 'pink.dae'
    dae.write_text(effect.format(one))
    fixed = convert.repair(dae)
    assert fixed.parent == dae.parent and fixed != dae                 # beside it: its images where they were
    assert fixed.read_text() == effect.replace('<texture ', '<texture texcoord="UVSET0" ').format(one)
    dae.write_text(effect.format(one.replace('"#b-map" offset="2" set="0"', '"#b-map" offset="2" set="1"')))
    assert convert.repair(dae) == dae                                   # which of two is meant is not in the file
    dae.write_text(effect.replace('<texture ', '<texture texcoord="UVMap" ').format(one))
    assert convert.repair(dae) == dae


# A box on one joint whose bind frame is turned half round about x. Its eight corners' smooth normals point out from the
# centre; the file stores them either in the box's own space (Collada's way) or in the joint's frame (a rip's way).
CORNERS = np.array([(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float)
TRIANGLES = [0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5]
HALF_TURN_X = np.diag([1., -1., -1., 1.])        # the joint's bind; it is its own inverse bind matrix too


def _numbers(a):
    return ' '.join(f'{v:g}' for v in np.ravel(a))


def _box_dae(frame):
    normals = CORNERS / np.sqrt(3)
    if frame == 'joints':
        normals = normals @ HALF_TURN_X[:3, :3].T
    accessor = ('<technique_common><accessor source="#{0}-array" count="8" stride="3"><param name="X" type="float"/>'
                '<param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common>')
    return f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
 <library_geometries><geometry id="box-mesh" name="box"><mesh>
  <source id="box-pos"><float_array id="box-pos-array" count="24">{_numbers(CORNERS)}</float_array>{accessor.format('box-pos')}</source>
  <source id="box-nrm"><float_array id="box-nrm-array" count="24">{_numbers(normals)}</float_array>{accessor.format('box-nrm')}</source>
  <vertices id="box-vtx"><input semantic="POSITION" source="#box-pos"/><input semantic="NORMAL" source="#box-nrm"/></vertices>
  <triangles count="12"><input semantic="VERTEX" source="#box-vtx" offset="0"/><p>{_numbers(TRIANGLES)}</p></triangles>
 </mesh></geometry></library_geometries>
 <library_controllers><controller id="box-skin"><skin source="#box-mesh">
  <bind_shape_matrix>{_numbers(np.eye(4))}</bind_shape_matrix>
  <source id="box-joints"><Name_array id="box-joints-array" count="1">root</Name_array><technique_common>
   <accessor source="#box-joints-array" count="1" stride="1"><param name="JOINT" type="name"/></accessor></technique_common></source>
  <source id="box-ibm"><float_array id="box-ibm-array" count="16">{_numbers(HALF_TURN_X)}</float_array><technique_common>
   <accessor source="#box-ibm-array" count="1" stride="16"><param name="TRANSFORM" type="float4x4"/></accessor></technique_common></source>
  <source id="box-weights"><float_array id="box-weights-array" count="1">1</float_array><technique_common>
   <accessor source="#box-weights-array" count="1" stride="1"><param name="WEIGHT" type="float"/></accessor></technique_common></source>
  <joints><input semantic="JOINT" source="#box-joints"/><input semantic="INV_BIND_MATRIX" source="#box-ibm"/></joints>
  <vertex_weights count="8"><input semantic="JOINT" source="#box-joints" offset="0"/>
   <input semantic="WEIGHT" source="#box-weights" offset="1"/><vcount>{_numbers([1] * 8)}</vcount><v>{_numbers([0] * 16)}</v>
  </vertex_weights>
 </skin></controller></library_controllers>
 <library_visual_scenes><visual_scene id="Scene" name="Scene">
  <node id="root" name="root" sid="root" type="JOINT"><matrix>{_numbers(HALF_TURN_X)}</matrix></node>
  <node id="box" name="box" type="NODE"><instance_controller url="#box-skin"><skeleton>#root</skeleton></instance_controller></node>
 </visual_scene></library_visual_scenes>
 <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
'''


@pytest.mark.fast
def test_a_joint_frame_normal_is_turned_into_the_mesh_space_by_the_joints_bind(tmp_path):
    dae = tmp_path / 'box.dae'
    dae.write_text(_box_dae('joints'), encoding='utf-8')
    rotations = convert.bind_rotations(dae)
    assert list(rotations) == ['box'] and list(rotations['box']) == ['root']
    stored = CORNERS[7] / np.sqrt(3) @ HALF_TURN_X[:3, :3].T
    assert np.allclose(rotations['box']['root'] @ stored, CORNERS[7] / np.sqrt(3))


@pytest.mark.fast
def test_the_frame_is_the_one_whose_normals_lie_nearer_their_faces():
    faces = np.tile([0., 0., 1.], (4, 1))
    near = np.tile([0., 0.6, 0.8], (4, 1))
    across = np.tile([0., 1., 0.], (4, 1))
    assert convert.normals_frame(across, near, faces) == 'joints'
    assert convert.normals_frame(near, across, faces) == 'mesh'
    assert convert.normals_frame(near, near, faces) == 'mesh'           # nothing to tell them apart: as stored
    assert convert.normals_frame(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))) == 'mesh'


@pytest.mark.blender
@pytest.mark.parametrize('frame', ['mesh', 'joints'])
def test_a_dae_s_normals_come_in_as_the_model_has_them_whichever_frame_the_file_wrote(tmp_path, frame):
    import bpy
    try:
        convert.blender4()
    except RuntimeError:
        pytest.skip('no Blender 4.x for .dae conversion')
    dae = tmp_path / 'box.dae'
    dae.write_text(_box_dae(frame), encoding='utf-8')
    listing = load.run(dae, tmp_path / 'load')
    assert listing[0]['normals']['frame'] == frame
    assert listing[0]['normals']['meshes']['box'] == {'stored': 54.7 if frame == 'mesh' else 125.3,
                                                      'turned': 125.3 if frame == 'mesh' else 54.7}
    box = next(o for o in bpy.data.objects if o.type == 'MESH')
    normals = np.array([n.vector[:] for n in box.data.corner_normals])
    faces = np.array([box.data.polygons[i].normal[:] for p in box.data.polygons for i in [p.index] * p.loop_total])
    assert np.allclose(np.degrees(np.arccos(np.clip((normals * faces).sum(1), -1, 1))), 54.74, atol=0.1)


def test_a_dae_s_joints_hang_as_its_nodes_do_and_are_bound_where_its_skins_say(tmp_path):
    shift = np.eye(4)
    shift[:3, 3] = [0, 2, 0]                     # the arm joint bound two units up: its inverse bind is the way back
    ibm = np.linalg.inv(shift)
    dae = tmp_path / 'joints.dae'
    dae.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <library_controllers><controller id="c"><skin source="#g">
  <source id="j"><Name_array id="j-a" count="2">j0 j1</Name_array></source>
  <source id="m"><float_array id="m-a" count="32">{_numbers(np.eye(4))} {_numbers(ibm)}</float_array></source>
  <joints><input semantic="JOINT" source="#j"/><input semantic="INV_BIND_MATRIX" source="#m"/></joints>
 </skin></controller></library_controllers>
 <library_visual_scenes><visual_scene id="s">
  <node id="n0" name="root" sid="j0" type="JOINT"><node id="grp" type="NODE">
   <node id="n1" name="arm_jnt" sid="j1" type="JOINT"/></node></node>
 </visual_scene></library_visual_scenes>
</COLLADA>
''', encoding='utf-8')
    parents, binds = convert.joint_binds(dae)
    assert parents == {'root': None, 'arm_jnt': 'root'}
    assert np.allclose(binds['arm_jnt'][:3, 3], [0, 2, 0]) and np.allclose(binds['root'], np.eye(4))
