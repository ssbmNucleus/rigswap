"""dae.py: Collada read the same way whoever wrote it; and the builders the Melee tests write .dae files and make
Models with."""
import json
import math
from pathlib import Path

import numpy as np

from rigswap.plugins.melee import dae as melee_dae
from rigswap.plugins.melee.export import checks as melee_checks


def _matrix(x=0., y=0., z=0., turn_z=0.):
    c, s = math.cos(turn_z), math.sin(turn_z)
    return f'{c} {-s} 0 {x} {s} {c} 0 {y} 0 0 1 {z} 0 0 0 1'


def _joint(name, matrix, *children):
    return (f'<node id="{name}" name="{name}" sid="{name}" type="JOINT"><matrix sid="transform">{matrix}</matrix>'
            + ''.join(children) + '</node>')


def _dae(path, scene):
    path.write_text('<?xml version="1.0" encoding="utf-8"?>'
                    '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
                    '<asset><up_axis>Z_UP</up_axis></asset><library_visual_scenes><visual_scene id="Scene">'
                    + scene + '</visual_scene></library_visual_scenes></COLLADA>', encoding='utf-8')
    return path


def _arm(tmp_path, name, hand_x=1.0, extra=''):
    # a root, an elbow turned a quarter about z, a hand 1 along the elbow's x: the hand sits 1 up from the elbow
    return _dae(tmp_path / f'{name}.dae', _joint('JOBJ_0', _matrix(),
                                                 _joint('JOBJ_1', _matrix(y=2, turn_z=math.pi / 2),
                                                        _joint('JOBJ_2', _matrix(x=hand_x)), extra)))


def _dae_with_materials(path, textured):
    """A .dae with the materials skin and eye, as Blender writes one; those in ``textured`` sample an image file
    beside it."""
    def effect(m):
        if m not in textured:
            return f'<effect id="{m}-effect"><profile_COMMON><technique sid="common"><lambert/></technique>'                    '</profile_COMMON></effect>'
        return (f'<effect id="{m}-effect"><profile_COMMON><newparam sid="{m}-surface"><surface type="2D">'
                f'<init_from>{m}-png</init_from></surface></newparam><newparam sid="{m}-sampler"><sampler2D>'
                f'<source>{m}-surface</source></sampler2D></newparam><technique sid="common"><lambert><diffuse>'
                f'<texture texture="{m}-sampler" texcoord=""/></diffuse></lambert></technique></profile_COMMON>'
                '</effect>')
    images = ''.join(f'<image id="{m}-png" name="{m}"><init_from>{m}.png</init_from></image>' for m in textured)
    path.write_text('<?xml version="1.0" encoding="utf-8"?>'
                    '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
                    f'<library_images>{images}</library_images>'
                    f'<library_effects>{effect("skin")}{effect("eye")}</library_effects><library_materials>'
                    '<material id="skin-material" name="skin"><instance_effect url="#skin-effect"/></material>'
                    '<material id="eye-material" name="eye"><instance_effect url="#eye-effect"/></material>'
                    '</library_materials></COLLADA>', encoding='utf-8')
    for m in textured:
        (path.parent / f'{m}.png').write_bytes(b'png')
    return path


def _hsdraw_read_back(path, sampled, named):
    """A .dae as HSDRawViewer's exporter writes one: a triangle for each material, named Material_<n>; those in
    ``sampled`` sample their image in the .dae, and <dae>.textures.json beside it names ``named`` ({material: image
    file}), every image written."""
    effects, images, materials, meshes, nodes = [], [], [], [], []
    for k, m in enumerate(named):
        if m in sampled:
            image = Path(named[m]).stem
            images.append(f'<image id="{image}-image" name="{image}"><init_from>{named[m]}</init_from></image>')
            effects.append(f'<effect id="{m}-effect" name="{m}"><profile_COMMON><newparam sid="{image}-surface">'
                           f'<surface type="2D"><init_from>{image}-image</init_from></surface></newparam>'
                           f'<newparam sid="{image}-sampler"><sampler2D><source>{image}-surface</source></sampler2D>'
                           f'</newparam><technique sid="standard"><phong><diffuse><texture texture="{image}-sampler" '
                           'texcoord="CHANNEL0"/></diffuse></phong></technique></profile_COMMON></effect>')
        else:
            effects.append(f'<effect id="{m}-effect" name="{m}"><profile_COMMON><technique sid="standard"><phong>'
                           '<diffuse><color sid="diffuse">1 1 1 1</color></diffuse></phong></technique>'
                           '</profile_COMMON></effect>')
        materials.append(f'<material id="{m}" name="{m}"><instance_effect url="#{m}-effect"/></material>')
        meshes.append(f'<geometry id="mesh{k}"><mesh><source id="mesh{k}-p"><float_array id="mesh{k}-a" count="9">'
                      f'{10 * k} 0 0 {10 * k + 1} 0 0 {10 * k} 1 0</float_array></source><vertices id="mesh{k}-v">'
                      f'<input semantic="POSITION" source="#mesh{k}-p"/></vertices><triangles count="1" material="{m}">'
                      f'<input semantic="VERTEX" source="#mesh{k}-v" offset="0"/><p>0 1 2</p></triangles></mesh>'
                      '</geometry>')
        nodes.append(f'<node id="Joint_0_Object_{k}" name="Joint_0_Object_{k}" type="NODE"><instance_geometry '
                     f'url="#mesh{k}"><bind_material><technique_common><instance_material target="#{m}" symbol="{m}"/>'
                     '</technique_common></bind_material></instance_geometry></node>')
    _dae(path, ''.join(nodes))
    text = path.read_text(encoding='utf-8').replace(
        '<library_visual_scenes>', f'<library_images>{"".join(images)}</library_images><library_effects>'
        f'{"".join(effects)}</library_effects><library_materials>{"".join(materials)}</library_materials>'
        f'<library_geometries>{"".join(meshes)}</library_geometries><library_visual_scenes>')
    path.write_text(text, encoding='utf-8')
    for image in named.values():
        (path.parent / image).write_bytes(image.encode())
    path.with_name(path.name + '.textures.json').write_text(json.dumps(named), encoding='utf-8')
    return path


def _triangle_model():
    return melee_dae.Model(points=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                       weights=[{'JOBJ_0': 1.} for _ in range(3)], faces=[[0, 1, 2]], triangles=1, materials=1)


def _objects_model(objects):
    """A Model of one triangle for each (mesh name, material), apart from each other, as read_dae gives them."""
    points = np.array([[10. * k + x, y, 0.] for k in range(len(objects)) for x, y in ((0, 0), (1, 0), (0, 1))])
    return melee_dae.Model(points=points, weights=[{'JOBJ_0': 1.} for _ in points],
                       faces=[[3 * k, 3 * k + 1, 3 * k + 2] for k in range(len(objects))], triangles=len(objects),
                       materials=len({m for _, m in objects}), face_objects=[o for o, _ in objects],
                       face_materials=[m for _, m in objects])


def test_a_texture_drawn_after_lighting_reads_back_with_its_format(tmp_path):
    """runs/pilot3/golduck-zip-Mario re-exported: the import's alpha Test moved Mesh_0043's cutout CMPR texture to the
    stage after lighting (TObj flags 0x340010 -> 0x440080, LIGHTMAP_EXT). HSDRawViewer's exporter wrote it as
    Texture_7_CMP.png and named it for Material_952538CE in readback.dae.textures.json, but sampled no image in that
    material's effect, and the format check read the texture as missing."""
    back = melee_dae.read_dae(_hsdraw_read_back(tmp_path / 'readback.dae', {'Material_22BB968B'},
                                            {'Material_22BB968B': 'Texture_2_CMP.png',
                                             'Material_952538CE': 'Texture_7_CMP.png'}))
    assert back.material_images == {'Material_22BB968B': [str(tmp_path / 'Texture_2_CMP.png')],
                                    'Material_952538CE': [str(tmp_path / 'Texture_7_CMP.png')]}
    assert len(set(back.textures.values()) - {None}) == 2
    source = _objects_model([('body', 'body-material'), ('fin', 'fin-material')])
    source.material_images = {'body-material': ['body.png'], 'fin-material': ['fin.png']}
    asked = tmp_path / 'formats.json'
    asked.write_text(json.dumps({'body-material': 'CMPR', 'fin-material': 'CMPR'}), encoding='utf-8')
    formats = melee_checks.texture_format_check(asked, source, back)
    assert formats['pass'] and formats['materials']['fin-material']['read_back'] == ['CMPR']
    # A material with no texture at all still reads back as none.
    (tmp_path / 'readback.dae.textures.json').write_text(json.dumps({'Material_22BB968B': 'Texture_2_CMP.png'}),
                                                       encoding='utf-8')
    back = melee_dae.read_dae(tmp_path / 'readback.dae')
    assert back.material_images['Material_952538CE'] == []
    assert melee_checks.texture_format_check(asked, source, back)['materials']['fin-material']['read_back'] == [None]
