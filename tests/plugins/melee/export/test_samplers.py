"""Source wrap modes survive Collada without changing UVs or encoded images."""
import json
import struct

import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee.dat import Dat, joint_dobjs
from rigswap.plugins.melee.export.write import preserve_samplers
from tests.plugins.melee.test_dat import _DatBuilder, _costume


def _files(tmp_path, samplers, texture_samplers):
    materials = [{'name': f'paint-{i}', 'pbrMetallicRoughness': {'baseColorTexture': {'index': i}}}
                 for i in range(len(texture_samplers))]
    materials.append({'name': 'plain'})
    doc = json.dumps({'materials': materials, 'samplers': samplers,
                      'textures': [dict(source=0, **({'sampler': s} if s is not None else {}))
                                   for s in texture_samplers]}).encode()
    glb = tmp_path / 'source.glb'
    glb.write_bytes(struct.pack('<5I', 0x46546C67, 2, 20 + len(doc), len(doc), 0x4E4F534A) + doc)
    dae = tmp_path / 'model.dae'
    # Deliberately different material IDs and names. All texture materials share one source image.
    ids = [f'm-{i}' for i in range(len(materials))]
    dae.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
                   '<library_materials>' + ''.join(f'<material id="{key}" name="{m["name"]}"/>'
                                                 for key, m in zip(ids, materials)) + '</library_materials>'
                   '<library_geometries><geometry id="geometry"><mesh>'
                   + ''.join(f'<triangles count="1" material="{key}"/>' for key in ids)
                   + '</mesh></geometry></library_geometries><library_visual_scenes><visual_scene>'
                   '<node name="surface"><instance_geometry url="#geometry"/></node>'
                   '</visual_scene></library_visual_scenes></COLLADA>', encoding='utf-8')
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [((0,), True) for _ in texture_samplers] + [((), True)])], {})
    dat = b.save(tmp_path / 'imported.dat', [('root', root)])
    d = Dat.read(dat)
    for obj in joint_dobjs(d, root)[:-1]:
        t = d.ptr(d.ptr(obj + 8) + 8)
        struct.pack_into('>2I', d.data, t + 0x34, 1, 1)
    d.save(dat)
    return glb, dae, dat


def test_wrap_modes_are_per_material_and_axis_even_with_one_shared_image(tmp_path):
    glb, dae, dat = _files(tmp_path, [{'wrapS': 33071, 'wrapT': 33071},
                                     {'wrapS': 33648}, {'wrapT': 33071}], [0, 1, 2, None])
    before = Dat.read(dat)
    result = preserve_samplers(glb, dae, dat, 'root')
    after = Dat.read(dat)
    expected = [(0, 0), (2, 1), (1, 0), (1, 1)]
    fields = set()
    for obj, wrap in zip(joint_dobjs(after, 0), expected):
        t = after.ptr(after.ptr(obj + 8) + 8)
        assert (after.u32(t + 0x34), after.u32(t + 0x38)) == wrap
        fields.update(range(t + 0x34, t + 0x3C))
    assert all(a == b for i, (a, b) in enumerate(zip(before.data, after.data)) if i not in fields)
    assert len(before.data) == len(after.data)
    assert before.relocations == after.relocations and before.tail == after.tail
    assert [r['written'] for r in result['materials']] == [list(v) for v in expected]
    assert all(r['before'] == [1, 1] for r in result['materials'])
    from rigswap.plugins.melee.export.write import wraps_written      # what written_alike counts TObjs by
    assert list(wraps_written(result).values()) == expected[:len(wraps_written(result))]


def test_sampler_transfer_refuses_an_unmatched_import_without_writing(tmp_path):
    glb, dae, dat = _files(tmp_path, [{}], [0])
    dae.write_text(dae.read_text().replace('count="1"', 'count="0"'), encoding='utf-8')
    before = dat.read_bytes()
    with pytest.raises(Failed, match='DAE groups but 2 imported DObjs'):
        preserve_samplers(glb, dae, dat, 'root')
    assert dat.read_bytes() == before


def test_sampler_transfer_refuses_a_missing_texture_without_writing(tmp_path):
    glb, dae, dat = _files(tmp_path, [{}], [0])
    d = Dat.read(dat)
    d.point(d.ptr(joint_dobjs(d, 0)[0] + 8) + 8, None)
    d.save(dat)
    before = dat.read_bytes()
    with pytest.raises(Failed, match='0 imported textures, expected one'):
        preserve_samplers(glb, dae, dat, 'root')
    assert dat.read_bytes() == before
