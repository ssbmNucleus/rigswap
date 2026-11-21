import struct

import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee.dat import Dat, joints, joint_dobjs
from rigswap.plugins.melee.export import write
from tests.plugins.melee.test_dat import _DatBuilder, _costume


def test_material_modes_reach_every_pobj_without_changing_other_bytes(tmp_path, monkeypatch):
    b = _DatBuilder()
    root, _ = _costume(b, [(None, [([], True), ([], True)])], {})
    path = b.save(tmp_path / 'model.dat', [('body', root)])
    d = Dat.read(path)
    objects = [o for j in joints(d, 'body') for o in joint_dobjs(d, j)]
    first = d.ptr(objects[0] + 12)
    extra = d.add(bytes(24))
    d.point(first + 4, extra)
    polys = [first, extra, d.ptr(objects[1] + 12)]
    for p in polys:
        struct.pack_into('>H', d.data, p + 12, 0xa004)
    d.save(path)
    before = path.read_bytes()
    monkeypatch.setattr(write, '_gltf', lambda _: {'materials': [
        {'name': 'open', 'doubleSided': True}, {'name': 'closed'}]})
    from rigswap.plugins.melee.export import eyes
    monkeypatch.setattr(eyes, 'dae_dobjs', lambda _: [('mesh', 'a'), ('mesh', 'b')])
    dae = tmp_path / 'model.dae'
    dae.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">'
                   '<material id="a" name="open"/><material id="b" name="closed"/></COLLADA>')
    assert write.preserve_sidedness(None, dae, path, 'body') == [
        {'two_sided': True, 'pobjs': 2}, {'two_sided': False, 'pobjs': 1}]
    after = path.read_bytes()
    assert len(after) == len(before)
    assert {i for i, (a, b) in enumerate(zip(before, after)) if a != b} == {32 + p + 12 for p in polys[:2]}
    d = Dat.read(path)
    assert [struct.unpack_from('>H', d.data, p + 12)[0] for p in polys] == [0x2004, 0x2004, 0xa004]
    with pytest.raises(Failed, match='material groups'):
        write.write_culling(d, objects, [True])
    monkeypatch.setattr(eyes, 'dae_dobjs', lambda _: [('mesh', 'missing')])
    with pytest.raises(Failed, match='absent from GLB'):
        write.preserve_sidedness(None, dae, path, 'body')
