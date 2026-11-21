"""dat.py: the .dat file read and written; and the builders the Melee tests make .dat files with (_DatBuilder,
_costume, _fighter_file)."""
import math
import struct

import numpy as np
import pytest

from rigswap.plugins import Failed
from rigswap.plugins.melee import dat as melee_dat


def test_rest_skeleton_read_from_a_dat(tmp_path):
    # Three joints: a root 1 up, turned a quarter about z; a child 1 along its x; the child's sibling 1 along its z.
    def jobj(child=0, next_=0, rz=0., t=(0., 0., 0.)):
        return struct.pack('>5I9f2I', 0, 0, child, next_, 0, 0., 0., rz, 1., 1., 1., *t, 0, 0)
    data = jobj(child=0x40, rz=math.pi / 2, t=(0, 1, 0)) + jobj(next_=0x80, t=(1, 0, 0)) + jobj(t=(0, 0, 1))
    relocations = struct.pack('>2I', 0x08, 0x40 + 0x0C)
    roots = struct.pack('>2I', 0, 0)
    strings = b'PlyTest_Share_joint\0'
    header = struct.pack('>8I', 0x20 + len(data) + len(relocations) + len(roots) + len(strings), len(data), 2, 1, 0,
                         0, 0, 0)
    dat = tmp_path / 'PlTsNr.dat'
    dat.write_bytes(header + data + relocations + roots + strings)

    assert melee_dat.dat_roots(dat.read_bytes()) == [('PlyTest_Share_joint', 0)]
    joints = melee_dat.dat_joints(dat, 'PlyTest_Share_joint')
    assert list(joints) == ['JOBJ_0', 'JOBJ_1', 'JOBJ_2']
    assert np.allclose([joints[n][:3, 3] for n in joints], [[0, 1, 0], [0, 2, 0], [0, 1, 1]], atol=1e-6)
    with pytest.raises(Failed, match='no root'):
        melee_dat.dat_joints(dat, 'PlyOther_Share_joint')


class _DatBuilder:
    """A .dat file put together a structure at a time."""

    def __init__(self):
        self.data, self.relocations = bytearray(), []

    def add(self, payload, align=4):
        self.data.extend(b'\0' * (-len(self.data) % align))
        at = len(self.data)
        self.data.extend(payload)
        return at

    def point(self, at, target):
        struct.pack_into('>I', self.data, at, target)
        self.relocations.append(at)

    def save(self, path, roots):
        names, table = b'', b''
        for name, offset in roots:
            table += struct.pack('>2I', offset, len(names))
            names += name.encode() + b'\0'
        body = bytes(self.data) + struct.pack(f'>{len(self.relocations)}I', *self.relocations) + table + names
        path.write_bytes(struct.pack('>8I', 0x20 + len(body), len(self.data), len(self.relocations), len(roots), 0,
                                     0, 0, 0) + body)
        return path


def _link_tree(b, offsets, parents, child_at, next_at):
    for k, parent in enumerate(parents):
        if parent is not None:
            siblings = [i for i, p in enumerate(parents) if p == parent]
            at = offsets[parent] + child_at if siblings[0] == k else offsets[siblings[siblings.index(k) - 1]] + next_at
            b.point(at, offsets[k])


def _costume(b, joints, matanims):
    """A costume: ``joints`` [(parent index or None, [(TObj texture map ids, has geometry[, render mode]) for each
    DObj])], and its matanim tree ``matanims`` {joint index: [texture map ids animated, for each material animation]}.
    -> (root joint, matanim root)"""
    offsets = [b.add(bytes(0x40)) for _ in joints]
    _link_tree(b, offsets, [p for p, _ in joints], 0x08, 0x0C)
    for k, (_, dobjs) in enumerate(joints):
        link = offsets[k] + 0x10
        for textures, drawn, *mode in dobjs:
            dobj, mobj = b.add(bytes(0x10)), b.add(struct.pack('>2I', 0, mode[0] if mode else 0) + bytes(0x10))
            b.point(link, dobj)
            b.point(dobj + 0x08, mobj)
            if drawn:
                b.point(dobj + 0x0C, b.add(bytes(0x18)))
            tlink = mobj + 0x08
            for texture in textures:
                tobj = b.add(struct.pack('>3I', 0, 0, texture) + bytes(0x5C - 12))
                b.point(tlink, tobj)
                tlink = tobj + 0x04
            link = dobj + 0x04
    mjoints = [b.add(bytes(0x0C)) for _ in joints]
    _link_tree(b, mjoints, [p for p, _ in joints], 0x00, 0x04)
    for k, lists in matanims.items():
        link = mjoints[k] + 0x08
        for ids in lists:
            mat = b.add(bytes(0x10))
            b.point(link, mat)
            alink = mat + 0x08
            for texture in ids:
                anim = b.add(struct.pack('>2I', 0, texture) + bytes(0x10))
                b.point(alink, anim)
                b.point(anim + 0x08, b.add(bytes(0x10)))
                alink = anim
            link = mat
    return offsets[0], mjoints[0]


def _fighter_file(path, animated, options):
    """A Pl<code>.dat with one model: ``options`` {set: DObj places of option 0}, and the TObj places it animates."""
    b = _DatBuilder()
    root, x8 = b.add(bytes(0x60)), b.add(bytes(0x14))
    b.point(root + 0x08, x8)
    struct.pack_into('>I', b.data, x8, 1)
    struct.pack_into('>I', b.data, x8 + 0x08, len(animated))
    places = b.add(b''.join(struct.pack('>H', a) for a in animated))
    lists = b.add(bytes(4))
    b.point(lists, places)
    b.point(x8 + 0x0C, lists)
    sets = b.add(bytes(16))
    b.point(x8 + 0x04, sets)
    for k, dobjs in options.items():
        lookup, option = b.add(struct.pack('>I', 1) + bytes(4)), b.add(struct.pack('>I', len(dobjs)) + bytes(4))
        b.point(lookup + 4, option)
        b.point(option + 4, b.add(bytes(dobjs)))
        b.point(sets + 4 * k, lookup)
    return b.save(path, [('ftDataTest', root)])


def test_bytes_at_reads_the_data_block_by_offset(tmp_path):
    b = _DatBuilder()
    at = b.add(bytes([1, 2, 3, 4, 5]))
    d = melee_dat.Dat.read(b.save(tmp_path / 'x.dat', [('root', at)]))
    assert d.bytes_at(at + 1, 3) == [2, 3, 4]
