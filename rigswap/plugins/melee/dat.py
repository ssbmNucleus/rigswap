"""The .dat file, read and written: an HSD archive's data block, its relocation table (the data's pointer fields) and
its root table (Dat); the joint tree under a root and each joint's DObjs (joints, joint_dobjs); the root table and the
rest skeleton read straight from the bytes (dat_roots, dat_joints); and two numbers of a fighter read from its files,
which facts/fighters.py records (model_scale, vanilla_max_influences)."""
from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .. import Failed
from . import FILES_VAR, _folder


# HSD structures, by the fields read here and in export/ (sysdolphin baselib; HSDRaw/Common in HSDLib):
#   joint (0x40): +0x04 flags, +0x08 first child, +0x0C next sibling, +0x10 first DObj
#   DObj (0x10): +0x04 next, +0x08 MObj, +0x0C first PObj
#   MObj (0x18): +0x04 render mode, +0x08 first TObj, +0x0C material (0x14 bytes), +0x14 pixel engine settings
#   TObj (0x5C): +0x04 next, +0x08 texture map id, ..., +0x4C image (0x18 bytes: data, width, height, format, ...)
#   matanim joint (0x0C): +0x00 first child, +0x04 next sibling, +0x08 first material animation
#   material animation (0x10): +0x00 next, +0x04 colour animation, +0x08 first texture animation, +0x0C render anim
#   texture animation (0x18): +0x00 next, +0x04 texture map id, +0x08 its animation
JOINT_INSTANCE = 1 << 12                    # the joint's child is an instance: the game does not walk into it
JOINT_NO_DOBJ = (1 << 5) | (1 << 14)        # particle or spline joints: +0x10 is not a DObj


class Dat:
    """A .dat file read for its data block, relocation table (the data's pointer fields) and root table."""

    def __init__(self, raw):
        self.raw = bytes(raw)
        _, data_size, relocations, roots, references = struct.unpack_from('>5I', self.raw, 0)
        self.data = bytearray(self.raw[0x20:0x20 + data_size])
        table = 0x20 + data_size
        self.relocations = set(struct.unpack_from(f'>{relocations}I', self.raw, table))
        self.tail = self.raw[table + relocations * 4:]       # the root and reference tables and their names
        self.counts = (roots, references)
        strings = (roots + references) * 8
        self.roots = []
        for i in range(roots):
            offset, name = struct.unpack_from('>2I', self.tail, i * 8)
            end = self.tail.index(b'\0', strings + name)
            self.roots.append((self.tail[strings + name:end].decode('ascii'), offset))

    @classmethod
    def read(cls, path):
        return cls(Path(path).read_bytes())

    def u32(self, at):
        return struct.unpack_from('>I', self.data, at)[0]

    def u16(self, at):
        return struct.unpack_from('>H', self.data, at)[0]

    def ptr(self, at):
        return self.u32(at) if at in self.relocations else None

    def bytes_at(self, at, n):
        return list(self.data[at:at + n])

    def root(self, name):
        return next((offset for n, offset in self.roots if n == name), None)

    def chain(self, head, next_at):
        out, seen = [], set()
        while head is not None:
            if head in seen:
                raise Failed(f'a looped chain at {head:#x}')
            seen.add(head)
            out.append(head)
            head = self.ptr(head + next_at)
        return out

    def add(self, payload, align=4):
        """Append ``payload`` to the data block; its offset."""
        self.data.extend(b'\0' * (-len(self.data) % align))
        at = len(self.data)
        self.data.extend(payload)
        return at

    def point(self, at, target):
        """Make the field at ``at`` point to ``target`` (an offset in the data block), or to nothing."""
        struct.pack_into('>I', self.data, at, target or 0)
        if target is None:
            self.relocations.discard(at)
        else:
            self.relocations.add(at)

    def save(self, path):
        relocations = sorted(self.relocations)
        body = bytes(self.data) + struct.pack(f'>{len(relocations)}I', *relocations) + self.tail
        header = struct.pack('>5I', 0x20 + len(body), len(self.data), len(relocations), *self.counts)
        Path(path).write_bytes(header + self.raw[0x14:0x20] + body)


# ---- reading -------------------------------------------------------------------------------------------------------

def joints(d, symbol):
    """The joints under root ``symbol``, as offsets, in the game's tree order."""
    root = d.root(symbol)
    if root is None:
        raise Failed(f'the file has no root {symbol}')
    out = []

    def walk(j):
        while j is not None:
            out.append(j)
            if not d.u32(j + 4) & JOINT_INSTANCE:
                walk(d.ptr(j + 0x08))
            j = d.ptr(j + 0x0C)
    walk(root)
    return out


def joint_dobjs(d, joint):
    """The DObjs of one joint, as offsets, in chain order."""
    if d.u32(joint + 4) & JOINT_NO_DOBJ:
        return []
    return d.chain(d.ptr(joint + 0x10), 0x04)


def dat_roots(data):
    """[(symbol, offset in the data block)] from a .dat file's root table."""
    _, data_size, relocations, roots, references = struct.unpack_from('>5I', data, 0)
    table = 0x20 + data_size + relocations * 4
    strings = table + (roots + references) * 8
    out = []
    for i in range(roots):
        offset, name = struct.unpack_from('>2I', data, table + i * 8)
        end = data.index(b'\0', strings + name)
        out.append((data[strings + name:end].decode('ascii'), offset))
    return out


def dat_joints(path, symbol, local=False):
    """{JOBJ_<n>: 4x4 rest matrix in world space (or local with local=True)} under ``symbol``, from each joint's own
    rotation, scale and translation (what the game's animations start from), numbered depth first as HSDRawViewer
    numbers them. Read straight from the file: a joint is 0x40 bytes, flags at +0x04, child at +0x08, next at +0x0C,
    rotation, scale and translation as nine floats from +0x14. Pointers are the fields the relocation table lists."""
    data = Path(path).read_bytes()
    _, data_size, relocations, _, _ = struct.unpack_from('>5I', data, 0)
    pointers = set(struct.unpack_from(f'>{relocations}I', data, 0x20 + data_size))
    root = next((off for name, off in dat_roots(data) if name == symbol), None)
    if root is None:
        raise Failed(f'{Path(path).name} has no root {symbol}')

    def ptr(at):
        return struct.unpack_from('>I', data, 0x20 + at)[0] if at in pointers else None

    joints = {}

    def walk(at, parent_world):
        while at is not None:
            rx, ry, rz, sx, sy, sz, tx, ty, tz = struct.unpack_from('>9f', data, 0x20 + at + 0x14)
            cx, sx_, cy, sy_, cz, sz_ = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
            r = (np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
                 @ np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]]))
            matrix = np.eye(4)
            matrix[:3, :3] = r * np.array([sx, sy, sz])
            matrix[:3, 3] = (tx, ty, tz)
            world = parent_world @ matrix
            joints[f'JOBJ_{len(joints)}'] = matrix if local else world
            walk(ptr(at + 0x08), world)
            at = ptr(at + 0x0C)

    walk(root, np.eye(4))
    return joints


def model_scale(f, files_dir=None):
    """The fighter's in-game size: its model is drawn this many times larger (Pl<code>.dat: the ftData root points to
    the common attributes, whose ``model_scaling`` float is at +0x8C)."""
    path = _folder(files_dir, FILES_VAR, 'vanilla game files') / f'Pl{f.code}.dat'
    data = path.read_bytes()
    root = next((off for name, off in dat_roots(data) if name.startswith('ftData')), None)
    if root is None:
        raise Failed(f'{path.name} has no ftData root')
    attributes = struct.unpack_from('>I', data, 0x20 + root)[0]
    scale = struct.unpack_from('>f', data, 0x20 + attributes + 0x8C)[0]
    if not (np.isfinite(scale) and scale > 0):
        raise Failed(f'{path.name}: model scale {scale!r}')
    return float(scale)


def vanilla_max_influences(path, symbol):
    """Largest positive envelope in the vanilla DAT, including hidden body states.

    PObj type/envelope offsets: HSDRaw/Common/HSD_POBJ.cs; envelopes are a
    null-terminated array of (joint pointer, float weight), HSD_Envelope.cs.
    Rigid geometry has one influence. No meshes are exported or guessed here.
    """
    d = Dat.read(path)
    maximum = 0
    for joint in joints(d, symbol):
        for dobj in joint_dobjs(d, joint):
            for pobj in d.chain(d.ptr(dobj + 0x0c), 4):
                maximum = max(maximum, 1)
                if d.u16(pobj + 0x0c) & 0x3000 != 0x2000:
                    continue
                table = d.ptr(pobj + 0x14)
                if table is None:
                    raise Failed(f'{Path(path).name}: a skinned PObj has no envelope table')
                while (envelope := d.ptr(table)) is not None:
                    count = 0
                    while d.ptr(envelope) is not None:
                        weight = struct.unpack_from('>f', d.data, envelope + 4)[0]
                        if not np.isfinite(weight) or weight < 0:
                            raise Failed(f'{Path(path).name}: invalid envelope weight {weight}')
                        count += weight > 0
                        envelope += 8
                    maximum = max(maximum, count)
                    table += 4
    if not maximum:
        raise Failed(f'{Path(path).name}: no vanilla geometry to measure')
    return maximum
