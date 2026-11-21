"""Costume storage measured from the fighter's own HSDRawViewer round trip, not a downloaded model.

Relocation targets delimit blocks, as in export.write.copy_parts. Geometry is every unique block reached from a
PObj, stopping at skeleton joints. Texture payloads are reached from each model TObj's image and palette data.
The rest of the model is retained overhead; bytes outside the model are a fixed kept cost. Relocation entries
belong to the block containing their pointer. Shared blocks and PObjs are counted once, including all body states.

Placement reserves a fresh vanilla DObj list with blank materials plus one copy of every PObj header, using the
exporter's own writers. It is a structural estimate: an imported model can have different strip packing, materials,
and unreachable data. Export must still check the real file size.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from ....vocabulary import Target
from ... import Failed
from ..dat import Dat, joints, joint_dobjs


@dataclass
class MeleeTarget(Target):
    memory: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)


def read_record(f, targets=None):
    """Require preparation rather than launching the importer during target selection or a build."""
    from .. import _folder, TARGETS_VAR
    path = _folder(targets, TARGETS_VAR, 'Melee targets') / f'{f.name}.memory.json'
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        counts = [record['triangles'], record['kept'], *(record['bytes'][k]
                  for k in ('geometry', 'retained', 'placement'))]
        valid = (record['schema'] == 1 and record['fighter'] == f.name and record['triangles'] > 0
                 and all(type(n) is int and n >= 0 for n in counts)
                 and record['method'] and record['importer']['version'] and record['vanilla_sha256'])
        if not valid:
            raise ValueError('invalid measurement')
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Failed(f'{path}: missing or invalid importer memory record; run melee.prepare() for this targets '
                     'folder before selecting or prepping the fighter') from e
    return record


def write_record(f, files, targets):
    """Round-trip the fighter's spawn DObjs with export's own options; save only measurements and provenance.

    Fresh copies and a fresh DAE avoid both writes to game files and a stale cached DAE. The importer retains
    non-model bytes once; its model overhead and the exporter's placement allocation scale with triangle count.
    These rates describe this fighter's geometry/weights, not a bound on an arbitrary build's packing.
    """
    from ..export.write import HSDRAW_DEFAULT, HSDRAW_VAR, export_dae, import_dae, vanilla_copies
    files, targets = Path(files), Path(targets)
    importer = importer_build(Path(os.environ.get(HSDRAW_VAR) or HSDRAW_DEFAULT).resolve())
    targets.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='memory-', dir=targets) as tmp:
        dat, fighter_dat = vanilla_copies(tmp, files / f'Pl{f.code}Nr.dat', files / f'Pl{f.code}.dat')
        dae = export_dae(dat, f.symbol, Path(tmp) / 'vanilla.dae', drawn_by=f)
        imported = import_dae(dat, f.symbol, dae, Path(tmp) / 'imported.dat')
        stat = imported.stat()
        record = dict(_measure(str(imported), stat.st_mtime_ns, stat.st_size, f.symbol, str(dat)))
        record['method'] = 'HSDRawViewer spawn DAE round trip; ' + record['method']
        record.update(schema=1, fighter=f.name, importer=importer, vanilla_sha256=_sha256(dat),
                      fighter_sha256=_sha256(fighter_dat), dae_sha256=_sha256(dae),
                      options={'names': 'place', 'dobjs': 'high', 'show': [f'{m}:0' for m in f.shown],
                               'alpha_mode': 'Test'})
    path = targets / f'{f.name}.memory.json'
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    pending.replace(path)
    return record


def importer_build(exe):
    """Which HSDRawViewer build ``exe`` is: its version from ``<exe>.deps.json`` (the library named as the executable:
    HSDRawViewer on Windows, hsdrawviewer for the headless build on macOS and Linux) and the sha256 of the executable,
    its .dll and HSDRaw.dll. Failed when it can't be told."""
    exe = Path(exe)
    deps = exe.with_suffix('.deps.json')
    try:
        libraries = json.loads(deps.read_text(encoding='utf-8'))['libraries']
        version = next(k.split('/', 1)[1] for k in libraries if k.split('/', 1)[0].lower() == exe.stem.lower())
        binaries = [exe, exe.with_suffix('.dll'), exe.with_name('HSDRaw.dll')]
        return {'version': version, 'version_source': deps.name, 'sha256': {p.name: _sha256(p) for p in binaries}}
    except (OSError, ValueError, KeyError, StopIteration) as e:
        raise Failed(f'cannot identify the HSDRawViewer build at {exe}: {e}') from e


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(f, files=None):
    from .. import _folder, FILES_VAR
    path = _folder(files, FILES_VAR, 'vanilla game files') / f'Pl{f.code}Nr.dat'
    stat = path.stat()
    return _measure(str(path.resolve()), stat.st_mtime_ns, stat.st_size, f.symbol)


@lru_cache(maxsize=None)
def _measure(path, modified, size, symbol, vanilla=None):
    from ..export import dobjs
    d = Dat.read(path)
    tree = set(joints(d, symbol))
    objects = [o for j in joints(d, symbol) for o in joint_dobjs(d, j)]
    pobjs = {p for o in objects for p in d.chain(d.ptr(o + 0x0C), 4)}
    triangles = sum(_triangles(d, p) for p in pobjs)
    if not triangles:
        raise Failed(f'{Path(path).name}: no triangles to measure storage against')
    starts = sorted({0, len(d.data), *(d.u32(p) for p in d.relocations), *(p for _, p in d.roots)})
    fields = {s: [] for s in starts[:-1]}
    for p in sorted(d.relocations):
        fields[starts[bisect_right(starts, p) - 1]].append(p)

    def reach(roots, stop=()):
        seen, pending = set(), list(roots)
        while pending:
            at = pending.pop()
            if at is None or at in seen or at in stop:
                continue
            if at not in fields:
                raise Failed(f'{Path(path).name}: pointer outside a data block: {at:#x}')
            seen.add(at)
            pending.extend(d.u32(p) for p in fields[at])
        return seen

    sizes = {s: e - s + 4 * len(fields[s]) for s, e in zip(starts, starts[1:])}
    model = reach([d.root(symbol)])
    geometry = reach(pobjs, tree)
    payloads = []
    for o in objects:
        material = d.ptr(o + 8)
        if material is None:
            continue
        for t in d.chain(d.ptr(material + 8), 4):
            for field in (0x4C, 0x50):  # HSD_TObj's image and palette descriptors; each starts with its data pointer
                descriptor = d.ptr(t + field)
                if descriptor is not None:
                    payloads.append(d.ptr(descriptor))
    textures = reach(payloads) - geometry
    retained = model - geometry - textures
    totals = {name: sum(sizes[p] for p in blocks) for name, blocks in
              (('geometry', geometry), ('textures', textures), ('retained', retained))}
    kept = len(d.raw) - sum(totals.values())
    before_data, before_relocations = len(d.data), len(d.relocations)
    image_data = d.add(bytes(32), 32)
    image = d.add(bytes(0x18))
    d.point(image, image_data)
    material = d.add(bytes(0x14))
    cache, head = {}, None
    for slot in reversed(dobjs.layout(Dat.read(vanilla or path), symbol)):
        obj = d.add(bytes(0x10))
        d.point(obj + 4, head)
        d.point(obj + 8, dobjs._blank_material(d, slot.textures, image, material, cache))
        head = obj
    # Joining each source chain to an empty last chain copies every header with the exporter's exact relocations.
    empty = d.add(bytes(0x10))
    for o in objects:
        dobjs._joined_pobjs(d, [o, empty])
    placement = len(d.data) - before_data + 4 * (len(d.relocations) - before_relocations) - 0x10
    totals['placement'] = placement
    return {'file': Path(path).name, 'file_bytes': size, 'triangles': triangles, 'pobjs': len(pobjs),
            'bytes': totals, 'kept': kept,
            'per_triangle': {k: totals[k] / triangles for k in ('geometry', 'retained', 'placement')},
            'method': 'unique PObj geometry blocks / nondegenerate display-list triangles; model texture payloads '
                      'excluded; remaining model blocks retained; bytes outside model kept once; placement is '
                      'blank vanilla slots plus one copy of every PObj header, including relocations'}


def _triangles(d, pobj):
    """Count triangles with distinct position indices in GX triangle, strip, fan and quad commands.

    Vertex descriptors: HSDRaw GX_VertexAttribute (0x18 bytes); GX attribute types 0 absent, 1 direct, 2/3 indexed.
    Matrix indices are bytes, position/normal/UV components use GX u8/s8/u16/s16/f32, colours packed GX formats.
    """
    at, stride, position = d.ptr(pobj + 8), 0, None
    while at is not None:
        name, kind, count, component = struct.unpack_from('>4I', d.data, at)
        if name == 0xFF:
            break
        if kind == 0:
            width = 0
        elif kind in (2, 3):
            width = kind - 1
        elif kind == 1:
            if name <= 8:
                width = 1
            elif name in (11, 12):
                width = (2, 3, 4, 2, 3, 4)[component]
            else:
                components = {9: 2 + count, 10: 9 if count == 1 else 3}.get(name, 1 + count)
                width = components * (1, 1, 2, 2, 4)[component]
        else:
            raise Failed(f'unknown GX attribute type {kind}')
        if name == 9:
            position = stride, width
        stride += width
        at += 0x18
    at = d.ptr(pobj + 0x10)
    if at is None:
        return 0
    if position is None or not stride:
        raise Failed('a vanilla display list has no positions')
    end, total = at + 32 * d.u16(pobj + 0x0E), 0
    while at < end and d.data[at]:
        opcode, n = d.data[at] & 0xF8, d.u16(at + 1)
        if at + 3 + n * stride > end:
            raise Failed('a vanilla GX command exceeds its display list')
        offset, width = position
        positions = [bytes(d.data[at + 3 + i * stride + offset:at + 3 + i * stride + offset + width])
                     for i in range(n)]
        if opcode in (0x80, 0x88):
            faces = [(i, i + 1, i + 2) for i in range(0, n - 3, 4)]
            faces += [(i + 2, i + 3, i) for i in range(0, n - 3, 4)]
        elif opcode == 0x90:
            faces = [(i, i + 1, i + 2) for i in range(0, n - 2, 3)]
        elif opcode == 0x98:
            faces = [(i, i + 1, i + 2) for i in range(n - 2)]
        elif opcode == 0xA0:
            faces = [(0, i + 1, i + 2) for i in range(n - 2)]
        elif opcode in (0xA8, 0xB0, 0xB8):
            faces = []
        else:
            raise Failed(f'unknown vanilla GX primitive {opcode:#x}')
        total += sum(len({positions[i] for i in face}) == 3 for face in faces)
        at += 3 + n * stride
    return total
