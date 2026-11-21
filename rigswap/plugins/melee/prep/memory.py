"""4. memory: the costume file, estimated, must fit the fighter's share of the game's preload memory; the costliest
textures are halved until it does (hold_memory_budget). The vanilla parts the costume keeps count in it (budget).
Here too, as prep.py had them: the triangle palettes, at most ten envelopes each (check_mesh_palettes). export checks
the written file against the same share (export/checks.py: check_costume_size)."""
from __future__ import annotations

from ... import Failed
from ..facts.parts import KEPT_TRIANGLES, not_kept
from .textures import TEXTURE_MIN_SIDE, fit_inside, resample, write_pixels


# Melee loads a fighter's Pl<code>.dat, its costume and its effect file into preload heap 4, one fixed arena it shares
# with the stage and every other fighter's files. A costume that does not fit hangs the match on the loading screen
# (assertion "memp_kouho" in lbmemory.c). A costume's share is the heap, less the heaviest legal stage (Pokemon
# Stadium), the heaviest vanilla opponent (Zelda, who brings Sheik), the files preloaded with the fighter (the table's
# ``preloaded`` column, each rounded up to 32 bytes plus an archive header), the costume's own archive header and slack
# for small allocations. Measured with Dolphin logs and heap walks, 2026-09-23. (shared/modellab/preload_heap.py)
HEAP_BYTES = 6_599_680
STAGE_BYTES = 2_411_520
OPPONENT_BYTES = 1_172_000
ARCHIVE_HEADER_BYTES = 96
SLACK_BYTES = 64 * 1024
# The game reads a file in 32-byte units; a costume fits when its size rounded up to them is within its share.
FILE_UNIT = 32

# ---- 4. memory -------------------------------------------------------------------------------------------------------

def check_mesh_palettes(meshes):
    """Plan triangle palettes per mesh/material from full influence sets, never from the number of joint names.

    src/sysdolphin/baselib/pobj.c:1131 loads ten envelopes per palette, :1154 walks all joints within each envelope.
    The importer packs triangles/strips into palettes (HSDRaw/Tools/TriangleConverter/PrimitiveGroup.cs:51,72).
    This triangle packing is a prep estimate; export checks the actual palettes after the importer's strip packing.
    """
    from ..export.dobjs import MAX_ENVELOPES
    rows = []
    for o in meshes:
        o.data.calc_loop_triangles()
        groups = {g.index: g.name for g in o.vertex_groups}
        influences = {v.index: tuple(sorted((groups[g.group], float(g.weight)) for g in v.groups if g.weight > 0))
                      for v in o.data.vertices}
        materials = {}
        for triangle in o.data.loop_triangles:
            material = triangle.material_index
            entries = {influences[v] for v in triangle.vertices}
            if () in entries:
                raise Failed(f'{o.name}, material {material}: a drawn vertex has 0 envelope joints')
            palettes = materials.setdefault(material, [])
            palette = next((p for p in palettes if len(p | entries) <= MAX_ENVELOPES), None)
            if palette is None:
                palette = set()
                palettes.append(palette)
            palette.update(entries)
            check_palette_size(len(palette), f'{o.name}, material {material}, palette {len(palettes)}')
        for material, palettes in sorted(materials.items()):
            rows.append({'mesh': o.name, 'material': material, 'sizes': [len(p) for p in palettes]})
    return {'meshes': rows, 'limit': MAX_ENVELOPES, 'pass': True}


def check_palette_size(count, where):
    from ..export.dobjs import MAX_ENVELOPES
    if count > MAX_ENVELOPES:
        raise Failed(f'{where}: palette has {count} envelopes; limit {MAX_ENVELOPES}')


def costume_budget(preloaded):
    """The bytes a fighter's costume file may take of preload heap 4, given the sizes of the files preloaded with it
    (the table's ``preloaded`` column)."""
    files = sum(_in_units(size) + ARCHIVE_HEADER_BYTES for size in preloaded if size)
    return HEAP_BYTES - STAGE_BYTES - OPPONENT_BYTES - files - ARCHIVE_HEADER_BYTES - SLACK_BYTES


def _in_units(size):
    return -(-int(size) // FILE_UNIT) * FILE_UNIT


def costume_estimate(textures, triangles, kept, measurement):
    """{bytes, textures, geometry, kept}: the costume file prep expects export to write, in 32-byte units, for these
    textures, this many triangles and ``kept`` bytes kept from the vanilla file. ``measurement`` is facts.memory's
    importer round-trip measurement; it is required, with no guessed fallback."""
    texture_bytes = sum(t.bytes() for t in textures)
    count = measurement['triangles']
    geometry, retained, placement = (-(-measurement['bytes'][k] * int(triangles) // count)
                                     for k in ('geometry', 'retained', 'placement'))
    return {'bytes': _in_units(kept + texture_bytes + geometry + retained + placement),
            'textures': texture_bytes, 'geometry': geometry, 'retained': retained, 'placement': placement,
            'kept': int(kept)}


def hold_memory_budget(textures, triangles, budget, kept, measurement):
    """Hold the costume to ``budget`` bytes (see costume_estimate). When the estimate is over: the CI8 textures picked
    for hard edges go back to CMPR, then the texture that costs the most is halved on its long side (never below
    TEXTURE_MIN_SIDE), and again, until it fits. Halved textures are resized from the build's own pixels. Failed with
    the numbers when it cannot fit. -> record"""
    before = costume_estimate(textures, triangles, kept, measurement)
    record = {'budget_bytes': budget, 'triangles': int(triangles), 'before': before, 'to_cmpr': [], 'halved': []}
    record['measurement'] = measurement
    if before['bytes'] <= budget:
        record.update(after=before, status='fits')
        return record
    for t in textures:
        if t.automatic:
            t.format, t.automatic, t.why = 'CMPR', False, 'CMPR: the costume was over its memory with CI8'
            record['to_cmpr'].append(t.name)
    start = {t.name: tuple(t.size) for t in textures}
    while (now := costume_estimate(textures, triangles, kept, measurement))['bytes'] > budget:
        smaller = [(t.bytes(), t.name, t) for t in textures if halved(t.size) is not None]
        if not smaller:
            raise Failed(f'the costume comes to about {now["bytes"]:,} bytes ({now["textures"]:,} of textures, '
                         f'{now["geometry"]:,} for {int(triangles):,} triangles, {now["kept"]:,} kept from the vanilla '
                         f'file) and the fighter\'s share of preload memory is {budget:,}: {now["bytes"] - budget:,} '
                         f'over with every texture at {TEXTURE_MIN_SIDE} px or less')
        _, _, t = max(smaller, key=lambda c: c[:2])
        t.size = halved(t.size)
    for t in textures:
        if tuple(t.size) != start[t.name]:
            write_pixels(t.image, resample(t.pixels, t.size, t.alpha))
            t.why += f'; halved to fit the memory, from {start[t.name][0]}x{start[t.name][1]}'
            record['halved'].append({'image': t.name, 'from': list(start[t.name]), 'to': list(t.size)})
    record.update(after=now, status='textures reduced')
    return record


def halved(size, floor=TEXTURE_MIN_SIDE):
    """``size`` with its long side halved, not below ``floor``, aspect kept; None when it is already at the floor."""
    w, h = size
    if max(w, h) <= floor:
        return None
    return fit_inside((w, h), max(floor, max(w, h) // 2))


def budget(f, files=None, replaced=()):
    """Reserve copied resources and up to 31 bytes of initial alignment before prep. ``replaced``: the weapons the
    build replaced (kept_places), whose places and triangles are not kept, nor their other forms' (facts/parts.py:
    not_kept, hidden_forms)."""
    hidden = not_kept(f.name, replaced)
    gone = {p for r in replaced for p in r['places']} | {p for part in hidden.values() for p in part['places']}
    keep = tuple(p for p in f.keep if p not in gone)
    if not keep:
        return {'bytes': 0, 'triangles': 0, 'places': []}
    from .. import _folder, FILES_VAR
    from ..dat import Dat
    from ..export.write import copy_parts
    path = _folder(files, FILES_VAR, 'vanilla game files') / f'Pl{f.code}Nr.dat'
    source = Dat.read(path)
    _, report = copy_parts(source, Dat.read(path), f.symbol, keep)
    out = {'bytes': report['bytes'] + 31,
           'triangles': KEPT_TRIANGLES[f.name] - sum(part['triangles'] for part in hidden.values()),
           'places': list(keep)}
    if replaced:
        out['not_kept'] = hidden
    return out
