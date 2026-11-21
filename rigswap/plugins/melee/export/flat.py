"""A flat fighter's costume (facts/flat.py): what the file has to say of the build prep/flat.py made, once
fill_vanilla_dobjs has laid it out. The hull's DObj goes to the first of the places of the hull's model 0, on the root
as every skinned DObj is, which drives every form of it (write.py: fill_forms); nothing else draws it, in low detail
or at all. Every material the build drew is CONSTANT: unlit, so the black stays black. The eyes' texture is REPLACE:
it alone shows through the fill.
"""
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from ... import Failed
from ..dae import NS
from ..dat import Dat
from ..facts.flat import COLOUR_REPLACE, RENDER_CONSTANT, RENDER_LIGHT, TOBJ_COLOUR
from . import dobjs as melee_dobjs
from .eyes import _dobj_list, dae_dobjs


def _record(build_dir):
    path = Path(build_dir) / 'flat.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def hull_of(build_dir):
    """The objects of the build that draw the outline hull: none for a fighter who isn't flat."""
    made = _record(build_dir)
    return [made['hull']] if made else []


def drawn_dae(dae, build_dir):
    """``dae``, or a copy beside it without the hull, which no low detail draws: what the costume's read-back, with the
    hull's own places left out of it, is held to."""
    hull = hull_of(build_dir)
    if not hull:
        return dae
    tree = ET.parse(dae)
    for scene in tree.getroot().iter(NS + 'visual_scene'):
        for node in scene.findall(NS + 'node'):
            if node.get('name') in hull:
                scene.remove(node)
    path = Path(dae).with_name('drawn.dae')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return path


def apply(build_dir, out, vanilla_dat, fighter_dat, symbol, placed, dae, vanilla_dae=None):
    """Do to the costume ``out`` (``placed`` is what fill_vanilla_dobjs returned) what the module's docstring says.
    ``dae`` is the .dae HSDRawViewer imported (the order of the build's DObjs). -> what was done"""
    made = _record(build_dir)
    order = dae_dobjs(dae)
    if len(order) != placed['build_dobjs']:
        raise Failed(f'{Path(dae).name} makes {len(order)} DObjs and the import has {placed["build_dobjs"]}: the hull '
                     f'and the eyes cannot be told from the rest')
    d = Dat.read(out)
    listed = _dobj_list(d, symbol)
    tables = melee_dobjs.fighter_tables(fighter_dat)
    deferred = {int(n): tuple(source) for n, source in placed.get('deferred', {}).items()}

    def drawn_by():         # {place: the build's DObjs it draws, its material the first's}
        places = {}
        for n, p in enumerate(placed['places']):
            if n not in deferred:
                places.setdefault(p, []).append(n)
        for p, ns in placed['low_detail']['places'].items():
            places.setdefault(int(p), []).extend(ns)
        return places

    def unlit(mobj):
        struct.pack_into('>I', d.data, mobj + 0x04, d.u32(mobj + 0x04) & ~RENDER_LIGHT | RENDER_CONSTANT)

    seen, eyes = {}, []
    for place, ns in sorted(drawn_by().items()):
        mobj = d.ptr(listed[place][1] + 0x08)
        unlit(mobj)
        seen[place] = order[ns[0]][0]
        if order[ns[0]][0] in made['eyes']:
            eyes.append(place)
            for tobj in d.chain(d.ptr(mobj + 0x08), 0x04):
                struct.pack_into('>I', d.data, tobj + 0x40, d.u32(tobj + 0x40) & ~TOBJ_COLOUR | COLOUR_REPLACE)
    seats = iter(tables.options[melee_dobjs.HULL, 0, 0])
    hulls = {}
    for n, (joint, dobj) in deferred.items():
        if order[n][0] != made['hull']:
            continue
        to = next(seats)
        unlit(d.ptr(dobj + 0x08))
        d.point(listed[to][1] + 0x08, d.ptr(dobj + 0x08))
        d.point(listed[to][1] + 0x0C, d.ptr(dobj + 0x0C))
        placed['places'][n] = to
        placed['blank'] -= 1
        hulls[n] = to
    d.save(out)
    return {'hull': hulls, 'eyes': eyes, 'constant': seen, 'turned': made['turned'], 'depth': made['depth']}
