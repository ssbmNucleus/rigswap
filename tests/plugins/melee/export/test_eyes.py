"""export/eyes.py: the build's eyes as the fighter's own blinking eyes (the Melee side of rigswap.core.eyes)."""
import json
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from rigswap.core import blink as eyes
from rigswap.plugins import Failed
from rigswap.plugins.melee import dae as melee_dae, dat as melee_dat
from rigswap.plugins.melee.export import dobjs as melee_dobjs, eyes as melee_eyes, write as export_write


# ---- Melee: the fighter's blink, the texture animation's keys, GX images ------------------------------------------

def test_the_blink_is_half_closed_closed_half_closed_open():
    assert melee_eyes.blink_of([[1, 2, 1, 0]]) == {'open': 0, 'half': 1, 'closed': 2}
    assert melee_eyes.blink_of([[0, 1, 1, 2, 1, 0], [5, 4, 0]]) == {'open': 0, 'half': 1, 'closed': 2}
    assert melee_eyes.blink_of([[4, 1], [3, 3]]) is None


def test_marios_eye_track_keys_decode():
    # PlMrNr.dat, the TIMG track of DObj 14's texture animation: CON, 7 keys of u8 / 32, each with its wait
    keys = melee_eyes.track_keys(bytes.fromhex('6100012001400160018001a005a000'), 0., 0x85, 0)
    assert keys == [(0., 0.), (1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.), (10., 5.)]
    assert [melee_eyes.value_at(keys, f) for f in (0, 1, 2, 7, 12)] == [0, 1, 2, 5, 5]
    assert melee_eyes.states_of(keys, 6, {'open': 0, 'half': 1, 'closed': 2}) == \
        ['open', 'half', 'closed', 'open', 'open', 'open']


def test_an_image_the_blink_never_shows_takes_the_state_it_is_nearest_in_openness():
    skin, white, iris, line = (230, 170, 120, 255), (240, 240, 250, 255), (30, 60, 200, 255), (20, 10, 10, 255)
    opened = np.full((16, 16, 4), skin, np.uint8)
    opened[4:12, 2:14] = white
    opened[5:11, 6:10] = iris
    shut = np.full((16, 16, 4), skin, np.uint8)
    half = shut.copy()
    half[8:12] = opened[8:12]                       # the lid down to the middle
    squeezed = shut.copy()
    squeezed[7:9, 2:14] = line                      # shut tight: a line on the skin, nothing of the eye
    aside = opened.copy()
    aside[5:11, 6:10] = white
    aside[5:11, 9:13] = iris                        # looking aside: the eye open, its iris moved
    squint = shut.copy()
    squint[7:11] = opened[7:11]                     # a squint: half the eye shows, shifted down
    blink = {'open': 0, 'half': 1, 'closed': 2}
    keys = [(0., 0.), (1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.)]
    pictures = [opened, half, shut, squeezed, aside, squint]
    assert melee_eyes.states_of(keys, 6, blink, pictures) == ['open', 'half', 'closed', 'closed', 'open', 'half']
    # without vanilla's pictures (or one that can't be read) an image the blink never shows stays open
    assert melee_eyes.states_of(keys, 6, blink) == ['open', 'half', 'closed', 'open', 'open', 'open']
    assert melee_eyes.states_of(keys, 6, blink, pictures[:5] + [None])[5] == 'open'
    # a blink whose half closed frame shows the open image: two references
    assert melee_eyes.states_of([(0., 0.), (2., 1.), (3., 2.)], 3, blink, [opened, shut, squeezed]) == \
        ['open', 'closed', 'closed']


def test_cmpr_decodes_its_two_colours_and_the_thirds_between():
    # one 8x8 tile: four 4x4 blocks, each red (0xF800) over blue (0x001F), indices 0, 1, 2, 3 by column
    block = struct.pack('>HH', 0xF800, 0x001F) + bytes([0b00011011] * 4)
    clear = struct.pack('>HH', 0x001F, 0xF800) + bytes([0b11111111] * 4)       # first not above second: index 3 clear
    px = melee_eyes.decode(block * 3 + clear, 8, 8, melee_eyes.GX_CMPR)
    assert px.shape == (8, 8, 4)
    assert tuple(px[0, 0]) == (255, 0, 0, 255) and tuple(px[0, 1]) == (0, 0, 255, 255)
    assert tuple(px[0, 2]) == (170, 0, 85, 255) and tuple(px[0, 3]) == (85, 0, 170, 255)
    assert tuple(px[3, 7]) == (85, 0, 170, 255)                                 # the top right block
    assert (px[4:, 4:, 3] == 0).all() and (px[4:, :4, 3] == 255).all()           # the bottom right block is clear


@pytest.mark.parametrize('palette', [False, True])
def test_gx_pictures_round_trip(palette):
    rng = np.random.default_rng(3)
    px = rng.integers(0, 256, (12, 20, 4), dtype=np.uint8)
    px[..., 3] = 255
    px[:4, :4, 3] = 96
    data, fmt, colours, colours_fmt = melee_eyes.encode(px, palette)
    assert len(data) == melee_eyes.gx_bytes(fmt, 20, 12)
    back = melee_eyes.decode(data, 20, 12, fmt, colours, colours_fmt)
    assert back.shape == px.shape
    if not palette:     # RGB5A3 keeps 5 bits of an opaque colour, 4 and 3 of alpha of a clear one
        assert np.abs(back[4:, :, :3].astype(int) - px[4:, :, :3]).max() <= 8
        assert np.abs(back[:4, :4, 3].astype(int) - 96).max() <= 37
    else:
        assert fmt == melee_eyes.GX_CI8 and colours_fmt == melee_eyes.TLUT_RGB5A3 and len(colours) == 512
    flat = np.zeros((8, 8, 4), np.uint8)
    flat[:] = (10, 200, 30, 255)
    data, fmt, colours, colours_fmt = melee_eyes.encode(flat, True)
    assert colours_fmt == melee_eyes.TLUT_RGB565
    assert np.abs(melee_eyes.decode(data, 8, 8, fmt, colours, colours_fmt).astype(int) - flat).max() <= 8


def test_the_pictures_shrink_to_fit_where_vanilla_eye_images_were():
    room = [36864] * 6                       # Mario: six CI8 images of 190 x 190
    assert melee_eyes.picture_size((256, 256), melee_eyes.GX_CI8, room) == (128, 128)
    assert melee_eyes.picture_size((128, 64), melee_eyes.GX_CI8, room) == (128, 64)
    assert melee_eyes.picture_size((256, 256), melee_eyes.GX_CI8, [36864] * 2) == (256, 256)
    assert melee_eyes.picture_size((64, 64), melee_eyes.GX_RGB5A3, [100] * 3) == (32, 32)


# ---- Melee: the eyes laid into a synthetic costume ----------------------------------------------------------------

class _Dat:
    """A .dat put together a structure at a time."""

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
        Path(path).write_bytes(struct.pack('>8I', 0x20 + len(body), len(self.data), len(self.relocations),
                                           len(roots), 0, 0, 0, 0) + body)
        return Path(path)


def _image(b, w, h, fmt=melee_eyes.GX_CI8):
    desc = b.add(struct.pack('>I2H2I2f', 0, w, h, fmt, 0, 0., 0.))
    b.point(desc, b.add(bytes(melee_eyes.gx_bytes(fmt, w, h)), 32))
    return desc


def _texture_animation(b, images):
    """A texture animation of ``images`` CI8 8 x 8 images, each with its palette, both tracks keyed frame k -> k."""
    keys = bytes([0x01 | (images - 1) << 4]) + b''.join(bytes([k * 32, 1]) for k in range(images))
    aobj = b.add(struct.pack('>If', 0, float(images)) + bytes(8))
    fobjs = []
    for track, slope in ((melee_eyes.TRACK_IMAGE, 0), (melee_eyes.TRACK_PALETTE, 0x50)):
        fobj = b.add(struct.pack('>2If4B', 0, len(keys), 0., track, 0x85, slope, 0) + bytes(4))
        b.point(fobj + 0x10, b.add(keys))
        fobjs.append(fobj)
    b.point(aobj + 0x08, fobjs[0])
    b.point(fobjs[0], fobjs[1])
    table, luts = b.add(bytes(4 * images)), b.add(bytes(4 * images))
    for k in range(images):
        b.point(table + 4 * k, _image(b, 8, 8))
        lut = b.add(struct.pack('>3IH2x', 0, melee_eyes.TLUT_RGB565, 0, 256))
        b.point(lut, b.add(bytes(512), 32))
        b.point(luts + 4 * k, lut)
    anim = b.add(struct.pack('>2I', 0, 0) + bytes(12) + struct.pack('>2H', images, images))
    b.point(anim + 0x08, aobj)
    b.point(anim + 0x0C, table)
    b.point(anim + 0x10, luts)
    return anim


def _triangle(b, head, uv=True, colour=False):
    """A real GX triangle: indexed positions/normals, optional UV0/color0, and one head envelope."""
    attributes = [(0, 1, 0, 0, 1, None),
                  (9, 2, 1, 4, 12, struct.pack('>9f', 0., 0., 0., 1., 0., 0., 0., 1., 0.)),
                  (10, 2, 0, 4, 12, struct.pack('>3f', 0., 0., 1.))]
    if colour:
        attributes.append((11, 2, 1, 5, 4, bytes([255, 128, 64, 255])))
    if uv:
        attributes.append((13, 2, 1, 4, 8, struct.pack('>6f', 0., 0., 1., 0., 0., 1.)))
    attrs = b.add(bytes(24 * (len(attributes) + 1)))
    for k, (name, kind, count, fmt, stride, data) in enumerate(attributes):
        at = attrs + 24 * k
        struct.pack_into('>4IBxHI', b.data, at, name, kind, count, fmt, 0, stride, 0)
        if data:
            b.point(at + 20, b.add(data, 32))
    struct.pack_into('>I', b.data, attrs + 24 * len(attributes), 255)
    display = b'\x90\x00\x03' + b''.join(bytes([0, k, 0] + ([0] if colour else []) + ([k] if uv else []))
                                           for k in range(3))
    pobj = b.add(struct.pack('>3I2H2I', 0, 0, 0, 0x2000, 1, 0, 0))
    b.point(pobj + 8, attrs)
    b.point(pobj + 16, b.add(display.ljust(32, b'\0'), 32))
    envelope = b.add(struct.pack('>If2I', 0, 1., 0, 0))
    b.point(envelope, head)
    table = b.add(bytes(8))
    b.point(table, envelope)
    b.point(pobj + 20, table)
    return pobj


def _costume(path, dobjs, animated, image_size=None, eye_textures=1):
    """A root's DObjs with triangles weighted to its head child; one TObj each, ``eye_textures`` for animated
    DObjs, whose first TObj has four animation images. With ``image_size`` every TObj gets that size image."""
    b = _Dat()
    root = b.add(bytes(0x40))
    head = b.add(bytes(0x40))
    b.point(root + 8, head)
    for joint in (root, head):
        struct.pack_into('>3f', b.data, joint + 0x20, 1., 1., 1.)
    link = root + 0x10
    for n in range(dobjs):
        dobj, mobj = b.add(bytes(0x10)), b.add(bytes(0x18))
        b.point(link, dobj)
        b.point(dobj + 0x08, mobj)
        b.point(dobj + 0x0C, _triangle(b, head))
        b.point(mobj + 0x0C, b.add(struct.pack('>12B2f', *([255] * 12), 1., 50.)))
        tlink = mobj + 0x08
        for k in range(eye_textures if n in animated else 1):
            tobj = b.add(bytes(0x5C))
            b.point(tlink, tobj)
            struct.pack_into('>2I', b.data, tobj + 8, k, 4)
            struct.pack_into('>3f', b.data, tobj + 0x1C, 1., 1., 1.)
            struct.pack_into('>I', b.data, tobj + 0x40, 0x40010)
            if image_size:
                b.point(tobj + 0x4C, _image(b, *image_size, fmt=14))
            tlink = tobj + 4
        link = dobj + 0x04
    mroot = b.add(bytes(0x0C))
    link = mroot + 0x08
    for n in range(max(animated) + 1):
        mat = b.add(bytes(0x10))
        b.point(link, mat)
        if n in animated:
            b.point(mat + 0x08, _texture_animation(b, 4))
        link = mat
    return b.save(path, [('PlyTest_Share_joint', root), ('PlyTest_Share_matanim_joint', mroot)])


def _fighter(path, animated, body, low=()):
    """A Pl<code>.dat: the TObjs it animates, one model with ``body`` drawn at spawn, and an idle that blinks."""
    b = _Dat()
    root, x8 = b.add(bytes(0x60)), b.add(bytes(0x14))
    b.point(root + 0x08, x8)
    struct.pack_into('>3I', b.data, x8, 1, 0, len(animated))
    places = b.add(b''.join(struct.pack('>H', a) for a in animated))
    lists = b.add(bytes(4))
    b.point(lists, places)
    b.point(x8 + 0x0C, lists)
    sets = b.add(bytes(16))
    b.point(x8 + 0x04, sets)
    lookup, option = b.add(struct.pack('>I', 1) + bytes(4)), b.add(struct.pack('>I', len(body)) + bytes(4))
    b.point(lookup + 4, option)
    b.point(option + 4, b.add(bytes(body)))
    b.point(sets, lookup)
    if low:
        lookup, option = b.add(struct.pack('>I', 1) + bytes(4)), b.add(struct.pack('>I', len(low)) + bytes(4))
        b.point(lookup + 4, option)
        b.point(option + 4, b.add(bytes(low)))
        b.point(sets + 4, lookup)
    idle, attack = melee_eyes.IDLE_ROWS[0], melee_eyes.IDLE_ROWS[-1] + 1
    actions = b.add(bytes((attack + 1) * melee_eyes.ACTION_BYTES))
    b.point(root + 0x0C, actions)
    texture = lambda frame: struct.pack('>I', melee_eyes.ANIMATE_TEXTURE << 26 | frame)     # noqa: E731
    wait = lambda frames: struct.pack('>I', 0x01 << 26 | frames)                           # noqa: E731
    for k, (name, script) in [
            (idle, ('PlyTest5K_Share_ACTION_Wait1_figatree', texture(1) + wait(2) + texture(2) + wait(4) + texture(1)
                    + wait(1) + texture(0) + bytes(4))),
            (attack, ('PlyTest5K_Share_ACTION_Attack11_figatree', texture(3) + bytes(4)))]:
        b.point(actions + k * melee_eyes.ACTION_BYTES, b.add(name.encode() + b'\0'))
        b.point(actions + k * melee_eyes.ACTION_BYTES + 0x0C, b.add(script))
    return b.save(path, [('ftDataTest', root)])


DAE = '''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<library_geometries>{geometries}</library_geometries>
<library_visual_scenes><visual_scene id="Scene" name="Scene">
<node id="JOBJ_0" name="JOBJ_0" sid="JOBJ_0" type="JOINT"/>{nodes}
</visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>'''


def _dae(path, meshes):
    """A .dae with mesh nodes ``meshes`` [(name, x)]: each one triangle near x, its own material."""
    geometries, nodes = '', ''
    for name, x in meshes:
        geometries += (f'<geometry id="{name}-mesh"><mesh><source id="{name}-pos"><float_array id="{name}-pos-array" '
                       f'count="9">{x} 0 0 {x + .1} 0 0 {x} 1 0</float_array></source>'
                       f'<vertices id="{name}-v"><input semantic="POSITION" source="#{name}-pos"/></vertices>'
                       f'<triangles material="m" count="1"><input semantic="VERTEX" source="#{name}-v" offset="0"/>'
                       f'<p>0 1 2</p></triangles></mesh></geometry>')
        nodes += (f'<node id="{name}" name="{name}"><instance_geometry url="#{name}-mesh"><bind_material>'
                  f'<technique_common><instance_material symbol="m" target="#{name}-material"/></technique_common>'
                  f'</bind_material></instance_geometry></node>')
    Path(path).write_text(DAE.format(geometries=geometries, nodes=nodes), encoding='utf-8')
    return Path(path)


@pytest.mark.parametrize('eye_textures', [1, 2])
@pytest.mark.parametrize('direct, shared_material', [(False, False), (True, False), (True, True)])
def test_the_eyes_take_the_animated_dobjs_with_the_builds_pictures(tmp_path, monkeypatch, eye_textures, shared_material, direct):
    symbol = 'PlyTest_Share_joint'
    # vanilla: five DObjs; the game animates the TObjs of DObjs 1 and 2 (the eyes) and draws all five at spawn
    vanilla = _costume(tmp_path / 'PlTsNr.dat', 6 if direct else 5, animated={1, 2}, eye_textures=eye_textures)
    animated = (1, 1 + eye_textures)
    fighter = _fighter(tmp_path / 'PlTs.dat', animated, [0, 1, 2] if direct else [0, 1, 2, 3, 4],
                       low=(3, 4, 5) if direct else ())
    # the import: three DObjs (the left eye, the right eye, the body), each TObj a 16 x 16 image; vanilla's matanim
    imported = _costume(tmp_path / 'imported.dat', 3, animated={1, 2}, image_size=(16, 16))
    if shared_material:
        d = melee_dat.Dat.read(imported)
        units = melee_dat.joint_dobjs(d, melee_dat.joints(d, symbol)[0])
        d.point(units[1] + 8, d.ptr(units[0] + 8))
        d.save(imported)
    # the build: its eye objects and the three pictures (open red, half green, closed blue)
    build = tmp_path / 'build'
    (build / 'eyes').mkdir(parents=True)
    for state, colour in (('open', (255, 0, 0, 255)), ('half', (0, 255, 0, 255)), ('closed', (0, 0, 255, 255))):
        Image.new('RGBA', (16, 16), colour).save(build / 'eyes' / f'{state}.png')
    (build / 'eyes.json').write_text(json.dumps({
        'objects': [{'name': 'eyes_left', 'side': 'left'}, {'name': 'eyes_right', 'side': 'right'}],
        'frames': {s: f'eyes/{s}.png' for s in eyes.STATES}, 'made': {}}), encoding='utf-8')
    dae = _dae(tmp_path / 'model.dae', [('eyes_left', 0), ('eyes_right', 0), ('body', 0)])
    if shared_material:
        dae.write_text(dae.read_text(encoding='utf-8').replace('eyes_right-material', 'eyes_left-material'),
                       encoding='utf-8')
    # where the vanilla eyes are: DObj 1 on the character's right (-x), DObj 2 on its left
    vanilla_dae = _dae(tmp_path / 'vanilla.dae', [(f'Joint_0_Object_{k}', x) for k, x in
                                                   ((0, 0.), (1, -1.), (2, 1.), (3, 0.), (4, 0.))])

    if direct:
        # One ordinary place cannot temporarily hold the body and two eyes. Their final animated places can.
        with pytest.raises(Failed, match=f'{3 - shared_material} DObjs.*1 places'):
            export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat')
    placed = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, tmp_path / 'out.dat',
                                            deferred=(0, 1) if direct else ())
    assert placed['places'] == ([None, None, 0] if direct else [3, 4, 0])
    if direct:
        # The serialized plan is the one apply must honor, without deciding the destinations again.
        planned = melee_eyes.plan(build, vanilla, fighter, symbol, vanilla_dae)
        placed['eye_plan'] = json.loads(json.dumps(planned))
        assert planned['destinations'] == {'eyes_left': 2, 'eyes_right': 1}
        monkeypatch.setattr(melee_eyes, 'plan', lambda *args: pytest.fail('eye destinations were decided twice'))

    rec = melee_eyes.apply(build, tmp_path / 'out.dat', vanilla, fighter, symbol, placed, dae, vanilla_dae)
    assert rec['added_by_eyes'] == {'materials': int(shared_material)}
    before_model = melee_dae.read_dae(dae)
    # Independently count materials on drawn high-detail DObjs after applying the eye placement.
    result = melee_dat.Dat.read(tmp_path / 'out.dat')
    drawn_materials = {result.ptr(o + 8) for o in melee_dat.joint_dobjs(result, melee_dat.joints(result, symbol)[0])[:3 if direct else 5]
                       if result.ptr(o + 12) is not None}
    assert len(drawn_materials) == before_model.materials + rec['added_by_eyes']['materials']
    assert rec['blink_frames'] == {'open': 0, 'half': 1, 'closed': 2}
    assert [(m['object'], m['from_place'], m['to_place']) for m in rec['moved']] == [
        ('eyes_left', None if direct else 3, 2), ('eyes_right', None if direct else 4, 1)]
    assert rec['format'] == 'CI8' and rec['size'] == [16, 16]
    assert all(t['images'] == ['open', 'half', 'closed', 'open'] for t in rec['tables'])
    # the four vanilla 8 x 8 images had room for none of the 16 x 16 pictures: they are added
    assert rec['bytes_in_vanilla_images'] == 0 and rec['bytes_added'] > 3 * 256

    before, after = melee_dobjs.layout(vanilla, symbol), melee_dobjs.layout(tmp_path / 'out.dat', symbol)
    assert melee_dobjs.layout_differs(before, after, animated) is None
    assert [s.drawn for s in after] == ([True] * 5 + [not shared_material] if direct
                                      else [True, True, True, False, False])
    d = melee_dat.Dat.read(tmp_path / 'out.dat')
    assert all(at + 4 <= len(d.data) and d.u32(at) < len(d.data) for at in d.relocations)
    dobjs = d.chain(d.ptr(melee_dat.joints(d, symbol)[0] + 0x10), 0x04)
    src = melee_dat.Dat.read(imported)
    src_dobjs = src.chain(src.ptr(melee_dat.joints(src, symbol)[0] + 0x10), 0x04)
    if direct:
        assert placed['places'] == [2, 1, 0] and placed['blank'] == int(shared_material)
        assert rec['freed_places'] == []
        assert placed['low_detail']['left_out'] == []
        for place, ns in placed['low_detail']['places'].items():
            assert d.ptr(dobjs[place] + 8) == src.ptr(src_dobjs[ns[0]] + 8)
            chain = d.chain(d.ptr(dobjs[place] + 12), 4)
            assert len(chain) == len(ns)
            for p, n in zip(chain, ns):
                original = src.ptr(src_dobjs[n] + 12)
                assert d.data[p + 8:p + 24] == src.data[original + 8:original + 24]
    # the left eye's own PObjs now draw at DObj 2, the right eye's at DObj 1
    assert d.ptr(dobjs[2] + 0x0C) == src.ptr(src_dobjs[0] + 0x0C)
    assert d.ptr(dobjs[1] + 0x0C) == src.ptr(src_dobjs[1] + 0x0C)
    for place in (1, 2):
        material = d.ptr(dobjs[place] + 8)
        tobjs = d.chain(d.ptr(material + 8), 4)
        assert len(tobjs) == len(before[place].textures) == eye_textures
        assert [d.u32(t + 8) for t in tobjs] == list(before[place].textures)
        if eye_textures == 2:
            assert d.u32(tobjs[1] + 0x40) == melee_dobjs.TOBJ_PASS
        pobj = d.ptr(dobjs[place] + 12)
        attrs = d.ptr(pobj + 8)
        assert [d.u32(attrs + k * 24) for k in range(5)] == [0, 9, 10, 13, 255]
        uv = d.ptr(attrs + 3 * 24 + 20)
        assert struct.unpack_from('>6f', d.data, uv) == (0., 0., 1., 0., 0., 1.)
        display = d.ptr(pobj + 16)
        assert d.data[display] == 0x90 and d.u16(display + 1) == 3
        assert d.u16(pobj + 12) & 0x2000
        envelope = d.ptr(d.ptr(pobj + 20))
        assert d.ptr(envelope) == melee_dat.joints(d, symbol)[1]
        assert struct.unpack_from('>f', d.data, envelope + 4)[0] == 1.
    # The low-detail/source materials are not expanded along with the animated ones.
    assert all(len(d.chain(d.ptr(d.ptr(o + 8) + 8), 4)) == 1 for o in src_dobjs[:2])

    animations = melee_dobjs._material_animations(d, symbol, melee_dat.joints(d, symbol))
    mats = animations[melee_dat.joints(d, symbol)[0]]
    want = {'open': (255, 0, 0), 'half': (0, 255, 0), 'closed': (0, 0, 255)}
    for place in (1, 2):
        anim = d.chain(d.ptr(mats[place] + 0x08), 0x00)[0]
        for k, state in enumerate(['open', 'half', 'closed', 'open']):
            desc = d.ptr(d.ptr(anim + 0x0C) + 4 * k)
            lut = d.ptr(d.ptr(anim + 0x10) + 4 * k)
            colours = d.data[d.ptr(lut):d.ptr(lut) + 2 * d.u16(lut + 0x0C)]
            px = melee_eyes.decode(d.data[d.ptr(desc):], d.u16(desc + 4), d.u16(desc + 6), d.u32(desc + 8),
                                   colours, d.u32(lut + 4))
            assert px.shape == (16, 16, 4) and (np.abs(px[..., :3].astype(int) - want[state]) <= 8).all()


def test_eye_cleanup_keeps_texture_indices_when_its_source_used_fewer_slots(tmp_path):
    symbol = 'PlyTest_Share_joint'
    vanilla = _costume(tmp_path / 'vanilla.dat', 7, animated={4, 5})
    d = melee_dat.Dat.read(vanilla)
    objects = melee_dat.joint_dobjs(d, melee_dat.joints(d, symbol)[0])
    # Every available place before the eyes has two textures; each incoming material has one.
    for obj in objects[:4]:
        first = d.ptr(d.ptr(obj + 8) + 8)
        extra = d.add(bytes(d.data[first:first + 0x5C]))
        for offset in range(0, 0x5C, 4):
            if first + offset in d.relocations:
                d.relocations.add(extra + offset)
        struct.pack_into('>I', d.data, extra + 8, 1)
        d.point(first + 4, extra)
    d.save(vanilla)
    fighter = _fighter(tmp_path / 'fighter.dat', (8, 9), list(range(7)))
    imported = _costume(tmp_path / 'imported.dat', 7, animated={4, 5}, image_size=(16, 16))
    d = melee_dat.Dat.read(imported)
    objects = melee_dat.joint_dobjs(d, melee_dat.joints(d, symbol)[0])
    d.point(objects[2] + 4, None)  # three incoming DObjs, retaining the vanilla animation tree
    d.save(imported)
    out = tmp_path / 'out.dat'
    placed = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, out)
    assert placed['places'] == [6, 0, 1]
    build = tmp_path / 'build'
    (build / 'eyes').mkdir(parents=True)
    for state in eyes.STATES:
        Image.new('RGBA', (16, 16), (255, 0, 0, 255)).save(build / 'eyes' / f'{state}.png')
    (build / 'eyes.json').write_text(json.dumps({
        'objects': [{'name': 'left', 'side': 'left'}, {'name': 'right', 'side': 'right'}],
        'frames': {s: f'eyes/{s}.png' for s in eyes.STATES}, 'made': {}}), encoding='utf-8')
    dae = _dae(tmp_path / 'model.dae', [('left', 0), ('right', 0), ('body', 0)])
    vanilla_dae = _dae(tmp_path / 'vanilla.dae', [('Joint_0_Object_4', -1), ('Joint_0_Object_5', 1)])
    melee_eyes.apply(build, out, vanilla, fighter, symbol, placed, dae, vanilla_dae)
    before, after = melee_dobjs.layout(vanilla, symbol), melee_dobjs.layout(out, symbol)
    assert not after[0].drawn and len(after[0].textures) == 1
    assert melee_dobjs.layout_differs(before, after, (8, 9)) is None


def test_no_blink_in_the_idle_leaves_the_costume_alone(tmp_path):
    symbol = 'PlyTest_Share_joint'
    vanilla = _costume(tmp_path / 'PlTsNr.dat', 3, animated={1})
    b = _Dat()
    root = b.add(bytes(0x60))
    fighter = b.save(tmp_path / 'PlTs.dat', [('ftDataTest', root)])       # animates nothing
    build = tmp_path / 'build'
    build.mkdir()
    (build / 'eyes.json').write_text(json.dumps({'objects': [{'name': 'eyes', 'side': 'both'}]}), encoding='utf-8')
    out = tmp_path / 'out.dat'
    out.write_bytes(vanilla.read_bytes())
    rec = melee_eyes.apply(build, out, vanilla, fighter, symbol, {'build_dobjs': 1, 'places': [0]}, None)
    assert rec == {'skipped': 'the fighter animates no textures: it has no blink to give the eyes'}
    assert out.read_bytes() == vanilla.read_bytes()
    (build / 'eyes.json').write_text(json.dumps({'skipped': 'the description names no eyes'}), encoding='utf-8')
    assert melee_eyes.apply(build, out, vanilla, fighter, symbol, {}, None) == \
        {'skipped': 'the description names no eyes'}


@pytest.mark.parametrize('invalid', ['joint', 'textures', 'chain'])
def test_deferred_eyes_cannot_be_silently_dropped_or_overfill_an_animated_place(tmp_path, monkeypatch, invalid):
    symbol = 'PlyTest_Share_joint'
    vanilla = _costume(tmp_path / 'vanilla.dat', 3, animated={1, 2})
    fighter = _fighter(tmp_path / 'fighter.dat', (1, 2), [0, 1, 2])
    imported = _costume(tmp_path / 'imported.dat', 3, animated={1, 2})
    out = tmp_path / 'out.dat'
    placed = export_write.fill_vanilla_dobjs(imported, vanilla, fighter, symbol, out, deferred=(0, 1))
    (tmp_path / 'eyes.json').write_text(json.dumps({'objects': [
        {'name': 'left', 'side': 'left'}, {'name': 'right', 'side': 'right'}]}), encoding='utf-8')
    dae = _dae(tmp_path / 'model.dae', [('left', 0), ('right', 0), ('body', 0)])
    if invalid == 'joint':
        _, obj = placed['deferred'][0]
        placed['deferred'][0] = (1, obj)
        reason = 'another joint'
    elif invalid == 'textures':
        d = melee_dat.Dat.read(out)
        d.point(d.ptr(placed['deferred'][0][1] + 8) + 8, None)
        d.save(out)
        reason = 'incompatible textures'
    else:
        # Without a side-specific destination the two eyes share one animated place: two one-PObj chains.
        monkeypatch.setattr(melee_dobjs, 'MAX_POBJS', 1)
        reason = 'would draw 2 PObjs; limit 1'
    before = out.read_bytes()
    with pytest.raises(Failed, match=reason):
        melee_eyes.apply(tmp_path, out, vanilla, fighter, symbol, placed, dae)
    assert out.read_bytes() == before


def test_the_import_order_of_dobjs_follows_the_dae(tmp_path):
    dae = _dae(tmp_path / 'm.dae', [('eyes_left', 0), ('main', 1)])
    assert melee_eyes.dae_dobjs(dae) == [('eyes_left', 'eyes_left-material'), ('main', 'main-material')]


@pytest.mark.plugin
@pytest.mark.parametrize('textured_first', [False, True])
def test_export_of_joined_pobjs_with_different_uv_and_colour_channels(tmp_path, textured_first):
    if not Path(os.environ.get(export_write.HSDRAW_VAR) or export_write.HSDRAW_DEFAULT).is_file():
        pytest.skip('needs HSDRawViewer with --model export')
    symbol = 'PlyTest_Share_joint'
    costume = _costume(tmp_path / 'mixed.dat', 1, animated={0}, image_size=(16, 16))
    d = melee_dat.Dat.read(costume)
    tree = melee_dat.joints(d, symbol)
    dobj = melee_dat.joint_dobjs(d, tree[0])[0]
    textured = d.ptr(dobj + 12)
    untextured = _triangle(d, tree[1], uv=False, colour=True)
    first, second = (textured, untextured) if textured_first else (untextured, textured)
    d.point(dobj + 12, first)
    d.point(first + 4, second)
    d.save(costume)
    dae = export_write.export_dae(costume, symbol, tmp_path / 'mixed.dae')
    model = melee_dae.read_dae(dae)
    assert model.triangles == 2
    # Both triangles survive; existing channels retain their values and missing ones get neutral values.
    ns = melee_eyes.NS
    root = ET.parse(dae).getroot()
    mesh = next(root.iter(ns + 'mesh'))
    inputs = {i.get('semantic'): i.get('source')[1:] for i in mesh.iter(ns + 'input')}
    sources = {s.get('id'): s for s in mesh.findall(ns + 'source')}
    uv = np.fromstring(sources[inputs['TEXCOORD']].find(ns + 'float_array').text, sep=' ').reshape(-1, 2)
    colours = np.fromstring(sources[inputs['COLOR']].find(ns + 'float_array').text, sep=' ').reshape(-1, 4)
    assert {tuple(v) for v in uv} == {(0., 1.), (1., 1.), (0., 0.)}   # export flips V
    drawn_colours = []
    for triangles in mesh.findall(ns + 'triangles'):
        bindings = triangles.findall(ns + 'input')
        stride = max(int(i.get('offset')) for i in bindings) + 1
        offset = int(next(i for i in bindings if i.get('semantic') == 'COLOR').get('offset'))
        indices = np.fromstring(triangles.find(ns + 'p').text, sep=' ', dtype=int).reshape(-1, stride)
        drawn_colours.extend(colours[indices[:, offset]])
    assert len(drawn_colours) == 6
    assert np.count_nonzero(np.all(np.isclose(drawn_colours, [1., 1., 1., 1.]), axis=1)) == 3
    assert np.count_nonzero(np.all(np.isclose(drawn_colours, [1., 128 / 255, 64 / 255, 1.]), axis=1)) == 3
