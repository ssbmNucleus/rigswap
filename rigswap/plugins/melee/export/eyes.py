"""The build's eyes as the fighter's own blinking eyes: the export step that puts the build's eye objects (written by
rigswap.core.eyes, listed in ``<build>/eyes.json``) into the DObjs whose textures the game animates, with
the build's three pictures (open, half closed, closed) in the texture animation the vanilla costume gives them.

What the game does, from the decomp (the doldecomp melee project, read 2026-10-10) and the
vanilla files:

  Which textures blink. Pl<code>.dat lists the TObjs the game animates (dobjs.fighter_tables); each must carry
  a texture animation, which the costume's matanim tree gives it by its texture map id (dobjs.py). Mario's are the
  only TObjs of DObjs 14 and 15, his eyes.

  The texture animation (sysdolphin tobj.c HSD_TObjAddAnim, TObjUpdateFunc): an AObj whose image track (TIMG) picks,
  at each frame, an image from the animation's image table, and whose palette track (TCLT), when it has one, picks the
  palette for a palette image from its palette table. Mario's: six images (CI8, 190 x 190, each with its palette) and
  both tracks keyed frame k -> image k, k = 0 to 5, held to frame 10.

  When it blinks. The animation does not play by itself: the fighter's own scripts set its frame with the command
  "Animate Texture" (opcode 0x28, ft/ftaction.c ftAction_800726F4: frame in the low 11 bits). Its idle scripts (the
  action table's IDLE_ROWS) blink: half closed, closed, half closed, open (Mario's Wait1: frame 1, wait 2, frame 2,
  wait 4, frame 1, wait 1, frame 0), and its other scripts set other frames for expressions (Mario's squint and looks
  aside: images 3 to 5).

So the vanilla AObj stays as it is, its keys and timing what the fighter's scripts expect. Only the tables change:
each image the idle blink shows open, half closed or closed takes the build's picture of that state, and every other
image (an expression the build has no picture for) takes the open one. The pictures are written in the vanilla
images' own kind: CI8 with a palette each when the animation has a palette track, else RGB5A3; at the size prep gave
the eye texture (the eye TObj's image), halved while they would not fit where vanilla's eye images were
(``picture_size``); a state the build drew from the open picture (no picture of the download shows it) is drawn again
at that size (rigswap.core.blink: pictures_at), so its lid is drawn in the texels the game shows. They go there (nothing else points at those, and prep's memory budget held them), else at the
end of the file.

The eyes' destinations are planned before ordinary places are allocated. fill_vanilla_dobjs retains their imported
geometry without spending temporary body places; apply puts it straight into the animated DObjs. For callers that
already placed the eyes, their old places become blank (dobjs.py's rules). Each eye takes the animated DObj on its side of the face (the
character's left is +x in the costume's model space), an eye for both sides to the first; several to one place draw
from one DObj, the later ones' PObjs copied into its chain. The eye material's TObj takes the texture map id of the
vanilla TObj at that place, so the matanim tree pairs with it. When vanilla has extra TObjs (Zelda's specular map),
a single-texture eye material keeps those slots as passthroughs. The low-detail places drawing the eyes keep
drawing them, open."""
from __future__ import annotations

import bisect
import json
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from ....core.blink import pictures_at
from ... import Failed
from ..dae import NS
from ..dat import Dat, joints, joint_dobjs
from . import dobjs as melee_dobjs

STATES = ('open', 'half', 'closed')

# ---- the fighter's scripts ----------------------------------------------------------------------------------------
# Each script command's size in bytes, by its opcode (its first byte's top six bits): HSDRawViewer's
# Scripts/command_fighter.yml (what its script editor compiles with), the same table the old importer read
# (randall vault/echo/_command_table.py). A command past the table stops the walk of that script.
COMMAND_BYTES = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 8, 6: 4, 7: 8, 8: 4, 9: 4, 10: 20, 11: 20, 12: 4, 13: 4, 14: 4,
                 15: 4, 16: 4, 17: 12, 18: 4, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4,
                 29: 4, 30: 4, 31: 4, 32: 4, 33: 4, 34: 12, 35: 4, 36: 4, 37: 4, 38: 28, 39: 16, 40: 4, 41: 4, 42: 4,
                 43: 4, 44: 4, 45: 4, 46: 4, 47: 4, 48: 4, 49: 4, 50: 4, 51: 4, 52: 4, 53: 4, 54: 12, 55: 12, 56: 8,
                 57: 4, 58: 16}
END, SUBROUTINE, RETURN, GO_TO, ANIMATE_TEXTURE = 0x00, 0x05, 0x06, 0x07, 0x28
FRAME_BITS = 0x7FF          # Animate Texture: the frame, its low 11 bits
TEXTURE_SHIFT, TEXTURE_BITS = 18, 0x7F      # ... and which of the fighter's animated textures, bits 18-24
ACTION_BYTES = 0x18         # ftData +0x0C: the actions, each +0x00 its animation's name, +0x0C its script
# The idle rows of that table. Its rows are the game's own, the same in every fighter's file (ft/chara/ftCommon/
# forward.h ftCo_Submotion: Wait1_0, Wait2, two more, and Wait1_1, the idle holding an item). A row's name can't tell
# them: it names the row's animation, and a row with no animation of its own has no name, as all of Nana's but her
# eight specials, though each has its script (ftData_80085FD4 takes only the animation from Popo's row).
IDLE_ROWS = range(2, 7)

# ---- texture animations (sysdolphin) ------------------------------------------------------------------------------
#   texture animation (0x18): +0x00 next, +0x04 texture map id, +0x08 AObj, +0x0C image table, +0x10 palette table,
#                             +0x14 u16 images, +0x16 u16 palettes
#   AObj (0x10): +0x00 flags, +0x04 end frame, +0x08 first FObj
#   FObj (0x14): +0x00 next, +0x04 bytes of keys, +0x08 start frame, +0x0C u8 track, +0x0D u8 value format,
#                +0x0E u8 slope format, +0x10 the keys
#   image (0x18): +0x00 pixels, +0x04 u16 width, +0x06 u16 height, +0x08 format, +0x0C mipmap, +0x10, +0x14 LOD
#   palette (0x10): +0x00 colours, +0x04 format, +0x08 TLUT name, +0x0C u16 colours
TRACK_IMAGE, TRACK_PALETTE = 1, 10          # HSD_A_T_TIMG, HSD_A_T_TCLT
OP_CON, OP_LIN, OP_SPL0, OP_SPL, OP_SLP, OP_KEY = 1, 2, 3, 4, 5, 6
GX_RGB5A3, GX_CI8, GX_CMPR = 5, 9, 14
TLUT_RGB565, TLUT_RGB5A3 = 1, 2
GX_BLOCK = {GX_RGB5A3: (4, 4, 16), GX_CI8: (8, 4, 8), 14: (8, 8, 4), 0: (8, 8, 4), 1: (8, 4, 8), 2: (8, 4, 8),
            3: (4, 4, 16), 4: (4, 4, 16), 6: (4, 4, 32), 8: (8, 8, 4), 10: (4, 4, 16)}   # format: block w, h, bits
PALETTE_COLOURS = 256
ALIGN = 32
SMALLEST_SIDE = 32          # px: the pictures are not halved below this side to fit vanilla's room; they are added
                            # instead


# ---- the idle blink -----------------------------------------------------------------------------------------------

def _ends(d):
    """Every place a pointer or a root points at, sorted: where one structure of the file ends and the next begins
    (as HSDRaw cuts the data block)."""
    return sorted({d.u32(r) for r in d.relocations} | {offset for _, offset in d.roots})


def _script_frames(d, at, ends, seen=None):
    """[(texture, frame)] each Animate Texture of the script at ``at`` sets, in order, through its subroutines and
    jumps."""
    seen = set() if seen is None else seen
    out = []
    if at is None or at in seen:
        return out
    seen.add(at)
    end = ends[bisect.bisect_right(ends, at)] if bisect.bisect_right(ends, at) < len(ends) else len(d.data)
    while at < end:
        op = d.data[at] >> 2
        size = COMMAND_BYTES.get(op)
        if size is None or at + size > end:
            break
        word = d.u32(at)
        if op == ANIMATE_TEXTURE:
            out.append(((word >> TEXTURE_SHIFT) & TEXTURE_BITS, word & FRAME_BITS))
        elif op in (SUBROUTINE, GO_TO):
            out += _script_frames(d, d.ptr(at + 4), ends, seen)
            if op == GO_TO:
                break
        elif op in (END, RETURN):
            break
        at += size
    return out


def idle_frames(fighter_dat):
    """{idle row of the action table: [(texture, frame)]}: what the fighter's idle scripts (IDLE_ROWS) set its animated
    textures to."""
    d = Dat.read(fighter_dat)
    root = next((offset for name, offset in d.roots if name.startswith('ftData')), None)
    if root is None:
        raise Failed(f'{Path(fighter_dat).name} has no ftData root')
    table = d.ptr(root + 0x0C)
    if table is None:
        return {}
    ends = _ends(d)
    return {row: _script_frames(d, d.ptr(table + row * ACTION_BYTES + 0x0C), ends) for row in IDLE_ROWS}


def blink_of(runs):
    """The blink in ``runs`` (lists of frames, one per script and texture): a frame x, then y, then x again, then r,
    x, y and r all different, as a blink steps half closed, closed, half closed, open; the one the scripts step through
    most (the first of those on a tie). A frame set again at once is no step. -> {open, half, closed} or None"""
    found = {}
    for run in runs:
        frames = [f for k, f in enumerate(run) if k == 0 or f != run[k - 1]]
        for k in range(len(frames) - 3):
            x, y, back, rest = frames[k:k + 4]
            if x == back and len({x, y, rest}) == 3:
                found[rest, x, y] = found.get((rest, x, y), 0) + 1
    if not found:
        return None
    best = max(found, key=lambda key: (found[key], -list(found).index(key)))
    return dict(zip(STATES, best))


def idle_blink(fighter_dat):
    """The frames the fighter's idle blink shows its eyes open, half closed and closed, or None when its idle never
    blinks its eye texture; and what each idle script set."""
    frames = idle_frames(fighter_dat)
    runs = []
    for sets in frames.values():
        for texture in sorted({t for t, _ in sets}):
            runs.append([f for t, f in sets if t == texture])
    return blink_of(runs), frames


# ---- reading a texture animation's keys ---------------------------------------------------------------------------

def _value(data, at, fmt):
    """A key's value as fobj.c parseFloat reads it: little-endian, a float or an integer over 2**(fmt & 31)."""
    kind, frac = fmt & 0xE0, fmt & 0x1F
    if kind == 0:
        return struct.unpack_from('<f', data, at)[0], at + 4
    size, code = {0x20: (2, '<h'), 0x40: (2, '<H'), 0x60: (1, 'b'), 0x80: (1, 'B')}.get(kind, (0, None))
    if code is None:
        raise Failed(f'a key value of unknown format {fmt:#x}')
    return struct.unpack_from(code, data, at)[0] / (1 << frac), at + size


def _varint(data, at):
    value, shift = 0, 0
    while True:
        b = data[at]
        at += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, at


def track_keys(keys, start, value_format, slope_format):
    """The keys of one FObj as [(frame, value)], each value held from its frame (fobj.c FObjLoadData)."""
    out, at, frame = [], 0, start
    while at < len(keys):
        op = keys[at] & 0x0F
        b = keys[at]
        at += 1
        count, shift = ((b >> 4) & 7) + 1, 3
        while b & 0x80:
            b = keys[at]
            at += 1
            count += (b & 0x7F) << shift
            shift += 7
        for _ in range(count):
            if op == OP_SLP:
                _, at = _value(keys, at, slope_format)
                continue
            value, at = _value(keys, at, value_format)
            if op == OP_SPL:
                _, at = _value(keys, at, slope_format)
            if op not in (OP_CON, OP_LIN, OP_SPL0, OP_SPL, OP_KEY):
                raise Failed(f'a key of unknown kind {op}')
            wait, at = _varint(keys, at)
            out.append((frame, value))
            frame += wait
    return out


def value_at(keys, frame):
    """What a track shows at ``frame``: the last key at or before it (the first key before any)."""
    shown = keys[0][1] if keys else 0
    for f, value in keys:
        if f <= frame:
            shown = value
    return int(shown)


def tracks(d, texanim):
    """{track: [(frame, value)]} of a texture animation's AObj."""
    aobj = d.ptr(texanim + 0x08)
    out = {}
    for fobj in (d.chain(d.ptr(aobj + 0x08), 0x00) if aobj is not None else []):
        keys = d.ptr(fobj + 0x10)
        size = d.u32(fobj + 0x04)
        start = struct.unpack_from('>f', d.data, fobj + 0x08)[0]
        data = bytes(d.data[keys:keys + size]) if keys is not None else b''
        out[d.data[fobj + 0x0C]] = track_keys(data, start, d.data[fobj + 0x0D], d.data[fobj + 0x0E])
    return out


def states_of(keys, count, blink, pictures=None):
    """For each of ``count`` images a track picks from, the state it shows: the one the blink's open, half closed or
    closed frame shows it in; any other (an expression: eyes squeezed shut, a squint, a look aside) the state whose
    vanilla picture it is nearest in ``openness``, given ``pictures`` (the vanilla images, (h, w, 4) each, None where
    one can't be read), else open."""
    known = {}
    for state in STATES:
        known.setdefault(value_at(keys, blink[state]), state)
    ref = {state: value_at(keys, blink[state]) for state in STATES}
    pics = pictures or []
    usable = len(pics) == count and all(pics[k] is not None for k in ref.values() if k < count)
    level = {state: openness(pics[k], pics[ref['open']], pics[ref['closed']]) for state, k in ref.items()} \
        if usable else {}
    out = []
    for k in range(count):
        o = openness(pics[k], pics[ref['open']], pics[ref['closed']]) \
            if k not in known and usable and pics[k] is not None else None
        out.append(known.get(k) or ('open' if o is None or None in level.values()
                                    else min(STATES, key=lambda s: abs(level[s] - o))))
    return out


def _first_frame(keys, value):
    return next((f for f, v in keys if int(v) == value), None)


def _shown_with(lut_keys, image_keys, count):
    """For each of ``count`` palettes, the image the image track shows at the first frame the palette track picks it
    (None where it never does)."""
    return [None if _first_frame(lut_keys, j) is None else value_at(image_keys, _first_frame(lut_keys, j))
            for j in range(count)]


def vanilla_pictures(d, texanim, keys):
    """The images of a texture animation as pixels, (h, w, 4) each: a palette image with the palette the palette
    track picks at the first frame the image track shows it (else its own number's). None where an image can't be
    read (a format decode does not read)."""
    out = []
    luts = d.u16(texanim + 0x16)
    for k in range(d.u16(texanim + 0x14)):
        desc = d.ptr(d.ptr(texanim + 0x0C) + 4 * k)
        try:
            w, h, fmt = d.u16(desc + 0x04), d.u16(desc + 0x06), d.u32(desc + 0x08)
            colours = colours_fmt = None
            if luts:
                f = _first_frame(keys.get(TRACK_IMAGE, []), k)
                j = min(luts - 1, value_at(keys[TRACK_PALETTE], f) if f is not None and TRACK_PALETTE in keys else k)
                lut = d.ptr(d.ptr(texanim + 0x10) + 4 * j)
                colours, colours_fmt = d.data[d.ptr(lut):d.ptr(lut) + 2 * d.u16(lut + 0x0C)], d.u32(lut + 0x04)
            out.append(decode(d.data[d.ptr(desc):], w, h, fmt, colours, colours_fmt))
        except (Failed, TypeError, ValueError, KeyError):
            out.append(None)
    return out


def openness(picture, opened, shut):
    """How open the eye in ``picture`` is, against the blink's ``opened`` and ``shut`` pictures (each (h, w, 4), resized
    to ``opened``'s size): of the eye's texels (where those two differ, a channel more than SAME_TEXEL apart), the
    share the same as the open picture's among those the same as either's. None when no texel is the same as either's
    (or the two do not differ)."""
    from ....core.blink import SAME_TEXEL
    size = opened.shape[1::-1]

    def fitted(px):
        return np.asarray(Image.fromarray(px, 'RGBA').resize(size, Image.BILINEAR)) if px.shape[1::-1] != size \
            else px
    a, b, c = (fitted(p)[..., :3].astype(np.int16) for p in (picture, opened, shut))
    eye = np.abs(b - c).max(-1) > SAME_TEXEL
    like_open = (np.abs(a - b).max(-1) <= SAME_TEXEL)[eye].sum()
    like_shut = (np.abs(a - c).max(-1) <= SAME_TEXEL)[eye].sum()
    return float(like_open / (like_open + like_shut)) if like_open + like_shut else None


# ---- GX images ----------------------------------------------------------------------------------------------------

def gx_bytes(fmt, width, height):
    bw, bh, bits = GX_BLOCK[fmt]
    return -(-width // bw) * bw * -(-height // bh) * bh * bits // 8


def _blocks(values, bw, bh):
    """(h, w, ...) -> GX block order: blocks of bh rows x bw columns, left to right, top to bottom, each row by row."""
    h, w = values.shape[:2]
    H, W = -(-h // bh) * bh, -(-w // bw) * bw
    padded = np.zeros((H, W) + values.shape[2:], values.dtype)
    padded[:h, :w] = values
    tiles = padded.reshape(H // bh, bh, W // bw, bw, *values.shape[2:]).swapaxes(1, 2)
    return tiles.reshape(-1, *values.shape[2:])


def _unblocks(flat, w, h, bw, bh):
    H, W = -(-h // bh) * bh, -(-w // bw) * bw
    tiles = flat[:H * W].reshape(H // bh, W // bw, bh, bw, *flat.shape[1:]).swapaxes(1, 2)
    return tiles.reshape(H, W, *flat.shape[1:])[:h, :w]


def rgb5a3(rgba):
    """(..., 4) uint8 -> (...) uint16: 1RRRRRGGGGGBBBBB when opaque, else 0AAARRRRGGGGBBBB."""
    r, g, b, a = (rgba[..., i].astype(np.uint16) for i in range(4))
    opaque = 0x8000 | (r >> 3) << 10 | (g >> 3) << 5 | (b >> 3)
    clear = (a >> 5) << 12 | (r >> 4) << 8 | (g >> 4) << 4 | (b >> 4)
    return np.where(a == 255, opaque, clear).astype(np.uint16)


def rgb565(rgba):
    r, g, b = (rgba[..., i].astype(np.uint16) for i in range(3))
    return ((r >> 3) << 11 | (g >> 2) << 5 | (b >> 3)).astype(np.uint16)


def colours_of(values, fmt):
    """uint16 GX colours of palette ``fmt`` (TLUT_RGB565 or TLUT_RGB5A3) -> (..., 4) uint8."""
    v = values.astype(np.int64)
    if fmt == TLUT_RGB565:
        rgb = [((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31, np.full_like(v, 255)]
    else:
        opaque = (v & 0x8000) != 0
        rgb = [np.where(opaque, ((v >> five) & 31) * 255 // 31, ((v >> four) & 15) * 17)
               for five, four in ((10, 8), (5, 4), (0, 0))]
        rgb.append(np.where(opaque, 255, ((v >> 12) & 7) * 255 // 7))
    return np.stack(rgb, -1).astype(np.uint8)


def encode(pixels, palette):
    """(h, w, 4) uint8 -> (pixel bytes, GX format, palette bytes or None, palette format or None): CI8 with a palette of
    PALETTE_COLOURS when ``palette``, RGB565 colours when every pixel is opaque, else RGB5A3; else RGB5A3."""
    h, w = pixels.shape[:2]
    if not palette:
        return _blocks(rgb5a3(pixels), 4, 4).astype('>u2').tobytes(), GX_RGB5A3, None, None
    img = Image.fromarray(pixels, 'RGBA')
    if (pixels[..., 3] == 255).all():
        q = img.convert('RGB').quantize(PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        colours = np.array(q.getpalette()[:3 * PALETTE_COLOURS], np.uint8).reshape(-1, 3)
        colours = np.concatenate([colours, np.full((len(colours), 1), 255, np.uint8)], 1)
        fmt, values = TLUT_RGB565, rgb565
    else:
        q = img.quantize(PALETTE_COLOURS, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        colours = np.array(q.getpalette('RGBA')[:4 * PALETTE_COLOURS], np.uint8).reshape(-1, 4)
        fmt, values = TLUT_RGB5A3, rgb5a3
    table = np.zeros((PALETTE_COLOURS, 4), np.uint8)
    table[:len(colours)] = colours
    index = np.asarray(q, np.uint8)
    return (_blocks(index, 8, 4).tobytes(), GX_CI8, values(table).astype('>u2').tobytes(), fmt)


def decode(data, w, h, fmt, palette=None, palette_format=None):
    """Pixel bytes of GX format RGB5A3, CI8 (with its palette bytes) or CMPR -> (h, w, 4) uint8."""
    if fmt == GX_CMPR:
        return _cmpr(data, w, h)
    if fmt == GX_RGB5A3:
        flat = np.frombuffer(bytes(data[:gx_bytes(fmt, w, h)]), '>u2')
        return _unblocks(colours_of(flat, TLUT_RGB5A3), w, h, 4, 4)
    if fmt == GX_CI8:
        index = np.frombuffer(bytes(data[:gx_bytes(fmt, w, h)]), np.uint8)
        table = colours_of(np.frombuffer(bytes(palette), '>u2'), palette_format)
        return _unblocks(table[index], w, h, 8, 4)
    raise Failed(f'decoding GX format {fmt} is not written here')


def _cmpr(data, w, h):
    """CMPR: 8x8 tiles, each four 4x4 blocks (top left, top right, bottom left, bottom right) of 8 bytes: two RGB565
    colours, then sixteen 2-bit indices row by row; the two colours' thirds between them, or with the first not above
    the second their half and a clear colour."""
    across, down = -(-w // 8), -(-h // 8)
    raw = np.frombuffer(bytes(data[:across * down * 32]), np.uint8).reshape(down, across, 2, 2, 8).astype(np.int64)
    c0, c1 = raw[..., 0] << 8 | raw[..., 1], raw[..., 2] << 8 | raw[..., 3]
    a, b = colours_of(c0, TLUT_RGB565).astype(np.int64), colours_of(c1, TLUT_RGB565).astype(np.int64)
    more = (c0 > c1)[..., None]
    table = np.stack([a, b, np.where(more, (2 * a + b) // 3, (a + b) // 2),
                      np.where(more, (a + 2 * b) // 3, 0)], -2)                    # (down, across, 2, 2, 4, rgba)
    index = (raw[..., 4:8, None] >> np.array([6, 4, 2, 0])) & 3                  # (down, across, 2, 2, row, column)
    texels = np.take_along_axis(table[..., None, None, :, :], index[..., None, None], axis=-2)[..., 0, :]
    # texels: (down, across, block row, block column, row, column, rgba) -> (down * 8, across * 8, rgba)
    return texels.transpose(0, 2, 4, 1, 3, 5, 6).reshape(down * 8, across * 8, 4)[:h, :w].astype(np.uint8)


# ---- the import's DObjs, and where the vanilla eyes are -----------------------------------------------------------

def dae_dobjs(dae):
    """[(mesh node name, material id)] for each DObj HSDRawViewer's import makes from ``dae``, in its order: the mesh
    nodes at the top of the scene in the file's order, each node's triangle lists in order (ModelImporter.cs: one
    DObj for each mesh's polygon group; checked against its log, 2026-10-10)."""
    root = ET.parse(dae).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    out = []
    for scene in root.iter(NS + 'visual_scene'):
        for node in scene.findall(NS + 'node'):
            if node.get('type') == 'JOINT':
                continue
            for inst in node:
                if inst.tag not in (NS + 'instance_controller', NS + 'instance_geometry'):
                    continue
                target = ids.get(inst.get('url', '').lstrip('#'))
                skin = target.find(NS + 'skin') if target is not None else None
                geometry = ids.get(skin.get('source').lstrip('#')) if skin is not None else target
                mesh = geometry.find(NS + 'mesh') if geometry is not None else None
                symbols = {m.get('symbol'): m.get('target', '').lstrip('#')
                           for m in inst.iter(NS + 'instance_material')}
                for prims in (mesh if mesh is not None else []):
                    if prims.tag in (NS + 'triangles', NS + 'polylist') and int(prims.get('count', '0')):
                        out.append((node.get('name'), symbols.get(prims.get('material'), prims.get('material'))))
    return out


def dobj_centre(dae, names):
    """The middle of what the first mesh node of ``dae`` called one of ``names`` draws, at rest in the model's space
    (dae.read_dae on the file with only that node), or None when it has none."""
    from ..dae import read_dae
    tree = ET.parse(dae)
    for scene in tree.getroot().iter(NS + 'visual_scene'):
        meshes = [n for n in scene.findall(NS + 'node') if n.get('type') != 'JOINT']
        keep = next((n for name in names for n in meshes if n.get('name') == name), None)
        if keep is None:
            return None
        for n in meshes:
            if n is not keep:
                scene.remove(n)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(dae).name
        tree.write(path, encoding='utf-8', xml_declaration=True)
        points = read_dae(path).points
    return points.mean(0) if len(points) else None


def _dobj_list(d, symbol):
    return [(k, dobj) for k, j in enumerate(joints(d, symbol)) for dobj in joint_dobjs(d, j)]


def _blank(d, textures, cache):
    """A blank material with ``textures`` (texture map ids), as fill_vanilla_dobjs makes one."""
    if 'image' not in cache:
        data = d.add(bytes(32), melee_dobjs.IMAGE_ALIGN)
        cache['image'] = d.add(struct.pack('>I2H2I2f', 0, 8, 8, melee_dobjs.GX_I4, 0, 0., 0.))
        d.point(cache['image'], data)
        cache['material'] = d.add(struct.pack('>4B4B4B2f', 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255,
                                              255, 1., 50.))
    return melee_dobjs._blank_material(d, tuple(textures), cache['image'], cache['material'], cache)


def _eye_material(d, mobj, textures, animated):
    """A private eye material with vanilla's TObj slots, keeping unused slots as UV0 passthroughs.

    Zelda has an extra specular TObj. A build with one eye texture still needs that slot so the game's
    texture indices stay fixed. Copy descriptors, not image/geometry data; low-detail eyes keep their material.
    """
    source = d.chain(d.ptr(mobj + 0x08), 0x04) if mobj is not None else []
    if not source or (len(source) != len(textures) and len(source) != 1):
        return None, []

    def copy(at, size):
        new = d.add(bytes(d.data[at:at + size]))
        for field in range(0, size, 4):
            if at + field in d.relocations:
                d.relocations.add(new + field)
        return new

    material = copy(mobj, 0x18)
    tobjs = []
    for k, texture in enumerate(textures):
        tobj = copy(source[k] if len(source) == len(textures) else source[0], 0x5C)
        struct.pack_into('>I', d.data, tobj + 0x08, texture)
        if len(source) != len(textures) and k not in animated:
            struct.pack_into('>I', d.data, tobj + 0x40, melee_dobjs.TOBJ_PASS)
            d.point(tobj + 0x58, None)     # no custom TEV overriding the passthrough
        tobjs.append(tobj)
    for k, tobj in enumerate(tobjs):
        d.point(tobj + 0x04, tobjs[k + 1] if k + 1 < len(tobjs) else None)
    d.point(material + 0x08, tobjs[0])
    mode = d.u32(material + 0x04) | sum(melee_dobjs.RENDER_TEX0 << k for k in range(len(textures)))
    struct.pack_into('>I', d.data, material + 0x04, mode)
    return material, tobjs


# ---- apply --------------------------------------------------------------------------------------------------------

def plan(build_dir, vanilla_dat, fighter_dat, symbol, vanilla_dae=None):
    """Decide the eyes' final places before ordinary DObjs are allocated. No temporary body places are needed."""
    path = Path(build_dir) / 'eyes.json'
    eyes = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if not eyes.get('objects'):
        return {'skipped': eyes.get('skipped') or 'the build has no eye objects'}
    tables = melee_dobjs.fighter_tables(fighter_dat)
    vanilla = melee_dobjs.layout(vanilla_dat, symbol)
    spots = [p for p in melee_dobjs.texture_places(vanilla, tables.animated) if p is not None]
    if not spots:
        return {'skipped': 'the fighter animates no textures: it has no blink to give the eyes'}
    blink, idle = idle_blink(fighter_dat)
    if blink is None:
        return {'skipped': "the fighter's idle scripts never blink its animated textures", 'idle': idle}
    places = list(dict.fromkeys(p for p, _ in spots))
    body = set(range(len(vanilla)) if tables.body is None else tables.body)
    if any(p not in body or p in tables.elsewhere or vanilla[p].material_animated for p in places):
        return {'skipped': 'animated eye places are not independent body places with static material colours'}
    sides = {}
    for p in places:
        s = vanilla[p]
        centre = dobj_centre(vanilla_dae, [f'Joint_{s.joint}_Object_{s.local}', f'Joint_{s.joint}_Object_{p}']) \
            if vanilla_dae and Path(vanilla_dae).is_file() else None
        sides[p] = None if centre is None else ('left' if centre[0] > 0 else 'right')
    destinations = {obj['name']: next((p for p in places if sides[p] == obj['side']), places[0])
                    for obj in eyes['objects']}
    return {'destinations': destinations, 'sides': sides, 'blink_frames': blink, 'idle': idle}


def apply(build_dir, out, vanilla_dat, fighter_dat, symbol, placed, dae, vanilla_dae=None):
    """Make the costume ``out`` (fill_vanilla_dobjs has written it; ``placed`` is what it returned) blink with the
    build's eyes, in place, as the module's docstring says. ``dae`` is the .dae HSDRawViewer imported (the order of
    the build's DObjs); ``vanilla_dae`` the vanilla costume's every DObj as HSDRawViewer exports them (where the
    vanilla eyes are), when there is one. -> what was done, or {'skipped': why} with the costume untouched."""
    build_dir = Path(build_dir)
    eyes = json.loads((build_dir / 'eyes.json').read_text(encoding='utf-8'))
    planned = placed.get('eye_plan') or plan(build_dir, vanilla_dat, fighter_dat, symbol, vanilla_dae)
    if 'skipped' in planned:
        return planned
    tables = melee_dobjs.fighter_tables(fighter_dat)
    vanilla = melee_dobjs.layout(vanilla_dat, symbol)
    spots = [p for p in melee_dobjs.texture_places(vanilla, tables.animated) if p is not None]
    blink, idle = planned['blink_frames'], planned['idle']

    order = dae_dobjs(dae)
    if len(order) != placed['build_dobjs']:
        raise Failed(f'{Path(dae).name} makes {len(order)} DObjs and the import has {placed["build_dobjs"]}: the eyes '
                     f'cannot be told apart from the rest')
    d = Dat.read(out)
    listed = _dobj_list(d, symbol)          # the costume's DObjs, by place
    eye_dobjs = []          # (object record, its index among the build's DObjs)
    for obj in eyes['objects']:
        mine = [i for i, (node, _) in enumerate(order) if node == obj['name']]
        if len(mine) != 1:
            raise Failed(f"eye object {obj['name']} makes {len(mine)} DObjs in the import; it should make one")
        eye_dobjs.append((obj, mine[0]))

    sides = {int(p): side for p, side in planned['sides'].items()}
    goes = {}
    for obj, n in eye_dobjs:
        to = planned['destinations'][obj['name']]
        goes.setdefault(to, []).append((obj, n))

    deferred = {int(n): tuple(source) for n, source in placed.get('deferred', {}).items()}
    if deferred and set(deferred) != {n for _, n in eye_dobjs}:
        raise Failed('deferred DObjs do not match the planned eye objects')

    def source(n):
        return deferred[n] if n in deferred else listed[placed['places'][n]]

    # Validate all destinations before changing any. Deferred geometry must never silently disappear on a skip.
    for to, group in goes.items():
        for obj, n in group:
            if source(n)[0] != vanilla[to].joint:
                raise Failed(f"eye object {obj['name']} is on another joint than animated DObj {to}")
            mobj = d.ptr(source(n)[1] + 0x08)
            textures = d.chain(d.ptr(mobj + 0x08), 4) if mobj is not None else []
            if not textures or len(textures) not in (1, len(vanilla[to].textures)):
                raise Failed(f'eye object {obj["name"]} has incompatible textures for animated DObj {to}')
        count = sum(len(d.chain(d.ptr(source(n)[1] + 0x0C), 4)) for _, n in group)
        if count > melee_dobjs.MAX_POBJS:
            raise Failed(f'animated DObj {to} would draw {count} PObjs; limit {melee_dobjs.MAX_POBJS}')

    # Each animated destination gets a private material: the MObjs this adds, by address, against the source DAE's
    # materials (two eyes sharing one source material become two). Not what the read-back counts: that names a
    # material by its bytes, so identical private copies read back as one (Zelda's two eye places) and a copy can read
    # back as the eye's old material (Mario's place 14). The export's material check leaves the eyes out instead
    # (checks.material_check).
    expected_materials = [('source', material) for _, material in order]
    for to, group in goes.items():
        for _, n in group:
            expected_materials[n] = ('eye', to)
    added_materials = len(set(expected_materials)) - len({material for _, material in order})

    # the build's eye DObjs into the animated places; their own places blank
    cache, moved, freed, eye_tobjs = {}, [], set(), []
    for to, group in goes.items():
        first = source(group[0][1])[1]
        mobj = d.ptr(first + 0x08)
        mobj, tobjs = _eye_material(d, mobj, vanilla[to].textures, {t for p, t in spots if p == to})
        eye_tobjs += [tobjs[t] for p, t in spots if p == to]
        head = d.ptr(first + 0x0C) if len(group) == 1 else \
            melee_dobjs._joined_pobjs(d, [source(n)[1] for _, n in group])[0]
        target = listed[to][1]
        d.point(target + 0x08, mobj)
        d.point(target + 0x0C, head)
        for obj, n in group:
            at = placed['places'][n]
            if n not in deferred:
                textures = vanilla[at].textures
                if at < max(p for p, _ in spots):
                    # Before an animated texture, retain the count the placement planner already balanced.
                    source_material = d.ptr(listed[at][1] + 0x08)
                    textures = tuple(d.u32(t + 0x08) for t in d.chain(d.ptr(source_material + 0x08), 0x04))
                d.point(listed[at][1] + 0x08, _blank(d, textures, cache))
                d.point(listed[at][1] + 0x0C, None)
                freed.add(at)
            else:
                placed['places'][n] = to
            moved.append({'object': obj['name'], 'side': obj['side'], 'from_place': at, 'to_place': to,
                          'vanilla_side': sides[to]})

    # the pictures into every animated texture's tables
    animations = melee_dobjs._material_animations(d, symbol, joints(d, symbol))
    tree = joints(d, symbol)
    texanims = []
    for p, t in spots:
        s = vanilla[p]
        mats = animations.get(tree[s.joint], [])
        mat = mats[s.local] if s.local < len(mats) else None
        found = [a for a in (d.chain(d.ptr(mat + 0x08), 0x00) if mat is not None else [])
                 if d.u32(a + 0x04) == s.textures[t]]
        if not found:
            raise Failed(f'the animated TObj {t} of DObj {p} has no texture animation in the costume')
        texanims.append(found[0])
    image = d.ptr(eye_tobjs[0] + 0x4C)
    prep_size = (d.u16(image + 0x04), d.u16(image + 0x06))
    # palette images only when every animation picks a palette with each image: one that doesn't would draw a palette
    # image with none (tobj.c, the assert on line 1595)
    palette = all(TRACK_PALETTE in tracks(d, a) and d.u16(a + 0x16) for a in texanims)
    room = _vanilla_image_room(d, texanims)
    size = picture_size(prep_size, GX_CI8 if palette else GX_RGB5A3, [cap for _, cap in room])
    # at the size they are written: a state the build drew from the open picture is drawn again at that size
    # (core.blink.pictures_at), so its lid is a texel thick at the least where a resize would blur it away
    shown = pictures_at(build_dir, eyes, size)
    pictures = {state: encode(shown[state], palette) for state in STATES}
    # the state each vanilla image (and palette) shows, read off vanilla's own pictures before they are written over
    states = []
    for a in texanims:
        keys = tracks(d, a)
        image_keys = keys.get(TRACK_IMAGE, [])
        images = states_of(image_keys, d.u16(a + 0x14), blink, vanilla_pictures(d, a, keys))
        lut_states = []
        if palette and d.u16(a + 0x16):
            lut_keys = keys.get(TRACK_PALETTE, image_keys)
            lut_states = [images[k] if k is not None and k < len(images) else state for k, state in
                          zip(_shown_with(lut_keys, image_keys, d.u16(a + 0x16)),
                              states_of(lut_keys, d.u16(a + 0x16), blink))]
        states.append((a, images, lut_states))

    descs, palettes, added, reused = {}, {}, 0, 0
    for state, (data, fmt, colours, colours_fmt) in pictures.items():
        spot = next((at for at, cap in room if cap >= len(data)), None)
        if spot is None:
            at = d.add(data, ALIGN)
            added += len(data)
        else:
            room = [(a, c) for a, c in room if a != spot]
            d.data[spot:spot + len(data)] = data
            at, reused = spot, reused + len(data)
        descs[state] = d.add(struct.pack('>I2H2I2f', 0, size[0], size[1], fmt, 0, 0., 0.))
        d.point(descs[state], at)
        added += 0x18
        if colours is not None:
            lut = d.add(colours, ALIGN)
            palettes[state] = d.add(struct.pack('>3IH2x', 0, colours_fmt, 0, len(colours) // 2))
            d.point(palettes[state], lut)
            added += len(colours) + 0x10
    shown = []
    for a, images, lut_states in states:
        for k, state in enumerate(images):
            d.point(d.ptr(a + 0x0C) + 4 * k, descs[state])
        for k, state in enumerate(lut_states):
            d.point(d.ptr(a + 0x10) + 4 * k, palettes[state])
        shown.append({'texture_animation': a, 'images': images, 'palettes': lut_states})
    d.save(out)
    if deferred:
        placed['blank'] -= len(goes)
        placed['longest_chain'] = max(placed['longest_chain'],
                                     max(len(d.chain(d.ptr(listed[p][1] + 0x0C), 4)) for p in goes))
    return {'blink_frames': blink, 'idle': {k: v for k, v in idle.items() if v}, 'moved': moved,
            'added_by_eyes': {'materials': added_materials},
            'freed_places': sorted(freed), 'size': list(size), 'prep_size': list(prep_size),
            'format': 'CI8' if palette else 'RGB5A3', 'drawn_at_size': sorted(eyes.get('drawn') or {}),
            'tables': shown, 'bytes_added': added, 'bytes_in_vanilla_images': reused,
            'made': eyes.get('made'), 'notes': eyes.get('notes', [])}


def picture_size(size, fmt, room, smallest=SMALLEST_SIDE):
    """The size the three pictures are written at: ``size`` (prep's, the eye texture's), halved while one picture
    would not fit the third largest of the vanilla images' places ``room`` (bytes each) and both sides stay at least
    ``smallest``. The pictures then take the room of vanilla's eye images, which prep's memory budget already held
    (Fighter.kept); a fighter with fewer than three vanilla eye images has its pictures added at prep's size."""
    if len(room) < len(STATES):
        return tuple(size)
    fits = sorted(room, reverse=True)[len(STATES) - 1]
    w, h = size
    while gx_bytes(fmt, w, h) > fits and min(w, h) // 2 >= smallest:
        w, h = w // 2, h // 2
    return w, h


def _vanilla_image_room(d, texanims):
    """[(pixels offset, bytes)] of the vanilla images the animations' tables point at that nothing else points at,
    the largest first."""
    descs = {d.ptr(d.ptr(a + 0x0C) + 4 * k) for a in texanims for k in range(d.u16(a + 0x14))}
    pointers = {}
    for r in d.relocations:
        pointers.setdefault(d.u32(r), set()).add(r)
    room = {}
    for desc in descs:
        if desc is None:
            continue
        data = d.ptr(desc)
        if data is None or pointers.get(data, set()) - descs:
            continue
        fmt = d.u32(desc + 0x08)
        if fmt in GX_BLOCK:
            room[data] = max(room.get(data, 0), gx_bytes(fmt, d.u16(desc + 0x04), d.u16(desc + 0x06)))
    return sorted(room.items(), key=lambda kv: -kv[1])
