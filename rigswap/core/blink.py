"""The pictures of a blink: the eyes open, half closed and closed, as a game that blinks by swapping the eye's image
needs them (plugins/melee/export/eyes.py reads them, and redraws a drawn state at the size it shows: pictures_at).

The durable facts, lifted from the old middle (details/eyes):
- The eye's picture is the part of its image the eye faces draw: their footprint, each triangle read as the image's
  wrap draws it (tiled), with CROP_MARGIN texels round it for the texture filter's reach. The eye objects' UVs move
  into that crop.
- A rip keeps the eye's other states in one of two places. In the same image: an atlas with the eyes drawn open, half
  shut and shut side by side, found by masked colour difference (a place is alike when its difference is under what a
  flat patch of the picture's own average colour scores), its mirror image too, since a rip may keep the other eye
  flipped. Or in an image file of the download the model doesn't load: one of the same size sharing SAME_LAYOUT of its
  texels (each channel within SAME_TEXEL) is that picture with something drawn differently, and the eye's place in it
  is the eye in another state.
- A state the download lacks is drawn from the open picture: half closed, its top half in the lid's colour, the lid's
  edge soft over SOFT of its height; closed, all of it in the lid's colour with a line LINE thick at LINE_SHADE of its
  brightness across the middle. The lid comes down from the side of the picture the eye's top is on, on the model: a
  texture may lie on the model turned.
- The lid's colour is the download's half closed picture's own lid (its texels that differ from the open one), else
  the model's surface bordering the eye (eyes.lid_on_model); never the image round the open eye, which is padding or
  another island.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

STATES = ('open', 'half', 'closed')
CROP_MARGIN = 2         # texels kept round what the eye faces draw
SEARCH_SIDE = 512       # px: an image is searched at most this large
OVERLAP = 0.25          # a place found may cover at most this share of its box over one already taken
GREY = 0.5              # what a texel's transparent part counts as, for the colour difference
SAME_LAYOUT = 0.5       # another image is the same picture in another state when this share of its texels are the same
SAME_TEXEL = 24         # ... each channel within this many steps of 255
SOFT = 1 / 16           # the half closed lid's edge blends over this share of the picture's height
LINE = 1 / 16           # the closed lid's line, this share of the height (a texel at least)
LINE_SHADE = 0.45       # ... at this share of the lid's brightness
TURNS = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}    # quarter turns, anticlockwise, bringing a side to the top


# ---- what faces draw ----------------------------------------------------------------------------------------------

def tiled(corners, extension):
    """A face's UVs (..., n, 2) as an image of that wrap (a Blender extension) draws them. Every tile of a repeating
    image is alike, so the face moves by whole tiles to the tile of its lowest corner. A clamped image has one tile and
    draws its edge past it, so the face stays where it is: a rip's eye faces run past the eye's picture onto the edge's
    skin colour, and moved they would draw the eye again."""
    if extension == 'EXTEND':
        return corners
    return corners - np.floor(corners.min(axis=-2, keepdims=True))


def footprint(tris, size, extension):
    """The texels of an image of ``size`` (w, h) and wrap ``extension`` the triangles ``tris`` (k, 3, 2) in UV space
    cover: (h, w) bool."""
    w, h = size
    img = Image.new('L', (w, h), 0)
    if len(tris):
        moved = tiled(tris, extension)
        pts = np.stack([moved[..., 0] * w, (1. - moved[..., 1]) * h], -1)
        draw = ImageDraw.Draw(img)
        for tri in pts:
            draw.polygon([tuple(p) for p in tri], fill=255)
    return np.asarray(img) > 0


def islands(mask):
    """The parts of ``mask`` (h, w) bool, texels touching at a side or a corner: [(h, w) bool], the largest first."""
    runs = []
    for y in np.nonzero(mask.any(axis=1))[0]:
        step = np.diff(np.concatenate([[0], mask[y].astype(np.int8), [0]]))
        runs += [(int(y), int(a), int(b)) for a, b in zip(np.nonzero(step == 1)[0], np.nonzero(step == -1)[0])]
    parent = list(range(len(runs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    by_row = {}
    for k, (y, _, _) in enumerate(runs):
        by_row.setdefault(y, []).append(k)
    for y, mine in by_row.items():
        for k in mine:
            for j in by_row.get(y + 1, ()):
                if runs[j][1] <= runs[k][2] and runs[k][1] <= runs[j][2]:
                    parent[find(j)] = find(k)
    parts = {}
    for k, (y, a, b) in enumerate(runs):
        parts.setdefault(find(k), np.zeros_like(mask))[y, a:b] = True
    return sorted(parts.values(), key=lambda p: -p.sum())


def inside(mask):
    """The texels of ``mask`` whose four neighbours are in it too. A part with none (a sliver a seam split off) is no
    picture: any patch of its colour looks as like it as it does."""
    p = np.pad(mask, 1)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def crop_box(mask, margin=CROP_MARGIN):
    """The box of ``mask`` grown by ``margin`` texels, inside the image."""
    x, y, w, h = box_of(mask)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    return [x0, y0, min(mask.shape[1], x + w + margin) - x0, min(mask.shape[0], y + h + margin) - y0]


def crop_uvs(uv, size, crop, extension):
    """One face's UVs (n, 2) in an image of ``size`` and wrap ``extension``, moved into the image ``crop`` cuts out
    (tiled first, as the footprint reads them)."""
    w, h = size
    cx, cy, cw, ch = crop
    moved = tiled(uv, extension)
    return np.stack([(moved[:, 0] * w - cx) / cw, 1. - ((1. - moved[:, 1]) * h - cy) / ch], -1)


def layered(base, top, tris_base, tris_top, extension):
    """``base`` (h, w, 4) of wrap ``extension`` with ``top`` laid over it, alpha over, where the triangles draw: each
    texel of the base the triangles ``tris_base`` (k, 3, 2 UVs) cover takes the texel of ``top`` at the same point of
    the triangle in ``tris_top`` (its UVs on another map). A material that draws one image through one UV map and a
    second over it through another becomes one image on the first map."""
    out = base.astype(float)
    h, w = base.shape[:2]
    th, tw = top.shape[:2]
    upper = top.astype(float)
    for a, b in zip(tris_base, tris_top):
        a = tiled(a, extension)
        q = np.stack([a[:, 0] * w, (1 - a[:, 1]) * h], -1)
        x0, y0 = np.maximum(np.floor(q.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int), [w - 1, h - 1])
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        p = np.stack([xs.ravel() + .5, ys.ravel() + .5], -1)
        t = np.array([q[1] - q[0], q[2] - q[0]]).T
        if abs(np.linalg.det(t)) < 1e-12:
            continue
        l12 = np.linalg.solve(t, (p - q[0]).T).T
        bc = np.column_stack([1 - l12.sum(1), l12])
        on = (bc >= -1e-6).all(1)
        uv = bc[on] @ b
        c = upper[np.clip(((1 - uv[:, 1] % 1) * th).astype(int), 0, th - 1),
                  np.clip((uv[:, 0] % 1 * tw).astype(int), 0, tw - 1)]
        px, py = p[on, 0].astype(int), p[on, 1].astype(int)
        al = c[:, 3:] / 255.
        out[py, px, :3] = out[py, px, :3] * (1 - al) + c[:, :3] * al
    return out.round().astype(np.uint8)


# ---- the search for the other states ------------------------------------------------------------------------------

def flat(pixels):
    """(h, w, 4) uint8 -> (h, w, 3) colours 0-1, a texel's transparent part counted as GREY."""
    p = pixels.astype(float) / 255.
    return p[..., :3] * p[..., 3:] + GREY * (1. - p[..., 3:])


def same_layout(a, b):
    """The share of the texels of two pictures of one size that are the same, each channel within SAME_TEXEL."""
    return float((np.abs(a.astype(np.int16) - b).max(axis=2) <= SAME_TEXEL).mean())


def _correlate(a, k, valid):
    got = np.fft.irfft2(np.fft.rfft2(a, a.shape) * np.conj(np.fft.rfft2(k, a.shape)), a.shape)
    return got[:valid[0], :valid[1]]


def masked_difference(base, template, mask):
    """The mean squared colour difference between ``template`` (h, w, 3) on its ``mask`` texels and ``base`` (H, W, 3)
    at every place it fits, all at once by FFT correlation: (H - h + 1, W - w + 1)."""
    (H, W), (h, w) = base.shape[:2], mask.shape
    valid = (H - h + 1, W - w + 1)
    m = mask.astype(float)
    total = _correlate((base ** 2).sum(2), m, valid)
    for c in range(3):
        total -= 2 * _correlate(base[..., c], template[..., c] * m, valid)
    total += ((template ** 2).sum(2) * m).sum()
    return np.maximum(total, 0.) / max(m.sum() * 3, 1.)


def _overlaps(valid, size, boxes):
    w, h = size
    ys, xs = np.arange(valid[0]), np.arange(valid[1])
    out = np.zeros(valid, bool)
    for bx, by, bw, bh in boxes:
        oy = np.clip(np.minimum(ys + h, by + bh) - np.maximum(ys, by), 0, None)
        ox = np.clip(np.minimum(xs + w, bx + bw) - np.maximum(xs, bx), 0, None)
        out |= np.outer(oy, ox) > OVERLAP * w * h
    return out


def find_alike(pixels, box, mask, taken, most=4):
    """The places in ``pixels`` (an image, (h, w, 4)) where its part ``box`` (on its ``mask`` texels, (h, w) bool of
    the box) or the mirror image of it looks alike, best first, none over ``taken`` or each other by more than OVERLAP;
    searched at SEARCH_SIDE. -> [(box, mirrored)]"""
    scale = min(1., SEARCH_SIDE / max(pixels.shape[:2]))
    img = Image.fromarray(pixels, 'RGBA')
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BOX)
    base = flat(np.asarray(img))
    x, y, w, h = box
    sx, sy, sw, sh = int(x * scale), int(y * scale), max(1, round(w * scale)), max(1, round(h * scale))
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((sw, sh), Image.BOX)) >= 128
    if not m.any() or sh > base.shape[0] - sy or sw > base.shape[1] - sx:
        return []
    template = base[sy:sy + sh, sx:sx + sw]
    texels = template[m]
    bar = float(((texels - texels.mean(0)) ** 2).mean())
    straight = masked_difference(base, template, m)
    mirror = masked_difference(base, template[:, ::-1], m[:, ::-1])
    diff, mirrored = np.minimum(straight, mirror), mirror < straight
    blocked = _overlaps(diff.shape, (sw, sh), [[round(v * scale) for v in b] for b in taken] + [[sx, sy, sw, sh]])
    out = []
    while len(out) < most:
        open_ = np.where(blocked, np.inf, diff)
        fy, fx = np.unravel_index(int(np.argmin(open_)), open_.shape)
        if not open_[fy, fx] < bar:
            break
        out.append(([round(fx / scale), round(fy / scale), w, h], bool(mirrored[fy, fx])))
        blocked |= _overlaps(diff.shape, (sw, sh), [(fx, fy, sw, sh)])
    return out


# ---- the lid ------------------------------------------------------------------------------------------------------

def from_half(opened, half):
    """The lid in a half closed picture ``half`` of the open picture ``opened`` (both (h, w, 4)): the median of its
    texels that differ from the open picture's. -> (4,) uint8, or None where none differ."""
    differ = np.abs(half.astype(np.int16) - opened).max(axis=2) > SAME_TEXEL
    if not differ.any():
        return None
    return np.median(half[differ].reshape(-1, 4), axis=0).round().astype(np.uint8)


def half_closed(region, skin):
    h = region.shape[0]
    soft = max(1., SOFT * h)
    lid = np.clip((h / 2 - (np.arange(h) + 0.5)) / soft + 0.5, 0., 1.)[:, None, None]
    return (region.astype(float) * (1 - lid) + skin.astype(float) * lid).round().astype(np.uint8)


def closed(region, skin):
    h = region.shape[0]
    out = np.empty_like(region)
    out[:] = skin
    thick = max(1, round(LINE * h))
    top = (h - thick) // 2
    out[top:top + thick, :, :3] = (skin[:3].astype(float) * LINE_SHADE).round().astype(np.uint8)
    out[top:top + thick, :, 3] = 255
    return out


def lidded(state, region, skin, up='top'):
    """``region`` (h, w, 4), one eye open, drawn ``state`` with the lid coming down from its side ``up``."""
    k = TURNS[up]
    drawn = (half_closed if state == 'half' else closed)(np.rot90(region, k), skin)
    return np.ascontiguousarray(np.rot90(drawn, -k))


def fitted(content, w, h, mirrored=False):
    img = Image.fromarray(np.ascontiguousarray(content), 'RGBA')
    if mirrored:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(img if img.size == (w, h) else img.resize((w, h), Image.LANCZOS))


# ---- the three pictures -------------------------------------------------------------------------------------------

def blink(base, crop, parts, found, skin):
    """The three pictures, each the part ``crop`` of the eye's image ``base`` (h, w, 4): {state: pixels} and {state:
    how it was made}, and what drawn_states records. ``parts``: the eyes in the image, [{'box', 'up'}]; ``found``:
    {state: [(pixels of its image, box, the part it is laid on, mirrored)]}, the download's own pictures of a state;
    a state with none is drawn on each part in the lid's colour ``skin``, or is the open picture with no lid."""
    cx, cy, cw, ch = crop
    pictures, made, drawn = {'open': base[cy:cy + ch, cx:cx + cw].copy()}, {'open': 'the eye texture'}, {}
    for state in STATES[1:]:
        full = base.copy()
        if found.get(state):
            for px, (x, y, w, h), k, mirrored in found[state]:
                bx, by, bw, bh = parts[k]['box']
                full[by:by + bh, bx:bx + bw] = fitted(px[y:y + h, x:x + w], bw, bh, mirrored)
            made[state] = 'the download\'s own picture, laid on the open eye' + ('s' if len(parts) > 1 else '')
        elif skin is not None:
            for part in parts:
                x, y, w, h = part['box']
                full[y:y + h, x:x + w] = lidded(state, base[y:y + h, x:x + w], skin, part['up'])
            made[state] = f'drawn from the open eye in the lid\'s colour {tuple(int(c) for c in skin)}'
            drawn[state] = {'parts': [{'box': [p['box'][0] - cx, p['box'][1] - cy, *p['box'][2:]], 'up': p['up']}
                                      for p in parts], 'skin': [int(c) for c in skin]}
        else:
            made[state] = 'the open eye: no picture of it and no lid to draw it in'
        pictures[state] = full[cy:cy + ch, cx:cx + cw].copy()
    return pictures, made, drawn


def pictures_at(build_dir, record, size):
    """The three pictures of the build's blink (``record``, its eyes.json) at ``size`` (w, h), the size a game shows
    them: {state: (h, w, 4) uint8}. A picture of the download is resized; a state drawn from the open picture
    (``record['drawn']``) is drawn again at that size on the open picture resized, its parts' boxes at that size with
    their texels whole, so the lid is drawn in the texels the game shows and the closed lid's line stays a texel thick:
    drawn at the texture's size and then resized, a line under a texel of the shown size blurs away (Goku's eyes on
    Zelda, 2026-10: a line 2 texels thick in a picture 155 tall, shown 32 tall, and no blink showed in game)."""
    build_dir, size = Path(build_dir), tuple(int(v) for v in size)

    def resized(path):
        with Image.open(path) as img:
            return np.asarray(img.convert('RGBA').resize(size, Image.LANCZOS))
    drawn = record.get('drawn') or {}
    out = {state: resized(build_dir / record['frames'][state]) for state in STATES if state not in drawn}
    if not drawn:
        return out
    with Image.open(build_dir / record['frames']['open']) as img:
        sx, sy = size[0] / img.width, size[1] / img.height
    for state, how in drawn.items():
        picture = out['open'].copy()
        for part in how['parts']:
            x, y, w, h = part['box']
            x0, y0 = math.floor(x * sx + 1e-9), math.floor(y * sy + 1e-9)
            x1, y1 = max(x0 + 1, math.ceil((x + w) * sx - 1e-9)), max(y0 + 1, math.ceil((y + h) * sy - 1e-9))
            vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x1, size[0]), min(y1, size[1])
            if vx1 > vx0 and vy1 > vy0:
                region = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
                region[vy0 - y0:vy1 - y0, vx0 - x0:vx1 - x0] = out['open'][vy0:vy1, vx0:vx1]
                made = lidded(state, region, np.array(how['skin'], np.uint8), part['up'])
                picture[vy0:vy1, vx0:vx1] = made[vy0 - y0:vy1 - y0, vx0 - x0:vx1 - x0]
        out[state] = picture
    return out
