"""The colours a game paints on a character in play, laid into the build's images (STATE, plan step 4).

Some games colour a character in play: Splatoon paints an Inkling's hair, eyes and clothes in her team's ink. Its colour
image is then plain or near black where the game paints, and a mask beside it in the download says where: Inkling's
``m_teamcolor_alb.png`` is black where ``m_teamcolor_tcl.png`` is white. The model file wires only the colour, normal
and roughness images, so nothing loads the mask, and drawn from her files alone she is black and grey.

Which image is such a mask, and in what colour it paints, is meaning: one ask (route 'core paint', VOTES answers), each
image the build draws its colours from beside its candidates, numbered, and the model in its own colours. A mask is one
more than half the answers name; its colour is the one of the answers' colours nearest the rest, never a blend (Inkling
has no one colour: three answers said pink, green and orange). The candidates are structure: the image files in the
colour image's folder the build doesn't load, with its stem (the file name less its last ``_suffix``, case aside:
``M_TeamColor_Alb`` and ``m_teamcolor_tcl``) and its shape (the same ratio of width to height). An image with none isn't
asked about, and a build with none asks nothing.

The paint is structure: colour image x (1 - m) + colour x m, the mask resampled to the colour image, whose alpha stays.
Where it paints is the files', while they agree: a mask marks with its light, as alpha always does, and the game ships
the colour image blank (near black, NEAR_BLACK) where it paints; where both hold, ``m`` is the mask's light. Where the
image draws its colours under the mask's light and is blank only under its dark, the files contradict themselves, and
only the game knows whether the blank side is a hole left for the colour or something drawn black: Inkling's eye mask
is dark on the iris ring the image leaves black, and her shorts' is light on a white patch and dark on the black cloth,
one picture in mirror. There a second ask (route 'core paint way', VOTES answers) shows the parts drawing that image,
close on the model, as the files draw them and painted on the blank side, and the side paints where every answer
picks it: one doubting answer leaves the files' own picture. (Asked every mask's way, the answers wavered and missed
what the files say plainly: her tank's cap, blank under a graded mask, kept black. A majority painted her shorts orange
two of three, and a mask's light is her shorts' colour: of her ten legwear masks, two are light on parts drawn white.)
An image blank under neither side has nothing missing to paint (Psyduck's Pokemon X/Y shading layers lie over his drawn
yellow body, Bowser Odyssey's metalness map over his dark grey metal).
Every material drawing that image as its colour draws the painted one instead, a packed image of its own; the image and
its other uses stay as they were (load/materials.convert does the same).

A layer's image says only where it shows. Of two copies of one shape in one place both worn (look.layers), one
drawing only images in greys with no alpha of their own lies over the other, and its grey is its alpha: light where it
shows, as alpha always is. Inkling's eyelid band is her eye's shape, over it; its image is a white domino with a black H
where her eyes show through, and its blink frames close the H from below. Its colour is the game's, as a mask's is:
an ask of its own (route 'core layer', VOTES answers), each layer's image lettered beside the model, the colour more
than half the answers give, nearest the rest. (A row in the paint ask made one answer give every row its part's own
colour.) The layer draws a packed image of its own, the colour where its grey is light, clear where dark; with no
colour said, its grey, marked ``layer``. ``paint.json`` beside build.blend says what was painted or layered, or why
nothing was.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from .render import still

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.dds', '.tif', '.tiff', '.webp', '.gif'}
SAME_SHAPE = 0.01       # two images have one shape when their ratios of width to height differ by less than this share
MOST = 8                # candidates offered per colour image, at most
ROWS = 5                # colour images per picture
TILE = 200
TILE_ON_MODEL = 300       # a view of the parts a colour image draws, as drawn or painted (drawn_or_dark)
VOTES = 3
# Near black: the colour image's mean light where a mask paints is under an eighth of white. Inkling's painted sides
# read 2 to 25 of 255. Two of three answers took Psyduck's shading layers for masks, over his drawn body (186) and eyes
# (146), and Bowser Odyssey's metalness map, over his dark grey metal (56).
NEAR_BLACK = 32


def stem(name):
    """A file name less its extension and its last ``_suffix``, lower case; None for a name with no ``_``."""
    s = Path(name).stem.lower()
    return s.rsplit('_', 1)[0] if '_' in s else None


def siblings(colour, sizes, loaded):
    """The candidates for a colour image's mask (the module's docstring): ``colour``, its file; ``sizes``, {file in
    its folder: (width, height)}, the colour image's own among them; ``loaded``, the files the build loads, as
    _key gives them. A name with no stem has none (Kate's are hashes). -> files, sorted, at most MOST."""
    w, h = sizes[colour]
    own = stem(Path(colour).name)
    return sorted(f for f, (fw, fh) in sizes.items()
                  if own and f != colour and _key(f) not in loaded and stem(Path(f).name) == own
                  and abs(fw / fh - w / h) < SAME_SHAPE * w / h)[:MOST]


def tally(answers, rows):
    """Per lettered row ({letter: its candidates' numbers}), the mask more than half the answers name, and of the
    colours the answers give the row, the one nearest the rest (the first, of those as near). A row with no numbers is
    a layer's: its colour alone, given by more than half the answers. -> {letter: (n, (r, g, b))}, n 0 for a layer's;
    a row with no such mask, or no colour, is left out."""
    out = {}
    for letter, numbers in rows.items():
        said = [r for a in answers for r in (a or {}).get('rows') or [] if r.get('row') == letter]
        masks = Counter(r.get('mask') for r in said if r.get('mask') in numbers)
        colours = [c for c in (_rgb(r.get('colour')) for r in said) if c]
        if not colours or numbers and not masks:
            continue
        n, count = masks.most_common(1)[0] if numbers else (0, len(colours))
        if 2 * count > len(answers):
            out[letter] = (n, nearest(colours))
    return out


def nearest(colours):
    """Of the colours (r, g, b), the one nearest the rest (the first, of those as near)."""
    c = np.array(colours, float)
    return colours[int(np.argmin(np.linalg.norm(c[:, None] - c[None], axis=2).sum(axis=1)))]


def under(pixels, mask):
    """The colour image ``pixels`` (h, w, 4) uint8's mean light under ``mask`` (h, w) uint8's light, and under its
    dark: (light, dark), 0 to 255."""
    light = pixels[..., :3].astype(np.float32).mean(axis=2)
    m = mask.astype(np.float32) / 255
    return tuple(float((light * w).sum() / max(float(w.sum()), 1e-6)) for w in (m, 1 - m))


def compose(pixels, mask, rgb, dark=False):
    """``pixels`` (h, w, 4) uint8 painted ``rgb`` through ``mask`` (h, w) uint8: light paints, or dark with ``dark``.
    The alpha stays."""
    m = mask.astype(np.float32)[..., None] / 255
    m = 1 - m if dark else m
    out = pixels.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - m) + np.asarray(rgb, np.float32) * m
    return np.round(out).astype(np.uint8)


def cover(pixels, rgb=None):
    """A layer's image ``pixels`` (h, w, 4) uint8, grey with no alpha, as it shows: its grey for its alpha, and ``rgb``
    where it shows (its grey, with none)."""
    out = pixels.copy()
    out[..., 3] = pixels[..., 0]
    if rgb is not None:
        out[..., :3] = np.asarray(rgb, np.uint8)
    return out


def grey(pixels):
    """Whether an image ``pixels`` (h, w, 4) uint8 is in greys with no alpha of its own: opaque, its channels equal."""
    return bool((pixels[..., 3] == 255).all() and (np.ptp(pixels[..., :3].astype(np.int16), axis=2) <= 2).all())


# ---- in the build scene -------------------------------------------------------------------------------------------

def ready(build_dir, load_dir, out, ask, character=None, picture=None, layers=()):
    """The open build scene's colour images painted where the game paints them, and its layers' images drawn as they
    show, as the module's docstring says; ``picture``, the model in its own colours, the ask's first picture where it
    exists; ``character``, who labels said it is; ``layers``, the worn copies in one place (look.layers). Saves
    build.blend when it changed an image. -> what paint.json holds."""
    import bpy
    build_dir, out = Path(build_dir), Path(out)
    images = colour_images(load_dir)
    over = layered(layers, load_dir)
    shown = {i['image'] for i in over}
    rows = [dict(r, letter=chr(ord('A') + k)) for k, r in enumerate(
        i for i in images if i['candidates'] and i['image'] not in shown)]
    rec = {'images': {i['image']: {k: i[k] for k in ('file', 'materials', 'candidates')} for i in images}}
    front = bool(picture) and Path(picture).is_file()
    shows = []
    if over:
        shows, rec['layer_answers'] = show(over, out, ask, character, picture if front else None)
        rec.update(layers=over, layered=shows)
    if not rows:
        if shows:
            bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
        return _write(build_dir, dict(rec, skipped='no colour image has a candidate mask beside it'))
    n = 0
    for r in rows:
        r['numbers'] = list(range(n + 1, n + 1 + len(r['candidates'])))
        n += len(r['candidates'])
    pictures = ([str(picture)] if front else []) + [str(p) for p in sheets(rows, out / 'paint.png')]
    answers = []
    for k in range(VOTES):
        answer, _ = ask(question(rows, character, front) + f' (Ask {k + 1} of {VOTES}.)', pictures, schema(rows),
                        {'rows': []}, route='core paint')
        answers.append(answer)
    picked = tally(answers, {r['letter']: r['numbers'] for r in rows})
    rec['answers'] = answers
    named = []
    for r in rows:
        if r['letter'] not in picked:
            continue
        number, rgb = picked[r['letter']]
        mask_file = r['candidates'][r['numbers'].index(number)]
        image = bpy.data.images[r['image']]
        w, h = image.size
        with Image.open(mask_file) as m:
            mask = np.asarray(m.convert('L').resize((w, h), Image.BILINEAR))
        named.append(dict(r, mask_file=mask_file, mask=mask, rgb=rgb, before=_pixels(image)))
    sides = {r['letter']: side(r['before'], r['mask']) for r in named}
    asked = [r for r in named if sides[r['letter']] == 'ask']
    dark, rec['way_answers'] = (dark_ways(asked, out, ask, character, picture if front else None) if asked
                                else ({}, []))
    painted, unpainted = [], []
    for r in named:
        mask_file, mask, rgb, before = r['mask_file'], r['mask'], r['rgb'], r['before']
        way = sides[r['letter']]
        if way is None:
            unpainted.append({'image': r['image'], 'mask': mask_file,
                              'under': [round(v, 1) for v in under(before, mask)],
                              'why': 'the colour image already draws its colours under both sides of the mask'})
            continue
        if way == 'ask':
            if not dark.get(r['letter']):
                unpainted.append({'image': r['image'], 'mask': mask_file,
                                  'why': 'not every answer paints it where the files contradict themselves'})
                continue
            way = 'dark'
        new = _packed(bpy.data.images[r['image']], '-painted', compose(before, mask, rgb, dark=way == 'dark'),
                      r['file'])
        painted.append({'image': r['image'], 'mask': mask_file, 'paints': way, 'colour': '#%02x%02x%02x' % rgb,
                        'as': new.name})
    rec.update(painted=painted, unpainted=unpainted)
    if painted or shows:
        bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
    if not painted:
        rec['skipped'] = ('no mask painted: ' + '; '.join(u['why'] for u in unpainted) if unpainted
                          else 'no answer named a mask more than half the answers agree on')
    return _write(build_dir, rec)


def side(pixels, mask):
    """Where ``mask`` paints the colour image ``pixels``, as the module's docstring says: 'light' where the image is
    near black under its light, 'ask' where it is near black only under its dark, None where under neither."""
    light, dark = under(pixels, mask)
    return 'light' if light < NEAR_BLACK else 'ask' if dark < NEAR_BLACK else None


def dark_ways(named, out, ask, character=None, picture=None):
    """Whether each mask of ``named`` (the paint rows the files leave in doubt, each with its 'mask', the colour image
    as it was, 'before', and its colour, 'rgb') paints its dark side, as the module's docstring says: one ask, VOTES
    answers, the parts drawing each colour image as drawn and painted, on the model; ``picture``, the model in its own
    colours. -> ({letter: True to paint, where every answer says so}, the answers)."""
    pictures = ([str(picture)] if picture else []) + [str(p) for p in drawn_or_dark(named, Path(out) / 'way.png')]
    answers = []
    for k in range(VOTES):
        answer, _ = ask(way_question(named, character, bool(picture)) + f' (Ask {k + 1} of {VOTES}.)', pictures,
                        way_schema(named), {'rows': []}, route='core paint way')
        answers.append(answer)
    return picked_dark(answers, [r['letter'] for r in named]), answers


def drawn_or_dark(named, path):
    """Per row of ``named`` (as dark_ways takes them), a picture: the open scene's parts that draw its colour image,
    close, from the front and the back (the build faces -y, z up), as the files draw them (1), then painted where its
    mask is dark (2), the rest of the model in its own colours around them, laid out by way_sheet. The scene is left
    as it was. -> their paths."""
    import bpy
    from mathutils import Matrix
    from .render import _prepare_scene
    hidden = {ob.name: ob.hide_render for ob in bpy.context.scene.objects}
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE_ON_MODEL
    cam = bpy.data.objects.new('core_paint_way', bpy.data.cameras.new('core_paint_way'))
    cam.data.type = 'ORTHO'
    scene.collection.objects.link(cam)
    scene.camera = cam
    paths = []
    try:
        for r in named:
            image = bpy.data.images[r['image']]
            pts = _drawn_by(image)
            lo, hi = pts.min(0), pts.max(0)
            centre, size = (lo + hi) / 2, float(max(hi[0] - lo[0], hi[2] - lo[2], 1e-6))
            cam.data.ortho_scale = 1.4 * size
            depth = float(np.ptp(_drawn_by(None)[:, 1])) + size
            cam.data.clip_start, cam.data.clip_end = depth * 1e-3, depth * 4
            tiles = []
            for drawn in (r['before'], compose(r['before'], r['mask'], r['rgb'], dark=True)):
                new = _packed(image, '-way', drawn)
                for side in (-1., 1.):          # the front, then the back
                    m = np.eye(4)
                    m[:3, :3] = np.column_stack([[-side, 0., 0.], [0., 0., 1.], [0., side, 0.]])
                    m[:3, 3] = centre + np.array([0., side * 2 * depth, 0.])
                    cam.matrix_world = Matrix(m.tolist())
                    scene.render.filepath = str(path.with_name(f'{path.stem}-tile.png'))
                    still()
                    with Image.open(scene.render.filepath) as img:
                        rgba = img.convert('RGBA')
                    tiles.append(Image.new('RGB', rgba.size, (205, 210, 218)))
                    tiles[-1].paste(rgba, (0, 0), rgba)
                _unpacked(new, image)
            paths.append(path.with_name(f'{path.stem}-{r["letter"]}.png'))
            way_sheet(r['letter'], tiles).save(paths[-1])
        Path(scene.render.filepath).unlink(missing_ok=True)
    finally:
        bpy.data.objects.remove(cam)
        for name, hide in hidden.items():
            bpy.data.objects[name].hide_render = hide
    return paths


WAY_GAP = 48        # between a way sheet's two pairs of tiles, more than between a pair's own two


def way_sheet(letter, tiles):
    """Row ``letter``'s way picture from its four tiles (drawn front, drawn back, painted front, painted back), laid
    out to be read: the row named on a line of its own, each way's number over its pair and on each of its tiles,
    the two ways apart, every tile whole. -> the sheet."""
    from .render import _font
    t, head = tiles[0].width, 110
    sheet = Image.new('RGB', (4 * t + 2 * 8 + WAY_GAP, t + head), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((8, 4), f'Row {letter}', fill=(200, 0, 0), font=_font(36, bold=True))
    for i, tile in enumerate(tiles):
        k, v = divmod(i, 2)
        left = i * t + v * 8 + k * (8 + WAY_GAP)
        sheet.paste(tile, (left, head))
        d.rectangle([left, head, left + 44, head + 52], fill='white')
        d.text((left + 10, head + 4), str(k + 1), fill='black', font=_font(40, bold=True))
        if v == 0:
            d.text((left + t + 4, 50), str(k + 1), fill='black', font=_font(48, bold=True), anchor='mt')
    return sheet


def _drawn_by(image):
    """The world points of the open scene's faces whose material draws ``image`` as its colour (every mesh's, for
    None), (n, 3)."""
    import bpy
    out = [np.zeros((0, 3))]
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        mine = {k for k, s in enumerate(ob.material_slots) if image is None or (
            _colour_node(s.material) is not None and _colour_node(s.material).image == image)}
        if not mine:
            continue
        me = ob.data
        rows = sorted({v for p in me.polygons if p.material_index in mine for v in p.vertices})
        co = np.array([me.vertices[v].co[:] for v in rows]).reshape(-1, 3)
        mw = np.array(ob.matrix_world)
        out.append(co @ mw[:3, :3].T + mw[:3, 3])
    return np.vstack(out)


def _unpacked(new, image):
    """``image`` drawn again where ``new`` (_packed's) was, and ``new`` gone."""
    import bpy
    for mat in bpy.data.materials:
        node = _colour_node(mat)
        if node is not None and node.image == new:
            node.image = image
    bpy.data.images.remove(new)


def picked_dark(answers, letters):
    """Per letter, whether every answer paints it (2), as the module's docstring says. -> {letter: True or False}."""
    def paints(a, letter):
        return any(r.get('row') == letter and r.get('way') == 2 for r in (a or {}).get('rows') or [])
    return {letter: bool(answers) and all(paints(a, letter) for a in answers) for letter in letters}


def show(over, out, ask, character=None, picture=None):
    """The layers' images (``over``, as layered gives them) drawn as they show, as the module's docstring says: one
    ask, VOTES answers, for the colour each shows in; ``picture``, the model in its own colours. -> ([{image, colour,
    as}], the answers)."""
    import bpy
    rows = [dict(r, letter=chr(ord('A') + k), candidates=[], numbers=[]) for k, r in enumerate(over)]
    pictures = ([str(picture)] if picture else []) + [str(p) for p in sheets(rows, Path(out) / 'layer.png')]
    answers = []
    for k in range(VOTES):
        answer, _ = ask(layer_question(rows, character, bool(picture)) + f' (Ask {k + 1} of {VOTES}.)', pictures,
                        layer_schema(rows), {'rows': []}, route='core layer')
        answers.append(answer)
    picked = tally(answers, {r['letter']: [] for r in rows})
    shows = []
    for r in rows:
        image = bpy.data.images[r['image']]
        rgb = picked[r['letter']][1] if r['letter'] in picked else None
        new = _packed(image, '-layer', cover(_pixels(image), rgb), r['file'])
        new['layer'] = True             # the eyes pass draws a layer's grey see-through, as it is
        shows.append({'image': r['image'], 'colour': '#%02x%02x%02x' % rgb if rgb else None, 'as': new.name})
    return shows, answers


def colour_images(load_dir):
    """The images the scene's meshes draw as their colour, each with its file and its candidate masks (siblings).
    -> [{image, file, materials, candidates}], by image name."""
    import bpy
    mats = {}
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            node = _colour_node(slot.material)
            if node is not None and node.image.size[0]:
                mats.setdefault(node.image.name, set()).add(slot.material.name)
    loaded = {_key(f) for f in (_file(im, load_dir) for im in bpy.data.images) if f}
    out = []
    for name in sorted(mats):
        f = _file(bpy.data.images[name], load_dir)
        candidates = []
        if f:
            sizes = {}
            for p in sorted(Path(f).parent.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and stem(p.name) == stem(Path(f).name):
                    try:
                        with Image.open(p) as img:
                            sizes[str(p)] = img.size
                    except (OSError, ValueError):
                        continue
            sizes[f] = sizes.get(f) or tuple(bpy.data.images[name].size)
            candidates = siblings(f, sizes, loaded)
        out.append({'image': name, 'file': f, 'materials': sorted(mats[name]), 'candidates': candidates})
    return out


def layered(pairs, load_dir):
    """Of each pair of worn copies in one place (look.layers: their mesh names), the one drawing only images in
    greys with no alpha of their own, and not both: a layer over the other. -> [{image, file, materials}], the images
    the layers draw."""
    import bpy
    out = {}
    for pair in pairs or ():
        drawn = []
        for meshes in pair:
            d = {}
            for name in meshes:
                ob = bpy.data.objects.get(name)
                for slot in (ob.material_slots if ob is not None else ()):
                    node = _colour_node(slot.material)
                    if node is not None and node.image.size[0]:
                        d.setdefault(node.image.name, set()).add(slot.material.name)
            drawn.append(d)
        greys = [bool(d) and all(grey(_pixels(bpy.data.images[i])) for i in d) for d in drawn]
        if sum(greys) == 1:
            for name, mats in drawn[greys.index(True)].items():
                out.setdefault(name, set()).update(mats)
    return [{'image': n, 'file': _file(bpy.data.images[n], load_dir), 'materials': sorted(m)}
            for n, m in sorted(out.items())]


# ---- the ask ------------------------------------------------------------------------------------------------------

QUESTION = (
    '{first}Some games paint a colour of their own on a character in play: a team or ink colour, or a colour the '
    'player picks. There, the character\'s colour image is plain or near black where the game paints, and a mask '
    'beside it in the files says where: an image in greys, its light (or its dark) where the game paints. Such a '
    'mask comes with a colour image left blank where it marks, as if unfinished: the game fills that colour in. A map '
    'of metal, roughness, shading, ambient occlusion, skin or emission also comes in greys, but it marks parts the '
    'colour image already draws, and is not one. '
    '{sheets} show, row by row, an image the model draws its colours from, lettered, then the other images in its '
    'folder that share its name, numbered: {legend}. For each lettered row, say which numbered image is such a mask, '
    'or 0 for none. For every row, say the colour (#rrggbb) the character wears where its game paints, in its '
    'best-known look. Most images have no such mask.')
LAYER_QUESTION = (
    '{first}Some parts of this character are drawn over another part of the same shape, as layers: {legend}. '
    '{sheets} show each layer\'s image, lettered, in greys with no colour of its own. Such an image says only where '
    'its layer shows, where it is light; where it is dark, the part under it shows through. Where it shows, the game '
    'draws the layer in a colour of its own. For each lettered layer, say the colour (#rrggbb) the character wears '
    'where it shows, in its best-known look.')
WAY_QUESTION = (
    '{first}Its game may paint a colour of its own on some of its parts in play, a team or ink colour ({legend}). '
    '{sheets} show, one per lettered row, the parts of the character that draw one colour image, close, from the '
    'front and the back: as its files draw them (1), and painted in that colour (2). For each lettered row, say which '
    'of the two shows the character as it looks in its best-known look.')


def question(rows, character, front):
    first = (f'Picture 1 shows a 3D character{f" ({character})" if character else ""} in its own colours, as its '
             'files draw it. ') if front else ''
    sheets = 'The pictures after it' if front else 'The pictures'
    legend = '; '.join(f'{r["letter"]}: {Path(r["file"]).name}, then '
                       + ', '.join(f'{n} {Path(c).name}' for n, c in zip(r['numbers'], r['candidates']))
                       for r in rows)
    return QUESTION.format(first=first, sheets=sheets, legend=legend)


def layer_question(rows, character, front):
    first = (f'Picture 1 shows a 3D character{f" ({character})" if character else ""} in its own colours, as its '
             'files draw it. ') if front else ''
    sheets = 'The pictures after it' if front else 'The pictures'
    legend = '; '.join(f'{r["letter"]}: {_name(r)}, drawn by {", ".join(r["materials"])}' for r in rows)
    return LAYER_QUESTION.format(first=first, sheets=sheets, legend=legend)


def way_question(rows, character, front):
    first = (f'Picture 1 shows a 3D character{f" ({character})" if character else ""} in its own colours, as its '
             'files draw it. ') if front else ''
    sheets = 'The pictures after it' if front else 'The pictures'
    legend = '; '.join(f'{r["letter"]}: in ' + '#%02x%02x%02x' % tuple(r['rgb']) for r in rows)
    return WAY_QUESTION.format(first=first, sheets=sheets, legend=legend)


def way_schema(rows):
    row = {'type': 'object', 'properties': {'row': {'type': 'string', 'enum': [r['letter'] for r in rows]},
                                            'way': {'type': 'integer', 'enum': [1, 2]}},
           'required': ['row', 'way'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'rows': {'type': 'array', 'items': row}},
            'required': ['rows'], 'additionalProperties': False}


def layer_schema(rows):
    row = {'type': 'object', 'properties': {'row': {'type': 'string', 'enum': [r['letter'] for r in rows]},
                                            'colour': {'type': 'string'}},
           'required': ['row', 'colour'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'character': {'type': 'string'},
                                             'rows': {'type': 'array', 'items': row}},
            'required': ['character', 'rows'], 'additionalProperties': False}


def schema(rows):
    numbers = [0] + [n for r in rows for n in r['numbers']]
    row = {'type': 'object', 'properties': {
        'row': {'type': 'string', 'enum': [r['letter'] for r in rows]},
        'mask': {'type': 'integer', 'enum': numbers},
        'colour': {'type': 'string'}},
        'required': ['row', 'mask', 'colour'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'character': {'type': 'string'},
                                             'rows': {'type': 'array', 'items': row}},
            'required': ['character', 'rows'], 'additionalProperties': False}


def sheets(rows, path):
    """Pictures of ROWS rows each: a colour image lettered, then its candidates numbered, each in a TILE square, its
    clear texels over a checker. -> their paths."""
    from .render import _font
    font = _font(22, bold=True)
    width = (1 + max(len(r['candidates']) for r in rows)) * (TILE + 8) + 8
    paths = []
    for k in range(0, len(rows), ROWS):
        part = rows[k:k + ROWS]
        sheet = Image.new('RGB', (width, len(part) * (TILE + 40) + 8), (235, 235, 235))
        d = ImageDraw.Draw(sheet)
        for y, r in enumerate(part):
            tiles = [(r['letter'], r['file'] or _image(r['image']))] + [
                (str(n), c) for n, c in zip(r['numbers'], r['candidates'])]
            for x, (label, f) in enumerate(tiles):
                left, top = 8 + x * (TILE + 8), 8 + y * (TILE + 40)
                d.text((left, top), label, fill=(200, 0, 0) if x == 0 else (0, 0, 0), font=font)
                sheet.paste(_tile(f), (left, top + 32))
        p = Path(path).with_name(f'{Path(path).stem}-{k // ROWS}.png')
        sheet.save(p)
        paths.append(p)
    return paths


def _tile(f):
    """An image (a file, or a PIL image) filling a TILE square, larger or smaller (a 64x16 strip is as readable as the
    rest)."""
    if isinstance(f, Image.Image):
        img = f.convert('RGBA')
    else:
        with Image.open(f) as img:
            img = img.convert('RGBA')
    k = TILE / max(img.size)
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))),
                     Image.NEAREST if k > 1 else Image.LANCZOS)
    board = Image.new('RGBA', (TILE, TILE), (255, 255, 255, 255))
    d = ImageDraw.Draw(board)
    for i in range(0, TILE, 16):
        for j in range(0, TILE, 16):
            if (i + j) // 16 % 2:
                d.rectangle([i, j, i + 15, j + 15], fill=(200, 200, 200, 255))
    board.alpha_composite(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    return board.convert('RGB')


# ---- helpers ------------------------------------------------------------------------------------------------------

def _rgb(s):
    s = (s or '').strip().lstrip('#')
    try:
        return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4)) if len(s) == 6 else None
    except ValueError:
        return None


def _name(r):
    return Path(r['file']).name if r['file'] else r['image']


def _image(name):
    """A Blender image as a PIL image."""
    import bpy
    return Image.fromarray(_pixels(bpy.data.images[name]), 'RGBA')


def _packed(image, suffix, pixels, made_from=None):
    """A packed image of ``pixels`` (h, w, 4) uint8, named for ``image`` with ``suffix``, drawn instead of it by every
    material whose colour it is; ``made_from``, the file ``image`` was read from, kept as the new image's
    ``made_from``: it is the model's own picture still, which the eyes pass mustn't take for another state of it
    (Inkling's unpainted eye, a black disc, was taken for her closed eye). -> the new image."""
    import bpy
    w, h = image.size
    new = bpy.data.images.new(image.name + suffix, width=w, height=h, alpha=True)
    if made_from:
        new['made_from'] = str(made_from)
    new.colorspace_settings.name = image.colorspace_settings.name
    new.pixels.foreach_set((pixels[::-1].astype(np.float32) / 255).ravel())
    new.update()
    new.file_format = 'PNG'
    new.pack()
    for mat in bpy.data.materials:
        node = _colour_node(mat)
        if node is not None and node.image == image:
            node.image = new
    return new


def _key(f):
    import os
    return os.path.normcase(os.path.abspath(f))


def _file(image, load_dir):
    """The file an image was read from, if it is on disk: its path in the build, or the same path read from the
    load's models folder (a path relative to the .blend load wrote it in)."""
    import bpy
    if image.source != 'FILE' or not image.filepath:
        return None
    for p in (bpy.path.abspath(image.filepath),
              str(Path(load_dir) / 'models' / image.filepath.lstrip('/').replace('\\', '/'))):
        if Path(p).is_file():
            return str(Path(p).resolve())
    return None


def _colour_node(mat):
    from .render import _base_color_image
    return _base_color_image(mat) if mat is not None else None


def _pixels(image):
    """A Blender image's pixels, (h, w, 4) uint8, the top row first."""
    w, h = image.size
    values = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(values)
    return np.round(np.clip(values, 0, 1) * 255).astype(np.uint8).reshape(h, w, 4)[::-1].copy()


def _write(build_dir, rec):
    (Path(build_dir) / 'paint.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return rec
