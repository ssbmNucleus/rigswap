"""The eyes made ready to blink: which of the model's surfaces are its eyes, and the build's eye faces as the objects
a plugin's export makes blink (contracts.md section 2, eyes.json).

A surface is one mesh drawn with one of its materials. The candidates are the head's surfaces that draw an image: the
faces of a mesh and material riding the model's own head (most of their weight on its spine above the chest or a bone
that carries: run._on_head), whichever fighter bone the correspondence gave them. Which of them are the eyes is
semantic: one ask (route 'core eyes'), the head drawn from the front with its surfaces numbered, each surface alone,
and the pictures code found that may be the eyes in another state (blink.py: the same picture elsewhere in its image,
or in an image file the model doesn't load). Where the file says it, the name says it:

- Ultimate rips (the community MaxScript's FBX and DAE) name the eye materials EyeL and EyeR, the character's own left
  and right eye (EyeL1, EyeR1: a second pair); EyeLD, EyeLG, EyeLL... are other looks, on meshes look.py leaves
  out, and so are the lids the game blinks with (Mario_Blink, Mario_Halfblink1..3): no image holds another state of
  the eye, so the half closed and closed pictures are drawn. Their colour image is the eye's white, eye_<name>_w_col;
  the iris is drawn over it through the mesh's second UV map (the game slides that map to move the iris) from
  eye_<name>_b_col beside it (Marth: eyel_ and eyer_), which neither the FBX nor the DAE names, so no importer loads
  it: drawn from its material alone an Ultimate eye is all white, as it was in game (2026-10-25). The iris is laid
  over the white here (blink.layered). Another rip (Diddy Kong's DAE, 2026-11-10) carries the second UV map as a copy
  of the eye mesh on itself, named <mesh>_UV1, drawing the white too, and its files name white and iris otherwise
  (eye_diddy_ouch_col, eye_diddy_003_col): the copy is read as the eye's second map, and where no iris is found by
  name, which of the download's unloaded images is the everyday iris is semantic: one ask (route 'core eyes iris'),
  each image that is see-through in part where the second map draws it laid over the white and shown on the
  character's face (as flat tiles of the texture the ask took Diddy's pinprick pupil, his eye in shock, for his
  everyday eye).

Then, in the open build scene (build.write's): the eye faces leave their meshes for objects of their own, eyes_left
and eyes_right (by the side of the face their middle is on: the character's left is +x), each with one material,
``eyes``, whose image is the crop of the eye texture they draw (blink.crop_box, the UVs moved into it); the three
pictures go in ``eyes/`` beside build.blend, and ``eyes.json`` says what was done, or why nothing was (``skipped``):
then the plugin's export leaves the vanilla eyes blank and the model's eyes as they are, with no blink.
"""
import json
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import blink
from .render import still

OBJECTS = {'left': 'eyes_left', 'right': 'eyes_right'}
MATERIAL, IMAGE = 'eyes', 'eyes_open'
ON_HEAD = 0.5           # a face is the head's when this share of its corners' weight is on the model's head
LID_SHARE = 0.5         # the lid is the surface bordering more than this share of the eye's upper outline
MOST_FOUND = 24         # pictures of other states offered in the ask, at most
FACE = 2.5              # an iris is judged on the face: the view is this many times the eyes' own span
TILE = 360
GREY = (0.42, 0.44, 0.47)                       # the rest, in the parts' pictures (linear)
GREY_SRGB = tuple(1.055 * c ** (1 / 2.4) - 0.055 for c in GREY)     # the same grey in an image
NAMED = re.compile(r'^Eye[LR]\d?$')             # Ultimate's eye materials (above)
_CHANNEL = re.compile(r'_UV\d+$')               # a rip's copy of a mesh carrying one of its further UV maps (above)
_TAKEN = re.compile(r'\.\d{3}$')                # Blender's suffix for a name already taken
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.dds', '.tif', '.tiff', '.webp', '.gif'}


class Surface:
    """One mesh's faces on the head drawn with one of its materials: the candidates for the eyes."""

    def __init__(self, n, ob, slot, faces, image, extension):
        self.n, self.object, self.slot, self.faces, self.image = n, ob.name, slot, faces, image
        self.extension = extension      # its image's wrap, a Blender extension
        mat = ob.material_slots[slot].material
        self.material = mat.name
        self.key = f'{ob.name} / {mat.name}'
        self.named = bool(NAMED.match(_TAKEN.sub('', mat.name)))


def ready(build_dir, head, load_dir, out, ask):
    """The eyes of the open build scene made ready to blink, as the module's docstring says; ``head``: {mesh name: the
    share of each vertex's weight on the model's head}; pictures and asks go in ``out``. Saves build.blend again when
    it made eye objects. -> what eyes.json holds."""
    import bpy
    build_dir, out = Path(build_dir), Path(out)
    (build_dir / 'eyes.json').unlink(missing_ok=True)
    surfaces, box = candidates(head)
    if not surfaces:
        return _write(build_dir, {'skipped': 'no surface on the head draws an image'})
    pixels = {s.image: _pixels(bpy.data.images[s.image]) for s in surfaces}
    named = [s for s in surfaces if s.named]
    found = [] if named else others(surfaces, pixels, Path(load_dir) / 'source')
    rec = {'surfaces': {s.n: s.key for s in surfaces}, 'named': [s.n for s in named],
           'found': [{k: f[k] for k in ('n', 'surface', 'from', 'box', 'mirrored')} for f in found]}
    if named:           # the file names the eyes, and blinks them with lids: no picture of another state to find
        eyes, half, shut = named, [], []
    else:
        answer, info = ask_eyes(surfaces, found, pixels, box, out, ask)
        rec.update(answer=answer, source=(info or {}).get('source'))
        eyes = [s for s in surfaces if s.n in (answer.get('eyes') or [])]
        mine = {s.n for s in eyes}
        picked = {state: [] if answer.get('cant_tell') else answer.get(state) or [] for state in ('half', 'closed')}
        half, shut = ([f for n in picked[state] for f in found if f['n'] == n and f['surface'] in mine]
                      for state in ('half', 'closed'))
    if not eyes:
        return _write(build_dir, dict(rec, skipped='no surface draws the eyes apart from the rest of the head'))
    made = make(build_dir, eyes, {'half': half, 'closed': shut}, pixels, Path(load_dir) / 'source', ask, out)
    if made.get('objects'):
        bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
    return _write(build_dir, dict(rec, **made))


# ---- the candidates -----------------------------------------------------------------------------------------------

def candidates(head):
    """The head's surfaces drawing an image, numbered from 1 by mesh name and material slot, and the box (lo, hi) of
    every face on the head (``head``: ready's)."""
    import bpy
    out, points = [], []
    for ob in sorted((o for o in bpy.context.scene.objects if o.type == 'MESH' and len(o.data.polygons)
                      and o.name in head), key=lambda o: o.name):
        me = ob.data
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', lv)
        start, count = np.empty(len(me.polygons), np.int64), np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get('loop_start', start)
        me.polygons.foreach_get('loop_total', count)
        on = np.add.reduceat(np.asarray(head[ob.name])[lv], start) / count >= ON_HEAD
        if not on.any():
            continue
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
        points.append(co[lv[np.repeat(on, count)]])
        mats = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get('material_index', mats)
        for slot in sorted(set(mats[on].tolist())):
            mat = ob.material_slots[slot].material if slot < len(ob.material_slots) else None
            node = _colour_image(mat)
            if node is None or not node.image.size[0] or _render_uv(me) is None:
                continue
            out.append(Surface(len(out) + 1, ob, slot, np.flatnonzero(on & (mats == slot)), node.image.name,
                               node.extension))
    pts = np.vstack(points) if points else np.zeros((1, 3))
    return out, (pts.min(0), pts.max(0))


def others(surfaces, pixels, source):
    """The pictures that may be a surface's picture in another state (blink.py): for each part of the image a
    surface draws (its islands with texels inside them), the places alike in the same image, and the same place in
    each image file of the download the model doesn't load that has that image's layout (it loads the file a painted
    image was made from: paint.py, ``made_from``; a file with the same pixels as one it loads is that picture, as
    Inkling's eye is in her download twice). -> [{n, surface, image (pixels' key or a file), from, box, like (the
    part's box), mirrored}], numbered from 1, at most MOST_FOUND."""
    import bpy
    loaded = {str(Path(bpy.path.abspath(im.filepath)).resolve()).lower() for im in bpy.data.images if im.filepath}
    loaded |= {str(Path(im['made_from']).resolve()).lower() for im in bpy.data.images if im.get('made_from')}
    sizes = {pixels[s.image].shape[:2] for s in surfaces}
    own, files = [], []
    for p in sorted(source.rglob('*')) if source.is_dir() else []:
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(p) as img:
                if img.size[::-1] in sizes:
                    mine = str(p.resolve()).lower() in loaded
                    (own if mine else files).append((str(p), np.asarray(img.convert('RGBA'))))
        except (OSError, ValueError):
            continue
    files = [(f, fp) for f, fp in files if not any(np.array_equal(fp, o) for _, o in own)]
    found, seen = [], set()
    for s in surfaces:
        px = pixels[s.image]
        parts = [m for m in blink.islands(blink.footprint(_tris(s), px.shape[1::-1], s.extension))
                 if blink.inside(m).any()]
        boxes = [blink.box_of(m) for m in parts]
        twins = [(f, fp) for f, fp in files if fp.shape == px.shape and blink.same_layout(px, fp) >= blink.SAME_LAYOUT
                 and int(np.abs(fp.astype(np.int16) - px).max()) > 1]        # not the model's own copy of it
        for mask, box in zip(parts, boxes):
            x, y, w, h = box
            here = [(s.image, b, 'alike in its own image', m) for b, m in
                    blink.find_alike(px, box, mask[y:y + h, x:x + w], boxes)]
            here += [(f, box, 'the same place in ' + Path(f).name, False) for f, fp in twins
                     if (blink.flat(fp[y:y + h, x:x + w]) - blink.flat(px[y:y + h, x:x + w]))[mask[y:y + h, x:x + w]]
                     .any()]
            for image, b, how, mirrored in here:
                if (image, tuple(b), tuple(box)) not in seen and len(found) < MOST_FOUND:
                    seen.add((image, tuple(b), tuple(box)))
                    found.append({'n': len(found) + 1, 'surface': s.n, 'image': image, 'from': how, 'box': b,
                                  'like': box, 'mirrored': mirrored})
    return found


# ---- the ask ------------------------------------------------------------------------------------------------------

QUESTION = ('Picture 1 shows the head of a 3D character from the front: it faces you, so its left eye is on your '
            'right. Its surfaces are numbered, each one mesh drawn with one material. Picture 2 shows each numbered '
            'surface alone, in its own colours, the rest of the head grey.{states} Which numbered surfaces draw the '
            "character's eyes as they look open: the eyeballs, the irises and pupils, or a patch drawn with the eyes "
            'on it? List them under "eyes". List none when no surface draws the eyes apart from the rest of the '
            "head: eyes painted on the whole face or head with the rest of it, a mask's lenses, goggles or a visor "
            'over them, or a character with no eyes. Eyebrows, eyelids, glasses and the face round the eyes are not '
            'eyes.{state_answers} Set cant_tell only if you cannot tell.')
STATES = (' Picture 3 holds, row by row, a picture a surface draws, first in its row and marked "part" with the '
          "surface's number (as the model shows it), then numbered pictures code found in the model's texture files "
          'that look like it: the same picture elsewhere in the texture or mirrored, or the same place in a texture '
          'file the model does not use. A texture file may keep the eyes in other states this way.')
STATE_ANSWERS = (' Then, for the eyes you list, look in picture 3 at the row of their surface: under "half", the '
                 'number of the picture showing those same eyes half closed (the lids part way down), and under '
                 '"closed", the number of the one showing them shut (the lids all the way down), one for each '
                 'picture of the open eyes; leave a state empty when no picture shows it. Only the lids closing '
                 'counts: a squint, a wink, a look aside, a happy or an angry face is no half closed or closed eye.')


def ask_eyes(surfaces, found, pixels, box, out, ask):
    """The ask: (answer, info). The default (no answer) names no eyes."""
    pictures = draw(surfaces, box, out)
    if found:
        pictures.append(str(sheet_of_found(surfaces, found, pixels, out / 'eyes-states.png')))
    numbers = [s.n for s in surfaces]
    props = {'eyes': {'type': 'array', 'items': {'type': 'integer', 'enum': numbers}}}
    if found:
        tiles = [f['n'] for f in found]
        props.update(half={'type': 'array', 'items': {'type': 'integer', 'enum': tiles}},
                     closed={'type': 'array', 'items': {'type': 'integer', 'enum': tiles}})
    props['cant_tell'] = {'type': 'boolean'}
    schema = {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    question = QUESTION.format(states=STATES if found else '', state_answers=STATE_ANSWERS if found else '')
    default = {'eyes': [], 'cant_tell': True}
    answer, info = ask(question, pictures, schema, default, route='core eyes')
    return answer or default, info


def draw(surfaces, box, out):
    """Picture 1, the head from the front textured, each surface's number on it; picture 2, each surface alone in its
    colours, the rest grey (a layer see-through where it is), framed alike. -> their paths."""
    import bpy
    from mathutils import Matrix
    from .render import _base_color_image, _font, _prepare_scene
    hidden = {ob.name: ob.hide_render for ob in bpy.context.scene.objects}
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE * 2
    lo, hi = box
    centre, size = (lo + hi) / 2, float(max(hi[0] - lo[0], hi[2] - lo[2]))
    cam = bpy.data.objects.new('core_eyes', bpy.data.cameras.new('core_eyes'))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1.15 * size
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 20
    scene.collection.objects.link(cam)
    scene.camera = cam
    m = np.eye(4)
    m[:3, :3] = np.column_stack([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])     # from the front (-y), z up
    m[:3, 3] = centre + np.array([0., -5. * size, 0.])
    cam.matrix_world = Matrix(m.tolist())
    out.mkdir(parents=True, exist_ok=True)
    grey = bpy.data.materials.new('core_eyes_grey')
    grey.diffuse_color = GREY + (1.0,)
    saved = [(slot, slot.link) for ob in scene.objects if ob.type == 'MESH' for slot in ob.material_slots]
    greys = {}

    def grey_of(mat):
        # A layer drawn grey is see-through where it is (paint marks its image): Inkling's eyelid band, solid, hid the
        # eye under it. Workbench draws a material's active image, colour and alpha, so the grey is an image
        node = _base_color_image(mat) if mat else None
        if node is None or not node.image.get('layer'):
            return grey
        if mat.name not in greys:
            px = _pixels(node.image)
            px[..., :3] = np.round(np.array(GREY_SRGB) * 255).astype(np.uint8)
            img = bpy.data.images.new(f'core_eyes_grey_{node.image.name}', px.shape[1], px.shape[0], alpha=True)
            img.pixels.foreach_set((px[::-1].astype(np.float32) / 255).ravel())
            copy = greys[mat.name] = mat.copy()
            mine = _base_color_image(copy)
            mine.image = img
            copy.node_tree.nodes.active = mine
        return greys[mat.name]

    def render(path):
        scene.render.filepath = str(path)
        still()
        with Image.open(path) as img:
            rgba = img.convert('RGBA')
        tile = Image.new('RGB', rgba.size, (205, 210, 218))
        tile.paste(rgba, (0, 0), rgba)
        return tile
    try:
        head = render(out / 'eyes-head.png')
        d = ImageDraw.Draw(head)
        scale, placed = head.width / cam.data.ortho_scale, []
        for s in surfaces:          # each number on the face of the surface nearest its middle, of those facing you
            me = bpy.data.objects[s.object].data
            centres = np.array([me.polygons[int(f)].center[:] for f in s.faces])
            facing = np.array([me.polygons[int(f)].normal[1] < 0 for f in s.faces])
            pool = centres[facing] if facing.any() else centres
            c = pool[np.argmin(np.linalg.norm((pool - pool.mean(0))[:, [0, 2]], axis=1))]
            x, y = head.width / 2 + (c[0] - centre[0]) * scale, head.height / 2 - (c[2] - centre[2]) * scale
            while any(abs(x - a) < 30 and abs(y - b) < 30 for a, b in placed):
                y += 30
            placed.append((x, y))
            d.text((x, y), str(s.n), fill='black', font=_font(30, bold=True), anchor='mm', stroke_width=3,
                   stroke_fill='white')
        head.save(out / 'eyes-head.png')
        scene.render.resolution_x = scene.render.resolution_y = TILE
        tiles = []
        for s in surfaces:
            for ob in scene.objects:
                if ob.type != 'MESH':
                    continue
                for k, slot in enumerate(ob.material_slots):
                    mine = ob.name == s.object and k == s.slot
                    own = slot.material
                    slot.link = 'OBJECT'
                    slot.material = own if mine else grey_of(own)
            tile = render(out / f'eyes-part-{s.n}.png')
            (out / f'eyes-part-{s.n}.png').unlink()
            ImageDraw.Draw(tile).text((10, 6), str(s.n), fill='black', font=_font(40, bold=True))
            tiles.append(tile)
            for slot, link in saved:
                slot.material = None
                slot.link = link
    finally:
        for slot, link in saved:
            if slot.link != link:
                slot.material = None
                slot.link = link
        bpy.data.objects.remove(cam)
        for m in greys.values():
            img = _base_color_image(m).image
            bpy.data.materials.remove(m)
            bpy.data.images.remove(img)
        bpy.data.materials.remove(grey)
        for name, hide in hidden.items():
            bpy.data.objects[name].hide_render = hide
    cols = min(4, len(tiles))
    sheet = Image.new('RGB', (cols * TILE + 8 * (cols - 1), -(-len(tiles) // cols) * (TILE + 8) - 8), 'white')
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * (TILE + 8), (k // cols) * (TILE + 8)))
    sheet.save(out / 'eyes-parts.png')
    return [str(out / 'eyes-head.png'), str(out / 'eyes-parts.png')]


def sheet_of_found(surfaces, found, pixels, path):
    """Picture 3: a row for each part of a surface's picture with pictures found like it, the part first."""
    from .render import _font
    rows = {}
    for f in found:
        rows.setdefault((f['surface'], tuple(f['like'])), []).append(f)
    image_of = {s.n: s.image for s in surfaces}
    cell = TILE // 2
    width = 1 + max(len(r) for r in rows.values())
    sheet = Image.new('RGB', (width * (cell + 6), len(rows) * (cell + 6)), 'white')
    for r, ((n, like), row) in enumerate(rows.items()):
        cells = [(pixels[image_of[n]], like, f'part {n}')] + [(_read(f['image'], pixels), f['box'], str(f['n']))
                                                              for f in row]
        for c, (px, (x, y, w, h), label) in enumerate(cells):
            tile = Image.new('RGB', (cell, cell), (205, 210, 218) if c else (240, 220, 160))
            pic = Image.fromarray(np.ascontiguousarray(px[y:y + h, x:x + w]), 'RGBA')
            k = (cell - 30) / max(w, h)
            pic = pic.resize((max(1, round(w * k)), max(1, round(h * k))), Image.NEAREST if k >= 2 else Image.LANCZOS)
            tile.paste(pic, ((cell - pic.width) // 2, 28 + (cell - 30 - pic.height) // 2), pic)
            ImageDraw.Draw(tile).text((6, 2), label, fill='black', font=_font(22, bold=True))
            sheet.paste(tile, (c * (cell + 6), r * (cell + 6)))
    sheet.save(path)
    return path


# ---- the eye objects ----------------------------------------------------------------------------------------------

def make(build_dir, eyes, chosen, pixels, source, ask=None, out=None):
    """The eye faces as eye objects with the pictures of their blink (see the module's docstring). ``chosen``:
    {state: [found pictures]}; ``ask`` and ``out``: for the ask of an iris no name finds (ask_iris), none without
    them. -> what eyes.json holds."""
    import bpy
    notes = []
    count = Counter()
    for s in eyes:
        count[s.image] += len(s.faces)
    texture = count.most_common(1)[0][0]
    others = [s for s in eyes if s.image != texture]
    eyes = [s for s in eyes if s.image == texture]
    wrap = eyes[0].extension            # the eye objects draw with the first surface's material (_split)
    base = pixels[texture].copy()
    size = base.shape[1::-1]
    laid, second = [], {}
    copy = lambda s: bool(_CHANNEL.search(_TAKEN.sub('', s.object)))        # noqa: E731
    for c in [s for s in eyes if copy(s)]:      # a copy of an eye on itself, face for face: that eye's second UV map
        for e in eyes:
            pair = None if e is c or copy(e) or e.n in second else _coincident(c, [e])
            if pair is not None:
                second[e.n] = (*pair, c.object)
                laid.append(c)
                break
    eyes = [s for s in eyes if s not in laid]
    layers = []                 # (an eye, its UV triangles on its own map, on its second map, what that map is)
    for s in eyes:
        me = bpy.data.objects[s.object].data
        if s.n in second:
            layers.append((s, *second[s.n]))
        elif len(me.uv_layers) > 1:
            layers.append((s, _tris(s), _tris(s, me.uv_layers[1]), me.uv_layers[1].name))
    asked, rec = None, {}
    for s, below, above, how in layers:                   # Ultimate: the iris over the white (the docstring)
        if not s.named:
            continue
        iris = _iris(bpy.data.images[texture], source, s)
        if iris is None and ask is not None and out is not None:      # no file of the name: one ask for all the eyes
            if asked is None:
                files = [f for f in unloaded(source, size, list(pixels.values()))
                         if any(layer_over(f[1], a, wrap) for s2, _, a, _ in layers if s2.named)][:MOST_FOUND]
                asked = (None, None, None)
                if files:
                    with on_face([s2 for s2, *_ in layers if s2.named], [c.object for c in laid], out) as show:
                        asked = ask_iris(base, [(b, a) for s2, b, a, _ in layers if s2.named], files, wrap, out,
                                         ask, show)
                rec = {'iris': {'offered': [Path(f).name for f, _ in files], 'answer': asked[1],
                                'source': (asked[2] or {}).get('source')}}
            iris = asked[0]
        if iris is not None:
            base = blink.layered(base, iris[1], below, above, wrap)
            notes.append(f'{s.key}: the iris {Path(iris[0]).name} laid over the white through {how}')
    for s in others:            # Smash 4: the iris a second mesh lying on the eye face for face, in its own image
        pair = _coincident(s, eyes)
        if pair is None:
            notes.append(f'surface {s.key} draws {s.image}, not the eye texture {texture}: left as it is')
            continue
        base = blink.layered(base, pixels[s.image], *pair, wrap)
        laid.append(s)
        notes.append(f'{s.key}: lies on the eye face for face; its image {s.image} laid over the eye texture')
    tris = np.concatenate([_tris(s) for s in eyes])
    mask = blink.footprint(tris, size, wrap)
    if not mask.any():
        return {'skipped': 'the eye faces draw no texel of their image', 'notes': notes}
    crop = blink.crop_box(mask)
    grown = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(2 * blink.CROP_MARGIN + 1))) > 0     # an eye split at a seam stays one part
    parts = [{'box': blink.box_of(m & mask)} for m in blink.islands(grown) if (m & mask).any()]
    _tops(parts, eyes, size)
    found = {}
    for state in ('half', 'closed'):            # the first picture named for each part of the open eyes
        for f in chosen.get(state) or []:
            k = _nearest(parts, f['like'])
            if k not in {g[2] for g in found.get(state, [])}:
                found.setdefault(state, []).append((_read(f['image'], pixels), f['box'], k, f['mirrored']))
    skin, lid = None, None
    if found.get('half'):
        px, (x, y, w, h), k, mirrored = found['half'][0]
        bx, by, bw, bh = parts[k]['box']
        skin = blink.from_half(base[by:by + bh, bx:bx + bw], blink.fitted(px[y:y + h, x:x + w], bw, bh, mirrored))
        lid = {'from': 'the half closed picture'} if skin is not None else None
    if skin is None:
        skin, lid = lid_on_model(eyes)
    pictures, made, drawn = blink.blink(base, crop, parts, found, skin)
    folder = Path(build_dir) / 'eyes'
    folder.mkdir(parents=True, exist_ok=True)
    frames = {}
    for state, px in pictures.items():
        Image.fromarray(px, 'RGBA').save(folder / f'{state}.png')
        frames[state] = f'eyes/{state}.png'
    objects = _split(eyes, pictures['open'], texture, size, crop, laid)
    return {'objects': objects, 'material': MATERIAL, 'image': IMAGE, 'eye_texture': texture,
            'eye_texture_size': list(size), 'crop': crop, 'parts': parts, 'frames': frames, 'made': made,
            'drawn': drawn, 'lid': dict(lid or {}, rgba=None if skin is None else [int(c) for c in skin]),
            'notes': notes, **rec}


# ---- an iris no name finds ----------------------------------------------------------------------------------------

IRIS = ('The eyes of a 3D character are drawn in two layers: the white of the eye, and over it a picture of the iris '
        "and pupil, which the model's file does not name. Each numbered tile shows the character's eyes on its face, "
        "from the front, with one picture from the model's texture files laid over the white. Which tile shows the "
        'eyes as the character has them every day, open and looking ahead: its usual iris and pupil? The other tiles '
        'may show the eye in another state (the pupil shrunk or shaking in shock or pain, knocked out, glowing in a '
        'special move, lit in another colour) or a picture that is no iris at all. Give that tile\'s number under '
        '"iris", or 0 when no tile shows an everyday iris. Set cant_tell only if you cannot tell.')


def unloaded(source, size, loaded):
    """The download's image files of ``size`` (w, h) that the model loads none of: [(path, pixels (h, w, 4))], of
    those with the same pixels (a file for each costume) the first. ``loaded``: the pixels of the images it loads."""
    out = []
    for p in sorted(source.rglob('*')) if source.is_dir() else []:
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(p) as img:
                if img.size != tuple(size):
                    continue
                px = np.asarray(img.convert('RGBA'))
        except (OSError, ValueError):
            continue
        if not any(px.shape == o.shape and np.array_equal(px, o) for o in [*loaded, *(o for _, o in out)]):
            out.append((str(p), px))
    return out


def layer_over(px, tris, extension):
    """Whether the UV triangles ``tris`` draw of the image ``px`` both texels that show and texels that don't: a
    picture laid over another, as an iris over its white (a white, a skin, a wholly clear image is none)."""
    alpha = px[..., 3][blink.footprint(tris, px.shape[1::-1], extension)]
    return bool((alpha > 127).any() and (alpha < 128).any())


@contextmanager
def on_face(eyes, hide, out):
    """Inside: a function drawing the surfaces ``eyes`` on the character's face from the front, the face round them
    in view, with the eye texture it is given (pixels, the eye image's layout) in place of their image; the objects
    ``hide`` (an eye's copies carrying its second UV map) are left out. The scene is as it was after."""
    import bpy
    from mathutils import Matrix
    from .render import _base_color_image, _prepare_scene
    scene = bpy.context.scene
    hidden = {ob.name: ob.hide_render for ob in scene.objects}
    resolution = scene.render.resolution_x, scene.render.resolution_y
    _prepare_scene()
    for name in hide:
        bpy.data.objects[name].hide_render = True
    points, far = [], []
    for ob in scene.objects:
        if ob.type == 'MESH' and len(ob.data.vertices):
            m = np.array(ob.matrix_world)
            co = np.empty(3 * len(ob.data.vertices))
            ob.data.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            far.append(co[:, 1])
            points += [co[list(ob.data.polygons[int(f)].vertices)]
                       for s in eyes if s.object == ob.name for f in s.faces]
    points, far = np.vstack(points), np.concatenate(far)
    lo, hi = points.min(0), points.max(0)
    centre, size = (lo + hi) / 2, float(max(hi[0] - lo[0], hi[2] - lo[2]))
    scene.render.resolution_x = scene.render.resolution_y = TILE
    cam = bpy.data.objects.new('core_eyes_iris', bpy.data.cameras.new('core_eyes_iris'))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = FACE * size
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, float(far.max() - far.min()) + 2 * size
    scene.collection.objects.link(cam)
    scene.camera = cam
    m = np.eye(4)
    m[:3, :3] = np.column_stack([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])     # from the front (-y), z up
    m[:3, 3] = [centre[0], float(far.min()) - size, centre[2]]
    cam.matrix_world = Matrix(m.tolist())
    nodes = {}
    for s in eyes:
        node = _base_color_image(bpy.data.objects[s.object].material_slots[s.slot].material)
        if node is not None and node.name + s.material not in nodes:
            nodes[node.name + s.material] = (node, node.image)
    w, h = next(iter(nodes.values()))[1].size
    image = bpy.data.images.new('core_eyes_iris', w, h, alpha=True)
    for node, _ in nodes.values():
        node.image = image
    path = Path(out) / 'eyes-iris-face.png'

    def show(pixels):
        image.pixels.foreach_set((pixels[::-1].astype(np.float32) / 255).ravel())
        image.update()
        scene.render.filepath = str(path)
        still()
        with Image.open(path) as img:
            rgba = img.convert('RGBA')
        tile = Image.new('RGB', rgba.size, (205, 210, 218))
        tile.paste(rgba, (0, 0), rgba)
        return tile
    try:
        yield show
    finally:
        for node, was in nodes.values():
            node.image = was
        bpy.data.images.remove(image)
        camera = cam.data
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(camera)
        scene.render.resolution_x, scene.render.resolution_y = resolution
        for name, hide_render in hidden.items():
            bpy.data.objects[name].hide_render = hide_render
        path.unlink(missing_ok=True)


def ask_iris(base, layers, files, wrap, out, ask, show=None):
    """The ask: which of ``files`` [(path, pixels)] is the everyday iris of the eyes drawing ``base`` through
    ``layers`` [(their UV triangles on the eye's map, on the second map)]. Each is laid over the white and drawn on
    a numbered tile by ``show`` (on_face's: the eyes on the face), or without it as the texture is where the eyes
    draw it. -> ((path, pixels) or None, the answer, the ask's info)."""
    from .render import _font
    if not files or not layers:
        return None, None, None
    mask = blink.footprint(np.concatenate([b for b, _ in layers]), base.shape[1::-1], wrap)
    if not mask.any():
        return None, None, None
    x, y, w, h = blink.crop_box(mask)
    cell = TILE
    across = min(len(files), 4)
    sheet = Image.new('RGB', (across * (cell + 6), -(-len(files) // across) * (cell + 6)), 'white')
    for k, (_, px) in enumerate(files):
        drawn = base
        for below, above in layers:
            drawn = blink.layered(drawn, px, below, above, wrap)
        tile = Image.new('RGB', (cell, cell), (205, 210, 218))
        if show is not None:
            tile.paste(show(drawn).resize((cell, cell), Image.LANCZOS), (0, 0))
        else:
            pic = Image.fromarray(np.ascontiguousarray(drawn[y:y + h, x:x + w]), 'RGBA')
            f = (cell - 40) / max(w, h)
            pic = pic.resize((max(1, round(w * f)), max(1, round(h * f))),
                             Image.NEAREST if f >= 2 else Image.LANCZOS)
            tile.paste(pic, ((cell - pic.width) // 2, 36 + (cell - 40 - pic.height) // 2), pic)
        ImageDraw.Draw(tile).text((8, 2), str(k + 1), fill='black', font=_font(30, bold=True), stroke_width=3,
                                  stroke_fill='white')
        sheet.paste(tile, ((k % across) * (cell + 6), (k // across) * (cell + 6)))
    path = Path(out) / 'eyes-iris.png'
    sheet.save(path)
    schema = {'type': 'object', 'properties': {'iris': {'type': 'integer', 'enum': list(range(len(files) + 1))},
                                               'cant_tell': {'type': 'boolean'}},
              'required': ['iris', 'cant_tell'], 'additionalProperties': False}
    default = {'iris': 0, 'cant_tell': True}
    answer, info = ask(IRIS, [str(path)], schema, default, route='core eyes iris')
    answer = answer or default
    n = 0 if answer.get('cant_tell') else int(answer.get('iris') or 0)
    return (files[n - 1] if 0 < n <= len(files) else None), answer, info


def _split(eyes, opened, texture, size, crop, laid=()):
    """The eye faces out of their meshes into eyes_left and eyes_right, one material ``eyes`` whose image is the open
    picture, their UVs moved into the crop; the faces of the surfaces ``laid`` into that picture go. -> [{name, side,
    surfaces, triangles}]"""
    import bpy
    import bmesh
    source = bpy.data.objects[eyes[0].object].material_slots[eyes[0].slot].material
    material = source.copy()
    material.name = MATERIAL
    image = bpy.data.images.new(IMAGE, opened.shape[1], opened.shape[0], alpha=True)
    image.pixels.foreach_set((opened[::-1].astype(np.float32) / 255).ravel())
    image.file_format = 'PNG'
    image.pack()
    for node in material.node_tree.nodes:   # every use of the eye texture (its colour, its alpha) takes the crop
        if node.type == 'TEX_IMAGE' and node.image is not None and (
                node.image.name == texture or node.image.filepath and node.image.filepath ==
                bpy.data.images[texture].filepath):
            node.image = image
    pieces, taken, keys = {}, {}, set()
    for s in eyes:
        keys.add((s.object, s.material))
        ob = bpy.data.objects[s.object]
        co = np.empty(3 * len(ob.data.vertices))
        ob.data.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
        for f in s.faces:
            side = 'left' if co[list(ob.data.polygons[int(f)].vertices)].mean(0)[0] > 0 else 'right'
            pieces.setdefault(side, {}).setdefault(ob.name, []).append(int(f))
            taken.setdefault(ob.name, set()).add(int(f))
    objects = []
    for side, by_object in sorted(pieces.items()):
        copies = [_copy_with(bmesh, bpy.data.objects[name], faces, f'{OBJECTS[side]}_{k}')
                  for k, (name, faces) in enumerate(sorted(by_object.items()))]
        eye = _join(bpy, copies, OBJECTS[side])
        eye.data.materials.clear()
        eye.data.materials.append(material)
        eye.data.polygons.foreach_set('material_index', np.zeros(len(eye.data.polygons), np.int64))
        layer = _render_uv(eye.data)
        for other in [l for l in eye.data.uv_layers if l.name != layer.name]:
            eye.data.uv_layers.remove(other)
        layer = eye.data.uv_layers[0]
        uv = np.empty(2 * len(eye.data.loops))
        layer.data.foreach_get('uv', uv)
        uv = uv.reshape(-1, 2)
        for poly in eye.data.polygons:
            s, n = poly.loop_start, poly.loop_total
            uv[s:s + n] = blink.crop_uvs(uv[s:s + n], size, crop, eyes[0].extension)
        layer.data.foreach_set('uv', uv.ravel())
        eye.data.update()
        objects.append({'name': eye.name, 'side': side, 'surfaces': sorted(f'{o} / {m}' for o, m in keys if o in by_object),
                        'triangles': int(sum(len(p.vertices) - 2 for p in eye.data.polygons))})
    for s in laid:
        taken.setdefault(s.object, set()).update(int(f) for f in s.faces)
    for name, faces in taken.items():
        ob = bpy.data.objects[name]
        _delete_faces(bmesh, ob, sorted(faces))
        if not len(ob.data.polygons):
            bpy.data.objects.remove(ob, do_unlink=True)
    return objects


# ---- the lid on the model -----------------------------------------------------------------------------------------

def lid_on_model(eyes):
    """The lid's colour from the model round the eyes: (rgba (4,) uint8 or None, how it was found). At each vertex of
    the outline of the eye faces, the nearest point of the rest of the model: its surface and its colour there. The
    surface the eye lies on (an iris on its eye white; the first surface behind most of the eye faces) is no lid, and
    is left out unless nothing else borders the eye. The lid is the surface bordering the most of what is left, in the
    median of its colours there, when that is more than LID_SHARE of it; else none (an eye bordered by a beak, a hat
    and a neck), and the eye stays open."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import barycentric_transform
    taken = {(s.object, int(f)) for s in eyes for f in s.faces}
    corners, owner = [], []
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH' or not len(ob.data.polygons):
            continue
        me, m = ob.data, np.array(ob.matrix_world)
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        verts, loops = np.empty(3 * n, np.int64), np.empty(3 * n, np.int64)
        polys, mats = np.empty(n, np.int64), np.empty(n, np.int64)
        me.loop_triangles.foreach_get('vertices', verts)
        me.loop_triangles.foreach_get('loops', loops)
        me.loop_triangles.foreach_get('polygon_index', polys)
        me.loop_triangles.foreach_get('material_index', mats)
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        uv = _uvs(me)
        for t in range(n):
            if (ob.name, int(polys[t])) not in taken:
                corners.append(co[verts[3 * t:3 * t + 3]])
                owner.append((ob, int(mats[t]), None if uv is None else uv[loops[3 * t:3 * t + 3]]))
    if not corners:
        return None, {'lid': None}
    tree = BVHTree.FromPolygons(np.concatenate(corners).tolist(), [(3 * k, 3 * k + 1, 3 * k + 2)
                                                                   for k in range(len(corners))])
    texels = {}

    def surface(k):
        ob, slot, _ = owner[k]
        mat = ob.material_slots[slot].material if slot < len(ob.material_slots) else None
        return f'{ob.name} / {mat.name if mat else None}', mat

    def colour(k, at):
        name, mat = surface(k)
        node = _colour_image(mat)
        uv = owner[k][2]
        if node is None or uv is None or not node.image.size[0]:
            bsdf = next((nd for nd in mat.node_tree.nodes if nd.type == 'BSDF_PRINCIPLED'), None) \
                if mat is not None and mat.node_tree is not None else None
            return None if bsdf is None else np.round(np.clip(np.array(
                bsdf.inputs['Base Color'].default_value), 0, 1) * 255).astype(np.uint8)
        px = texels.setdefault(node.image.name, _pixels(node.image))
        a, b, c = (Vector(p) for p in corners[k])
        u, v, _ = barycentric_transform(at, a, b, c, *(Vector((*p, 0.)) for p in uv))
        if not (np.isfinite(u) and np.isfinite(v)):
            u, v = uv.mean(axis=0)
        return px[min(px.shape[0] - 1, int((1 - v % 1.) * px.shape[0])), min(px.shape[1] - 1, int(u % 1. * px.shape[1]))]

    samples, under = [], Counter()
    for s in eyes:
        ob = bpy.data.objects[s.object]
        me, m = ob.data, ob.matrix_world
        edges = Counter(e for f in s.faces for e in me.polygons[int(f)].edge_keys)
        outline = {v for e, k in edges.items() if k == 1 for v in e} or \
            {v for f in s.faces for v in me.polygons[int(f)].vertices}
        world = {v: m @ me.vertices[v].co for v in outline}
        reach = max(1e-6, max(p.z for p in world.values()) - min(p.z for p in world.values()))
        for v, p in sorted(world.items()):
            at, _, k, _ = tree.find_nearest(p)
            if k is not None:
                c = colour(k, at)
                if c is not None:
                    samples.append((surface(k)[0], tuple(int(x) for x in c)))
        for f in s.faces:                       # what the eye lies on: the first surface behind each face
            normal = (m.to_3x3() @ me.polygons[int(f)].normal).normalized()
            hit, _, k, _ = tree.ray_cast(m @ me.polygons[int(f)].center - normal * 1e-4 * reach, -normal, 2 * reach)
            if k is not None:
                under[surface(k)[0]] += 1
    lies_on = under.most_common(1)[0][0] if under else None
    left = [x for x in samples if x[0] != lies_on] or samples
    how = {'bordering': dict(Counter(n for n, _ in samples)), 'lies_on': lies_on}
    if not left:
        return None, dict(how, lid=None)
    name, n = Counter(x for x, _ in left).most_common(1)[0]
    if n <= LID_SHARE * len(left):
        return None, dict(how, lid=None)
    colour_ = np.median(np.array([c for x, c in left if x == name], float), axis=0).round().astype(np.uint8)
    return colour_, dict(how, lid=name, share=round(n / len(left), 3), **{'from': 'the surface round the eye'})


# ---- helpers ------------------------------------------------------------------------------------------------------

def _iris(white, source, surface):
    """Ultimate's iris file for the eye white ``white`` (a Blender image): (path, pixels (h, w, 4)) or None."""
    import bpy
    path = Path(bpy.path.abspath(white.filepath)) if white.filepath else None
    name = path.name if path else white.name
    if '_w_' not in name:
        return None
    side = 'l' if surface.material.startswith('EyeL') else 'r'
    wants = [name.replace('_w_', '_b_'), re.sub(r'^eye_', f'eye{side}_', name.replace('_w_', '_b_'))]
    folders = [path.parent] if path and path.parent.is_dir() else []
    for want in wants:
        for found in [f / want for f in folders] + (sorted(source.rglob(want)) if source.is_dir() else []):
            if found.is_file():
                with Image.open(found) as img:
                    if img.size == tuple(white.size):
                        return str(found), np.asarray(img.convert('RGBA'))
    return None


def _nearest(parts, box):
    """The part whose box covers the most of ``box``."""
    def shared(a):
        ox = max(0, min(a[0] + a[2], box[0] + box[2]) - max(a[0], box[0]))
        oy = max(0, min(a[1] + a[3], box[1] + box[3]) - max(a[1], box[1]))
        return ox * oy
    return max(range(len(parts)), key=lambda k: shared(parts[k]['box']))


def _tops(parts, eyes, size):
    """Each part told the side of the image its eye's top is on (``up``): the side the height (+z) of the eye faces
    drawing it rises toward, by the slope of their corners' height across the image (a texture may lie turned)."""
    import bpy
    rows = []
    for s in eyes:
        ob = bpy.data.objects[s.object]
        m = np.array(ob.matrix_world)
        co = np.empty(3 * len(ob.data.vertices))
        ob.data.vertices.foreach_get('co', co)
        z = co.reshape(-1, 3) @ m[2, :3] + m[2, 3]
        uv = _uvs(ob.data)
        for f in s.faces:
            p = ob.data.polygons[int(f)]
            c = uv[p.loop_start:p.loop_start + p.loop_total]
            c = blink.tiled(c, s.extension)
            rows.append((np.column_stack([c[:, 0] * size[0], (1 - c[:, 1]) * size[1]]), z[list(p.vertices)]))
    for part in parts:
        x, y, w, h = part['box']
        mine = [(xy, z) for xy, z in rows if x <= xy[:, 0].mean() <= x + w and y <= xy[:, 1].mean() <= y + h]
        part['up'] = 'top'
        if mine:
            xy, z = np.vstack([a for a, _ in mine]), np.concatenate([b for _, b in mine])
            (gx, gy, _), *_ = np.linalg.lstsq(np.column_stack([xy, np.ones(len(xy))]), z, rcond=None)
            if np.isfinite(gx) and np.isfinite(gy) and (gx or gy):
                part['up'] = ('right' if gx > 0 else 'left') if abs(gx) > abs(gy) else ('bottom' if gy > 0 else 'top')


def _colour_image(mat):
    from .render import _base_color_image
    return _base_color_image(mat) if mat is not None else None


def _render_uv(me):
    return next((l for l in me.uv_layers if l.active_render), None) or me.uv_layers.active


def _uvs(me, layer=None):
    layer = layer or _render_uv(me)
    if layer is None:
        return None
    uv = np.empty(2 * len(me.loops))
    layer.data.foreach_get('uv', uv)
    return uv.reshape(-1, 2)


def _coincident(layer, eyes):
    """(the eye's UV triangles, ``layer``'s): ``layer``'s faces as triangles, each corner matched to the corner of the
    eye face at the same place; None when a face of it lies on no eye face."""
    def faces(s):
        import bpy
        ob = bpy.data.objects[s.object]
        me = ob.data
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
        lv = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', lv)
        uv = _uvs(me)
        out = []
        for f in s.faces:
            p = me.polygons[int(f)]
            loops = slice(p.loop_start, p.loop_start + p.loop_total)
            out.append((co[lv[loops]], uv[loops]))
        return out
    on = [f for s in eyes for f in faces(s)]
    if not on:
        return None
    centres = np.array([pos.mean(0) for pos, _ in on])
    tol = 1e-3 * (float(np.ptp(np.vstack([pos for pos, _ in on]), axis=0).max()) or 1.0)
    below, above = [], []
    for pos, uv in faces(layer):
        k = int(np.argmin(np.linalg.norm(centres - pos.mean(0), axis=1)))
        epos, euv = on[k]
        d = np.linalg.norm(epos[None, :, :] - pos[:, None, :], axis=2)
        match = d.argmin(axis=1)
        if np.linalg.norm(centres[k] - pos.mean(0)) > tol or d[np.arange(len(pos)), match].max() > tol:
            return None
        for i in range(1, len(pos) - 1):
            below.append(euv[match[[0, i, i + 1]]])
            above.append(uv[[0, i, i + 1]])
    return np.array(below, float), np.array(above, float)


def _tris(s, layer=None):
    """(k, 3, 2) UVs of a surface's faces as triangles (fans), on ``layer`` (the render map by default)."""
    import bpy
    me = bpy.data.objects[s.object].data
    uv = _uvs(me, layer)
    out = []
    for f in s.faces:
        p = me.polygons[int(f)]
        c = uv[p.loop_start:p.loop_start + p.loop_total]
        out += [(c[0], c[i], c[i + 1]) for i in range(1, len(c) - 1)]
    return np.array(out, float).reshape(-1, 3, 2)


def _pixels(image):
    """A Blender image's pixels, (h, w, 4) uint8, the top row first."""
    w, h = image.size
    values = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(values)
    return np.round(np.clip(values, 0, 1) * 255).astype(np.uint8).reshape(h, w, 4)[::-1].copy()


def _read(image, pixels):
    """Pixels of an image: a key of ``pixels`` (an image in the scene) or a file."""
    if image in pixels:
        return pixels[image]
    with Image.open(image) as img:
        return np.asarray(img.convert('RGBA'))


def _copy_with(bmesh, ob, faces, name):
    """A copy of ``ob`` (modifiers, vertex groups, attributes) holding only ``faces``."""
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = new.data.name = name
    for c in ob.users_collection:
        c.objects.link(new)
    keep = set(faces)
    _delete_faces(bmesh, new, [f for f in range(len(new.data.polygons)) if f not in keep])
    return new


def _delete_faces(bmesh, ob, faces):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[i] for i in faces], context='FACES')
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def _join(bpy, objects, name):
    if len(objects) > 1:
        for o in bpy.context.view_layer.objects:
            o.select_set(o in objects)
        bpy.context.view_layer.objects.active = objects[0]
        with bpy.context.temp_override(active_object=objects[0], object=objects[0], selected_objects=objects,
                                       selected_editable_objects=objects):
            bpy.ops.object.join()
    eye = objects[0]
    eye.name = eye.data.name = name
    return eye


def _write(build_dir, rec):
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    (Path(build_dir) / 'eyes.json').write_text(json.dumps(rec, indent=1, default=_plain), encoding='utf-8')
    return rec


def _plain(x):
    return x.tolist() if isinstance(x, np.ndarray) else int(x) if isinstance(x, np.integer) else float(x)
