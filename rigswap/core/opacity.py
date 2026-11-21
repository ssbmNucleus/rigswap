"""Whether an image's alpha is its opacity (STATE, the mechanisms, 5).

A material that wires no alpha (its shader's Alpha takes nothing: the .mtl or .dae says nothing of transparency) still
draws its image's alpha in the game: the export cuts its clear texels (Test, plugins/melee/facts/alpha.py), because a
rip that lost its wiring needs it (hair cards, lashes, lace), and those say opaque too. But a game can keep something
else in an image's alpha: Spore's Bean is clear over 86% of his texture, and in game his body was cut away, his feet
and arm flaps left. The files cannot tell the two apart; it is meaning: one ask (route 'core opacity', VOTES answers),
each such image a lettered row, where it would cut the parts drawing it on the model, close, from the front and the
back, cut where its alpha is clear (1) and drawn whole (2). An image more than half the answers draw whole is drawn opaque: every
material drawing it as its colour draws a packed copy with its alpha full (paint._packed), as the build then draws it.

Asked only where the alpha would cut: some sample over the faces a material wiring no alpha draws it on is below the
export's cut (CUT), as the image has it; an image a material wiring its alpha draws is its source's word, not asked.
Nor is an image of one colour whose alpha varies: its alpha is all the picture it has (brows, an eye's shadow, a hair
card, the shade under a cap's brim), and drawn whole it is a flat patch of that colour. It stays cut. The five such
images asked about before 2026-11-10 were all answered cut; Diddy Kong's cap shadow, black all over and clear where
his faces draw it, was answered cut on 10-28 and whole on 11-10, the two pictures alike, and lay as black strips
over his eyes.
The pictures draw the rest of the model as the game draws it: every other asked image as the answers draw it, and
before any answer as the export does (cut). Where the first answers draw an image whole, the ask is made once more
over the model drawn as they said, and its answers stand (Toon Link's brow card over his eye card, Rosalina's lid over
her eyeball: drawn whole, the one hid what the other's pictures were of). Workbench, which draws every picture the core
asks with, draws a straight image's alpha and the load's (channel-packed) whole: an image is drawn cut as a straight
copy, its alpha the export's cut, and whole as a copy with its alpha full; through the cut shows what lies behind,
every layer of it (Rosalina's eyeball behind her lid layer's opening).
``opacity.json`` beside build.blend says what was asked and drawn opaque, or why nothing was asked.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .paint import TILE_ON_MODEL, _colour_node, _drawn_by, _packed, _pixels, _unpacked, way_schema, way_sheet
from .render import still

CUT = 128                       # the export cuts a texel whose alpha is below this (Test: alpha below 0x80)
FLAT = 8                        # an image is one colour when its colours lie within this many steps of 255
CHECKS = 12, ((232, 232, 232), (178, 178, 184))   # behind the character: squares no part of it is drawn in
VOTES = 3
# a triangle is sampled at its corners, the middles of its edges and its centre
SAMPLES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]])

QUESTION = (
    'The pictures show {who}, one picture per lettered row: the parts of it that draw one image, close, from the front '
    'and the back, twice. In 1 they are cut where the image\'s alpha is clear, as a cut-out is (strands of hair, '
    'lashes, lace, a fringe), and what lies behind shows there; in 2 they are drawn whole, the alpha left out. The '
    'checks are nothing: where they show through, the character has a hole. A game can keep something else in an '
    'image\'s alpha (a shine, a mask for an effect), which never cuts what it draws. For each lettered row, say which '
    'of the two shows the character as it looks: 1 or 2.')


def ready(build_dir, out, ask, character=None):
    """The open build scene's images a material wiring no alpha draws cut (cutting), each asked whether it is cut or
    whole, as the module's docstring says; those more than half the answers draw whole drawn opaque. ``character``,
    who labels said it is. Saves build.blend when it made one. -> what opacity.json holds."""
    import bpy
    build_dir, out = Path(build_dir), Path(out)
    alone = []
    rows = [dict(r, letter=chr(ord('A') + k)) for k, r in enumerate(cutting(alone))]
    rec = {'asked': [{k: r[k] for k in ('letter', 'image', 'materials', 'cut')} for r in rows]}
    if alone:
        rec['alpha_is_the_picture'] = alone
    if not rows:
        return _write(build_dir, dict(rec, skipped='no material wiring no alpha draws an image its alpha cuts'))
    who = f'a 3D character ({character})' if character else 'a 3D character'
    whole, rec['asks'] = set(), []
    for stem in ('opacity', 'opacity-again'):
        pictures = [str(p) for p in cut_or_whole(rows, out / f'{stem}.png', whole)]
        answers = []
        for k in range(VOTES):
            answer, _ = ask(QUESTION.format(who=who) + f' (Ask {k + 1} of {VOTES}.)', pictures, way_schema(rows),
                            {'rows': []}, route='core opacity')
            answers.append(answer)
        rec['asks'].append({'others_whole': sorted(whole), 'answers': answers})
        said = {r['image'] for r in rows
                if sum(any(a.get('row') == r['letter'] and a.get('way') == 2 for a in (ans or {}).get('rows') or [])
                       for ans in answers) * 2 > VOTES}
        if said == whole or len(rows) == 1:     # the pictures drew the rest as the answers say
            break
        whole = said
    opaque = []
    for r in rows:
        if r['image'] not in said:
            continue
        image = bpy.data.images[r['image']]
        pixels = _pixels(image)
        pixels[..., 3] = 255
        made_from = image.get('made_from') or (bpy.path.abspath(image.filepath) if image.filepath else None)
        opaque.append({'image': r['image'], 'as': _packed(image, '-opaque', pixels, made_from).name})
    rec['opaque'] = opaque
    if opaque:
        bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
    return _write(build_dir, rec)


def cutting(alone=None):
    """The open scene's colour images that materials wiring no alpha draw, cut where they draw them: [{'image',
    'materials', 'cut': the share of the samples over their faces below CUT}], those with some. An image of one
    colour whose alpha varies is none of them (the module's docstring); its name goes in ``alone`` when given."""
    import bpy
    drawers = {}
    for mat in bpy.data.materials:
        node = _colour_node(mat)
        if node is not None and node.image is not None:
            drawers.setdefault(node.image.name, []).append(mat)
    out = []
    for name, mats in drawers.items():
        image = bpy.data.images[name]
        if any(_wired(m) for m in mats) or not image.size[0]:
            continue
        pixels = _pixels(image)
        alpha = pixels[..., 3]
        if alpha.min() >= CUT:
            continue
        if np.ptp(pixels[..., :3].reshape(-1, 3), axis=0).max() <= FLAT and np.ptp(alpha) > FLAT:
            if alone is not None:
                alone.append(name)
            continue
        drawn = _drawn_alpha(alpha, mats)[0]
        cut = float((drawn < CUT).mean()) if drawn.size else 0.0
        if cut > 0:
            out.append({'image': name, 'materials': sorted(m.name for m in mats), 'cut': round(cut, 3)})
    return out


def _wired(mat):
    """Whether ``mat``'s shader takes an alpha: its Principled BSDF's Alpha linked, or set below 1."""
    return any(n.type == 'BSDF_PRINCIPLED' and (n.inputs['Alpha'].is_linked or n.inputs['Alpha'].default_value < 1)
               for n in mat.node_tree.nodes) if mat.node_tree else False


def _drawn_alpha(alpha, mats):
    """``alpha`` (h, w, the top row first) where the open scene's faces of ``mats`` draw it: at SAMPLES over each of
    their triangles, through the UV map drawn (the render one), wrapped; and each triangle's world corners.
    -> (t, len(SAMPLES)), (t, 3, 3)"""
    import bpy
    h, w = alpha.shape
    out, corners = [np.zeros((0, len(SAMPLES)), alpha.dtype)], [np.zeros((0, 3, 3))]
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        mine = [k for k, s in enumerate(ob.material_slots) if s.material in mats]
        me = ob.data
        uv = next((u for u in me.uv_layers if u.active_render), me.uv_layers.active)
        if not mine or uv is None:
            continue
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        loops = np.empty(n * 3, np.int64)
        me.loop_triangles.foreach_get('loops', loops)
        index = np.empty(n, np.int64)
        me.loop_triangles.foreach_get('material_index', index)
        coords = np.empty(len(me.loops) * 2)
        uv.data.foreach_get('uv', coords)
        kept = np.isin(index, mine)
        tri = coords.reshape(-1, 2)[loops.reshape(-1, 3)[kept]]
        at = np.einsum('sk,tkd->tsd', SAMPLES, tri).reshape(-1, 2) % 1.0
        out.append(alpha[np.minimum(((1 - at[:, 1]) * h).astype(np.int64), h - 1),
                         np.minimum((at[:, 0] * w).astype(np.int64), w - 1)].reshape(-1, len(SAMPLES)))
        verts = np.empty(n * 3, np.int64)
        me.loop_triangles.foreach_get('vertices', verts)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        mw = np.array(ob.matrix_world)
        corners.append((co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])[verts.reshape(-1, 3)[kept]])
    return np.concatenate(out), np.concatenate(corners)


def cut_or_whole(rows, path, whole=()):
    """Per row of ``rows`` (cutting's), a picture: where the open scene's faces drawing its image would be cut, close,
    from the front and the back (the build faces -y, z up), cut where the image's alpha is clear (1: what lies behind
    showing there), then drawn whole (2), the rest of the model around them as the game draws it: every other row's
    image whole if it is in ``whole``, else cut. Laid out by paint.way_sheet. Framed on the cut faces, not every face
    drawing the image: Ganondorf's body image cuts only his cape's hem, a few pixels of the whole of him. The scene is
    left as it was. -> their paths."""
    import bpy
    from mathutils import Matrix
    from .render import _prepare_scene
    hidden = {ob.name: ob.hide_render for ob in bpy.context.scene.objects}
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE_ON_MODEL
    cam = bpy.data.objects.new('core_opacity', bpy.data.cameras.new('core_opacity'))
    cam.data.type = 'ORTHO'
    scene.collection.objects.link(cam)
    scene.camera = cam
    shot = path.with_name(f'{path.stem}-tile.png')
    scene.render.filepath = str(shot)

    def render():
        still()
        with Image.open(shot) as img:
            return _on_checks(np.asarray(img.convert('RGBA')))
    paths, drawing = [], {}
    try:
        for r in rows:
            drawing[r['image']] = _drawn_as(bpy.data.images[r['image']], r['image'] in whole)
        for r in rows:
            image = bpy.data.images[r['image']]
            mats = [bpy.data.materials[m] for m in r['materials']]
            samples, corners = _drawn_alpha(_pixels(image)[..., 3], mats)
            pts = corners[(samples < CUT).any(axis=1)].reshape(-1, 3)
            lo, hi = pts.min(0), pts.max(0)
            centre, size = (lo + hi) / 2, float(max(hi[0] - lo[0], hi[2] - lo[2], 1e-6))
            cam.data.ortho_scale = 1.4 * size
            depth = float(np.ptp(_drawn_by(None)[:, 1])) + size
            cam.data.clip_start, cam.data.clip_end = depth * 1e-3, depth * 4
            shots = []
            for way_whole in (False, True):
                new = _drawn_as(drawing[r['image']], way_whole, image)
                try:
                    for side in (-1., 1.):          # the front, then the back
                        m = np.eye(4)
                        m[:3, :3] = np.column_stack([[-side, 0., 0.], [0., 0., 1.], [0., side, 0.]])
                        m[:3, 3] = centre + np.array([0., side * 2 * depth, 0.])
                        cam.matrix_world = Matrix(m.tolist())
                        shots.append(render())
                finally:
                    _unpacked(new, drawing[r['image']])
            paths.append(path.with_name(f'{path.stem}-{r["letter"]}.png'))
            way_sheet(r['letter'], shots).save(paths[-1])
        shot.unlink(missing_ok=True)
    finally:
        for name, new in drawing.items():
            _unpacked(new, bpy.data.images[name])
        bpy.data.objects.remove(cam)
        bpy.context.view_layer.update()     # else the view layer lists the camera gone as None, to whatever walks it
        for name, hide in hidden.items():
            bpy.data.objects[name].hide_render = hide
    return paths


def _drawn_as(drawn, whole, image=None):
    """A copy of ``image`` (by default ``drawn``) drawn instead of ``drawn``: whole, its alpha full, or cut, straight
    with its alpha the export's cut (clear below CUT), which Workbench draws. -> the copy."""
    image = image or drawn
    pixels = _pixels(image)
    pixels[..., 3] = 255 if whole else np.where(pixels[..., 3] < CUT, 0, 255)
    new = _packed(drawn, '-whole' if whole else '-cut', pixels)
    new.alpha_mode = 'STRAIGHT'
    return new


def _on_checks(rgba):
    """``rgba`` over CHECKS, not a plain grey: a hole the cut opens onto nothing showed the grey, and Rosalina's eyeball,
    clear where it is white, read as white through it."""
    size, (light, dark) = CHECKS
    h, w = rgba.shape[:2]
    odd = ((np.arange(h)[:, None] // size + np.arange(w)[None, :] // size) % 2).astype(bool)
    out = np.where(odd[..., None], np.array(dark, np.uint8), np.array(light, np.uint8))
    a = rgba[..., 3:4].astype(np.float32) / 255
    return Image.fromarray((rgba[..., :3] * a + out * (1 - a)).round().astype(np.uint8), 'RGB')


def _write(build_dir, rec):
    (Path(build_dir) / 'opacity.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return rec
