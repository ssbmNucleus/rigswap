"""A piece of the model left out or brought back: a complaint about what the costume draws.

The record's ``drop`` lists the meshes the everyday look leaves out (look.py decides them, one look at every piece). That
call is a judgement, and the person may want something else than it gave: Diddy Kong's barrel jetpack was dropped by
every answer on 2026-10-28 and kept by every answer on 2026-11-10, the picture and the question the same, and "take
the jetpack off" was then a complaint the record could not hold. So a complaint can name a piece: complaint.py's look
says which way (leave out, bring back) and what, in the person's words, and the costume is built again with that
request beside its record (``pieces``: {'way': 'out' or 'back', 'piece': the words for it, 'said': the person's own}).

The build settles the request here, before it drops anything: the model's pieces (variants.units: a switch the file
names, else a mesh) that the costume draws, or those it leaves out, each drawn alone in its own colours with the rest
of the costume grey, and one ask (route 'core complaint piece') says which tiles are what the person means. Their
meshes go into ``drop`` or out of it and the request is spent: the record the build writes holds the new ``drop`` and
no request. A request no tile answers ends the build (NoPiece), said in words for the person.
"""
from pathlib import Path

import numpy as np

from . import variants
from .render import still

ROUTE = 'core complaint piece'
TILE = 256
ACROSS, PER = 5, 20         # tiles across a picture, and in one picture: 1280 wide, as the ask sends it
VIEW = 0.6                  # a tile shows at least this share of the model's size: where on the body the piece is

QUESTION = ('A 3D character model was put onto a fighting-game fighter as a costume. The person who imported it '
            'says: "{said}". What they want: a piece of the model {wanted}, namely "{piece}".\n\n'
            '{shown} Each numbered tile draws one piece alone in its own colours, the rest of the costume in plain '
            'grey; a tile marked back or side is seen from there. The pieces\' names in the model\'s file: {names}.'
            '\n\nWhich tiles are what the person means? List under "pieces" every tile that is part of it (one thing '
            'may be drawn as several pieces) and no tile that is not. Take the person at their word: do not judge '
            'whether the costume is better with or without it. List none when no tile shows it. Set cant_tell only '
            'if you cannot tell.')
WANTED = {'out': 'left out of the costume', 'back': 'brought back, which the costume leaves out now'}
SHOWN = {'out': 'The tiles show every piece the costume draws now.',
         'back': 'The tiles show every piece of the model the costume leaves out now.'}


class NoPiece(ValueError):
    """A request no piece of the model answers, said in words for the person who complained."""


def candidates(us, drop, way):
    """The pieces a request may mean: to leave one out, those the costume draws (a mesh of theirs not dropped); to
    bring one back, those it leaves out whole."""
    gone = set(drop)
    drawn = [u for u in us if any(m not in gone for m in u.meshes)]
    return drawn if way == 'out' else [u for u in us if u not in drawn]


def chosen(offered, answer):
    """The pieces an answer names (their tiles' numbers, from 1), each once, in the tiles' order; none when it cannot
    tell."""
    if (answer or {}).get('cant_tell'):
        return []
    numbers = {int(n) for n in (answer or {}).get('pieces') or [] if isinstance(n, (int, float))}
    return [u for n, u in enumerate(offered, 1) if n in numbers]


def changed(drop, picked, way):
    """``drop`` with the meshes of ``picked`` put in (left out) or taken out (brought back)."""
    meshes = [m for u in picked for m in u.meshes]
    if way == 'out':
        return list(drop) + [m for m in meshes if m not in set(drop)]
    return [m for m in drop if m not in set(meshes)]


def schema(count):
    return {'type': 'object',
            'properties': {'pieces': {'type': 'array', 'items': {'type': 'integer', 'enum': list(range(1, count + 1))}},
                           'cant_tell': {'type': 'boolean'}},
            'required': ['pieces', 'cant_tell'], 'additionalProperties': False}


def settle(rig, told, up, front, out, ask):
    """The request in the record ``told`` (``pieces``) settled against the open scene's rig, every mesh of the model
    still in it. -> (the new ``drop``, what was done: the way, the words, the pieces picked and their meshes).
    NoPiece when nothing is offered or picked."""
    request = told['pieces']
    way, piece = request.get('way'), ' '.join(str(request.get('piece') or '').split())
    drop = list(told.get('drop') or [])
    if way not in WANTED or not piece:
        raise NoPiece('the change names no piece to leave out or bring back')
    us, _ = variants.units(rig)
    offered = candidates(us, drop, way)
    if not offered:
        raise NoPiece('the costume draws every piece its model has: there is none to bring back' if way == 'back'
                      else 'the model has no piece to leave out')
    pictures = draw(us, offered, set(drop), up, front, Path(out) / 'pieces.png')
    said = ' '.join(str(request.get('said') or piece).split()).replace('"', "'")[:600]
    question = QUESTION.format(said=said, wanted=WANTED[way], piece=piece.replace('"', "'")[:200], shown=SHOWN[way],
                               names='; '.join(f'{n} {u.key}' for n, u in enumerate(offered, 1)))
    answer, info = ask(question, [str(p) for p in pictures], schema(len(offered)), {'pieces': [], 'cant_tell': True},
                       route=ROUTE)
    picked = chosen(offered, answer)
    if not picked:
        raise NoPiece(f'no piece of the model was found to be "{piece}": '
                      + ('the costume leaves out nothing like it' if way == 'back' else 'nothing it draws is that'))
    report = {'way': way, 'piece': piece, 'offered': [u.key for u in offered], 'answer': answer,
              'source': (info or {}).get('source'), 'pieces': [u.key for u in picked],
              'meshes': [m for u in picked for m in u.meshes]}
    return changed(drop, picked, way), report


def draw(us, offered, gone, up, front, path, over=False):
    """A numbered tile per piece of ``offered``: it alone in its own colours, every other piece the costume draws grey,
    those it leaves out (``gone``, mesh names) not drawn; seen from the side of the body the piece sits on, close
    enough to see it and wide enough to see where on the body it is (VIEW). With ``over``, the piece alone is laid
    over its tile again, so one inside another or stacked with others in one place (a file's ten face plates, its eyes
    in the head) is seen whole, not under the grey. -> the pictures, PER tiles in each."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    was = scene.render.resolution_x, scene.render.resolution_y
    scene.render.resolution_x = scene.render.resolution_y = TILE
    up = np.asarray(up, float) / np.linalg.norm(up)
    front = np.asarray(front, float) - (np.asarray(front, float) @ up) * up
    front /= np.linalg.norm(front)
    across = np.cross(up, front)
    drawn = [u for u in us if any(m not in gone for m in u.meshes)]
    body = np.vstack([[u.lo, u.hi] for u in drawn]) if drawn else np.vstack([[u.lo, u.hi] for u in us])
    size = float(np.ptp(body, axis=0).max())
    middle = (body.min(0) + body.max(0)) / 2
    grey = bpy.data.materials.new('core_pieces_grey')
    grey.diffuse_color = (0.4, 0.42, 0.45, 1.0)
    slots = {m: [(slot, slot.link) for slot in bpy.data.objects[m].material_slots] for u in us for m in u.meshes}

    def greyed(m, on):
        for slot, link in slots[m]:
            if on:
                slot.link = 'OBJECT'
                slot.material = grey
            else:
                slot.material = None
                slot.link = link
    cam = bpy.data.objects.new('core_pieces', bpy.data.cameras.new('core_pieces'))
    cam.data.type = 'ORTHO'
    scene.collection.objects.link(cam)
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    try:
        for u in us:
            for m in u.meshes:
                bpy.data.objects[m].hide_render = m in gone
                greyed(m, True)
        for n, u in enumerate(offered, 1):
            for m in u.meshes:
                bpy.data.objects[m].hide_render = False
                greyed(m, False)
            c = (u.lo + u.hi) / 2
            corners = np.array([[x, y, z] for x in (u.lo[0], u.hi[0]) for y in (u.lo[1], u.hi[1])
                                for z in (u.lo[2], u.hi[2])])
            facing = front if (c - middle) @ front >= -0.01 * size else -front
            flank = across if (c - middle) @ across >= 0 else -across

            def shown_area(v):
                return np.ptp(corners @ np.cross(up, v)) * np.ptp(corners @ up)
            side = flank if shown_area(flank) > 2 * shown_area(facing) else facing
            right = np.cross(up, side)
            span = max(np.ptp(corners @ right), np.ptp(corners @ up))
            cam.data.ortho_scale = max(1.5 * span, VIEW * size)
            cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 10
            m4 = np.eye(4)
            m4[:3, :3], m4[:3, 3] = np.column_stack([right, up, side]), c + side * size * 3
            cam.matrix_world = Matrix(m4.tolist())
            shot = path.with_name(f'{path.stem}-tile.png')
            scene.render.filepath = str(shot)
            still()
            tile = Image.new('RGB', (TILE, TILE), (205, 210, 218))
            with Image.open(shot) as rendered:
                rgba = rendered.convert('RGBA')
            shot.unlink()
            tile.paste(rgba, (0, 0), rgba)
            if over:                # the piece alone again, laid over what covers it or stands in its place
                rest = [bpy.data.objects[m] for v in us if v is not u for m in v.meshes
                        if not bpy.data.objects[m].hide_render]
                for ob in rest:
                    ob.hide_render = True
                still()
                for ob in rest:
                    ob.hide_render = False
                with Image.open(shot) as rendered:
                    alone = rendered.convert('RGBA')
                shot.unlink()
                tile.paste(alone, (0, 0), alone)
            d = ImageDraw.Draw(tile)
            d.text((8, 4), str(n), fill='black', font=_font(30, bold=True), stroke_width=3, stroke_fill='white')
            seen = 'back' if side @ front < -0.5 else 'side' if abs(side @ front) < 0.5 else ''
            if seen:
                d.text((TILE - 8, 6), seen, fill='black', font=_font(20), anchor='ra', stroke_width=2,
                       stroke_fill='white')
            tiles.append(tile)
            for m in u.meshes:
                bpy.data.objects[m].hide_render = m in gone
                greyed(m, True)
    finally:
        for m in slots:
            greyed(m, False)
            bpy.data.objects[m].hide_render = False
        camera = cam.data
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(camera)
        bpy.data.materials.remove(grey)
        scene.render.resolution_x, scene.render.resolution_y = was
    out = []
    for start in range(0, len(tiles), PER):
        part = tiles[start:start + PER]
        cols = min(ACROSS, len(part))
        sheet = Image.new('RGB', (cols * TILE, -(-len(part) // cols) * TILE), 'white')
        for k, t in enumerate(part):
            sheet.paste(t, ((k % cols) * TILE, (k // cols) * TILE))
        out.append(path.with_name(f'{path.stem}-{start // PER + 1}.png'))
        sheet.save(out[-1])
    return out
