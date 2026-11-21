"""Which way the model faces: the one thing about its stance the rig can't say.

The skeleton's mirror axis fixes the side axis, so the front lies along the other horizontal axis, one way or the
other. Feet and faces both mislead (heels, sandals, crests, hair), so the sign is asked: the model drawn from each of
the two allowed fronts, side by side, and the AI picks the one that shows its face.
"""
from pathlib import Path

import numpy as np
from .render import still

QUESTION = ('Pictures A and B show the same character from two opposite sides. Which picture shows the character '
            'from the front, so that you see its face and chest rather than its back? Answer "A" or "B", or '
            '"cant_tell" only if neither shows a face or a front.')
SCHEMA = {'type': 'object', 'properties': {'front': {'type': 'string', 'enum': ['A', 'B', 'cant_tell']}},
          'required': ['front'], 'additionalProperties': False}
TILE = 640


def choose(axis, out, ask, up=np.array([0., 0., 1.]), convention=np.array([0., -1., 0.])):
    """The front, +``axis`` or -``axis``, asked from a picture of each drawn into ``out`` standing on ``up``. On
    can't tell, the importers' convention when it is one of the two, else +``axis``."""
    candidates = [np.asarray(axis, float), -np.asarray(axis, float)]
    picture = draw(candidates, Path(out) / 'front.png', up)
    answer, info = ask(QUESTION, [str(picture)], SCHEMA, {'front': 'cant_tell'}, route='core front')
    pick = (answer or {}).get('front')
    if pick in ('A', 'B'):
        return candidates[0 if pick == 'A' else 1], pick
    if abs(candidates[0] @ convention) > 0.9:
        return (candidates[0] if candidates[0] @ convention > 0 else candidates[1]), 'cant_tell'
    return candidates[0], 'cant_tell'


UP_QUESTION = ('Pictures A, B, C and D show the same character turned four ways. In which picture is it standing '
               'upright, head at the top and feet at the bottom? Answer "A", "B", "C" or "D", or "cant_tell" only '
               'if none of them does.')
UP_VOTES = 3
UP_SCHEMA = {'type': 'object', 'properties': {'up': {'type': 'string', 'enum': ['A', 'B', 'C', 'D', 'cant_tell']}},
             'required': ['up'], 'additionalProperties': False}


def choose_up(side, out, ask, convention=np.array([0., 0., 1.])):
    """Which way is up, of the four directions across the axis ``side`` it is seen along: files don't always say it
    right (rips of one game come in lying down). The model is drawn once and shown turned four ways. Returns (up, the
    answer, whether most asks gave it)."""
    import bpy
    from mathutils import Vector
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene, _world_points
    side = np.asarray(side, float)
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    meshes = _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE
    pts = _world_points(meshes)
    lo, hi = pts.min(0), pts.max(0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    cam_data = bpy.data.cameras.new('core_up')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = size * 1.1
    cam_data.clip_start, cam_data.clip_end = size * 1e-3, size * 10
    cam = bpy.data.objects.new('core_up', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = Vector((centre + side * size * 3).tolist())
    cam.rotation_euler = _looking(-side, _any_across(side))
    m = np.array(cam.rotation_euler.to_matrix())
    right, screen_up = m[:, 0], m[:, 1]
    path = Path(out) / 'up.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = path.with_name('up-view.png')
    scene.render.filepath = str(tile)
    still()
    bpy.data.objects.remove(cam)
    base = Image.new('RGB', (TILE, TILE), 'white')
    rgba = Image.open(tile).convert('RGBA')
    base.paste(rgba, (0, 0), rgba)
    # turned so that each candidate points up on the page: as drawn, then a quarter turn each way, then half
    candidates = [screen_up, right, -right, -screen_up]
    turns = [0, 90, 270, 180]
    sheet = Image.new('RGB', (TILE * 4 + 60, TILE), 'white')
    for i, t in enumerate(turns):
        img = base.rotate(t, fillcolor='white')
        ImageDraw.Draw(img).text((12, 8), 'ABCD'[i], fill='black', font=_font(48, bold=True))
        sheet.paste(img, (i * (TILE + 20), 0))
    sheet.save(path)
    # asked UP_VOTES times: on a body that reads no way up at a glance (a ball with wings) the pick wavers between
    # asks; the most given wins, the first ask's on a tie, and one most asks didn't give is no sure answer
    picks = []
    for r in range(UP_VOTES):
        answer, info = ask(UP_QUESTION + ('' if r == 0 else f' (Ask {r + 1} of {UP_VOTES}.)'), [str(path)], UP_SCHEMA,
                           {'up': 'cant_tell'}, route='core up')
        picks.append((answer or {}).get('up'))
    given = [p for p in picks if p in ('A', 'B', 'C', 'D')]
    if given:
        pick = max(given, key=lambda p: (given.count(p), -given.index(p)))
        return (candidates['ABCD'.index(pick)], pick if len(set(given)) == 1 else f'{pick} ({"".join(picks)})',
                2 * given.count(pick) > UP_VOTES)
    near = max(candidates, key=lambda c: c @ convention)
    return near, 'cant_tell', False


def draw(fronts, path, up=np.array([0., 0., 1.])):
    """The open scene's meshes at rest, seen from each direction in ``fronts`` (looking back at the model), side by
    side and lettered A, B. Returns the path."""
    import bpy
    from mathutils import Vector
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene, _world_points
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    meshes = _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE
    pts = _world_points(meshes)
    lo, hi = pts.min(0), pts.max(0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    cam_data = bpy.data.cameras.new('core_front')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = size * 1.1
    cam_data.clip_start, cam_data.clip_end = size * 1e-3, size * 10
    cam = bpy.data.objects.new('core_front', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    tiles = []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(fronts):
        cam.location = Vector((centre + f * size * 3).tolist())
        cam.rotation_euler = _looking(-f, up)
        tile = path.with_name(f'{path.stem}-{i}.png')
        scene.render.filepath = str(tile)
        still()
        img = Image.new('RGB', (TILE, TILE), 'white')
        rgba = Image.open(tile).convert('RGBA')
        img.paste(rgba, (0, 0), rgba)
        ImageDraw.Draw(img).text((12, 8), 'AB'[i], fill='black', font=_font(48, bold=True))
        tiles.append(img)
    sheet = Image.new('RGB', (TILE * len(tiles) + 20 * (len(tiles) - 1), TILE), 'white')
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (TILE + 20), 0))
    sheet.save(path)
    bpy.data.objects.remove(cam)
    return path


def _looking(direction, up):
    """A camera rotation looking along ``direction`` with ``up`` up on the page."""
    from mathutils import Matrix
    back = -np.asarray(direction, float) / np.linalg.norm(direction)
    y = np.asarray(up, float) - (np.asarray(up, float) @ back) * back
    y /= np.linalg.norm(y)
    x = np.cross(y, back)
    return Matrix(np.column_stack([x, y, back]).tolist()).to_euler()


def _any_across(axis):
    """A unit vector across ``axis`` (a world axis): the next world axis round."""
    i = int(np.argmax(np.abs(axis)))
    out = np.zeros(3)
    out[(i + 1) % 3] = 1.0
    return out
