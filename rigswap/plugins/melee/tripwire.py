"""The tripwire: one cheap look at the finished draft, asking only whether it stands and is in one piece.

No look can be trusted to call a draft good (lab 2026-11-09: the strict models never
say it, the lenient ones said it of known-bad drafts), so none is asked that. Every model did call the wrecked draft
a wreck. So this asks two things and no more: does it stand as the fighter stands, is every part attached. An alarm
does not stop the import; the caller decides what a second try is (costume.make) and says plainly when the alarm
stands.

The draft is drawn as the game will pose it: the built meshes moved by the fighter's own frames (frame world x
inverse rest world, the game's skinning), idle and run, from a three-quarter side and from the front, the floor a
line. The fighter's own body is drawn the same way beside it: Donkey Kong and Bowser hunch far forward in their idle,
and asked only "is it upright" the look called their costumes fallen (fill-s2: Richard, Baragon; trip: Kritter).
No Dolphin."""
import json
import struct
from pathlib import Path

import numpy as np

from .targets import frames as fighter_frames
from ...core.render import still

ROUTE = 'core tripwire'
POSES = (('Wait1-1', 'idle'), ('Run-6', 'run'))
VIEWS = (('three-quarter side', (-.85, -.5)), ('front', (0, -1)))
TILE = 420
TO_BLENDER = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # game: y up, +z front

QUESTION = """You are checking a draft of a game costume. A 3D character model (the source) has been put onto a \
fighting-game fighter's skeleton, and the fighter's own animations now move it.

Picture 1: the source model as its maker left it, in its own colours.
Picture 2: the draft at two moments of the fighter's animations, idle (top row) and run (bottom row), each from a \
three-quarter side and from the front. The yellow line is the floor.
Picture 3: the fighter's own body at the same two moments, from the same sides. This is how the fighter stands and \
runs: some fighters crouch or hunch far forward in their idle, and the draft does what the fighter does. A pose the \
fighter also has is no fault.

Answer two things about the draft, and only these:
- upright: "yes" or "no". In the idle, does the draft stand on the floor the way the fighter in picture 3 stands? \
"no" means it lies down, is on its side, is upside down or is turned over where the fighter is not.
- attached: "yes" or "no". Is every part attached? "no" means a limb, the head or another part floats apart from \
the body or has come off.
- what: when either answer is "no", one sentence saying what you see; else "".
Small faults (a thick limb, a rough surface, a bent foot, a different size) are not asked about: answer "yes" unless \
the draft is plainly not standing as the fighter stands, or plainly in pieces."""

SCHEMA = {'type': 'object',
          'properties': {'upright': {'type': 'string', 'enum': ['yes', 'no']},
                         'attached': {'type': 'string', 'enum': ['yes', 'no']}, 'what': {'type': 'string'}},
          'required': ['upright', 'attached']}


def _local(node):
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] = m[:3, :3] @ np.diag(node.get('scale', [1, 1, 1]))
    m[:3, 3] = node.get('translation', [0, 0, 0])
    return m


def rests(glb):
    """{joint number: its rest world matrix} of a fighter's target file, in the game's space."""
    data = Path(glb).read_bytes()
    n = struct.unpack('<I', data[12:16])[0]
    nodes = json.loads(data[20:20 + n])['nodes']
    out = {}

    def walk(i, parent):
        world = parent @ _local(nodes[i])
        name = nodes[i].get('name', '')
        if name.startswith('JOBJ_'):
            out[int(name[5:])] = world
        for c in nodes[i].get('children', []):
            walk(c, world)
    children = {c for node in nodes for c in node.get('children', [])}
    for i in range(len(nodes)):
        if i not in children:
            walk(i, np.eye(4))
    return out


def moves(fighter_name, targets, pose):
    """{joint number: the 4x4 that carries a rest-pose point to where the frame ``pose`` has it}, in the game's space;
    None when the fighter has no such frame."""
    rest = rests(Path(targets) / f'{fighter_name}.glb')
    frame = fighter_frames.read(fighter_name, targets).get(pose)
    if frame is None:
        return None
    return {b['index']: np.array(b['matrix'], float).reshape(4, 4).T @ np.linalg.inv(rest[b['index']])
            for b in frame['bones'] if b['index'] in rest}


def posed(points, rows, by_joint):
    """Blender-space ``points`` skinned by ``rows`` ([vertex, joint number, weight]) through ``by_joint`` (moves)."""
    p = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(TO_BLENDER).T
    out, total = np.zeros((len(points), 4)), np.zeros(len(points))
    for v, j, w in rows:
        if j in by_joint:
            out[v] += w * (by_joint[j] @ p[v])
            total[v] += w
    held = total > 0
    out[held] /= total[held, None]
    out[~held] = p[~held]
    return (out @ TO_BLENDER.T)[:, :3]


def _sheet(by_pose, path):
    """The open scene's meshes drawn in POSES x VIEWS onto one sheet at ``path``: each moved by its own skin through
    the frames' moves, the floor a line."""
    import bpy
    from PIL import Image, ImageDraw
    from ...core.render import _font, _prepare_scene
    meshes = _prepare_scene()
    scene = bpy.context.scene
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new('w')
    scene.world.color = (0.30, 0.32, 0.36)
    scene.render.resolution_x = scene.render.resolution_y = TILE
    tables = {}
    for ob in meshes:
        me = ob.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        mw = np.array(ob.matrix_world)
        names = {g.index: g.name for g in ob.vertex_groups}
        rows = [(v.index, int(names[g.group][5:]), g.weight) for v in me.vertices for g in v.groups
                if names[g.group].startswith('JOBJ_') and g.weight > 0]
        tables[ob.name] = (co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3], rows)
        for m in list(ob.modifiers):
            ob.modifiers.remove(m)
        ob.parent = None
        ob.matrix_world = np.eye(4).tolist()
    cam = bpy.data.objects.new('tripwire', bpy.data.cameras.new('tripwire'))
    scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.clip_end = 100000
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = path.with_name(f'{path.stem}-tile.png')
    sheet = Image.new('RGB', (TILE * len(VIEWS), TILE * len(POSES)), (16, 18, 22))
    for r, (pose, label) in enumerate(POSES):
        spread = []
        for ob in meshes:
            pts = posed(*tables[ob.name], by_pose[pose])
            ob.data.vertices.foreach_set('co', pts.ravel())
            ob.data.update()
            spread.append(pts)
        spread = np.concatenate(spread)
        lo, hi = spread.min(0), spread.max(0)
        lo[2] = min(lo[2], 0.0)                 # the floor stays in the picture
        mid = (lo + hi) / 2
        size = float(max(hi[2] - lo[2], np.linalg.norm(hi[:2] - lo[:2])) * 1.08) or 1.0
        for c, (name, (dx, dy)) in enumerate(VIEWS):
            cam.location = (mid + np.array([dx, dy, 0]) * 400).tolist()
            cam.rotation_mode = 'XYZ'
            cam.rotation_euler = (np.pi / 2, 0, np.arctan2(dy, dx) + np.pi / 2)
            cam.data.ortho_scale = size
            scene.render.filepath = str(tile)
            still()
            with Image.open(tile) as rendered:
                img = rendered.convert('RGB')
            tile.unlink()
            d = ImageDraw.Draw(img)
            y = TILE / 2 - (0 - mid[2]) / size * TILE
            if 0 < y < TILE:
                d.line([(0, y), (TILE, y)], fill=(235, 235, 120), width=2)
            d.text((8, 6), f'{label}, {name}', fill=(235, 235, 235), font=_font(20, bold=True), stroke_width=2,
                   stroke_fill='black')
            sheet.paste(img, (c * TILE, r * TILE))
    sheet.save(path)
    return path


def _moves(fighter_name, targets):
    by_pose = {pose: moves(fighter_name, targets, pose) for pose, _ in POSES}
    return None if any(m is None for m in by_pose.values()) else by_pose


def draw(build_blend, fighter_name, targets, path):
    """Open ``build_blend`` and draw the draft in POSES x VIEWS onto one sheet at ``path``; None when the fighter has
    no such frames. Leaves the opened file as the scene: call it when nothing else needs the scene."""
    import bpy
    by_pose = _moves(fighter_name, targets)
    if by_pose is None:
        return None
    bpy.ops.wm.open_mainfile(filepath=str(Path(build_blend).resolve()))
    return _sheet(by_pose, path)


def draw_fighter(fighter_name, targets, path):
    """The fighter's own body (its target file) drawn as ``draw`` draws a draft: how the fighter itself stands and
    runs, which the look is shown beside the draft. Empties the scene first: call it when nothing else needs it."""
    import bpy
    by_pose = _moves(fighter_name, targets)
    if by_pose is None:
        return None
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(Path(targets) / f'{fighter_name}.glb'), disable_bone_shape=True)
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    return _sheet(by_pose, path)


def check(build_dir, fighter_name, targets, out, ask, source=None):
    """The tripwire's report on the build in ``build_dir``: {'alarm': bool, 'upright', 'attached', 'what'}, or
    {'skipped': why} when there is nothing to show or no answer came. ``source``: the source's own picture."""
    try:
        picture = draw(Path(build_dir) / 'build.blend', fighter_name, targets, Path(out) / 'tripwire.png')
        fighter = picture and draw_fighter(fighter_name, targets, Path(out) / 'tripwire-fighter.png')
    except Exception as e:      # noqa: BLE001 - a look that cannot be drawn must not fail a finished costume
        return {'skipped': f'could not draw the draft: {type(e).__name__}: {e}'}
    if not picture or not fighter:
        return {'skipped': 'the fighter has no idle or run frame'}
    if source is None or not Path(source).is_file():
        return {'skipped': 'no picture of the source'}
    return asked(ask, source, picture, fighter)


def asked(ask, source, picture, fighter):
    """The answer to the tripwire's question about one draft, as ``check`` reports it."""
    answer, info = ask(QUESTION, [str(source), str(picture), str(fighter)], SCHEMA, {}, route=ROUTE)
    if not answer:
        return {'skipped': 'no answer', 'error': (info or {}).get('error')}
    return {'alarm': answer.get('upright') == 'no' or answer.get('attached') == 'no',
            'upright': answer.get('upright'), 'attached': answer.get('attached'), 'what': answer.get('what') or ''}
