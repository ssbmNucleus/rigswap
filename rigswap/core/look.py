"""The everyday look: which of the model's pieces the costume shows, decided by one look at all of them.

A file from a game holds every form its pieces take, all present at once: the faces a character pulls, its mouths,
eyes open and shut, hand poses, a prop in the hand and on the back, effects, coarser copies for distance. The costume
shows one everyday look. Which pieces those are is meaning, not structure: names follow each game's own conventions,
and shape can't tell a hand round a cap from a head shell round its face plate. Every rule that tried (variants.py,
2026-10-25 to 11-18: forms read from overlapping boxes, one form per surface group, pieces hosted on others, copies,
three voted asks) dropped pieces the answers had kept: Kamek's hands, Diddy's eyes, Frieza's head shell and forearms.
Built from the answers alone, with no rule after them, Frieza came out whole (run keep-1, 2026-11-18). So one look
decides, seeing all of it, and nothing overrides it:

- every piece (variants.units: a switch the file names, else a mesh of one name riding bones in common; in a file of
  one mesh, each surface welded apart from the rest: variants.surfaces) is offered,
  on a numbered tile alone in its colours with the rest grey (pieces.draw), after the whole model drawn from four
  sides, with each piece's name and the bone it rides. What the files' names usually mean is told (GLOSSARY), never
  applied as a rule;
- the answer names the tiles to leave out, and what it doesn't name stays: a piece kept by mistake is a blemish the
  person can see and name (pieces.py, "take the jetpack off"), a piece left out by mistake is a body with a hole.
  Can't tell leaves nothing out, and so does an answer leaving out every piece (no look is left to wear);
- the answer stands: nothing asks again. A second ask (the look the answer leaves, from four sides beside the same
  tiles, to bring a piece back or leave one more out; 11-18) repaired no cheap model's first answer and broke right
  ones (lab 2026-11-19: Gemini Flash's dropped u-mario's face). The first answer is the whole result;
- the meshes left out go into the record (``drop``) and the build reads them from there, as a hand's record.

What the fighter stands on is a game fact and is told too: it fights on its own feet, and the game draws the floor
and the shadow, so a vehicle, mount, stand, floor or shadow card the file's game showed it with is the look's to
leave out (Taz's go-kart, 2026-11-19), never a rule's after it.

What the fighter holds is a game fact and is told, not applied after: a fighter that holds its weapon through play
holds it out once (variants.holding), one that holds nothing would hold a prop in its hand through every move
(EMPTY; correspond.empty_hands dropped those after the answers until 11-18).

A file whose own idle settles what shows (load.stance) is not asked: only its coarser levels of detail go
(variants.coarser), as before. A model of one piece has nothing to choose.

Worn copies in one place, both kept (Inkling's eyelid band over her eye), are layers: one draws over the other
(``layers``, read by paint.py). That is structure, read from the kept pieces whoever kept them.
"""
from pathlib import Path

import numpy as np

from . import pieces, variants
from .render import still

ROUTE = 'core look'
VIEW = 640                      # one side's square in the four-sides sheet: 1280 wide, as the ask sends it

GLOSSARY = ('What files usually name: Blink, Halfblink, Close and Shut are eyes shut or closing; N, Normal, Neutral, '
            'Default and Openblink the neutral state; a motion\'s name (Attack, Ouch, Talk, Damage, Appeal, Final) a '
            'face for that motion; LOD, Low or a higher level number a coarser copy of a finer piece. Blender adds '
            '.001, .002 to a name already taken, so a left and a right copy may share a name. Japanese romaji is '
            'common: kao face, kuchi mouth, me eye, te hand, gu fist, pa open hand, ude arm.')

EMPTY = (' This fighter holds nothing through play, so a prop in the character\'s hand would stay in it through every '
         'move: leave out props held in a hand, and keep one stowed on the body (on the back, at the belt).')

QUESTION = ('A 3D model{who} is becoming a costume for {fighter} in a fighting game: the costume is how the character '
            'looks in every move. A file from a game holds every form a part takes, all present at once: faces and '
            'mouths for each expression, eyes open and shut, hand poses (open, fist, gripping), effects, props in '
            'the hand and stowed, coarser copies for distance. Drawn together they overlap. It may also hold what '
            'the character is only shown with in its own game: the fighter fights on its own feet on the stage, '
            'and the game draws the floor and the shadow, so a vehicle or mount it rides, a stand or base it is '
            'posed on, and a floor or shadow under it are not part of its look.\n\n'
            'The first picture is the whole model, every piece shown, from the front, its own right side, the back '
            'and its own left side. Then each numbered tile shows '
            'one piece alone in its own colours with the rest of the model grey; a tile marked back or side is seen '
            'from there. The pieces, with the bone each rides: {names}.{held}\n\n'
            'Which pieces does the character\'s everyday look leave out? List under "leave_out" the tile of every '
            'form that is not the everyday one: each expression but the neutral face, each mouth but the neutral '
            'one, eyes shut or half shut, hand poses but the relaxed one, effects, a coarser copy where a finer one '
            'fills the same place; and every piece of a vehicle, mount, stand, floor or shadow. Keep exactly one '
            'form of each part. Never leave out a part of the body or of the '
            'outfit (a head or the shell round a face, an ear, a forearm, a guard, a tail). When you are not sure, '
            'keep the piece: a piece kept by mistake can be taken off later, a body part left out is a hole. Set '
            'cant_tell only if you cannot tell at all. {glossary}')


def schema(count):
    """The answer: the tiles to leave out (numbers 1 to ``count``) and whether it could not tell."""
    return {'type': 'object', 'properties': {
        'leave_out': {'type': 'array', 'items': {'type': 'integer', 'enum': list(range(1, count + 1))}},
        'cant_tell': {'type': 'boolean'}}, 'required': ['leave_out'], 'additionalProperties': False}


def numbered(offered, answer, key):
    """The pieces an answer lists under ``key`` (tile numbers from 1), each once, in the tiles' order."""
    picked = {int(n) for n in (answer or {}).get(key) or [] if isinstance(n, (int, float)) and not isinstance(n, bool)}
    return [u for n, u in enumerate(offered, 1) if n in picked]


def left_out(offered, answer):
    """The pieces the look leaves out: those the answer lists, none when it could not tell. An answer listing every
    piece names no look at all, the character gone with the rest: it could not tell either."""
    gone = [] if (answer or {}).get('cant_tell') else numbered(offered, answer, 'leave_out')
    return [] if len(gone) == len(offered) else gone


def layers(us, kept, size):
    """The worn copies in one place, both kept (``kept``: their keys): the same shape (variants._copies) filling
    mostly the same box. -> [(the meshes of one, of the other)]."""
    pad = 0.01 * size
    worn = [u for u in us if u.key in set(kept)]
    return [(a.meshes, b.meshes) for i, a in enumerate(worn) for b in worn[i + 1:]
            if variants._copies(a, b) and variants._boxes(a, b, pad)[0] >= 0.5]


def keep(rig, up, front, out, ask, fighter, character=None, weapons=(), settled=False):
    """Drop from the open scene the meshes the everyday look leaves out, and read the rig again without them.
    ``weapons``: the kinds the fighter holds through play (none: it holds nothing).
    -> (rig, report): {'offered' (keys), 'answer', 'dropped' (mesh names), 'layers', 'coarser', 'source'}."""
    size = float(np.ptp(rig.points, axis=0).max())
    us, _ = variants.units(rig)
    report = {'offered': [], 'answer': None, 'dropped': [], 'layers': [], 'coarser': []}
    if settled:                         # the file's own idle has said what shows: only coarser copies go
        finer = variants.coarser(us, size)
        report['coarser'] = [{'unit': k, 'finer': f} for k, f in finer.items()]
        gone = [u for u in us if u.key in finer]
    elif len(us) < 2:
        gone = []
    else:
        gone = ask_look(rig, us, up, front, Path(out), ask, fighter, character, weapons, report)
    report['dropped'] = sorted(m for u in gone for m in u.meshes)
    report['layers'] = layered(rig, us, report['dropped'])
    return variants.drop(rig, report['dropped']), report


def layered(rig, us, dropped):
    """The layers among the pieces ``us`` of the rig a look keeps, ``dropped`` (mesh names) gone, as the report and
    paint.py take them: [[the meshes of one, of the other]]. Whoever kept them: an ask, or a record."""
    gone = set(dropped)
    kept = [u.key for u in us if not set(u.meshes) <= gone]
    return [[list(a), list(b)] for a, b in layers(us, kept, float(np.ptp(rig.points, axis=0).max()))]


def ask_look(rig, us, up, front, out, ask, fighter, character, weapons, report):
    """The ask (the module's docstring). -> the pieces left out."""
    report['offered'] = [u.key for u in us]
    names = '; '.join(f'{n} {u.key} ({rig.names[u.bone] if u.bone >= 0 else "no bone"})'
                      for n, u in enumerate(us, 1))
    tiles = [str(p) for p in pieces.draw(us, us, set(), up, front, out / 'look.png', over=True)]
    whole = str(views(rig, [m for u in us for m in u.meshes], up, front, out / 'look-whole.png'))
    question = QUESTION.format(who=f' of {character}' if character else '', fighter=fighter, names=names,
                               held=variants.holding(list(weapons)) if weapons else EMPTY, glossary=GLOSSARY)
    answer, info = ask(question, [whole] + tiles, schema(len(us)),
                       {'leave_out': [], 'cant_tell': True}, ROUTE)
    report['answer'], report['source'] = answer, (info or {}).get('source')
    return left_out(us, answer)


def views(rig, meshes, up, front, path):
    """The meshes ``meshes`` alone, in their own colours, from the front, the right, the back and the left: one
    sheet of four squares, each labelled. -> its path."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    shown = set(meshes)
    hidden = [ob for ob in scene.objects if ob.type == 'MESH' and not ob.hide_render and ob.name not in shown]
    for ob in hidden:
        ob.hide_render = True
    up = np.asarray(up, float) / np.linalg.norm(up)
    front = np.asarray(front, float) - (np.asarray(front, float) @ up) * up
    front /= np.linalg.norm(front)
    across = np.cross(up, front)
    starts = dict(zip(rig.meshes, np.cumsum([0] + [len(bpy.data.objects[n].data.vertices) for n in rig.meshes])))
    rows = np.concatenate([np.arange(starts[n], starts[n] + len(bpy.data.objects[n].data.vertices))
                           for n in rig.meshes if n in shown]) if shown else np.arange(len(rig.points))
    pts = rig.points[rows] if len(rows) else rig.points
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    cam = bpy.data.objects.new('core_look', bpy.data.cameras.new('core_look'))
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 20
    scene.collection.objects.link(cam)
    before, was = scene.camera, (scene.render.resolution_x, scene.render.resolution_y)
    scene.camera = cam
    scene.render.resolution_x = scene.render.resolution_y = VIEW
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shot = path.with_name(f'{path.stem}-tile.png')
    sheet = Image.new('RGB', (2 * VIEW, 2 * VIEW), (16, 18, 22))
    try:
        # the model's own right is front x up (-across): the second square looks at it from there
        for i, (label, side) in enumerate((('front', front), ('right side', -across), ('back', -front),
                                           ('left side', across))):
            right = np.cross(up, side)
            u, v = pts @ right, pts @ up
            middle = right * (u.min() + u.max()) / 2 + up * (v.min() + v.max()) / 2 + side * float((pts @ side).mean())
            m4 = np.eye(4)
            m4[:3, :3], m4[:3, 3] = np.column_stack([right, up, side]), middle + side * size * 4
            cam.matrix_world = Matrix(m4.tolist())
            cam.data.ortho_scale = float(max(np.ptp(u), np.ptp(v)) * 1.1) or 1.0
            scene.render.filepath = str(shot)
            still()
            img = Image.new('RGB', (VIEW, VIEW), (150, 155, 165))
            with Image.open(shot) as rendered:
                rgba = rendered.convert('RGBA')
            shot.unlink()
            img.paste(rgba, (0, 0), rgba)
            ImageDraw.Draw(img).text((10, 6), label, fill='black', font=_font(30, bold=True), stroke_width=3,
                                     stroke_fill='white')
            sheet.paste(img, ((i % 2) * VIEW, (i // 2) * VIEW))
        sheet.save(path)
    finally:
        for ob in hidden:
            ob.hide_render = False
        scene.camera = before
        scene.render.resolution_x, scene.render.resolution_y = was
        camera = cam.data
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(camera)
    return path
