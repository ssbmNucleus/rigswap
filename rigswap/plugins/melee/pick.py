"""Which fighter a model suits, when no one names one (the owner, 2026-10-29, of Randall's importer: "I did like how
Randall could find the best match, like the user didn't always have to decide"; docs/STATE.md).

It is part structure and part meaning, and each goes where it belongs. Structure is the swap's own match, run on
every fighter at the seam in run.swap, after what every fighter reads alike (up, front, the labels, the forks): its
scale (run.scale) and correspond.match say what the swap would do to the model there (``fit``): whether parts that
meet in the model come apart at the fighter's rest (a round body's feet on a fighter with legs between), how far its
trunk, arms and legs are stretched or squashed (a fighter bone's rest length over its stretch of the model, where it
carries the model; the head, hands and feet keep their own size), a part of the fighter carrying none of it, how
tall the model stands there against the fighter, and what becomes of what it holds. Which fighters it comes apart on
least is a comparison over all of them, and code makes it: it offers those (the ones it holds together on, where it
holds on any), in order, the least stretched or resized first (``menu``); it never picks. Asked over every fighter,
the ask put Kamek on Bowser, 32% apart there against 8% on Ness, and said he came apart least there. How far a part
is stretched cuts no one: cut to the six least stretched, the list left Kirby off for Waddle Dee, a round body, by
arms a little shorter on him than on Link.
Meaning is one ask (``fighter``): the model upright in its own colours, the fighters offered with those facts in
words, the file's name and who the labels said it is. Whether the character is one of the fighters itself, in any
game's version, is an answer of its own, over the whole roster; else which of those offered suits its body (its
shape and build, not its kind) and its way of fighting. Its own size never counts: the swap scales it to whichever
fighter; the size it still stands at there is said. A skeleton that is a fighter's own (``own``) is that fighter,
unasked.

The fighters' skeletons and skins as the match reads them, and a picture of each, are read once per machine into the
targets folder (``prepare``).

    python -m rigswap.plugins.melee.pick <1-load dir> <out dir> [download name]
"""
import json
import sys
from pathlib import Path

import numpy as np
from ...core.render import still

AUTO = 'auto'           # the fighter: the one the model suits (run.swap, costume.py)
VOTES = 3              # asks: the answer wavers between repeats; the most given wins
NONE = 'none'
TILE = 640              # a view of the model
CARD = 256              # a fighter's picture
COLUMNS = 7             # fighters across picture B
CARRIES = .5            # a vertex over this share on a fighter bone is that bone's to carry (as run.swap's stretch)
APART = .05             # parts that meet in the model this share of the fighter's height apart at its rest, or more,
                        # have come off each other
TALL = 1.25             # scaled to the fighter, the model stands this many times its height, or as many times shorter,
                        # or more: said (run.scale reads room, not height)
PARTS = {'trunk': ('spine',), 'arms': ('left arm', 'right arm'), 'legs': ('left leg', 'right leg')}
SAID = {'DrMario': 'Dr. Mario', 'DonkeyKong': 'Donkey Kong', 'CaptainFalcon': 'Captain Falcon',
        'YoungLink': 'Young Link', 'GameAndWatch': 'Mr. Game & Watch'}
BODIES = {'biped': 'two legs', 'ball': 'a round body', 'quadruped': 'four legs', 'legless': 'no legs'}

QUESTION = ('Picture A shows one 3D game character in its own colours, from the front and from its left side. It will '
            "become a costume for one of the Super Smash Bros. Melee fighters, moving with that fighter's animations. "
            '{named}First, is this character itself one of Melee\'s fighters, in any game\'s version or art style? '
            'Answer "is" with that fighter, of these: {roster}; or "none". Then, which fighter in picture B suits '
            'it best? Match the body first, its shape and build: two legs, four or none, a round body, how long its '
            'arms and legs are, a tail or wings, its build; not what kind of creature it is, nor its game. Then '
            'how it fights and what it holds. Its own size never counts: the swap scales it to each fighter. Under '
            'each fighter, what the swap does to the costume on it: parts of it that come apart there, how tall it '
            "stands there where that is far from the fighter's height, how far its parts are stretched (their length "
            'there over their own: 2.0x twice as long, 0.5x half as long; the head, hands and feet keep their own '
            'size) and what becomes of what it holds: {offered}. A costume that comes apart, stands far taller or '
            'shorter than its fighter, or is stretched far looks broken: of the fighters its body suits, take the one '
            'it holds together on best. Answer that fighter as "fighter", as written here, and why, in one line.')
NAMED = 'Its file is named "{name}". '
KNOWN = 'Asked who it is, one look said: {character}. '


class Failed(RuntimeError):
    """The pick did not come to a fighter (costume.py says so as its own stage)."""


def prepare(targets):
    """{fighter: (its Rig as the match reads it, the room its surface walls in: run.room)} for every fighter in scope
    with a target in ``targets``, each read from its .glb once into targets/pick/<Name>.npz with its picture from the
    front (<Name>.png), and again when the .glb changes. It reads each in a scene of its own: run before the model's
    opens."""
    from .facts.fighters import in_scope
    folder = Path(targets) / 'pick'
    folder.mkdir(parents=True, exist_ok=True)
    fighters = {}
    for f in in_scope():
        glb = Path(targets) / f'{f.name}.glb'
        if not glb.is_file():
            continue
        stamp = np.array([glb.stat().st_mtime_ns, glb.stat().st_size], np.int64)
        path = folder / f'{f.name}.npz'
        if not (path.is_file() and path.with_suffix('.png').is_file() and np.array_equal(_stamp(path), stamp)):
            _read_fighter(f.name, glb, path, stamp)
        fighters[f.name] = _fighter(path, glb)
    return fighters


def _stamp(path):
    with np.load(path) as saved:
        return saved['stamp']


def _read_fighter(name, glb, path, stamp):
    """A fighter's Rig (rig.import_fighter) and room saved to ``path``, its picture beside it."""
    import bpy
    from ...core import rig as rigs, run
    bpy.ops.wm.read_homefile(use_empty=True)
    frig, _, meshes = rigs.import_fighter(glb)
    np.savez_compressed(path, stamp=stamp, names=np.array(frig.names), parents=frig.parents, heads=frig.heads,
                        tails=frig.tails, points=frig.points, weights=frig.weights, counts=np.array(frig.counts),
                        room=run.room(meshes))
    _card(name, meshes, path.with_suffix('.png'))


def _fighter(path, glb):
    from ...core.rig import Rig
    with np.load(path) as saved:
        rig = Rig(str(glb), [str(n) for n in saved['names']], saved['parents'], saved['heads'], saved['tails'],
                  points=saved['points'], weights=saved['weights'], counts=[int(c) for c in saved['counts']])
        return rig, float(saved['room'])


def own(rig, fighters):
    """The fighters whose skeleton is ``rig``'s own: each of their bones in it, under the same parent (a root under
    anything: a file may hang the skeleton off a node of its own)."""
    parent = {n: (rig.names[p] if p >= 0 else None) for n, p in zip(rig.names, rig.parents)}
    return [name for name, (frig, _) in fighters.items()
            if all(n in parent and (p < 0 or parent[n] == frig.names[p]) for n, p in zip(frig.names, frig.parents))]


def fit(rig, up, front, roles, forks, digits, fighters, targets):
    """Per fighter, what the swap would do to the model there: dict(fighter, scale, apart (``parted``), stretch
    (``stretched``), worst (the most any part is stretched or squashed, as a log: half and double alike), held
    (``held``)), or dict(fighter, failed) where its match fails. The swap's own steps (run.swap): the model stood up
    at its own size, scaled to the fighter (run.scale), stood at that size and read, matched (correspond.match)."""
    import contextlib
    import dataclasses
    import io
    from ...core import body as body_reading, correspond, repose, rig as rigs, run
    from .targets.bones import item_joints
    from .targets.weapons import read_weapons
    canonical = (np.array([0., 0., 1.]), np.array([0., -1., 0.]))

    def copy(r):
        return dataclasses.replace(r, points=r.points.copy(), heads=r.heads.copy(), tails=r.tails.copy())

    def quiet():
        """Body reading logs every reading: here, one per fighter."""
        return contextlib.redirect_stdout(io.StringIO())
    upright = copy(rig)
    repose.normalize(upright, up, front, 1.0)
    ours = run.room(rigs.objects(rig))
    rows = []
    for name, (frig, room) in fighters.items():
        try:
            description = correspond.read_description(Path(targets) / f'{name}.description.json')
            slots = read_weapons(Path(targets) / f'{name}.glb')
            with quiet():
                k = run.scale(ours, room)
                scaled = copy(upright)
                repose.normalize(scaled, *canonical, k)
                seen = body_reading.read(scaled, *canonical, roles, forks)
                grow, _ = run.grown(scaled, seen, frig, description)       # the size the swap will stand it at
                if abs(grow - 1.0) > 1e-3:
                    k *= grow
                    repose.normalize(scaled, *canonical, grow)
                    seen = body_reading.read(scaled, *canonical, roles, forks)
                m = correspond.match(scaled, seen, frig, description, items=item_joints(name), slots=slots,
                                     digits=digits)
            fch = correspond.fighter_chains(frig, description)
            stretch = stretched(scaled, m, fch)
            rows.append(dict(fighter=name, scale=round(float(k), 4), apart=parted(frig, m, fch),
                             stands=stands(scaled, frig), stretch=stretch, worst=worst(stretch),
                             held=held(scaled, seen, m, slots)))
            print(f'pick: {name}: {", ".join(_said(rows[-1]))}', flush=True)
        except Exception as e:      # a fighter the match fails on is not offered, and the record says why
            rows.append(dict(fighter=name, failed=f'{type(e).__name__}: {e}'[:200]))
            print(f'pick: {name}: {rows[-1]["failed"]}', flush=True)
    return rows


def stretched(rig, m, fch):
    """{part: how far the swap stretches it on this fighter}: over the bones along the fighter's trunk, its arms or
    its legs (``fch``: its chains) that carry the model, their rest lengths over their stretches of it; None where
    none of the part carries any of it. A part whose ends alone carry it (a round body whole on the chest, the feet
    with no leg above), or bones at a point (``_long``), keeps its own size: not said."""
    from ...core import repose
    bones = sorted(m.embedded)
    w = repose.fighter_weights(rig, m.owner, bones)
    carries = {fb: bool((w[:, i] > CARRIES).any()) for i, fb in enumerate(bones)}
    out = {}
    for part, chains in PARTS.items():
        chain = [fb for c in chains if c in fch for fb in fch[c].bones if fb in m.embedded]
        on = [fb for fb in chain if fb not in m.ends and fb not in m.placed and carries[fb] and _long(m.embedded[fb])]
        if not on:
            if chain and not any(carries[fb] for fb in chain):
                out[part] = None
            continue
        rest = sum(np.linalg.norm(np.subtract(*m.rest[fb])) for fb in on)
        inside = sum(np.linalg.norm(np.subtract(*m.embedded[fb])) for fb in on)
        out[part] = round(float(rest / inside), 2)
    return out


def _long(segment):
    """A bone with a length in the model; one at a point is a pivot (Fox's hips: nought at rest and in Sonic)."""
    return np.linalg.norm(np.subtract(*segment)) > 1e-6


def stands(rig, frig):
    """How many times the fighter's height the model stands, scaled to it (``rig``, upright)."""
    return round(float(np.ptp(rig.points[:, 2]) / (np.ptp(frig.points[:, 2]) or 1.)), 2)


def parted(frig, m, fch):
    """Where the model comes apart on this fighter: of its parts that meet on fighter bones not next to each other
    (Match.apart), the pair the re-pose takes farthest apart at the fighter's rest (repose.transforms, where they
    meet), as a share of the fighter's height: dict(gap, parts: [the part, the part it meets], as ``_parts`` names
    them); None where no pair is APART or more apart."""
    from ...core import repose
    moves = repose.transforms(m, sorted(m.embedded))
    index = {n: i for i, n in enumerate(frig.names)}
    height = float(np.ptp(frig.points[:, 2])) or 1.
    named = _parts(frig, fch)
    gaps = [(float(np.linalg.norm(moves[index[f]](p[None])[0] - moves[index[g]](p[None])[0])) / height,
             named[index[f]], named[index[g]])
            for (_, _, f, g), p in zip(m.apart, m.meets) if index[f] in moves and index[g] in moves]
    gap, a, b = max(gaps, default=(0., None, None))
    return dict(gap=round(gap, 3), parts=[a, b]) if gap >= APART else None


def _parts(frig, fch):
    """{fighter bone: the part of the body it is}: along the trunk, the trunk and the head; along a limb, the arm and
    the hand, or the leg and the foot; any other bone, its nearest such ancestor's."""
    named = {b: 'trunk' for b in fch['spine'].bones}
    named[fch['spine'].bones[-1]] = 'head'
    for side in ('left', 'right'):
        arm, leg = fch.get(f'{side} arm'), fch.get(f'{side} leg')
        if arm is not None:
            named.update({b: 'arm' for b in arm.bones})
            named[arm.bones[-1]] = 'hand'
        if leg is not None:
            ankle = leg.anchors[1] if len(leg.anchors) > 1 and leg.anchors[1] is not None else len(leg.bones)
            named.update({b: 'foot' if i >= ankle else 'leg' for i, b in enumerate(leg.bones)})

    def of(b):
        while b >= 0 and b not in named:
            b = frig.parents[b]
        return named.get(b, 'body')
    return {b: of(b) for b in range(len(frig.names))}


def worst(stretch):
    """The most any part is stretched or squashed, as a log: half and double alike."""
    return max((abs(float(np.log(r))) for r in stretch.values() if r), default=0.)


def held(rig, body, m, slots):
    """What becomes of what the model holds on this fighter, in words: a weapon slot takes it (correspond._match_props;
    where the slots are of two kinds, what it is decides, asked later), a fighter fighting empty-handed leaves it off
    (the look is told so: look.EMPTY), or what it carries outside its body goes in its hand; with nothing held, the fighter's own
    weapon stays."""
    from ...core import correspond
    kinds = list(dict.fromkeys(s['kind'] for s in correspond.armed(slots)))
    found = correspond.props(rig, body)
    if m.replaced:
        return 'what it holds takes the place of its ' + ' and '.join(dict.fromkeys(r['kind'] for r in m.replaced))
    if found and len(kinds) > 1:
        return f"what it holds takes the place of its {' or '.join(kinds)}, whichever it is"
    if any(p[4] for p in found):
        return 'what it holds is left off' + (f', and it keeps its own {kinds[0]}' if kinds else
                                              ': it fights empty-handed')
    if found:
        return 'what it carries goes in its hand'
    return f"it keeps its own {' and '.join(kinds)}" if kinds else ''


def menu(rows):
    """The fighters offered: of those the match read, the ones the model comes apart on least, no more than APART
    beyond the least (those it holds together on, where it holds on any), in order of how far apart, then the least
    stretched or resized (``worst``, ``stands``). Which comes apart least is a comparison over every fighter, code's
    to make: asked over all, the ask put Kamek on Bowser, 32% apart there and 8% on Ness, and said it came apart
    least there. How far it is stretched cuts no one (Kirby's 0.5x arms are Waddle Dee's)."""
    read = [r for r in rows if 'failed' not in r]
    least = min((gap(r) for r in read), default=0.)
    return sorted((r for r in read if gap(r) < least + APART),
                  key=lambda r: (gap(r), max(r['worst'], abs(float(np.log(r.get('stands') or 1.))))))


def gap(row):
    """How far apart the model comes on a fighter (``fit``'s row): nought where it holds together."""
    return (row['apart'] or {}).get('gap', 0.)


def tally(answers, said, offered):
    """The fighter most of ``answers`` give, the first ask's on a tie: each answer's "is" when it names a fighter
    (``said``: {as said: name}), else its "fighter" when one of ``offered``. -> (name, why, [each answer's fighter],
    whether it is the character itself), or None when none answers."""
    given = []
    for a in answers:
        a = a or {}
        if said.get(a.get('is')):
            given.append((said[a['is']], a.get('why') or '', True))
        elif said.get(a.get('fighter')) in offered:
            given.append((said[a['fighter']], a.get('why') or '', False))
    if not given:
        return None
    names = [n for n, _, _ in given]
    picked = max(names, key=lambda n: (names.count(n), -names.index(n)))
    _, why, itself = next(g for g in given if g[0] == picked)
    return picked, why, names, itself


def fighter(rig, up, front, roles, forks, digits, fighters, targets, out, ask, name=None, character=None):
    """(the fighter's name, the record {fighter, why, picked, votes, itself, own, offered: [{fighter, scale, apart,
    stands, stretch, held}], apart: [the same, for the fighters it comes apart on more], unread: {fighter: why}})
    for the model ``rig`` as run.swap has read it so far, the pictures drawn into ``out``. ``fighters``: prepare's;
    ``name``: the file's; ``character``: who the labels said it is. With no answer, the menu's first, said
    unpicked."""
    mine = own(rig, fighters)
    if len(mine) == 1:
        return mine[0], dict(fighter=mine[0], why="its skeleton is the fighter's own", picked=True, own=True)
    rows = fit(rig, up, front, roles, forks, digits, {n: fighters[n] for n in mine or fighters}, targets)
    offered = menu(rows)
    if not offered:
        raise ValueError('no fighter matched: ' + '; '.join(f"{r['fighter']}: {r['failed']}" for r in rows))
    facts = ('fighter', 'scale', 'apart', 'stands', 'stretch', 'held')
    record = dict(offered=[{k: r.get(k) for k in facts} for r in offered],
                  apart=[{k: r.get(k) for k in facts} for r in rows if 'failed' not in r and r not in offered],
                  unread={r['fighter']: r['failed'] for r in rows if 'failed' in r})
    pictures = [str(draw(rig, up, front, Path(out) / 'pick.png')),
                str(sheet(targets, [r['fighter'] for r in offered], Path(out) / 'pick-fighters.png'))]
    said = {SAID.get(n, n): n for n in fighters}
    question = QUESTION.format(
        named=(NAMED.format(name=name) if name else '') + (KNOWN.format(character=character) if character else ''),
        roster=', '.join(said), offered='; '.join(_words(r, targets) for r in offered))
    schema = {'type': 'object', 'properties': {'is': {'type': 'string', 'enum': [*said, NONE]},
                                               'fighter': {'type': 'string',
                                                           'enum': [SAID.get(r['fighter'], r['fighter'])
                                                                    for r in offered]},
                                               'why': {'type': 'string'}},
              'required': ['is', 'fighter', 'why'], 'additionalProperties': False}
    asked = [ask(question + ('' if r == 0 else f' (Ask {r + 1} of {VOTES}.)'), pictures, schema,
                 {'is': NONE, 'fighter': None, 'why': ''}, route='melee fighter') for r in range(VOTES)]
    answers = [a for a, _ in asked]
    record['answers'] = [dict(answer=a, source=(info or {}).get('source'), error=(info or {}).get('error'))
                         for a, info in asked]
    found = tally(answers, said, {r['fighter'] for r in offered})
    if found is None:
        first = offered[0]['fighter']
        return first, dict(record, fighter=first, why='', picked=False)
    picked, why, votes, itself = found
    return picked, dict(record, fighter=picked, why=why, picked=True, votes=votes, itself=itself)


def _words(row, targets):
    """A fighter offered, in words: its name, its body, and what the swap would do to the model on it."""
    return f"{SAID.get(row['fighter'], row['fighter'])} ({_about(targets, row['fighter'])}): {', '.join(_said(row))}"


def _said(row):
    """What the swap would do to the model on a fighter (``fit``'s row), in words."""
    apart = row['apart']
    said = [f"its {apart['parts'][0]} and {apart['parts'][1]} come apart, {apart['gap']:.0%} of the fighter's height"
            ] if apart else []
    tall = row.get('stands') or 1.
    if not 1 / TALL < tall < TALL:
        said.append(f"it stands {tall:.1f}x the fighter's height")
    said += [f"its {part} {'carries' if part == 'trunk' else 'carry'} none of it" if r is None else f'{part} {r:.1f}x'
             for part, r in row['stretch'].items()] or ['every part keeps its own size']
    return said + ([row['held']] if row['held'] else [])


def _about(targets, name):
    """A fighter's body in words: its kind and the parts it has (a tail), from its target's description."""
    d = json.loads((Path(targets) / f'{name}.description.json').read_text(encoding='utf-8'))
    return ', '.join([BODIES.get(d.get('body_type'), d.get('body_type') or 'two legs')] +
                     [f'a {h}' for h in d.get('has') or []])


def _camera(scene, name, size):
    import bpy
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = size * 1.1
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 10
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def _shoot(scene, tile, size):
    """The camera's view onto white, ``size`` pixels across."""
    import bpy
    from PIL import Image
    scene.render.filepath = str(tile)
    still()
    img = Image.new('RGB', (size, size), 'white')
    with Image.open(tile) as rendered:
        rgba = rendered.convert('RGBA')
    img.paste(rgba, (0, 0), rgba)
    tile.unlink()
    return img


def _card(name, meshes, path):
    """A fighter's picture: its target from the front at rest, its name under it."""
    import bpy
    from mathutils import Vector
    from PIL import Image, ImageDraw
    from ...core.front import _looking
    from ...core.render import _font, _prepare_scene, _world_points
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = CARD
    pts = _world_points(meshes)
    lo, hi = pts.min(0), pts.max(0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    cam = _camera(scene, 'melee_pick_card', size)
    cam.location = Vector((centre - np.array([0., 1., 0.]) * size * 3).tolist())      # targets face -y
    cam.rotation_euler = _looking(np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    card = Image.new('RGB', (CARD, CARD + 40), 'white')
    card.paste(_shoot(scene, path.with_name(f'{path.stem}-tile.png'), CARD), (0, 0))
    ImageDraw.Draw(card).text((8, CARD + 4), SAID.get(name, name), fill='black', font=_font(26, bold=True))
    card.save(path)


def sheet(targets, names, path):
    """Picture B: the fighters ``names`` in rows of COLUMNS, in order (prepare's pictures). Returns the path."""
    from PIL import Image, ImageDraw
    from ...core.render import _font
    rows = -(-len(names) // COLUMNS)
    page = Image.new('RGB', (CARD * min(len(names), COLUMNS), (CARD + 40) * rows + 60), 'white')
    ImageDraw.Draw(page).text((12, 8), 'B', fill='black', font=_font(48, bold=True))
    for i, n in enumerate(names):
        with Image.open(Path(targets) / 'pick' / f'{n}.png') as card:
            page.paste(card.convert('RGB'), ((i % COLUMNS) * CARD, 60 + (i // COLUMNS) * (CARD + 40)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)
    return path


def draw(rig, up, front, path):
    """Picture A: the model at rest in its own colours, upright, from the front and from its left side, only the meshes
    ``rig`` reads. Returns the path; the scene shows and hides what it did before."""
    import bpy
    from mathutils import Vector
    from PIL import Image, ImageDraw
    from ...core.front import _looking
    from ...core.render import _font, _prepare_scene
    scene = bpy.context.scene
    hidden = {ob: (ob.hide_render, ob.hide_viewport, ob.hide_get()) for ob in scene.objects}
    poses = {ob: ob.data.pose_position for ob in bpy.data.objects if ob.type == 'ARMATURE'}
    up, front = np.asarray(up, float), np.asarray(front, float)
    lo, hi = rig.points.min(0), rig.points.max(0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cam = None
    try:
        for ob in poses:
            ob.data.pose_position = 'REST'
        _prepare_scene()
        for ob in scene.objects:
            if ob.type == 'MESH' and ob.name not in rig.meshes:
                ob.hide_render = True
        scene.render.resolution_x = scene.render.resolution_y = TILE
        cam = _camera(scene, 'melee_pick', size)
        page = Image.new('RGB', (2 * TILE + 20, TILE), 'white')
        for i, side in enumerate((front, np.cross(up, front))):
            cam.location = Vector((centre + side * size * 3).tolist())
            cam.rotation_euler = _looking(-side, up)
            page.paste(_shoot(scene, path.with_name(f'{path.stem}-{i}.png'), TILE), (i * (TILE + 20), 0))
        ImageDraw.Draw(page).text((12, 8), 'A', fill='black', font=_font(48, bold=True))
        page.save(path)
    finally:
        if cam is not None:
            bpy.data.objects.remove(cam)
        for ob, pose in poses.items():
            ob.data.pose_position = pose
        for ob, (render, viewport, hide) in hidden.items():
            ob.hide_render, ob.hide_viewport = render, viewport
            ob.hide_set(hide)
    return path


if __name__ == '__main__':
    import os
    from ... import ai
    from ...core import run
    load_dir, out = sys.argv[1:3]
    r = run.swap(load_dir, AUTO, out, run._ask(ai.answers(os.environ.get('RIGSWAP_CACHE') or Path(out) / 'cache',
                                                          Path(out) / 'cache')),
                 export=False, name=sys.argv[3] if len(sys.argv) > 3 else None, pick_only=True)
    print(json.dumps(r['fighter_pick'], indent=1))
