"""Which way each limb goes where its bones branch: on toward its hand or foot, or no farther.

A limb's bones branch where it carries something besides itself (a sleeve, a panel, a strap, a flame, a held sword,
helper bones shaping the skin at a joint) and where its hand or foot spreads into fingers or toes. Which branch is the
limb, the skeleton doesn't say: Galahad's sleeve outweighs her hand, so her arm ran into the sleeve and the fighter's
hand went to her elbow; Ponyta's flame outweighs her hoof; Rhinedottir's sleeve strings outweigh her forearm; counting
branches stopped Marth's right arm at its forearm helpers (the fork dig, 10-27: no measurement settled them all). So it
is asked, as the labels are (labels.py): the model in its own colours from the front, then every fork of the limbs the
labels named, its branches each in a colour of its own, seen across the limb from two sides (the other limbs left
out). Per fork, the AI says what each branch is: the limb itself going on, digits (fingers, a thumb, toes, claws),
something the limb carries, or a helper bone; a branch's kind is the one more than half of VOTES asks give. A twin's
fork whose branches mirror an asked fork's takes its answers, mirror for mirror.

**The hand is where the fingers spread, the foot where the toes do** (ways): the limb runs from its root to the fork
nearest it with a branch of digits. Asked instead which way the limb went at each fork, the answers wavered where a
hand's skin and bones disagree (fk-2): tmario's wrist helper carries the back of his hand, and the asks took it for
the way on, though they saw his five fingers spread from the bone beside it; a hand whose fingers spread from its palm
one knuckle at a time read as going on into the finger fan. A limb whose digits don't branch (a hoof, a mitten) goes
on along its one limb branch at each fork, and ends where several are (a foot in parts). Body reading (body._limbs)
follows the ways out from each limb's root, and where there is none, the heaviest branch.
"""
from pathlib import Path

import numpy as np

from . import body, labels
from .render import still

VOTES = 3                   # asks: the answer wavers between repeats; per branch the most given wins
PER_PICTURE = 3             # forks drawn in one picture, a row each
TILE = 480
COLOURS = labels.COLOURS + [('yellow', (0.95, 0.85, 0.0)), ('pink', (1.0, 0.45, 0.75)),
                            ('brown', (0.45, 0.25, 0.1))]
LIGHT = (0.78, 0.8, 0.83)   # the limb itself, beside its fork's branches
KINDS = ('limb', 'digits', 'carried', 'helper')

QUESTION = (
    'The first picture shows one game character in its own colours from the front, in the pose its model file is '
    'bound in, often arms out in a T. The pictures after it show its limbs where their bones branch: each row is one '
    'fork, numbered, seen from two sides across the limb (an arm from the front and from above, a leg from the front '
    'and from its side). The limb is light grey and the rest of the character dark grey, its other limbs left out; '
    'each branch leaving the fork is in a colour of its own. A limb branches where it carries something besides '
    'itself (a sleeve, a cloth or armour panel, a strap, a flame, a held weapon or shield, small helper bones that '
    'shape the skin at a joint) and where its hand or foot spreads into fingers or toes. First say who the character '
    'is, if you know. Then, for every numbered fork and each of its colours, say what that branch is on the '
    'character, and its kind: "limb" if it is the arm or leg itself going on (the rest of the arm or leg, its hand or '
    'palm, its foot, paw or hoof); "digits" if it is a finger, a thumb, a toe or a claw, or several of them; '
    '"carried" if it is something the limb carries besides itself (a sleeve, cloth, a strap, armour, a flame, hair, '
    'a held weapon or shield); "helper" if it is a small bone that only shapes the skin near a joint. Judge each '
    'branch by what it is on the character, as its art shows it, not by its size or its bones\' names alone. Answer '
    'cant_tell only if you cannot tell. The forks, by the bone each is at, and per colour the branch\'s first bone '
    'and the bones at its ends: {forks}.')
SCHEMA = {'type': 'object', 'properties': {
    'character': {'type': 'string'},
    'forks': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'number': {'type': 'integer'},
        'branches': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'colour': {'type': 'string', 'enum': [c for c, _ in COLOURS]},
            'what': {'type': 'string'},
            'kind': {'type': 'string', 'enum': [*KINDS, 'cant_tell']}},
            'required': ['colour', 'what', 'kind'], 'additionalProperties': False}}},
        'required': ['number', 'branches'], 'additionalProperties': False}}},
    'required': ['character', 'forks'], 'additionalProperties': False}


def candidates(rig, up, roles):
    """The forks of the limbs labels named (``roles``, by root name), in order out from each limb's root: per fork its
    bone, its limb's root and role, and its branches (the children carrying skin, heaviest first, one per colour).
    A fork whose branches are all twins of its twin's, at a fork already listed, is left to that one. Returns (mass,
    forks, {mirrored fork: (its twin, {twin branch: own branch})}, the roots of every named limb)."""
    mass, _, pairs, twins = body.limb_pairs(rig, up)
    found, mirrored, roots = [], {}, []
    for pair in pairs:
        role = next((roles.get(rig.names[b]) for b in pair if roles.get(rig.names[b]) in body.ROLES), None)
        if role is None:
            continue
        roots += pair
        listed = set()
        for root in pair:
            out = [root]
            for b in out:
                out += mass.kids[b]
            for b in out:
                kids = sorted(mass.kids[b], key=lambda k: -mass.total[k])
                if len(kids) < 2:
                    continue
                t = twins.get(b)
                if t in listed and {twins.get(k) for k in kids} == set(mass.kids[t]):
                    mirrored[b] = (t, {twins[k]: k for k in kids})
                    continue
                listed.add(b)
                found.append(dict(bone=b, root=root, role=role, branches=kids[:len(COLOURS)]))
    return mass, found, mirrored, roots


def tally(answers, found):
    """Per listed fork and branch, the kind most answers give it, when more than half of them do: {fork: {branch:
    kind}}."""
    out = {}
    for n, f in enumerate(found, 1):
        out[f['bone']] = {}
        for (colour, _), k in zip(COLOURS, f['branches']):
            given = [b.get('kind') for b in _said(answers, n, colour)]
            best = max(set(given), key=given.count, default=None)
            if best in KINDS and 2 * given.count(best) > len(answers):
                out[f['bone']][k] = best
    return out


def ways(mass, roots, kinds):
    """Which way each limb goes out from its root, by what its forks' branches are (``kinds``, {fork: {branch:
    kind}}): {bone: the bone it goes on to, or None where it ends}. Its hand or foot is where its digits spread: the
    fork nearest the root with a branch of digits (of two as near, the one with more), and the limb runs to it. A limb
    with none goes on along its one limb branch at each fork, and ends where several are; where none is, the ways
    stop."""
    goes = {}
    for root in roots:
        out, depth = [root], {root: 0}
        for b in out:
            for k in mass.kids[b]:
                depth[k] = depth[b] + 1
                out.append(k)
        spread = [b for b in out if 'digits' in kinds.get(b, {}).values()]
        if spread:
            b = min(spread, key=lambda x: (depth[x], -list(kinds[x].values()).count('digits')))
            goes[b] = None
            while b != root:
                if len(mass.kids[mass.parents[b]]) > 1:
                    goes[mass.parents[b]] = b
                b = mass.parents[b]
            continue
        b = root
        while mass.kids[b]:
            if len(mass.kids[b]) == 1:
                b = mass.kids[b][0]
                continue
            own = [k for k in mass.kids[b] if kinds.get(b, {}).get(k) == 'limb']
            if len(own) != 1:
                if own:
                    goes[b] = None
                break
            goes[b] = b = own[0]
    return goes


def read(rig, up, front, out, ask, roles):
    """Ask what each branch is at the forks of the limbs the labels named, with pictures drawn into ``out``. Returns
    ({bone name: the name of the bone its limb goes on to, or None where the limb ends} (ways), the report)."""
    mass, found, mirrored, roots = candidates(rig, up, roles)
    if not found:
        return {}, {'forks': []}
    pictures = draw(rig, mass, found, roots, up, front, Path(out) / 'forks.png')
    kids = rig.children()
    listed = '; '.join(
        f'{n} (picture {(n - 1) // PER_PICTURE + 1}, {"an arm" if f["role"] == "arms" else "a leg"}, at '
        f'{rig.names[f["bone"]]}): ' + ', '.join(f'{COLOURS[i][0]} {_named(rig, kids, k)}'
                                                  for i, k in enumerate(f['branches']))
        for n, f in enumerate(found, 1))
    said = []
    for r in range(VOTES):
        answer, _ = ask(QUESTION.format(forks=listed) + f' (Ask {r + 1} of {VOTES}.)',
                        [str(p) for p in pictures], SCHEMA, {'forks': []}, route='core forks')
        said.append(answer)
    kinds = tally(said, found)
    for b, (t, mirror) in mirrored.items():
        kinds[b] = {mirror[k]: kind for k, kind in kinds[t].items()}
    goes = ways(mass, roots, kinds)
    named = {rig.names[b]: None if c is None else rig.names[c] for b, c in goes.items()}
    report = dict(
        character=[(a or {}).get('character') for a in said if (a or {}).get('character')],
        goes=named,
        kinds={rig.names[k]: kind for b in kinds for k, kind in kinds[b].items()},     # a twin's fork's too
        forks=[dict(number=n, bone=rig.names[f['bone']],
                    branches={c: rig.names[k] for (c, _), k in zip(COLOURS, f['branches'])},
                    kinds={rig.names[k]: kind for k, kind in kinds[f['bone']].items()},
                    said={c: [f"{b.get('what')} ({b.get('kind')})" for b in _said(said, n, c)]
                          for (c, _), k in zip(COLOURS, f['branches'])})
               for n, f in enumerate(found, 1)])
    print('forks: ' + str(named))
    return named, report


def _said(answers, number, colour):
    """What the answers said of fork ``number``'s branch in ``colour``."""
    return [b for a in answers for p in (a or {}).get('forks') or [] if p.get('number') == number
            for b in p.get('branches') or [] if b.get('colour') == colour]


def draw(rig, mass, found, roots, up, front, path):
    """The model at rest in its own colours from the front, then a picture per PER_PICTURE forks, a row each: the
    fork seen across its limb from two sides (an arm from the front and from above; a leg from the front and from the
    side it is on), framed on what the fork carries and the bone it hangs from. The fork's branches are in their
    colours, the rest of its limb light, the body grey, and the other limbs left out (masked). A vertex is a branch's
    when over half its weight is on the bones that branch carries; a face is when most of its corners are (labels'
    rule). Returns the pictures' paths."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = TILE
    up, front = np.asarray(up, float), np.asarray(front, float)
    left = np.cross(up, front)

    def on(bones):              # the vertices over half on the bones these joints carry
        return rig.weights[:, [x for j in bones for x in mass.groups[j]]].sum(axis=1) > 0.5
    limbs = {r: on(mass.sub[r]) for r in roots}
    paints = []
    for name, rgb in [('grey', labels.GREY), ('light', LIGHT)] + COLOURS:
        mat = bpy.data.materials.new(f'core_forks_{name}')
        mat.diffuse_color = (*rgb, 1.0)
        paints.append(mat)
    counts = [len(bpy.data.objects[name].data.vertices) for name in rig.meshes]
    starts = np.cumsum([0] + counts)
    saved, hidden, masks = {}, [], []
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name not in rig.meshes and not ob.hide_render:
            ob.hide_render = True
            hidden.append(ob)
    for name, start in zip(rig.meshes, starts):
        ob = bpy.data.objects[name]
        me = ob.data
        if me.name not in saved:
            index = np.empty(len(me.polygons), dtype=np.int32)
            me.polygons.foreach_get('material_index', index)
            loops = np.empty(len(me.loops), dtype=np.int64)
            me.loops.foreach_get('vertex_index', loops)
            first = np.empty(len(me.polygons), dtype=np.int64)
            me.polygons.foreach_get('loop_start', first)
            saved[me.name] = (me, index, len(me.materials), start, loops, first)
            for mat in paints:
                me.materials.append(mat)
        group = ob.vertex_groups.new(name='core_forks_shown')
        mask = ob.modifiers.new('core_forks_shown', 'MASK')
        mask.vertex_group, mask.threshold, mask.show_render = group.name, 0.5, False
        masks.append((ob, group, mask, start, len(me.vertices)))
    lo, hi = rig.points.min(0), rig.points.max(0)
    size = float((hi - lo).max())
    cam = bpy.data.objects.new('core_forks', bpy.data.cameras.new('core_forks'))
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 10
    scene.collection.objects.link(cam)
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def aim(side, screen_up, points):
        """Point the camera at ``points`` from ``side``, ``screen_up`` up the picture."""
        right = np.cross(screen_up, side)
        centre = (points.min(0) + points.max(0)) / 2
        span = max(np.ptp(points @ right), np.ptp(points @ screen_up), size * 1e-3)
        cam.data.ortho_scale = span * 1.15
        m4 = np.eye(4)
        m4[:3, :3], m4[:3, 3] = np.column_stack([right, screen_up, side]), centre + side * size * 3
        cam.matrix_world = Matrix(m4.tolist())

    def shoot(tile):
        scene.render.filepath = str(tile)
        still()
        img = Image.new('RGB', (TILE, TILE), (205, 210, 218))
        with Image.open(tile) as rendered:
            rgba = rendered.convert('RGBA')
        tile.unlink()
        img.paste(rgba, (0, 0), rgba)
        return img
    out = []
    try:
        aim(front, up, rig.points)
        out.append(path.with_name(f'{path.stem}-0.png'))
        img = shoot(out[-1])
        ImageDraw.Draw(img).text((10, 6), 'in its own colours', fill='black', font=_font(24, bold=True))
        img.save(out[-1])
        rows = []
        for n, f in enumerate(found, 1):
            # per vertex: -1 the body, 0 the limb, k the fork's k-th branch
            paint = np.where(limbs[f['root']], 0, -1)
            for k, b in enumerate(f['branches'], 1):
                paint[on(mass.sub[b])] = k
            for me, index, base, start, loops, first in saved.values():
                if not len(first):
                    continue
                face = _most(paint[start + loops], first)
                me.polygons.foreach_set('material_index', (base + np.where(face < 0, 0, 1 + face)).astype(np.int32))
            shown = ~np.any([v for r, v in limbs.items() if r != f['root']], axis=0)     # its twin is one
            for ob, group, mask, start, count in masks:
                group.add(list(range(count)), 0.0, 'REPLACE')
                group.add(np.flatnonzero(shown[start:start + count]).tolist(), 1.0, 'REPLACE')
                mask.show_render = True
            parent = mass.parents[f['bone']]
            framed = on(mass.sub[f['bone']]) | (on([parent]) if parent >= 0 else False)
            points = rig.points[framed] if framed.any() else rig.heads[[f['bone']]]
            if f['role'] == 'arms':
                views = [('from the front', front, up), ('from above', up, -front)]
            else:
                side = left if (rig.heads[f['bone']] - (lo + hi) / 2) @ left >= 0 else -left
                views = [('from the front', front, up), ('from its side', side, up)]
            tiles = []
            for j, (title, side, screen_up) in enumerate(views):
                aim(side, screen_up, points)
                img = shoot(path.with_name(f'{path.stem}-{n}-{j}.png'))
                ImageDraw.Draw(img).text((10, 6), f'fork {n}: {title}', fill='black', font=_font(24, bold=True))
                tiles.append(img)
            rows.append(tiles)
        for ob, group, mask, *_ in masks:
            mask.show_render = False
        for page, first in enumerate(range(0, len(rows), PER_PICTURE), 1):
            shown_rows = rows[first:first + PER_PICTURE]
            sheet = Image.new('RGB', (2 * TILE + 8, len(shown_rows) * (TILE + 8) - 8), 'white')
            for i, tiles in enumerate(shown_rows):
                for j, t in enumerate(tiles):
                    sheet.paste(t, (j * (TILE + 8), i * (TILE + 8)))
            out.append(path.with_name(f'{path.stem}-{page}.png'))
            sheet.save(out[-1])
    finally:
        for me, index, base, *_ in saved.values():
            me.polygons.foreach_set('material_index', index)
            for _ in paints:
                me.materials.pop()
        for ob, group, mask, *_ in masks:
            ob.modifiers.remove(mask)
            ob.vertex_groups.remove(group)
        for ob in hidden:
            ob.hide_render = False
        bpy.data.objects.remove(cam)
        for mat in paints:
            bpy.data.materials.remove(mat)
    return out


def _most(values, first):
    """Per face (its corners' values from ``first`` on, in order), the value most of its corners have, when more than
    half do and it isn't negative; else -1."""
    kinds = values.max() + 2
    hits = np.zeros((len(values), kinds), np.int64)
    hits[np.arange(len(values)), values + 1] = 1
    counts = np.add.reduceat(hits, first, axis=0)
    best = counts.argmax(axis=1)
    return np.where(2 * counts[np.arange(len(best)), best] > counts.sum(axis=1), best - 1, -1)


def _named(rig, kids, b):
    """A branch for the question: its first bone and the bones at the ends of all it carries."""
    out = [b]
    for x in out:
        out += kids[x]
    ends = [rig.names[x] for x in out[1:] if not kids[x]]
    return rig.names[b] + (f' (out to {", ".join(ends)})' if ends else '')
