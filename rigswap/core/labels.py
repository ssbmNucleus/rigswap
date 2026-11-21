"""What each of the model's twin pairs is to its body: its legs, its arms, a four-legged body's front legs, or none.

Code finds the pairs: twin chains leaving the trunk (body.limb_pairs). What a pair *is*, the skeleton doesn't say. A
gown's panels reach the floor beside the legs and can outweigh them (Gardevoir); a shoulder strap reaches out beside
the arm (Galahad); a fox's forelegs stand level with its hind legs (Ninetales). Every measurement tried for it (mass,
height, reach, branch count) stood in for the meaning and needed one more clause per model. So it is asked, as the
agreed design's labels step says (docs/DESIGN.md, step 3): the model in its own colours from the front (who it is
says what its parts are: in grey, nobody could tell Gardevoir, and her legs read as one more panel of her gown), then
in grey from the front, the side and the back, each pair in a colour of its own and numbered on both sides, and under
them each pair again on its own, the other pairs left out (a pair the others hide can't be named: Gardevoir's legs
inside her gown's panels) (PER_PICTURE pairs a picture, every picture in one ask).
The AI says what each pair is and its role; per role, the pair most of the VOTES asks give it, when more than half do
(tally). A pair given no role is an attachment, and no answer leaves no legs and no arms: the old measurements are no
fallback.

Then, once the body is read, every chain off it is named the same way (chains): each bone hanging off the spine, an
arm or a leg that is none of them, with all it carries, a twin and its mirror image one chain. What a chain is decides
what it does (docs/STATE.md, step 3): hair and cloth hang, because of what they are, from the bone they hang from on
the body or a limb (Galahad's sleeves off her forearms); everything else rides that bone as bound. Every geometric
stand-in for it misread some model: the hang ask's glued-on gate took cloth resting on the body for the body, skin
purity took a stiffly skinned braid for an object, a skirt strand was a strand reaching below the body's skin. A chain
carrying no surface of its own (no vertex over half on it) shows nothing and has nothing to swing, so it is not asked
about; nor is a branch the forks pass named digits or a helper. The pictures are the pairs', but each chain alone is
drawn close, in its own colours: what a part is, its colours say (a braid's from a strap's).

And where the fighter's hands hold things of more than one kind (Link's sword and shield), what each of the model's
props is, of those kinds or none, is asked the same way (equipment): a weapon slot takes only a prop of its kind
(correspond._match_props). Which is the sword and which the shield, no length or width says for every model.
"""
from pathlib import Path

import numpy as np

from . import body
from .render import still

VOTES = 3                   # asks: the answer wavers between repeats; per role the most given wins
PER_PICTURE = 6             # pairs drawn in one picture, each in a colour of its own
TILE = 640
SMALL = (3 * TILE + 16 - 8 * (PER_PICTURE - 1)) // PER_PICTURE    # a pair's own view, PER_PICTURE across the sheet
COLOURS = [('red', (0.85, 0.05, 0.05)), ('blue', (0.1, 0.3, 0.95)), ('green', (0.1, 0.7, 0.15)),
           ('orange', (1.0, 0.55, 0.0)), ('purple', (0.55, 0.15, 0.8)), ('cyan', (0.0, 0.75, 0.8))]
GREY = (0.4, 0.42, 0.45)
KINDS = ('hair', 'cloth', 'tail', 'ear', 'wing', 'armour', 'prop', 'face', 'body', 'other')
HANG = ('hair', 'cloth')    # the kinds that hang, because of what they are; the others ride the bone they hang from

QUESTION = (
    'The first picture shows one game character in its own colours from the front, in the pose its model file is '
    'bound in, often arms out in a T. Each picture after it (picture 1, 2...) shows the character in plain grey from '
    'the front, from its left side and from the back. Some of its parts come in left and right pairs, a part and its '
    'mirror image; each pair is drawn in a colour of its own and numbered on both sides: {legend}. Under the three '
    'views, each pair is drawn again from the front on its own, numbered in its corner, with the other pairs left '
    'out, so a pair the others hide can be seen. First say who the character is, if you know. Then, for every '
    'numbered pair, say what it is on the character (a '
    'leg, an arm, a wing, a tail, a skirt or coat panel, a sleeve, a strap, a lock of hair, an ear, an eyelid...) and '
    'its role: "legs" for what the character stands and walks on (on a character that walks on four legs, its hind '
    'legs); "front legs" for the front legs of a character that walks on four; "arms" for its arms, or the limbs it '
    'uses as arms; "other" for anything else. Judge each pair by what it is on the character, as its art and games '
    'show it, not by its size or by whether it reaches the floor here. A character has at most one pair of legs, one '
    'pair of front legs and one pair of arms, and may have none of them. Answer cant_tell only if you cannot tell. The '
    'pairs, by their two root bones and the bones at the ends of all each carries: {parts}.')
SCHEMA = {'type': 'object', 'properties': {
    'character': {'type': 'string'},
    'pairs': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'number': {'type': 'integer'}, 'what': {'type': 'string'},
                                         'role': {'type': 'string',
                                                  'enum': [*body.ROLES, 'other', 'cant_tell']}},
        'required': ['number', 'what', 'role'], 'additionalProperties': False}}},
    'required': ['character', 'pairs'], 'additionalProperties': False}


CHAINS = (
    'The first picture shows one game character in its own colours from the front, in the pose its model file is '
    'bound in, often arms out in a T. Each picture after it (picture 1, 2...) shows the character in plain grey from '
    'the front, from its left side and from the back, with some of its parts each in a colour of its own and '
    'numbered: {legend}. A part that comes in a left and a right, one the mirror image of the other, is one part, '
    'numbered on both sides. Under the three views, each part is drawn again on its own, close, in its own colours, '
    'numbered in its corner. First say who the character is, if you know. Then, for every numbered part, say what it '
    'is on the character (a ponytail, a skirt panel, a sleeve, a cape, a feeler, a wing, an ear, a horn, a helmet, a '
    'sheath, an eyelid, a jaw...) and its kind: "hair" for hair and what hangs as hair does (a lock, a braid, a '
    'ponytail, a fringe, a mane, a feeler or tendril that dangles); "cloth" for cloth and what hangs as cloth does (a '
    'skirt or a panel of one, a coat tail, a cape, a sleeve, a scarf, a sash, a ribbon, a hood, a strap or flap '
    'hanging loose), but cloth worn tight, so none of it hangs free (a cap, a bandana or band tied round the head, a '
    'belt, a glove), is "other"; "tail" for a tail; "ear" for an ear; "wing" for a wing; "armour" for a hard piece '
    'worn on the body (armour, a plate, a shoulder guard, a helmet, a crown, a shell); "prop" for an object worn or '
    'carried (a weapon, a sheath, a bag, a pouch, a buckle, jewellery, a device); "face" for a part of the face (an '
    'eye, an eyelid, a brow, a jaw, a mouth, a tongue, teeth, a beard, a moustache); "body" for a part of the body '
    'itself that holds its shape (a horn, a fin, a spike, a crest, an antenna standing up, a frill); "other" for '
    'anything else. Judge each part by what it is on the character, as its art and games show it, not by its size or '
    'by how it sticks out here. For hair and cloth, also say whether it is drawn as it rests when the character '
    'stands at ease, or whether the bind pose holds it out where it would not rest, as the T holds the arms out (a '
    'feeler sticking straight out sideways, a braid lifted off the back): a part shaped to flare, curl or bunch up, a '
    'skirt flared or a lock swept across the face, is drawn as it rests, and a bind pose holds parts out sideways or '
    'back, never straight up. For any other kind, answer "as drawn". Answer cant_tell only if you cannot tell. The '
    'parts, by their root bones and the bones at the ends of all each carries: {parts}.')
CHAINS_SCHEMA = {'type': 'object', 'properties': {
    'character': {'type': 'string'},
    'parts': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'number': {'type': 'integer'}, 'what': {'type': 'string'},
                                         'kind': {'type': 'string', 'enum': [*KINDS, 'cant_tell']},
                                         'rest': {'type': 'string',
                                                  'enum': ['as drawn', 'held out by the pose', 'cant_tell']}},
        'required': ['number', 'what', 'kind', 'rest'], 'additionalProperties': False}}},
    'required': ['character', 'parts'], 'additionalProperties': False}

EQUIPMENT = (
    'The first picture shows one game character in its own colours from the front, in the pose its model file is '
    'bound in, often arms out in a T. Each picture after it (picture 1, 2...) shows the character in plain grey from '
    'the front, from its left side and from the back, with some objects it holds or carries each in a colour of its '
    'own and numbered: {legend}. Under the three views, each object is drawn again on its own, close, in its own '
    'colours, numbered in its corner. First say who the character is, if you know. Then, for every numbered object, '
    'say what it is (a sword, a shield, a bow, a sheath, a staff, a bag...) and which of these it is: {kinds}, or '
    '"none" for anything else. Judge each object by what it is, as the character\'s art and games show it, not by its '
    'size or by where it is here. Answer cant_tell only if you cannot tell. The objects, by their root bones and the '
    'bones at the ends of all each carries: {parts}.')


def equipment_schema(kinds):
    return {'type': 'object', 'properties': {
        'character': {'type': 'string'},
        'objects': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'number': {'type': 'integer'}, 'what': {'type': 'string'},
                                             'kind': {'type': 'string', 'enum': [*kinds, 'none', 'cant_tell']}},
            'required': ['number', 'what', 'kind'], 'additionalProperties': False}}},
        'required': ['character', 'objects'], 'additionalProperties': False}


def candidates(rig, up):
    """The twin pairs, heaviest first (the order only sets the picture a pair is in): per pair its two roots, and per
    root the bones it carries (itself and all below it)."""
    mass, _, pairs, _ = body.limb_pairs(rig, up)
    kids = rig.children()

    def carried(b):
        out = [b]
        for x in out:
            out += kids[x]
        return out
    pairs = sorted(pairs, key=lambda p: -sum(mass.total[b] for b in p))
    return [dict(roots=list(p), bones=[carried(b) for b in p]) for p in pairs]


def tally(answers, numbers):
    """Per role, the pair numbers more than half of the answers give it: one, or several where the model cuts that
    limb into pieces, a pair each (Boomerang Bro's thighs, shins and feet: body.read joins them)."""
    out = {}
    for role in body.ROLES:
        given = [n for n in numbers if 2 * sum(any(p.get('number') == n and p.get('role') == role
                                                   for p in (a or {}).get('pairs') or []) for a in answers)
                 > len(answers)]
        if given:
            out[role] = given
    return out


def read(rig, up, front, out, ask):
    """Ask what each twin pair is, with pictures drawn into ``out``. Returns ({bone name: role} for both roots of every
    pair given a role, the report)."""
    found = candidates(rig, up)
    if not found:
        return {}, {'pairs': []}
    pictures = draw(rig, found, up, front, Path(out) / 'labels.png')
    numbers = list(range(1, len(found) + 1))
    legend = ', '.join(f'{n} {COLOURS[(n - 1) % PER_PICTURE][0]} (picture {(n - 1) // PER_PICTURE + 1})'
                       for n in numbers)
    kids = rig.children()
    parts = '; '.join(f'{n}: {_named(rig, kids, f)}' for n, f in zip(numbers, found))
    said = []
    for r in range(VOTES):
        answer, _ = ask(QUESTION.format(legend=legend, parts=parts) + f' (Ask {r + 1} of {VOTES}.)',
                        [str(p) for p in pictures], SCHEMA, {'pairs': []}, route='core labels')
        said.append(answer)
    picked = tally(said, numbers)
    roles = {rig.names[b]: role for role, ns in picked.items() for n in ns for b in found[n - 1]['roots']}
    report = dict(
        character=[(a or {}).get('character') for a in said if (a or {}).get('character')],
        roles={role: [rig.names[b] for n in ns for b in found[n - 1]['roots']] for role, ns in picked.items()},
        pairs=[dict(number=n, roots=[rig.names[b] for b in f['roots']],
                    said=[f"{p.get('what')} ({p.get('role')})" for a in said for p in (a or {}).get('pairs') or []
                          if p.get('number') == n]) for n, f in zip(numbers, found)])
    print('labels: ' + str(report['roles']))
    return roles, report


def branches(rig, shape, up, kinds=None):
    """The chains off the body (``shape``, body.read's), heaviest first: each bone hanging off its spine, an arm or a
    leg that is none of them and has none of them below it, with all it carries; a twin and its mirror image are one
    chain (per chain its roots, and per root the bones it carries, as candidates' pairs). Left out: one carrying no
    surface of its own (no vertex over half on what it carries), and a branch the forks pass named digits or a helper
    (``kinds``: {bone name: kind}, forks.read's)."""
    groups = shape.groups or {}
    paths = [*shape.spine, *(b for p in [*shape.arms.values(), *shape.legs.values()] for b in p)]
    main = {k for b in paths for k in groups.get(b, [b])}
    kids, carried = rig.children(), rig.carried()
    skip = {n for n, kind in (kinds or {}).items() if kind in ('digits', 'helper')}

    def below(b):
        out = [b]
        for x in out:
            out += kids[x]
        return out
    roots = []
    for b in range(len(rig.names)):
        if b in main or rig.parents[b] not in main or rig.names[b] in skip:
            continue
        sub = below(b)
        if not main & set(sub) and (rig.weights[:, sub].sum(axis=1) > 0.5).any():
            roots.append(b)
    mass, _, _, twins = body.limb_pairs(rig, up)
    joint = {k: b for b, group in mass.groups.items() for k in group}     # each bone's joint, as twins has them
    found, done = [], set()
    for b in roots:
        if b in done:
            continue
        twin = next((x for x in roots if x not in done and x != b and b in joint and x in joint
                     and twins.get(joint[b]) == joint[x]), None)
        pair = [b] if twin is None else [b, twin]
        done.update(pair)
        found.append(dict(roots=pair, bones=[below(r) for r in pair]))
    return sorted(found, key=lambda f: -sum(float(carried[bones].sum()) for bones in f['bones']))


def tally_chains(answers, numbers):
    """Per part number, (its kind, whether the pose holds it out): the kind more than half of the answers give it, else
    None (no answer, can't tell, or a split: it rides, as all but hair and cloth do); held out only for hair and cloth,
    and only when every answer giving that kind says so: turning it rewrites the shape its author gave it, which a
    wavering answer must not do (a Vulpix unrecognised on the grey body had its curled tails hung straight down)."""
    out = {}
    for n in numbers:
        said = [next((p for p in (a or {}).get('parts') or [] if p.get('number') == n), None) for a in answers]
        kinds = [p.get('kind') for p in said if p]
        kind = next((k for k in KINDS if 2 * kinds.count(k) > len(answers)), None)
        out[n] = (kind, kind in HANG and all(p.get('rest') == 'held out by the pose'
                                             for p in said if p and p.get('kind') == kind))
    return out


def chains(rig, shape, up, front, out, ask, kinds=None):
    """Ask what each chain off the body is (branches; ``kinds``: the forks pass's), with pictures drawn into ``out``.
    Returns ({bone name: {kind, held_out}} for every bone of every chain named, the report)."""
    found = branches(rig, shape, up, kinds)
    if not found:
        return {}, {'parts': []}
    pictures = draw(rig, found, up, front, Path(out) / 'chains.png', close=True)
    numbers = list(range(1, len(found) + 1))
    legend = ', '.join(f'{n} {COLOURS[(n - 1) % PER_PICTURE][0]} (picture {(n - 1) // PER_PICTURE + 1})'
                       for n in numbers)
    kids = rig.children()
    parts = '; '.join(f'{n}: {_named(rig, kids, f)}' for n, f in zip(numbers, found))
    said = []
    for r in range(VOTES):
        answer, _ = ask(CHAINS.format(legend=legend, parts=parts) + f' (Ask {r + 1} of {VOTES}.)',
                        [str(p) for p in pictures], CHAINS_SCHEMA, {'parts': []}, route='core chains')
        said.append(answer)
    picked = tally_chains(said, numbers)
    named = {rig.names[b]: dict(kind=picked[n][0], held_out=picked[n][1])
             for n, f in zip(numbers, found) if picked[n][0] for bones in f['bones'] for b in bones}
    report = dict(
        character=[(a or {}).get('character') for a in said if (a or {}).get('character')],
        parts=[dict(number=n, roots=[rig.names[b] for b in f['roots']], kind=picked[n][0], held_out=picked[n][1],
                    said=[f"{p.get('what')} ({p.get('kind')}, {p.get('rest')})" for a in said
                          for p in (a or {}).get('parts') or [] if p.get('number') == n])
               for n, f in zip(numbers, found)])
    print('chains: ' + str({' '.join(p['roots']): p['kind'] for p in report['parts']}))
    return named, report


def tally_equipment(answers, numbers, kinds):
    """Per object number, the kind of ``kinds`` more than half of the answers give it, else None (none, can't tell,
    no answer, or a split)."""
    out = {}
    for n in numbers:
        said = [next((p.get('kind') for p in (a or {}).get('objects') or [] if p.get('number') == n), None)
                for a in answers]
        out[n] = next((k for k in kinds if 2 * said.count(k) > len(answers)), None)
    return out


def equipment(rig, found, up, front, out, ask, kinds):
    """Ask which of ``kinds`` (the fighter's weapon slots': a sword, a shield) each of the model's props is
    (correspond.props; ``found``: per prop, its root bone, its bones and its vertices), with pictures drawn into
    ``out``, as the chains are asked: each prop drawn by its vertices, so one the file drew on the hand's own bone,
    with no bones of its own (correspond._on_the_hand), is drawn alone and named by that bone. Returns ({root bone
    name: kind} for every prop asked, None for one named no weapon, the report)."""
    found = [dict(roots=[b], bones=[sub], vertices=[sel]) for b, sub, sel in found]
    if not found:
        return {}, {'objects': []}
    pictures = draw(rig, found, up, front, Path(out) / 'equipment.png', close=True)
    numbers = list(range(1, len(found) + 1))
    legend = ', '.join(f'{n} {COLOURS[(n - 1) % PER_PICTURE][0]} (picture {(n - 1) // PER_PICTURE + 1})'
                       for n in numbers)
    kids = rig.children()
    parts = '; '.join(f'{n}: {_named(rig, kids, f)}' if f['bones'][0] else
                      f"{n}: an object drawn on the hand's bone {rig.names[f['roots'][0]]}"
                      for n, f in zip(numbers, found))
    said = []
    for r in range(VOTES):
        answer, _ = ask(EQUIPMENT.format(legend=legend, kinds=', '.join(f'"{k}"' for k in kinds), parts=parts)
                        + f' (Ask {r + 1} of {VOTES}.)', [str(p) for p in pictures], equipment_schema(kinds),
                        {'objects': []}, route='core equipment')
        said.append(answer)
    picked = tally_equipment(said, numbers, kinds)
    named = {rig.names[f['roots'][0]]: picked[n] for n, f in zip(numbers, found)}
    report = dict(
        character=[(a or {}).get('character') for a in said if (a or {}).get('character')],
        objects=[dict(number=n, root=rig.names[f['roots'][0]], kind=picked[n],
                      said=[f"{p.get('what')} ({p.get('kind')})" for a in said
                            for p in (a or {}).get('objects') or [] if p.get('number') == n])
                 for n, f in zip(numbers, found)])
    print('equipment: ' + str({o['root']: o['kind'] for o in report['objects']}))
    return named, report


def draw(rig, found, up, front, path, close=False, number=1):
    """The model at rest in its own colours from the front, then a picture per PER_PICTURE pairs (numbered from
    ``number``): the model at rest in grey from the front, its left side and the back, each
    pair's faces in its colour, its number on each twin where its colour shows in that view (from the side, once);
    under them, per pair, the front again with the other pairs' faces masked out, or with ``close``, the pair alone in
    its own colours, close, from the front or back (whichever side of the body it is on) or from its flank, whichever
    it spreads wider across. A pair may be one part on the midline (``found``'s roots one bone), numbered once. A
    vertex is a side's when over half its weight is on the bones that side carries; a face is a pair's when most of
    its corners are. Returns the pictures' paths."""
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
    # per vertex, the pair whose side carries over half its weight, else -1: the sides carry disjoint bones, so at
    # most one does
    shares = np.stack([rig.weights[:, bones].sum(axis=1) if 'vertices' not in f else f['vertices'][i].astype(float)
                       for f in found for i, bones in enumerate(f['bones'])], axis=1)
    owner = np.where(shares.max(axis=1) > 0.5, shares.argmax(axis=1), -1)
    of = np.array([k for k, f in enumerate(found) for _ in f['bones']])
    pair = np.where(owner >= 0, of[np.maximum(owner, 0)], -1)
    paints = []
    for name, rgb in [('grey', GREY)] + COLOURS:
        mat = bpy.data.materials.new(f'core_labels_{name}')
        mat.diffuse_color = (*rgb, 1.0)
        paints.append(mat)
    counts = [len(bpy.data.objects[name].data.vertices) for name in rig.meshes]
    starts = np.cumsum([0] + counts)
    saved, faces, hidden = {}, {}, []
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name not in rig.meshes and not ob.hide_render:
            ob.hide_render = True           # not the rig's: a mesh the file hides, shown again by _prepare_scene
            hidden.append(ob)
    for name, start in zip(rig.meshes, starts):
        me = bpy.data.objects[name].data
        if me.name in saved:
            continue
        index = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get('material_index', index)
        saved[me.name] = (me, index, len(me.materials))
        for mat in paints:
            me.materials.append(mat)
        mine = pair[start:start + len(me.vertices)]
        faces[me.name] = np.array([_most(mine[list(p.vertices)]) for p in me.polygons], dtype=np.int64)
    masks = []                              # per mesh, the vertices one pair's own view keeps (a mask modifier's group)
    for name, start in zip(rig.meshes, starts):
        ob = bpy.data.objects[name]
        group = ob.vertex_groups.new(name='core_labels_alone')
        mask = ob.modifiers.new('core_labels_alone', 'MASK')
        mask.vertex_group, mask.threshold, mask.show_render = group.name, 0.5, False
        masks.append((ob, group, mask, pair[start:start + len(ob.data.vertices)]))
    pts = rig.points
    lo, hi = pts.min(0), pts.max(0)
    centre, size = (lo + hi) / 2, float((hi - lo).max())
    span = size * 1.1
    cam = bpy.data.objects.new('core_labels', bpy.data.cameras.new('core_labels'))
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 10
    cam.data.ortho_scale = span
    scene.collection.objects.link(cam)
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def aim(side, middle=centre, width=span):
        """Point the camera at ``middle`` from ``side``, ``width`` across; returns the view's rightward direction."""
        right = np.cross(up, side)
        m4 = np.eye(4)
        m4[:3, :3], m4[:3, 3] = np.column_stack([right, up, side]), middle + side * size * 3
        cam.matrix_world = Matrix(m4.tolist())
        cam.data.ortho_scale = width
        return right

    def shoot(tile):
        """The camera's view rendered onto the backdrop."""
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
        aim(front)
        out.append(path.with_name(f'{path.stem}-0.png'))
        img = shoot(out[-1])
        ImageDraw.Draw(img).text((10, 6), 'in its own colours', fill='black', font=_font(26, bold=True))
        img.save(out[-1])
        for page, first in enumerate(range(0, len(found), PER_PICTURE), 1):
            shown = list(range(first, min(first + PER_PICTURE, len(found))))
            for me, index, base in saved.values():
                f = faces[me.name]
                me.polygons.foreach_set('material_index', np.where(
                    np.isin(f, shown), base + 1 + f - first, base).astype(np.int32))
            tiles = []
            for j, (side, title) in enumerate([(front, 'front'), (left, 'left side'), (-front, 'back')]):
                right = aim(side)
                tile = path.with_name(f'{path.stem}-{page}-{j}.png')
                seen = _flat(scene, tile)
                img = shoot(tile)
                ink = ImageDraw.Draw(img)
                ink.text((10, 6), f'picture {number + page - 1}: {title}', fill='black', font=_font(26, bold=True))
                for k in shown:
                    ys, xs = np.nonzero(seen == k - first + 1)
                    # the twins lie either side of the plane through their roots' middle; from the side, as one
                    middle = ((rig.heads[found[k]['roots']].mean(axis=0) - centre) @ right / span + 0.5) * TILE
                    halves = [xs >= 0] if side is left or len(found[k]['roots']) == 1 else [xs < middle,
                                                                                            xs >= middle]
                    rgb = tuple(int(255 * c) for c in COLOURS[k - first][1])
                    for h in halves:
                        if not h.any():
                            continue
                        # on the part where it shows: its visible pixel nearest their middle
                        i = np.argmin((xs[h] - xs[h].mean()) ** 2 + (ys[h] - ys[h].mean()) ** 2)
                        ink.text((int(xs[h][i]), int(ys[h][i])), str(k + 1), fill=rgb, font=_font(40, bold=True),
                                 anchor='mm', stroke_width=4, stroke_fill='white')
                tiles.append(img)
            aim(front)
            alone = []
            for k in shown:
                for ob, group, mask, mine in masks:
                    group.add(list(range(len(mine))), 0.0, 'REPLACE')
                    group.add(np.flatnonzero((mine == k) | ((mine < 0) & (not close))).tolist(), 1.0, 'REPLACE')
                    mask.show_render = True
                if close:
                    for me, index, base in saved.values():
                        me.polygons.foreach_set('material_index', np.where(
                            faces[me.name] == k, index, base).astype(np.int32))
                    part = pts[pair == k]
                    plo, phi = part.min(0), part.max(0)
                    c = (plo + phi) / 2
                    facing = front if (c - centre) @ front >= 0 else -front
                    flank = left if (c - centre) @ left >= 0 else -left
                    across = [np.ptp(part @ np.cross(up, v)) for v in (facing, flank)]
                    aim(facing if across[0] >= across[1] else flank, c,
                        max(1.6 * float((phi - plo).max()), 0.22 * size))
                img = shoot(path.with_name(f'{path.stem}-{page}-alone.png')).resize((SMALL, SMALL), Image.LANCZOS)
                rgb = tuple(int(255 * c) for c in COLOURS[k - first][1])
                ImageDraw.Draw(img).text((8, 4), str(k + 1), fill=rgb, font=_font(36, bold=True),
                                         stroke_width=4, stroke_fill='white')
                alone.append(img)
            for ob, group, mask, mine in masks:
                mask.show_render = False
            aim(front)
            sheet = Image.new('RGB', (3 * TILE + 16, TILE + 8 + SMALL), 'white')
            for j, t in enumerate(tiles):
                sheet.paste(t, (j * (TILE + 8), 0))
            for j, t in enumerate(alone):
                sheet.paste(t, (j * (SMALL + 8), TILE + 8))
            out.append(path.with_name(f'{path.stem}-{page}.png'))
            sheet.save(out[-1])
    finally:
        for me, index, base in saved.values():
            me.polygons.foreach_set('material_index', index)
            for _ in paints:
                me.materials.pop()
        for ob, group, mask, mine in masks:
            ob.modifiers.remove(mask)
            ob.vertex_groups.remove(group)
        for ob in hidden:
            ob.hide_render = False
        bpy.data.objects.remove(cam)
        for mat in paints:
            bpy.data.materials.remove(mat)
    return out


def _flat(scene, tile):
    """Per pixel of the camera's view, which paint shows there (0 grey, k the k-th colour, -1 none): the view drawn
    again in flat colour without smoothing, each pixel matched to the nearest paint as displayed."""
    import bpy
    from PIL import Image
    shading = scene.display.shading
    before = shading.light, shading.color_type, scene.display.render_aa, scene.render.dither_intensity
    shading.light, shading.color_type, scene.display.render_aa, scene.render.dither_intensity = \
        'FLAT', 'MATERIAL', 'OFF', 0.0
    try:
        scene.render.filepath = str(tile)
        bpy.ops.render.render(write_still=True)
        with Image.open(tile) as rendered:
            rgba = np.asarray(rendered.convert('RGBA'), dtype=float)
        tile.unlink()
    finally:
        shading.light, shading.color_type, scene.display.render_aa, scene.render.dither_intensity = before
    linear = np.array([GREY] + [rgb for _, rgb in COLOURS])
    shown = 255 * np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    gap = np.linalg.norm(rgba[..., None, :3] - shown, axis=-1)
    return np.where((rgba[..., 3] > 0) & (gap.min(axis=-1) < 24), gap.argmin(axis=-1), -1)


def _most(values):
    """The pair most of a face's corners belong to, when more than half do; else -1."""
    kinds, counts = np.unique(values, return_counts=True)
    k = int(np.argmax(counts))
    return int(kinds[k]) if kinds[k] >= 0 and 2 * counts[k] > len(values) else -1


def _named(rig, kids, found):
    """A pair for the question: its two roots (or its one, on the midline), and the bones at the ends of all the first
    carries (a hand's fingers and a sleeve's tips alike: the pair is all of it, not one path through it, and a path's
    end names one branch)."""
    ends = [rig.names[b] for b in found['bones'][0][1:] if not kids[b]]
    return ' and '.join(rig.names[r] for r in found['roots']) + (
        f', {"each " if len(found["roots"]) > 1 else ""}carrying bones out to {", ".join(ends)}' if ends else '')
