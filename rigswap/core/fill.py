"""One look fills the body of the record: which of the model's joints is each joint of a body, and its front.

What a part of a model *is* is meaning, and a look says it at a glance where rules about skeletons grow a clause per
model. So one ask shows the whole character (its own colours from four sides, two of the sides again with every joint
numbered, and a table of the joints) and takes the whole body in one answer: the front, the plan, the hips, chest,
neck and skull, and each limb's joints. Code does what code is sure of:
- left and right are read from where the limbs lie, never from the answer (the lab: 3 answers of 60 had them swapped);
- a named joint is a joint of the table, the chest and the neck lie on the spine between the hips and the skull, a
  limb has two ends;
- a limb whose joints are not one under another in the file's tree is listed joint by joint (a foot hung on a floor
  control beside the hips).
An answer that fails those is asked once more, told why; a second failure leaves the body to the reading
(labels.py, forks.py, body.read), as does a body the look calls a ball or neither two- nor four-legged: those plans
draw their own chains (plans.py).

A skeleton fitted inside a file that had none (onepiece.py) offers a dot every little way, not joints, and the
whole-body pictures hide a limb that points at the camera. There each arm is asked again alone, on a picture drawn
square to it (closer(), LIMB): the first answer says which part is which, the close-up where its joints are.

Lab, 2026-11-09: five models, three tries each, against records written by hand.
Gemini 3.7 Flash named every key joint right or one joint off (15 of 15) at about a cent a fill; GLM 5.3 Flash 10 of 15.
"""
import json
from pathlib import Path

import numpy as np

from . import body as body_reading, onepiece
from .render import still

ROUTE, AGAIN = 'core fill', 'core fill again'
VIEWS = 'ABCD'
TILE, BIG = 640, 1150
PLANS = ('biped', 'quadruped', 'ball', 'other')
LEG, ARM = ('hip', 'knee', 'ankle', 'toe'), ('clavicle', 'shoulder', 'elbow', 'hand')

QUESTION = """You are looking at a 3D character model that will be put onto a game fighter's skeleton, so that the \
fighter's animations move it. Say which of the model's own joints is each joint of a body.

Picture 1: the model in its own colours from four sides, A, B, C and D, a quarter turn apart.
Pictures 2 and 3: sides A and B again, dimmed, with every joint of the model's skeleton drawn as a red dot with its \
number, and a yellow line from each joint to its parent.

The table lists every joint: its number, its parent's number, its place (across, deep and high, as a percentage of \
the model's height; high = 0 is the floor), skin (how many points of the surface it carries), below (how many joints \
hang below it) and the names of its bones in the file (usually meaningless).

{table}
{fitted}
What to answer:
- character: who or what this is, in a few words.
- front_view: which of A, B, C, D has the camera in front of the character, looking at its face.
- plan: "biped" (stands on two legs, even with small arms), "quadruped" (walks on four), "ball" (feet straight off a \
round body, no legs to speak of) or "other".
- left and right are the character's own left and right, as for a person: facing you, its left is on your right.
- hips: the joint the legs part from. chest: the spine joint the arms hang from. {neck}
- each leg: hip (the thigh's joint, where the leg leaves the body), knee (null when the leg has no joint there), \
ankle (where the foot begins), toe (the joint in the foot the toes bend at, or null).
- each arm: clavicle (a collar joint before the shoulder, or null), shoulder (where the upper arm begins), elbow (or \
null), hand (the wrist, where the hand begins; never a finger).
- a quadruped's front legs are given as its arms.
- a body with no arms, or no legs, gives null for each of that limb's joints.
- notes: one sentence on anything unusual (a part that is an effect or lies collapsed, a pose that is not a stand).
Answer with joint numbers from the table only. Where a joint carrying no skin sits just above one that carries the \
part's skin, give the one that carries the skin.{again}"""

# The neck and skull entries, as the question words them. For a file's own skeleton the look is told what the build
# does with the neck joint: the fighter's head goes on it (body._from_record's head mark). Told only "the joint the
# head turns on", it named a skeleton's bone called neck, whose joint is at the neck's base, level with the arm roots:
# the head went there, the fighter's neck had no room and the neck bone was stretched fourteen times (the app's import
# of 2026-11-10). A fit keeps the old words (FITTED says the rest), so its question is as it was.
NECK_OWN = """neck: the joint the fighter's head is put on, where the head turns: the head and all that hangs from \
this joint will move with it as one piece, and what lies between the chest joint and it will be the fighter's neck. \
Where the skeleton has a neck bone and a head bone, that is the head bone's joint, at the top of the neck: the neck \
bone's own joint, at its base, would leave the fighter's neck no room. skull: the joint that carries the head's own \
surface, often that same joint."""
NECK_OFFERED = """neck: the joint the head turns on. \
skull: the joint that carries the head's own surface. In a simple skeleton two of these can be the same joint."""

# What the look is told of a skeleton fitted inside a file that had none (onepiece.py): its joints are offered, not
# real, and what the build does with the neck it names: the fighter's head goes on that joint (body._from_record's
# head mark, the spine's head anchor in correspond.model_chains), so all past it is stiff with the head and the
# stretch from the chest to it is the fighter's neck. (It was told the head is whole past the SKULL: Crash's neck was
# named at the top of his grin, dots-1, and the rest of his grin rode Mario's chest.) Told the truth it still named
# the neck within a joint or two of where it had (dots-3), so a body with no neck is answered as that, null, and the
# fit puts the head between the shoulders (onepiece.on_shoulders). Empty for any other model, whose question is then
# as it was.
FITTED = """
This model's file has no skeleton. The joints shown are a line fitted through the middle of its shape, one every \
little way along each part: none is a real joint yet, and the fighter's joints will be put where you say. For each \
joint asked, give the one nearest where the character's own joint would be (a knee where the leg would bend, even on \
a straight leg). The fighter's head is put on the neck joint: everything past it will move as one stiff piece with \
the head, and what lies between the chest joint and the neck joint will be the fighter's neck, squeezed or stretched \
to that neck's length and bent with it. Give a neck joint only where the character has a neck of its own, a stretch \
between its shoulders and its head that should bend: the joint where that neck meets the head. Where it has none \
(the head or the face sits straight on the shoulders, or head and body are one shape) give null for the neck: the \
head is then one stiff piece from the shoulders up. For the skull give a joint inside the head, past the \
neck's. Here "skin" is only how much surface lies round each joint.
"""

# One limb of a fitted skeleton, asked again on a picture of that limb alone (closer()). The whole-body pictures
# flatten the model: a limb that points at the camera hides its bend, and its dots, a few apart, bunch together. The
# look then names a limb's joints a dot or two off, the same way each time it is asked: Goku's bare right arm bends
# 47 degrees at one dot, and two asks named his elbow one and two dots past it, the second one dot from the hand, so
# Mario's forearm was 47 points stretched five times over (size-2, 2026-11-10; the owner: "twisted/shattered and not
# connected to his torso"). Where a limb bends is not left to code: under loose clothing the shape's bend is not the
# joint (the same Goku's front leg turns 71 degrees well above the middle of the leg, where his trousers hang), and
# a branch turns sharply where it leaves the trunk. So the look still says where the joints are, on a picture that
# shows them: the camera square to the limb, that limb's dots alone, and how far the line turns at each.
#
# The arms are asked so; a leg is not yet. On its close-up the look names a knee on the leg's real bend as well (both
# of Goku's, two asks alike), but a fitted leg begins where it parts from the other one, at the crotch, and a thigh
# that runs forward inside the trunk's slices (a lunge, wide trousers) is then two dots long: with the true knee
# Mario's thigh carried 14 points and Goku's whole lower leg was squeezed to half its length (look-1, read offline).
# Until a leg's first bone begins inside the hip, as an arm's begins inside the shoulder (onepiece.shoulder_joints),
# the first look's knee, a little low, serves a leg better.
LIMB_ROUTE, LIMB_AGAIN = 'core fill limb', 'core fill limb again'
CLOSER = ('arms',)      # the limbs asked again on a close-up
CLOSE = 900             # a close-up's tile, pixels a side: the limb in its own colours, and again with its dots
SPAN = 2                # a dot's turn is read this many dots each way: one dot's wobble is not a bend
LIMB = """This is one {limb} of a 3D character model{who}. The model will be put onto a game fighter's skeleton, and \
the fighter's {joints} will be put on the dots you name.

The picture shows one view twice. The camera stands square to the {limb}: no part of it points toward you or away \
from you, so its bend and its lengths are as they really are. Left: the model in its own colours. Right: the same \
view dimmed, with the line fitted through the middle of the {limb} drawn as numbered red dots, from inside the body \
(dot {first}) out to the {limb}'s tip (dot {last}). The file has no skeleton: none of the dots is a real joint yet.

The table lists the dots in that order: n (its number), along (how far along the line it lies, as a percentage of \
the line's length), turn (how many degrees the line turns at this dot: a bent {bend} shows as a large turn, but \
loose clothing and the place where the {limb} leaves the body turn the line too) and skin (how much surface lies \
round it).

{table}

What to answer, each a dot's number from the table:
{asked}
The dots you give must lie in that order along the line, each a different dot.{again}"""
ARM_ASKED = """- shoulder: the shoulder joint, where the upper {limb} begins and turns: the first dot of the {limb} \
itself, not one inside the chest.
- elbow: where the {limb} bends, or would bend if it is drawn straight; null only if it has no such joint.
- hand: the wrist, where the {end} begins; never a finger, and not the {end}'s tip."""
LEG_ASKED = """- hip: the hip joint, where the thigh begins and turns: the first dot of the {limb} itself.
- knee: where the {limb} bends, or would bend if it is drawn straight; null only if it has no such joint.
- ankle: where the foot begins.
- toe: the joint in the foot the toes bend at; null if the foot has none."""

_JOINT = {'anyOf': [{'type': 'integer'}, {'enum': [None]}]}     # a joint's number, or null (ai._check's forms)
_LIMB = lambda parts: {'type': 'object', 'properties': {p: _JOINT for p in parts}, 'required': list(parts)}  # noqa: E731
_PAIR = lambda parts: {'type': 'object', 'properties': {s: _LIMB(parts) for s in ('left', 'right')},     # noqa: E731
                       'required': ['left', 'right']}
SCHEMA = {'type': 'object',
          'properties': {'character': {'type': 'string'}, 'front_view': {'type': 'string', 'enum': list(VIEWS)},
                         'plan': {'type': 'string', 'enum': list(PLANS)},
                         'hips': _JOINT, 'chest': _JOINT, 'neck': _JOINT, 'skull': _JOINT,
                         'legs': _PAIR(LEG), 'arms': _PAIR(ARM), 'notes': {'type': 'string'}},
          'required': ['front_view', 'plan', 'hips', 'chest', 'neck', 'skull', 'legs', 'arms']}


class Unusable(ValueError):
    """An answer the body can't be built from; its words are told to the second ask."""


def sides(up):
    """The four ways the camera stands, a quarter turn apart about ``up``: A to D. On a file standing on +z they are
    -y, +x, +y, -x."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    first = min((np.array(a, float) for a in ((0, -1, 0), (1, 0, 0), (0, 0, 1))), key=lambda a: abs(a @ up))
    first = first - (first @ up) * up
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    return dict(zip(VIEWS, (first, second, -first, -second)))


def joints(rig, up):
    """The model's joints as the look is shown them (a joint is the bones at one place, body._Mass), numbered from 1
    down the tree: (the mass, [{n, parent, across, deep, high, skin, below, names, bone}])."""
    mass = body_reading._Mass(rig)
    order = []

    def walk(b):
        order.append(b)
        for k in sorted(mass.kids[b]):
            walk(k)
    walk(next(b for b in sorted(mass.bones) if mass.parents[b] not in mass.bones))
    number = {b: i + 1 for i, b in enumerate(order)}
    way = sides(up)
    surface = mass.weights(mass.bones) > 0
    pts = rig.points[surface]
    axes = np.vstack([way['B'], -way['A'], np.asarray(up, float) / np.linalg.norm(up)])   # across, deep, high
    low, high = (pts @ axes.T).min(0), (pts @ axes.T).max(0)
    middle, height = (low + high) / 2, float(high[2] - low[2]) or 1.0
    out = []
    for b in order:
        p = (rig.heads[b] @ axes.T - np.array([middle[0], middle[1], low[2]])) / height * 100
        out.append({'n': number[b], 'parent': number.get(int(mass.parents[b])), 'across': round(float(p[0]), 1),
                    'deep': round(float(p[1]), 1), 'high': round(float(p[2]), 1),
                    'skin': int((mass.weights([b]) > 0.5).sum()), 'below': len(mass.sub[b]) - 1,
                    'names': [rig.names[k] for k in mass.groups[b]], 'bone': b})
    return mass, out


def table(rows):
    lines = ['  n  parent  across    deep    high  skin  below  names']
    for j in rows:
        lines.append(f'{j["n"]:3d}  {j["parent"] if j["parent"] is not None else "-":>6}  {j["across"]:6.1f}  '
                     f'{j["deep"]:6.1f}  {j["high"]:6.1f}  {j["skin"]:4d}  {j["below"]:5d}  {", ".join(j["names"])}')
    return '\n'.join(lines)


def entries(rig, mass, rows, answer, up, settle=None):
    """The record's body entries an answer names (record.py's hand form: plan, front, hips, chest, neck, skull, legs,
    arms), its sides read from where the limbs lie. Raises Unusable, saying why, when the body can't be built from it;
    returns None for a plan left to the reading. ``settle``: for a skeleton fitted inside the model, what finishes it
    once its joints are named (onepiece.named: the entries in, the Rig and the entries as the fit holds them out);
    the body is checked on those."""
    plan = answer.get('plan')
    if plan not in ('biped', 'quadruped'):
        return None
    at = {j['n']: j['bone'] for j in rows}

    def bone(n, what):
        if n not in at:
            raise Unusable(f'{what} is {n!r}, which is no joint of the table')
        return at[n]
    if answer.get('front_view') not in VIEWS:
        raise Unusable(f'front_view is {answer.get("front_view")!r}, not one of A, B, C, D')
    front = sides(up)[answer['front_view']]
    left = np.cross(np.asarray(up, float), front)
    # a fit's look may say the body has no neck (FITTED): the fit then puts the head on the shoulders (``settle``)
    neckless = settle is not None and answer.get('neck') is None
    core = {k: bone(answer.get(k), k) for k in ('hips', 'chest', 'neck', 'skull') if not (neckless and k == 'neck')}
    out = {'plan': plan, 'front': [float(x) for x in front], **{k: rig.names[b] for k, b in core.items()}}
    if neckless:
        out['neck'] = None
    for limbs, parts, ends in (('legs', LEG, ('hip', 'ankle')), ('arms', ARM, ('shoulder', 'hand'))):
        found = []
        for side in ('left', 'right'):
            said = ((answer.get(limbs) or {}).get(side) or {})
            if all(said.get(p) is None for p in parts):
                continue                    # the body has no such limb
            for end in ends:
                if said.get(end) is None:
                    raise Unusable(f'the {side} {limbs[:-1]} names no {end}')
            named = [(p, bone(said[p], f'the {side} {limbs[:-1]}\'s {p}')) for p in parts if said.get(p) is not None]
            if len({b for _, b in named}) < 2:
                raise Unusable(f'the {side} {limbs[:-1]} is one joint: it needs two ends')
            # a fitted arm's close-up may say where its shoulder was seen (closer()): the record's ``begins``
            begins = said.get('begins') if settle is not None and limbs == 'arms' else None
            found.append((named, None if begins is None else bone(begins, f'where the {side} arm begins')))
        if len(found) == 1:
            raise Unusable(f'one of the {limbs} is named and not the other')
        if not found:
            continue
        # the character's left is where its limbs lie, not what the answer called them
        found.sort(key=lambda limb: -float(np.mean([rig.heads[b] for _, b in limb[0]], axis=0) @ left))
        out[limbs] = {}
        for side, (named, begins) in zip(('left', 'right'), found):
            limb = {p: rig.names[b] for p, b in named}
            if begins is not None:
                limb['begins'] = rig.names[begins]
            chain = [b for _, b in named]
            if any(not body_reading._descends(mass.parents, b, a) for a, b in zip(chain, chain[1:])):
                limb['bones'] = [rig.names[b] for b in chain]       # not one under another: listed joint by joint
            out[limbs][side] = limb
    if plan == 'quadruped' and ('legs' not in out or 'arms' not in out):
        raise Unusable('a quadruped has hind legs and, as its arms, front legs')
    if 'legs' not in out:
        return None                         # legless: the plan draws its legs (plans.py), left to the reading
    try:                                    # the body itself is the last check: it must build
        if settle is not None:
            rig, out = settle(out)
            mass = body_reading._Mass(rig)
        body_reading._from_record(rig, mass, np.asarray(up, float), front, out)
    except ValueError as e:
        raise Unusable(str(e)) from e
    return out


def _way_up(by, n):
    """``n`` and every joint above it in the table's tree, nearest first."""
    out = [n]
    while by[out[-1]]['parent'] in by and len(out) <= len(by):
        out.append(by[out[-1]]['parent'])
    return out


def between(rows, answer):
    """The joints of the table that lie between an answer's hips and its skull, both counted: the spine."""
    by = {j['n']: j for j in rows}
    hips, skull = _way_up(by, answer['hips']), _way_up(by, answer['skull'])
    return set(hips) ^ set(skull) | {next((n for n in hips if n in set(skull)), hips[-1])}


def line(rows, answer, limbs, side):
    """The dots of one limb an answer names, from the trunk out: (the dots, how many of them lie in the limb as it was
    named, to its last named joint). The first is the spine joint the limb leaves: on the way up the tree from the
    limb's last named joint, the first joint that lies between the hips and the skull (a fitted skeleton hangs from
    the top of the head, onepiece.read, so the way up from a hand or a toe reaches the spine). Then every joint down
    to that last one, and on to the limb's tip: at a fork, the way with the most joints below. ([], 0) where the
    answer names no such limb, or names it on joints that are not the table's or not one above another."""
    by = {j['n']: j for j in rows}
    parts = LEG if limbs == 'legs' else ARM
    said = ((answer.get(limbs) or {}).get(side) or {})
    named = [said[p] for p in parts if said.get(p) is not None]
    if len(named) < 2 or any(n not in by for n in named) or answer.get('hips') not in by or answer.get('skull') not in by:
        return [], 0
    spine = between(rows, answer)
    way = _way_up(by, named[-1])
    if any(n not in way for n in named) or way[0] in spine:
        return [], 0
    top = next((i for i, n in enumerate(way) if n in spine), max(way.index(n) for n in named))
    dots = way[:top + 1][::-1]
    held = len(dots)
    kids = {}
    for j in rows:
        kids.setdefault(j['parent'], []).append(j['n'])
    while kids.get(dots[-1]):
        dots.append(max(kids[dots[-1]], key=lambda k: (by[k]['below'], -k)))
    return dots, held


def turns(rows, dots):
    """Per dot of a limb's line, (how far along the line it lies, as a percentage of its length; how many degrees the
    line turns there). The turn is read SPAN dots each way and is 0 at the line's two ends."""
    by = {j['n']: j for j in rows}
    p = np.array([[by[n]['across'], by[n]['deep'], by[n]['high']] for n in dots], float)
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / (steps.sum() or 1.0) * 100
    out = []
    for i in range(len(p)):
        u, v = p[i] - p[max(i - SPAN, 0)], p[min(i + SPAN, len(p) - 1)] - p[i]
        size = np.linalg.norm(u) * np.linalg.norm(v)
        turn = float(np.degrees(np.arccos(np.clip(u @ v / size, -1, 1)))) if 0 < i < len(p) - 1 and size > 1e-9 else 0.0
        out.append((float(along[i]), turn))
    return out


def limb_table(rows, dots):
    by = {j['n']: j for j in rows}
    lines = ['  n  along  turn  skin']
    for n, (along, turn) in zip(dots, turns(rows, dots)):
        lines.append(f'{n:3d}  {along:5.0f}  {turn:4.0f}  {by[n]["skin"]:4d}')
    return '\n'.join(lines)


def facing(points, middle, up, side):
    """How a camera stands to see a limb square on: (the way from the limb to the camera, the picture's up).
    ``points``: the limb's dots, in order. The camera looks along the way the limb's steps reach least, each step
    counted by its length, so none of the limb's length is hidden: square onto a bent arm's own plane, and at a leg
    and its foot from the side. (Not the plane of the dots' places: a leg that only slants reads as flat there, and
    Goku's thigh, pointing forward, was looked at end on.) A limb whose steps all run one way has no such way of its
    own, and the camera looks across it from as near ``side`` (a side the whole model is shown from) as that allows.
    The camera stands on the limb's side of the body's ``middle``, so the body lies behind the limb, and the
    picture's up is the model's ``up`` as near as the view allows."""
    points = np.asarray(points, float)
    centre = points.mean(axis=0)
    out, along = onepiece.across(points)
    if out is None:
        ways = [np.asarray(a, float) for a in (side, np.cross(up, side), up)]      # a side of the model before above
        out = next((a for a in ways if abs(a @ along) < 0.5), min(ways, key=lambda a: abs(a @ along)))
        out = out - (out @ along) * along
    out = out / np.linalg.norm(out)
    if (centre - np.asarray(middle, float)) @ out < 0:
        out = -out
    up = np.asarray(up, float) / np.linalg.norm(up)
    top = up - (up @ out) * out
    if np.linalg.norm(top) < 0.3:       # looked at from above or below: the limb's own way is the picture's up
        top = along - (along @ out) * out
    return out, top / np.linalg.norm(top)


def _limb_fault(reply, dots, parts):
    """Why a limb's answer can't be used, or None: its two ends are named, every joint is a dot of the line, and they
    lie in order along it."""
    ends = (parts[0], parts[-2] if parts[-1] == 'toe' else parts[-1])
    for end in ends:
        if reply.get(end) is None:
            return f'no {end} is named'
    named = [(p, reply[p]) for p in parts if reply.get(p) is not None]
    for p, n in named:
        if n not in dots:
            return f'the {p} is {n!r}, which is no dot of this line'
    for (p, a), (q, b) in zip(named, named[1:]):
        if dots.index(b) <= dots.index(a):
            return f'the {q} ({b}) does not lie past the {p} ({a}) along the line'
    return None


def closer(rig, mass, rows, answer, up, out, ask, which=CLOSER):
    """Each limb of an answer asked again on a picture of that limb alone, for a skeleton fitted inside a file that
    had none (LIMB): -> (the answer with those limbs' joints, the report: per limb its line, picture and asks, and
    the joints taken). ``which``: 'arms', 'legs' or both (CLOSER says why the arms alone). A limb's answer that can't
    be used (_limb_fault) is asked once more of the second model, told why; a limb with no usable answer keeps the
    joints the first look gave it. No collar joint is asked: a fitted line has none, and what lies between the chest
    and the shoulder rides the chest."""
    found = [(limbs, side, *line(rows, answer, limbs, side)) for limbs in ('legs', 'arms') if limbs in which
             for side in ('left', 'right')]
    found = [f for f in found if f[2]]
    if not found:
        return answer, []
    pictures = closeups(rig, mass, rows, [(dots, held) for _, _, dots, held in found], up, Path(out) / 'fill-limb.png')
    four = answer.get('plan') == 'quadruped'
    who = f' ({answer["character"]})' if answer.get('character') else ''
    merged, report = json.loads(json.dumps(answer)), []
    for (limbs, side, dots, held), picture in zip(found, pictures):
        arm = limbs == 'arms'
        parts = ARM[1:] if arm else LEG
        limb = ('front leg' if four else 'arm') if arm else ('hind leg' if four else 'leg')
        asked = (ARM_ASKED if arm else LEG_ASKED).format(limb=limb, end='paw' if four else 'hand')
        entry = {'limb': limbs[:-1], 'side': side, 'line': dots, 'picture': str(picture), 'asked': []}
        again = ''
        for route in (LIMB_ROUTE, LIMB_AGAIN):
            question = LIMB.format(limb=limb, who=who, joints='shoulder, elbow and wrist' if arm else
                                   'hip, knee, ankle and toe joints', first=dots[0], last=dots[-1],
                                   bend='elbow' if arm else 'knee', table=limb_table(rows, dots), asked=asked,
                                   again=again)
            reply, info = ask(question, [str(picture)], _LIMB(parts), {}, route=route)
            said = {'route': route, 'source': (info or {}).get('source'), 'answer': reply,
                    'error': (info or {}).get('error'), 'cost': (info or {}).get('cost'),
                    'seconds': (info or {}).get('seconds')}
            entry['asked'].append(said)
            fault = 'no answer' if not reply else _limb_fault(reply, dots, parts)
            if fault:
                said['unusable'] = fault
                again = '' if not reply else (f'\n\nAn earlier answer could not be used: {fault}. '
                                              'Answer again with that put right.')
                continue
            joints = {**({'clavicle': None} if arm else {}), **{p: reply.get(p) for p in parts}}
            # A shoulder named at the spine joint the arm leaves was seen there, square onto the arm: the arm is held
            # against the body and its own dots begin by the elbow. The arm then begins at its first dot and the
            # record keeps where its shoulder was seen (``begins``: onepiece.shoulder_joints puts the joint there)
            if arm and joints['shoulder'] == dots[0] and dots[0] in between(rows, answer) and len(dots) > 2 \
                    and dots[1] not in (joints['elbow'], joints['hand']):
                joints.update(shoulder=dots[1], begins=dots[0])
            entry['joints'] = joints
            merged[limbs][side] = joints
            break
        report.append(entry)
    return merged, report


def _number(d, x, y, label, font, placed):
    """A dot's number beside it, moved out to the first place clear of the numbers already drawn (``placed``)."""
    w, h = d.textbbox((0, 0), label, font=font)[2:]
    spots = [(x + 9 + r * np.cos(a), y - h / 2 + r * np.sin(a)) for r in (0, 20, 38, 58, 80)
             for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    lx, ly = next(((sx, sy) for sx, sy in spots
                   if all(sx > px + pw + 1 or sx + w + 1 < px or sy > py + ph + 1 or sy + h + 1 < py
                          for px, py, pw, ph in placed)), spots[0])
    placed.append((lx, ly, w + 4, h + 4))
    if abs(lx - (x + 9)) > 1 or abs(ly - (y - h / 2)) > 1:
        d.line([(x, y), (lx + w / 2, ly + h / 2)], fill=(255, 255, 255), width=1)
    d.rectangle([lx - 1, ly, lx + w + 3, ly + h + 4], fill=(0, 0, 0))
    d.text((lx + 1, ly), label, fill=(255, 255, 255), font=font)


def closeups(rig, mass, rows, limbs, up, path):
    """A picture of each limb alone, the camera square to it (facing()): the model in its own colours, and beside it
    the same view dimmed with the limb's dots numbered as the table numbers them. ``limbs``: [(its dots, how many of
    them lie in the limb as named)]; the camera is squared to those, and frames every dot. Returns the paths."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    hidden = []
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name not in rig.meshes and not ob.hide_render:
            ob.hide_render = True
            hidden.append(ob)
    up = np.asarray(up, float) / np.linalg.norm(up)
    pts = rig.points[mass.weights(mass.bones) > 0]
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    cam = bpy.data.objects.new('core_fill_limb', bpy.data.cameras.new('core_fill_limb'))
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 20
    scene.collection.objects.link(cam)
    before = scene.camera
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = path.with_name(f'{path.stem}-tile.png')
    at = {j['n']: j['bone'] for j in rows}
    font = _font(22, bold=True)
    out = []
    try:
        for i, (dots, held) in enumerate(limbs):
            heads = np.array([rig.heads[at[n]] for n in dots], float)
            side, top = facing(heads[1:max(held, 3)], pts.mean(axis=0), up, sides(up)['A'])
            right = np.cross(top, side)
            u, v = heads @ right, heads @ top
            width = max(float(max(np.ptp(u), np.ptp(v))) * 1.35, size * 0.3)
            middle = right * (u.min() + u.max()) / 2 + top * (v.min() + v.max()) / 2 + side * float((heads @ side).mean())
            m4 = np.eye(4)
            m4[:3, :3], m4[:3, 3] = np.column_stack([right, top, side]), middle + side * size * 4
            cam.matrix_world = Matrix(m4.tolist())
            cam.data.ortho_scale = width
            scene.render.resolution_x = scene.render.resolution_y = CLOSE
            scene.render.filepath = str(tile)
            still()
            plain = Image.new('RGB', (CLOSE, CLOSE), (150, 155, 165))
            with Image.open(tile) as rendered:
                rgba = rendered.convert('RGBA')
            tile.unlink()
            plain.paste(rgba, (0, 0), rgba)
            dim = Image.blend(plain, Image.new('RGB', (CLOSE, CLOSE), (60, 62, 68)), 0.5)
            d = ImageDraw.Draw(dim)
            spots = [(CLOSE / 2 + (h - middle) @ right / width * CLOSE, CLOSE / 2 - (h - middle) @ top / width * CLOSE)
                     for h in heads]
            d.line(spots, fill=(255, 220, 90), width=3)
            placed = []
            for (x, y), n in zip(spots, dots):
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 80, 60), outline=(0, 0, 0))
            for (x, y), n in zip(spots, dots):
                _number(d, x, y, str(n), font, placed)
            sheet = Image.new('RGB', (2 * CLOSE, CLOSE), (16, 18, 22))
            sheet.paste(plain, (0, 0))
            sheet.paste(dim, (CLOSE, 0))
            out.append(path.with_name(f'{path.stem}-{i + 1}.png'))
            sheet.save(out[-1])
    finally:
        scene.camera = before
        bpy.data.objects.remove(cam, do_unlink=True)
        for ob in hidden:
            ob.hide_render = False
    return out


def shown(rig, up, out):
    """Draw what the look is shown into ``out`` and keep the numbered joints beside the pictures
    (``fill-joints.json``): a later complaint names a joint by its number (complaint.py), so a build whose body came
    from a record draws them too, with nothing asked. -> (mass, the joint rows, the pictures' paths)."""
    mass, rows = joints(rig, up)
    pictures = [str(p) for p in draw(rig, mass, rows, up, Path(out) / 'fill.png')]
    (Path(out) / 'fill-joints.json').write_text(json.dumps([{k: v for k, v in j.items() if k != 'bone'}
                                                             for j in rows]), encoding='utf-8')
    return mass, rows, pictures


def read(rig, up, out, ask, second_first=False, settle=None):
    """Ask the look and return (the body's entries for the record, or None when the body is left to the reading,
    the report). ``second_first``: the second model is asked first (a costume's second try). ``settle``: given for a
    skeleton fitted inside the model (entries()); the look is then told what its joints are (FITTED)."""
    mass, rows, pictures = shown(rig, up, out)
    report = {'joints': len(rows), 'asked': []}
    again = ''
    for route in ((AGAIN, ROUTE) if second_first else (ROUTE, AGAIN)):
        answer, info = ask(QUESTION.format(table=table(rows), again=again, fitted=FITTED if settle else '',
                                           neck=NECK_OFFERED if settle else NECK_OWN), pictures,
                           SCHEMA, {}, route=route)
        said = {'route': route, 'source': (info or {}).get('source'), 'answer': answer,
                'error': (info or {}).get('error')}
        report['asked'].append(said)
        if not answer:
            said['unusable'] = 'no answer'
            continue
        # a fitted skeleton's limbs are each asked again on a picture of the limb alone (closer()); a body that can't
        # be built from those joints is tried on the joints the first look gave. Drawing or asking them never fails
        # a build: the first look's joints stand
        tries = [answer]
        if settle is not None and answer.get('plan') in ('biped', 'quadruped'):
            try:
                closed, said['limbs'] = closer(rig, mass, rows, answer, up, out, ask)
                if closed != answer:
                    tries.insert(0, closed)
            except Exception as e:      # noqa: BLE001
                said['limbs_failed'] = f'{type(e).__name__}: {e}'
        made = fault = None
        for i, tried in enumerate(tries):
            try:
                made, fault = entries(rig, mass, rows, tried, up, settle), None
                break
            except Unusable as e:
                fault = e
                if len(tries) > 1 and i == 0:
                    said['limbs_unusable'] = str(e)
        if fault is not None:
            said['unusable'] = str(fault)
            again = f'\n\nAn earlier answer could not be used: {fault}. Answer again with that put right.'
            continue
        report['character'] = answer.get('character')
        report['notes'] = answer.get('notes')
        if made is None:
            report['left_to_the_reading'] = f'plan {answer.get("plan")}'
        print('fill: ' + json.dumps({'plan': answer.get('plan'), 'front_view': answer.get('front_view'),
                                     'body': made and {k: made[k] for k in ('hips', 'chest', 'neck', 'skull')}}))
        return made, report
    report['left_to_the_reading'] = 'no usable answer'
    return None, report


def draw(rig, mass, rows, up, path):
    """The pictures the look is shown: the model in its own colours from four sides (one sheet), and sides A and B
    dimmed with every joint numbered and joined to its parent (a picture each). Each side is framed on what it shows.
    Returns their paths."""
    import bpy
    from mathutils import Matrix
    from PIL import Image, ImageDraw
    from .render import _font, _prepare_scene
    for ob in bpy.data.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    _prepare_scene()
    scene = bpy.context.scene
    hidden = []
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.name not in rig.meshes and not ob.hide_render:
            ob.hide_render = True
            hidden.append(ob)
    up = np.asarray(up, float) / np.linalg.norm(up)
    way = sides(up)
    surface = mass.weights(mass.bones) > 0
    pts = rig.points[surface]
    size = float(np.ptp(pts, axis=0).max()) or 1.0
    cam = bpy.data.objects.new('core_fill', bpy.data.cameras.new('core_fill'))
    cam.data.type = 'ORTHO'
    cam.data.clip_start, cam.data.clip_end = size * 1e-3, size * 20
    scene.collection.objects.link(cam)
    before = scene.camera
    scene.camera = cam
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = path.with_name(f'{path.stem}-tile.png')

    def shoot(view, pixels):
        """(the picture, the view's rightward way, the middle of what it shows, how wide it is)."""
        side = way[view]
        right = np.cross(up, side)
        u, v = pts @ right, pts @ up
        middle = right * (u.min() + u.max()) / 2 + up * (v.min() + v.max()) / 2 + side * float((pts @ side).mean())
        width = float(max(np.ptp(u), np.ptp(v)) * 1.12) or 1.0
        m4 = np.eye(4)
        m4[:3, :3], m4[:3, 3] = np.column_stack([right, up, side]), middle + side * size * 4
        cam.matrix_world = Matrix(m4.tolist())
        cam.data.ortho_scale = width
        scene.render.resolution_x = scene.render.resolution_y = pixels
        scene.render.filepath = str(tile)
        still()
        img = Image.new('RGB', (pixels, pixels), (150, 155, 165))
        with Image.open(tile) as rendered:
            rgba = rendered.convert('RGBA')
        tile.unlink()
        img.paste(rgba, (0, 0), rgba)
        return img, right, middle, width
    out = []
    try:
        sheet = Image.new('RGB', (2 * TILE, 2 * TILE), (16, 18, 22))
        for i, view in enumerate(VIEWS):
            img = shoot(view, TILE)[0]
            ImageDraw.Draw(img).text((10, 6), view, fill='black', font=_font(34, bold=True), stroke_width=3,
                                     stroke_fill='white')
            sheet.paste(img, ((i % 2) * TILE, (i // 2) * TILE))
        out.append(path.with_name(f'{path.stem}-colours.png'))
        sheet.save(out[-1])
        font = _font(17, bold=True)
        bone = {j['bone']: j for j in rows}
        for view in 'AB':
            img, right, middle, width = shoot(view, BIG)
            img = Image.blend(img, Image.new('RGB', (BIG, BIG), (60, 62, 68)), 0.55)
            d = ImageDraw.Draw(img)

            def at(b):
                q = rig.heads[b] - middle
                return BIG / 2 + (q @ right) / width * BIG, BIG / 2 - (q @ up) / width * BIG
            for b, j in bone.items():
                if int(mass.parents[b]) in bone:
                    d.line([at(int(mass.parents[b])), at(b)], fill=(255, 220, 90), width=2)
            placed = []
            for b, j in bone.items():
                x, y = at(b)
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 80, 60), outline=(0, 0, 0))
                label = str(j['n'])
                w, h = d.textbbox((0, 0), label, font=font)[2:]
                # the number beside its dot, moved out to the first place clear of the numbers already drawn
                spots = [(x + 7 + r * np.cos(a), y - h / 2 + r * np.sin(a)) for r in (0, 16, 30, 46, 64)
                         for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
                lx, ly = next(((sx, sy) for sx, sy in spots
                               if all(sx > px + pw + 1 or sx + w + 1 < px or sy > py + ph + 1 or sy + h + 1 < py
                                      for px, py, pw, ph in placed)), spots[0])
                placed.append((lx, ly, w + 4, h + 4))
                if abs(lx - (x + 7)) > 1 or abs(ly - (y - h / 2)) > 1:
                    d.line([(x, y), (lx + w / 2, ly + h / 2)], fill=(255, 255, 255), width=1)
                d.rectangle([lx - 1, ly, lx + w + 3, ly + h + 4], fill=(0, 0, 0))
                d.text((lx + 1, ly), label, fill=(255, 255, 255), font=font)
            d.text((10, 6), f'{view}: joints numbered', fill='white', font=_font(26, bold=True), stroke_width=3,
                   stroke_fill='black')
            out.append(path.with_name(f'{path.stem}-joints-{view}.png'))
            img.save(out[-1])
    finally:
        scene.camera = before
        bpy.data.objects.remove(cam, do_unlink=True)
        for ob in hidden:
            ob.hide_render = False
    return out
