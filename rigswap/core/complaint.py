"""A complaint: the user says what looks wrong, and one entry of the record is changed.

A wrong import is put right in its record, not in the code (record.py). The user's words ("the legs look wrong",
"the head is too big") are no entry of the record, so one look turns them into one: it is shown the source, the
draft in two of the fighter's frames, the fighter's own body in those frames, the source's joints numbered where
the run kept them, and the record as it stands, and it names the one change most likely to fix what the user
describes: a size, the front, up to four joints, a part off the body that should swing or hold still, or a piece
of the model to leave out or bring back. Code checks the change is one the record can hold and applies it; the
costume is then built again from the changed record. A part that swings is one the record calls hair, cloth or a
tail (``parts``: labels.chains; correspond hangs those and lets a long tail swing on joints of its own), so making
one swing or hold still is a change of what it is called, for it and all that hangs from it. A
piece is named in the person's words and goes beside the record as a request (``pieces``): which of the model's
pieces it is, is settled by the build, which has the model open (pieces.py).

A size named outright ("make the head a fifth smaller") needs no look: ``sized`` applies it.

Lab, 2026-11-09 (the second record lab): shown a draft with no words, GPT-6 Luna named the fault written
down beforehand on 13 of 16 known-bad drafts. With the user's words, on six finished builds (work/rec/complaint, the
change expected written down first): five named it; the sixth ("she is too small") was refused because the look
judged the complaint untrue, so the question now says the person's word stands.
"""
import json
from pathlib import Path

from . import record as records

ROUTE = 'core complaint'
SIZES = ('size', 'head', 'limbs', 'trunk')
LEAST, MOST = 0.5, 2.0                  # one complaint changes a size by no more than this
STEP = {'smaller': 0.8, 'bigger': 1.25}     # the step a size takes when the person named no amount: the look says
                                        # which way, never how much (asked for both, it left the amount out or
                                        # flattened one the person had named: work/rec/complaint/heard*.json)
WAYS = {'leave out': 'out', 'bring back': 'back'}       # a piece of the model, in or out of the record's ``drop``
SWINGS = {'hair': 'hair', 'cloth': 'cloth', 'tail': 'tail', 'stiff': 'other'}   # what a part off the body is called
                                                        # to swing as that, or to ride the bone it hangs from
CORE = ('hips', 'chest', 'neck', 'skull')
LIMBS = {'leg': ('hip', 'knee', 'ankle', 'toe'), 'arm': ('clavicle', 'shoulder', 'elbow', 'hand')}
PARTS = list(CORE) + [f'{side} {limb} {part}' for limb, parts in LIMBS.items() for side in records.SIDES
                      for part in parts]

QUESTION = """A 3D character model (the source) was put onto a fighting-game fighter's skeleton, so that the \
fighter's animations move it. The person who imported it says something looks wrong. Say which one thing to change.

Picture 1: the source model in its own colours.
Picture 2: the draft at two moments of the fighter's animations, idle (top row) and run (bottom row), from a \
three-quarter side and from the front. The yellow line is the floor.
Picture 3: the fighter's own body at the same two moments: how the fighter itself stands and runs.{joint_pictures}

The person says: "{words}"{heard}

Take the person at their word: they have seen it move in the game and you have not. Do not judge whether they are \
right, and do not answer that it looks fine; find the change that does what they ask.

How the draft was made:
- sizes, each 1 as first made: whole body {size}, head {head}, limb width {limbs}, trunk width {trunk}.
{joints}
Answer with the ONE change most likely to fix what the person describes:
- "head", "limbs", "trunk" or "size": that size is wrong. Give `way`: "smaller" (or thinner) or "bigger" (or \
thicker). Give `times`, the factor to multiply it by, only when the person's own words name an amount: 0.7 for "30% \
smaller", 0.5 for "half the size", 1.5 for "half again as big"; stay between 0.5 and 2. When they name none, leave \
`times` out: a modest step is taken that way.
- "front": it faces the wrong way. Give `front_view`, the side of picture 1 (A, B, C or D) whose camera looks at \
its face.{joint_choice}
- "leave out": a piece of the model should not be on the costume (a pack, a prop, a weapon, an effect, a second \
outfit, a piece drawn twice). Give `piece`: a few words naming it as the person does, and where on the body it is.
- "bring back": a piece the source has in picture 1 is missing from the draft in picture 2 (a hat, a tail, a weapon, \
a part of the face). Give `piece` the same way.
- "nothing": none of these can do what they ask (a texture or a colour, how an animation moves, a piece the source \
never had). Say why.
- why: one sentence in plain words for that person, saying what you are changing and why."""

# What an assistant between the person and this look made of their words (``fix``'s ``heard``). It says which part
# they mean when their own words lean on the talk before ("still too big", "yes, do that"); what it adds of its own
# is not theirs. 2026-11-09, the first real run: "it's still too big" reached the look as "make it much smaller,
# around 0.6", and the head came out at 0.6.
HEARD = """
The assistant that passed this on summed it up as: "{heard}". Use that summary only to tell which part of the body \
the person means, or what they are agreeing to. A size, a number or an amount that is not in the person's own \
words is the assistant's guess, not theirs: leave it out. An amount the person's own words name is theirs."""

JOINT_PICTURES = """
Pictures 4 and 5: the source from sides A and B, dimmed, with every joint of its skeleton numbered."""
# What the look is told of a costume whose file had no skeleton (onepiece.py): what its numbered joints are, and what
# the neck's does (fill.FITTED says why: the fighter's head goes on it). Told "past the skull", it moved Crash's skull
# to his neck joint (dots-2), which changed nothing to the eye. Left out for any other costume, whose question is then
# as it was.
FITTED = """
This model's file had no skeleton: those joints are a line fitted through the middle of its shape, one every little \
way along each part, and the body's parts were put on the ones listed below. The fighter's head is put on the neck \
joint: everything past it moves as one stiff piece with the head, and what lies between the chest joint and the neck \
joint is the fighter's neck, squeezed or stretched to that neck's length and bent with it. The skull joint is the \
neck's own or one past it, inside the head: moving it alone changes almost nothing to the eye."""
# What the look is told of the neck joint on a costume from a file's own skeleton (fill.NECK_OWN says why).
OWN = """
The fighter's head is put on the neck joint: the head and all that hangs from that joint move with it as one piece, \
and what lies between the chest joint and it is the fighter's neck. Where the skeleton has a neck bone and a head \
bone, the neck joint belongs on the head bone's joint, at the top of the neck."""
JOINT_CHOICE = """
- "joints": a part of the body is on the wrong joint (a leg that bends in the wrong place, an arm that starts at \
the elbow, a head joint too low). Give `joints`: up to four entries, each the part (one of: {parts}) and `to`, the \
number of the joint it should be on.
- "swing": a part hanging off the body (hair, a ponytail, a cape, a skirt, a scarf, a tail, long ears) should swing \
as the fighter moves and holds stiff, or swings and should hold still. Give `joint`: the number of the joint where \
that part leaves the body, and `as`: what it is, "hair", "cloth" or "tail", for a part that should swing; "stiff" for \
one that should stay as it is drawn."""

SCHEMA = {'type': 'object',
          'properties': {'change': {'type': 'string', 'enum': ['head', 'limbs', 'trunk', 'size', 'front', 'joints',
                                                               'swing', 'leave out', 'bring back', 'nothing']},
                         'piece': {'type': 'string'},
                         'joint': {'type': 'integer'}, 'as': {'type': 'string', 'enum': list(SWINGS)},
                         'way': {'type': 'string', 'enum': ['smaller', 'bigger']},
                         'times': {'type': 'number'}, 'front_view': {'type': 'string'},
                         'joints': {'type': 'array', 'items': {'type': 'object', 'properties': {
                             'part': {'type': 'string'}, 'to': {'type': 'integer'}}, 'required': ['part', 'to']}},
                         'why': {'type': 'string'}},
          'required': ['change', 'why']}


class Unfixable(ValueError):
    """A change the record cannot hold, said in words for the person who complained."""


def sized(record, sizes):
    """``record`` with each of ``sizes`` ({'head': 0.8}) multiplied into that size, kept between LEAST and MOST of
    what it was. -> (the record, {size: [was, is]})."""
    out, changed = dict(record), {}
    for name, times in sizes.items():
        if name not in SIZES:
            raise Unfixable(f'{name!r} is no size of the record: the sizes are {", ".join(SIZES)}')
        was = float(record.get(name, 1.0))
        out[name] = round(was * min(max(float(times), LEAST), MOST), 4)
        changed[name] = [was, out[name]]
    return out, changed


def applied(record, answer, rows=None, fronts=None):
    """``record`` with the one change ``answer`` names. ``rows``: the joints the look was shown (fill.joints: n and
    names), for a change of joints; ``fronts``: {view: the front's vector} (fill.sides), for a change of front.
    -> (the record, {entry: [was, is]}); Unfixable when the answer names no change the record can hold."""
    change = answer.get('change')
    if change in SIZES:
        times = answer.get('times') if isinstance(answer.get('times'), (int, float)) else STEP.get(answer.get('way'))
        if times is None:
            raise Unfixable('the change names a size and no factor')
        return sized(record, {change: times})
    if change == 'front':
        if not fronts or answer.get('front_view') not in fronts:
            raise Unfixable('the change names no side the front could be')
        out = dict(record)
        out['front'] = [float(x) for x in fronts[answer['front_view']]]
        return out, {'front': [record.get('front'), out['front']]}
    if change in WAYS:
        piece = ' '.join(str(answer.get('piece') or '').split())
        if not piece:
            raise Unfixable('the change names no piece of the model')
        if 'drop' not in record:
            raise Unfixable('this costume kept no list of what it leaves out to change')
        out = dict(record)
        out['pieces'] = {'way': WAYS[change], 'piece': piece[:200]}
        return out, {'pieces': [None, f'{change}: {piece[:200]}']}
    if change == 'swing':
        if not rows:
            raise Unfixable('this costume kept no numbered joints to choose from')
        to, kind = answer.get('joint'), SWINGS.get(answer.get('as'))
        names = {j['n']: j['names'] for j in rows}
        if kind is None or to not in names:
            raise Unfixable('the change names no joint of the model, or no way for the part to move')
        below, grew = {to}, True                # the joint and all that hangs from it
        while grew:
            more = {j['n'] for j in rows if j.get('parent') in below} - below
            below, grew = below | more, bool(more)
        parts = dict(record.get('parts') or {})
        if not any(name in parts for name in names[to]):        # a joint of the body: every part below it is not meant
            raise Unfixable('that joint is on the body itself: no part hangs off the body there')
        mine = [name for n in sorted(below) for name in names[n] if name in parts]
        was = sorted({parts[b].get('kind') for b in mine})
        if was == [kind]:
            raise Unfixable('that part is already ' + ('held stiff' if kind == 'other' else f'swinging as {kind}'))
        for b in mine:
            parts[b] = dict(parts[b], kind=kind)
        out = dict(record)
        out['parts'] = parts
        return out, {f'parts from {names[to][0]}': ['/'.join(str(k) for k in was), kind]}
    if change == 'joints':
        if not rows:
            raise Unfixable('this costume kept no numbered joints to choose from')
        names = {j['n']: j['names'][0] for j in rows}
        out = records.as_joints(record)
        changed = {}
        for edit in (answer.get('joints') or [])[:4]:
            part, to = str(edit.get('part', '')).strip().lower(), edit.get('to')
            if part not in PARTS or to not in names:
                raise Unfixable(f'{edit!r} names no part of the body, or no joint of the model')
            if part in CORE:
                changed[part] = [out.get(part), names[to]]
                out[part] = names[to]
                continue
            side, limb, which = part.split()
            limbs = dict(out.get(limb + 's') or {})
            mine = dict(limbs.get(side) or {})
            if not mine:
                raise Unfixable(f'the body has no {side} {limb} to change')
            changed[part] = [mine.get(which), names[to]]
            mine[which] = names[to]
            mine.pop('bones', None)         # its joints are named anew: the chain is the tree's path between them
            if which in ('shoulder', 'clavicle'):
                mine.pop('begins', None)    # where a look saw the old shoulder says nothing of the new one
            limbs[side] = mine
            out[limb + 's'] = limbs
        if not changed:
            raise Unfixable('the change names no joint')
        return out, changed
    raise Unfixable(answer.get('why') or 'nothing in the record would fix that')


def described(record, rows):
    """The record's body in the look's words: each part and the number of its joint (None where the record keeps
    no joints the look could be shown)."""
    try:
        joints = records.as_joints(record)
    except ValueError:
        return None
    number = {name: j['n'] for j in rows for name in j['names']}
    lines = ['- joints now (numbers as in pictures 4 and 5): ' + ', '.join(
        f'{k} {number.get(joints.get(k), "?")}' for k in CORE)]
    for limb, parts in LIMBS.items():
        for side in records.SIDES:
            mine = (joints.get(limb + 's') or {}).get(side)
            if mine:
                lines.append(f'  {side} {limb}: ' + ', '.join(f'{p} {number.get(mine[p], "?")}'
                                                               for p in parts if mine.get(p)))
    return '\n'.join(lines) + '\n'


def fix(swap_dir, words, ask, heard=None):
    """Ask the look what to change for the complaint ``words`` about the costume in ``swap_dir`` (a run's folder: its
    record.json and the pictures it kept). ``words`` are the person's own; ``heard``, what an assistant between them
    and this look made of them, shown apart and only to say which part they mean (``HEARD``).
    -> (the changed record, {entry: [was, is]}, the look's sentence); Unfixable when it names nothing the record can
    hold."""
    from . import fill, onepiece
    swap_dir = Path(swap_dir)
    record = json.loads((swap_dir / 'record.json').read_text(encoding='utf-8'))
    source = next((p for p in (swap_dir / 'fill-colours.png', swap_dir / 'labels-0.png') if p.is_file()), None)
    draft, fighter = swap_dir / 'tripwire.png', swap_dir / 'tripwire-fighter.png'
    if source is None or not draft.is_file() or not fighter.is_file():
        raise Unfixable('this costume kept no pictures of itself to look at')
    pictures = [source, draft, fighter]
    rows, joints = None, None
    kept = swap_dir / 'fill-joints.json'
    if kept.is_file() and all((swap_dir / f'fill-joints-{v}.png').is_file() for v in 'AB'):
        rows = json.loads(kept.read_text(encoding='utf-8'))
        joints = described(record, rows)
    if joints:
        pictures += [swap_dir / 'fill-joints-A.png', swap_dir / 'fill-joints-B.png']
    said = ' '.join(str(words).split())
    heard = ' '.join(str(heard or '').split())
    fitted = (FITTED if onepiece.is_fit([name for j in rows for name in j['names']]) else OWN) if joints else ''
    question = QUESTION.format(
        words=said.replace('"', "'")[:600], joints=joints or '',
        heard=HEARD.format(heard=heard.replace('"', "'")[:600]) if heard and heard.lower() != said.lower() else '',
        joint_pictures=JOINT_PICTURES + fitted if joints else '',
        joint_choice=JOINT_CHOICE.format(parts=', '.join(PARTS)) if joints else '',
        **{name: record.get(name, 1.0) for name in SIZES})
    answer, info = ask(question, [str(p) for p in pictures], SCHEMA, {}, route=ROUTE)
    if not answer:
        raise Unfixable('the look gave no answer')
    fronts = fill.sides(record['up']) if record.get('up') else None
    changed_record, changed = applied(record, answer, rows if joints else None, fronts)
    if 'pieces' in changed_record:          # the build's ask is given the person's own words too (pieces.settle)
        changed_record['pieces'] = dict(changed_record['pieces'], said=said[:600])
    changed_record['source'] = 'complaint'
    return changed_record, changed, answer.get('why') or ''
