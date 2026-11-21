"""The record: what is known about one model, in a file. The build reads the body from it and from nothing else.

``<out>/record.json`` is written by every run. A run with no record beside its load reads the model as before (the
labels, the forks, the body reading, the chains' names) and writes what it concluded; the build then goes on from the
file, not from what the reading held in memory. A run whose load has a ``record.json`` beside it takes that one: a
hand's, a look's, or an earlier run's with an entry changed. So a wrong import is put right in its record.

A full record (as a run writes it):

    {"version": 1, "source": "reading",
     "up": [x, y, z], "front": [x, y, z],        the file's own axes
     "character": "...", "fighter": "Sheik",
     "drop": [mesh, ...],                          what the everyday look leaves out
     "body": {"plan": "biped", "root": bone, "spine": [bone, ...], "chest": bone, "skull": bone,
              "legs": {"left": [bone, ...], "right": [...]}, "arms": {"left": [...], "right": [...]},
              "forks": {bone: bone or null},
              "marks": {"neck": bone,
                        "arms": {"left": {"shoulder": bone, "hand": bone or null}, "right": {...}} or null,
                        "legs": {"left": {"chain": [bone, ...], "ankle": bone or null, "toe": bone or null},
                                 "right": {...}} or null}},
     "parts": {bone: {"kind": "hair", "held_out": false}, ...},     the chains off the body
     "digits": [bone, ...], "equipment": {bone: kind},
     "size": 1.0, "head": 1.0, "limbs": 1.0, "trunk": 1.0}

- ``marks``: the joints correspond.model_chains otherwise reads from geometry: the joint the head turns on, each
  arm's shoulder joint and wrist, each leg's chain down to its toe with its ankle and toe joints. ``arms`` or ``legs``
  null: the plan draws them (a ball's feet, a body with no arms: plans.py).
- ``size``: the model's size as a multiple of the one the run reads (run.grown). ``head``: the head's size as a
  multiple of the model's own, about the joint it turns on. ``limbs``, ``trunk``: how wide the limbs' segments and
  the trunk's come out, as a multiple of what the stretch gives them (repose.DIALS).

A hand record may name joints in place of ``body`` (body._from_record builds the chains from them):

    {"plan": "biped", "front": [0, -1, 0],
     "hips": bone, "chest": bone, "neck": bone, "skull": bone,
     "legs": {"left": {"hip": bone, "knee": bone, "ankle": bone, "toe": bone}, "right": {...}},
     "arms": {"left": {"clavicle": bone, "shoulder": bone, "elbow": bone, "hand": bone}, "right": {...}}}

- ``hips``: the joint the legs part from; ``chest``: the spine joint the arms hang from; ``neck``: the joint the head
  turns on; ``skull``: the bone carrying the head's own mesh (often the same joint as ``neck``). For a skeleton
  fitted inside a file that had none, ``"neck": null`` says the body has no neck: the head goes on the spine joint
  between the shoulders (onepiece.on_shoulders), and the record a run writes names that joint.
- a leg runs from its ``hip`` to its ``toe`` (or its ``ankle`` where it has no toe joint); an arm from its ``clavicle``
  (where the model has one) or its ``shoulder`` to its ``hand`` (the wrist). ``knee`` and ``elbow`` are optional.
- a limb is the tree's path between its ends; where the file hangs a piece of it off another bone (a foot on a floor
  control beside the hips), ``"bones": [...]`` lists its joints in order instead.
- for a skeleton fitted inside a file that had none, an arm may say ``"begins": bone``: the spine joint a look saw
  its shoulder at, on a picture square to the arm (fill.closer). The arm is held against the body and its own
  joints begin by the elbow; the fit puts the shoulder joint where that spine joint is seen, in the arm's own plane
  (onepiece.shoulder_joints).
- a four-legged body's front legs go under ``arms``.
A bone is named by its name in the file; in a hand record one of several at one place names the joint.
"""
import json
from pathlib import Path

VERSION = 1
SIDES = ('left', 'right')
DIALS = ('limbs', 'trunk', 'head')


def read(load_dir):
    """The record beside a load, or None."""
    path = Path(load_dir) / 'record.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def write(path, record):
    """Write ``record`` and return it as the file says it: what the build goes on from."""
    path = Path(path)
    path.write_text(json.dumps(record, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                    encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


def as_joints(record):
    """A record whose body is named by its joints (the hand form): itself when it already is, else a full record's
    body turned into that form, so one joint of it can be renamed. Each limb keeps its chain as the run read it
    (``bones``) until a joint of it is renamed. ValueError for a body its plan draws (a ball, a legless body): it
    names no limb joints to change."""
    if 'body' not in record:
        return dict(record)
    body = record['body']
    said = body['marks']
    if said.get('legs') is None or body.get('plan') not in ('biped', 'quadruped'):
        raise ValueError("this body's limbs are drawn by its plan: it names no joints to change")
    out = {key: value for key, value in record.items() if key != 'body'}
    out.update(plan=body['plan'], hips=body['spine'][0], chest=body['chest'], neck=said['neck'], skull=body['skull'],
               legs={}, arms={})
    for side, leg in said['legs'].items():
        chain = leg['chain']
        out['legs'][side] = {'hip': chain[0], 'ankle': leg['ankle'] or chain[-1], 'bones': list(chain)}
        if leg['toe']:
            out['legs'][side]['toe'] = leg['toe']
    for side, arm in (said.get('arms') or {}).items():
        chain = body['arms'][side]
        mine = {'shoulder': arm['shoulder'], 'hand': arm['hand'] or chain[-1], 'bones': list(chain)}
        if chain[0] != arm['shoulder']:
            mine['clavicle'] = chain[0]
        out['arms'][side] = mine
    if not out['arms']:
        del out['arms']
    return out


def marks(rig, shape, chains, shared):
    """The joints correspond.model_chains read for ``shape`` (its ``chains`` and the legs' ``shared`` start), by name."""
    from . import correspond

    def name(b):
        return None if b is None else rig.names[b]
    spine = chains['spine']
    out = {'neck': name(spine.bones[spine.anchors[2]]), 'arms': None, 'legs': None}
    if shape.arms.get('left'):                      # else the plan draws them (plans.arm_chains)
        out['arms'] = {}
        for side in SIDES:
            c = chains[f'{side} arm']
            out['arms'][side] = {'shoulder': name(c.bones[c.anchors[1]]),
                                 'hand': None if c.anchors[2] is None else name(c.bones[c.anchors[2]])}
    if shape.plan in ('biped', 'quadruped'):        # else the plan draws them (plans.leg_chains)
        out['legs'] = {}
        for side in SIDES:
            c = chains[f'{side} leg']
            out['legs'][side] = {
                'chain': [name(b) for b in correspond._forward_toe(rig, shape, shape.legs[side][len(shared):])],
                'ankle': None if c.anchors[1] is None else name(c.bones[c.anchors[1]]),
                'toe': None if c.anchors[2] is None else name(c.bones[c.anchors[2]])}
    return out


def of_reading(rig, shape, chains, shared, up, front, named, digits, equipment, drop, fighter, character):
    """The full record of what a run read: ``shape`` (body.read) and the joints of its ``chains``
    (correspond.model_chains), the chains' names (labels.chains), the digits (forks), what each prop is
    (labels.equipment) and the meshes dropped."""
    def name(b):
        return rig.names[b]
    return {
        'version': VERSION, 'source': 'reading',
        'up': [float(x) for x in up], 'front': [float(x) for x in front],
        'character': character, 'fighter': fighter, 'drop': list(drop),
        'body': {'plan': shape.plan, 'root': name(shape.root), 'spine': [name(b) for b in shape.spine],
                 'chest': name(shape.chest), 'skull': name(shape.skull),
                 'legs': {side: [name(b) for b in chain] for side, chain in shape.legs.items()},
                 'arms': {side: [name(b) for b in chain] for side, chain in shape.arms.items()},
                 'forks': {name(f): None if c is None else name(c) for f, c in shape.forks.items()},
                 'marks': marks(rig, shape, chains, shared)},
        'parts': {bone: dict(said) for bone, said in named.items()},
        'digits': sorted(digits), 'equipment': dict(equipment),
        'size': 1.0, 'head': 1.0, 'limbs': 1.0, 'trunk': 1.0}
