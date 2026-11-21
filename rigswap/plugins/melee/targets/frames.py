"""The fighter's game frames: its own animations at the frames the export's check poses a costume in (export/gate.py)
and the comparison draws (export/comparison.py). HSDRawViewer's ``--dump-pose`` evaluates the fighter's Pl<code>AJ.dat
(Fighter.motions: another fighter's, for a fighter that plays that one's) on its costume with the game's scale
compensation and writes each joint's world matrix (row-vector, as comparison.pose_matrices reads it). ``write`` dumps
them once, from copies of the vanilla files, into ``<targets>/frames/<Name>/<action>-<frame>.json``
(prepare calls it beside the other target files); ``read`` gives them to the export.

POSES owns both the gate's frames and the in-game harness's shot arguments. Each row is (action, frames, shot name),
with no shot name for the gate's additional coverage. The original 29 poses remain alongside every harness shot.
The harness's idle sample is a settled Wait, whose live animation phase varies; Wait1-1 remains the gate's idle."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

POSES = (
    ('Wait1', (1,), 'idle'),
    ('Run', (1, 6), 'run'),
    ('Squat', (4,), 'crouch'),
    ('Attack11', (3,), 'jab'),
    ('AttackHi3', (6,), 'utilt'),
    ('AttackS4S', (12,), 'fsmash'),
    ('AttackLw4', (8,), 'dsmash'),
    ('AttackAirN', (4,), 'nair'),
    ('AttackAirF', (8,), 'fair'),
    ('AttackAirLw', (10,), 'dair'),
    ('Appeal', (20,), 'taunt'),
    ('Landing', (2, 4), 'landing'),
    ('LandingAirN', (2, 6), 'landingair'),
    ('DamageHi1', (2, 4), 'damagehi'),
    ('DamageFlyHi', (4, 10), 'damagefly'),
    ('CliffWait', (10, 40), 'cliffwait'),
    ('WalkMiddle', (10,), None),
    ('Dash', (4,), None),
    ('JumpF', (8,), None),
    ('JumpAerialF', (5,), None),
    ('SquatWait', (1,), None),
    ('Guard', (5,), None),
    ('AttackDash', (6,), None),
    ('AttackS3S', (5,), None),
    ('AttackLw3', (8,), None),
    ('AttackHi4', (10,), None),
    ('AttackAirF', (15,), None),
    ('AttackAirB', (8,), None),
    ('AttackAirHi', (6,), None),
    ('Catch', (8,), None),
    ('ThrowF', (8,), None),
    ('CliffWait', (1,), None),
    ('Appeal', (30,), None),
)
FRAMES = tuple((action, frame) for action, frames, _ in POSES for frame in frames)
INGAME_SHOTS = tuple(f'{shot}:' + ','.join(map(str, frames)) for _, frames, shot in POSES if shot)
IDLE = 'Wait1-1'                # the frame the soles are measured in beside the rest pose
# The game's own names for an action that some fighters name otherwise: idle is Wait, the forward smash AttackS4
FAMILY = {'Wait1': 'Wait', 'AttackS4S': 'AttackS4'}


def name(action, frame):
    return f'{action}-{frame}'


def symbol(listing, action):
    """The animation of ``action`` in a fighter's listing (HSDRawViewer ``--anim list``), by its exact name between
    underscores (RunBrake is not Run): the action's own name, else with 1 after it (an aerial jump JumpAerialF1, a
    crouch SquatWait1), else with R (the taunt to the right, AppealR), else its family's name (FAMILY). None when the
    fighter has no such animation (a fighter whose shield is no Guard)."""
    lines = listing.splitlines()
    for n in (action, action + '1', action + 'R', FAMILY.get(action, action)):
        found = next((line.split()[0] for line in lines if f'_{n}_' in line), None)
        if found:
            return found
    return None


def folder(fighter_name, targets_dir=None):
    from .. import _folder, TARGETS_VAR
    return _folder(targets_dir, TARGETS_VAR, 'Melee targets') / 'frames' / fighter_name


def write(f, files, targets):
    """Dump the fighter's FRAMES not yet in ``<targets>/frames/<Name>/``. Returns {'made': [...], 'there': n,
    'none': [the actions the fighter has no animation for]}."""
    from ..export.write import hsdraw, vanilla_copies
    out = Path(targets) / 'frames' / f.name
    todo = [(a, n) for a, n in FRAMES if not (out / f'{name(a, n)}.json').is_file()]
    record = {'made': [], 'there': len(FRAMES) - len(todo), 'none': []}
    if not todo:
        return record
    out.mkdir(parents=True, exist_ok=True)
    files = Path(files)
    with tempfile.TemporaryDirectory() as tmp:
        costume, aj = vanilla_copies(tmp, files / f'Pl{f.code}Nr.dat', files / f'Pl{f.motions or f.code}AJ.dat')
        listing = hsdraw('--anim', 'list', aj)
        for action, frame in todo:
            found = symbol(listing, action)
            if found is None:
                record['none'].append(action)
                continue
            hsdraw('--dump-pose', costume, aj, found, out / f'{name(action, frame)}.json', frame)
            record['made'].append(name(action, frame))
    return record


def read(fighter_name, targets_dir=None):
    """{'<action>-<frame>': the frame as dumped ({'symbol', 'frame', 'bones'})}, in the order of the files' names; {}
    when none were dumped for the fighter (prepare writes them)."""
    root = folder(fighter_name, targets_dir)
    return {p.stem: json.loads(p.read_text(encoding='utf-8'))
            for p in sorted(root / f'{name(a, n)}.json' for a, n in FRAMES) if p.is_file()}
