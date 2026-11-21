"""A Melee fighter's weapon slots (Target.weapons): the weapons it fights with, each measured on its vanilla costume
(``weapon_slots``) and written beside its .glb (``write_weapons``), which ``target`` reads back (``read_weapons``).

The weapons a fighter fights with are kept vanilla parts (facts/parts.py, PARTS; what each one is, WEAPONS): a held
item of the download that takes one in place.held replaces that weapon, which is then not kept, nor are its other
forms, the sword put away and its scabbard, the shield put away (facts/parts.py: kept_places, hidden_forms); every
other kept part stays. A slot says whether the game swings it ('swung'): a prop taking a swung one is fitted to it, one
taking a shield's is held as the model holds it (correspond._match_props).
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ... import Failed
from ....vocabulary import roles_by_number
from ..dae import PLACE
from ..facts.parts import PARTS, WEAPONS


# The target's frame from the .glb's model space (the .dae's: y up, facing +z): Blender's glTF importer turns it z up,
# facing -y, and the build reads a target as it imports it (rigswap.vocabulary).
TO_TARGET = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
# How far a joint of the vanilla .dae may be from the same bone's head in the target's bone graph, which keeps six
# decimals (checks.JOINT_TOLERANCE is the .dae's own precision).
GRAPH_TOLERANCE = 1e-4 + 1e-6
WEAPON_METHOD = ("the vertices of its places in the vanilla costume (every DObj, HSDRawViewer --names place), in the "
                 "target's frame: its long axis the longest side of their own box (place_held.item_box), its grip end "
                 "the end of that axis nearer the head of the hand bone its places hang under, its grip the middle of "
                 "that end, its tip direction from the grip to the other end; its hand the part the first bone with a "
                 "hand's role above the joints its vertices are bound to; its bone the joint that carries the most of "
                 "its vertices' weight (on a tie, the one nearer the root); its flat the box's second side, across the "
                 "long axis along the weapon's width (its larger part positive)")


def weapon_slots(f, bones, roles, model):
    """The weapon slots of fighter ``f`` (Target.weapons): one for each part it fights with (WEAPONS), measured on
    ``model``, its vanilla costume with every DObj named by place (dae.read_dae), in the frame of its bone graph
    ``bones`` (bones.json's list: n, name, parent, head), whose ``roles`` ({bone number, or number as text: role},
    roles_by_number) say which bones are hands. How: WEAPON_METHOD. Failed when a part's places draw nothing or hang
    under no hand, or when the .dae's joints are not where the bone graph has them (the two frames would not be
    one)."""
    from .bones import joint_number
    from .geometry import item_box
    roles = roles_by_number(roles)
    by_name, by_n = {b['name']: b for b in bones}, {b['n']: b for b in bones}

    def at(name):
        """A joint of the .dae, in the target's frame; checked against the bone graph."""
        pos = TO_TARGET @ model.joints[name][:3, 3]
        if np.linalg.norm(pos - by_name[name]['head']) > GRAPH_TOLERANCE:
            raise Failed(f"{f.name}: the vanilla costume's {name} is at {np.round(pos, 6).tolist()} in the target's "
                         f"frame and its bone graph has it at {by_name[name]['head']}")
        return pos

    def hand_above(name):
        b = by_name[name]
        while b is not None and not str(roles.get(b['n'], '')).endswith('_hand'):
            b = by_n.get(b['parent'])
        return b

    out = []
    for part, places, _ in PARTS.get(f.name, ()):
        if part not in WEAPONS.get(f.name, {}):
            continue
        faces = [i for i, o in enumerate(model.face_objects) if (m := PLACE.fullmatch(o or '')) and int(m[1]) in places]
        if not faces:
            raise Failed(f'{f.name}: the places of its {part} ({", ".join(map(str, places))}) draw nothing in the '
                         'vanilla costume')
        verts = sorted({v for i in faces for v in model.faces[i]})
        carried = {}
        for v in verts:
            for j, w in model.weights[v].items():
                if w > 0:
                    carried[j] = carried.get(j, 0.0) + float(w)
        joints = sorted(carried, key=joint_number)
        bone = max(joints, key=lambda j: (carried[j], -joint_number(j)))
        above = {j: hand_above(j) for j in joints}
        named = {h['name'] if h else 'no hand' for h in above.values()}
        if len(named) != 1 or None in above.values():
            raise Failed(f'{f.name}: its {part} is bound to {", ".join(joints)}, which hang under '
                         f'{", ".join(sorted(named))}: not one hand')
        hand = above[joints[0]]
        for j in joints:
            at(j)
        head = at(hand['name'])
        pts = np.asarray(model.points, float)[verts] @ TO_TARGET.T
        centre, axes, ext = item_box(pts)
        length = float(ext[0])
        ends = {s: centre + s * axes[0] * length / 2 for s in (1, -1)}
        grip = min(ends, key=lambda s: float(np.linalg.norm(ends[s] - head)))
        flat = axes[1] * np.sign(axes[1][np.argmax(np.abs(axes[1]))])
        out.append({'part': part, 'kind': WEAPONS[f.name][part]['kind'], 'swung': WEAPONS[f.name][part]['swung'],
                    'hand': roles[hand['n']],
                    'grip': np.round(ends[grip], 6).tolist(), 'tip_direction': np.round(-grip * axes[0], 6).tolist(),
                    'flat': np.round(flat, 6).tolist(), 'width': round(float(ext[1]), 6),
                    'length': round(length, 6), 'hand_bone': hand['name'], 'bone': bone,
                    'bone_weights': {j: round(carried[j], 3) for j in joints},
                    'hand_to_grip': round(float(np.linalg.norm(ends[grip] - head)), 6), 'joints': joints,
                    'places': list(places), 'triangles': len(faces), 'method': WEAPON_METHOD})
    return out


def weapons_path(glb):
    """Where a target's weapon slots are: <Name>.weapons.json beside its .glb."""
    return Path(glb).with_suffix('.weapons.json')


def read_weapons(glb):
    """The weapon slots of the target in ``glb``, as write_weapons wrote them; [] when it wrote none."""
    path = weapons_path(glb)
    return json.loads(path.read_text(encoding='utf-8'))['weapons'] if path.is_file() else []


def write_weapons(f, bones, roles, glb, files_dir=None):
    """Measure fighter ``f``'s weapon slots (weapon_slots) and write them beside its .glb (weapons_path), with what they
    were measured on. ``bones`` and ``roles``: its bone graph and roles, as describe_from_tables made them. The vanilla
    costume with every DObj is exported once, to <the .glb's folder>/vanilla/<Name>/every/<Name>.dae, from copies of
    the game's files in ``files_dir`` (else RIGSWAP_MELEE_FILES). Returns the record written; None, writing nothing,
    when the fighter fights with a weapon and the game files are not there."""
    from .. import FILES_VAR
    from ..dae import read_dae
    from ..export.write import export_dae, vanilla_copies
    glb = Path(glb)
    record = {'fighter': f.name, 'frame': "the target's: +z up, facing -y, as Blender imports its .glb", 'weapons': []}
    if WEAPONS.get(f.name):
        folder = files_dir or os.environ.get(FILES_VAR)
        costume = Path(folder) / f'Pl{f.code}Nr.dat' if folder else None
        if costume is None or not costume.is_file():
            return None
        dae = glb.parent / 'vanilla' / f.name / 'every' / f'{f.name}.dae'
        if not dae.is_file():
            with tempfile.TemporaryDirectory() as tmp:
                export_dae(vanilla_copies(tmp, costume)[0], f.symbol, dae, by_place=True)
        record['weapons'] = weapon_slots(f, bones, roles, read_dae(dae))
        record['from'] = f'{costume.name}: every DObj, {dae.relative_to(glb.parent).as_posix()}'
    weapons_path(glb).write_text(json.dumps(record, indent=1), encoding='utf-8')
    return record
