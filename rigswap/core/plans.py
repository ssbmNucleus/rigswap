"""Correspondence for the plans named by the common body reader."""
import numpy as np

from . import body as B

LIFTED = 0.05           # the correspondence's floor contact band for hanging surfaces


def leg_chains(rig, body):
    """The model's leg chains (correspond.model_chains) for a body with no thigh or shin of its own: each from the
    fighter's hip in the body down to its ankle. A ball's from a hip level with the spine's first joint, straight above
    the foot, down through the foot's own joints (its bones go to the fighter's foot); a legless body's from the hips,
    where the dress hangs from the body, half way out to the side of the body there, straight down to the floor, its
    bones the hips' own (they carry nothing of the legs). Returns {'left leg': Chain, 'right leg': Chain}."""
    from .correspond import Chain, _ankle, _apart, _chain, _forward_toe, _last_joint
    root = body.spine[0]
    hips = rig.heads[root]
    up = body.up
    out = {}
    for side in ('left', 'right'):
        if body.plan == 'ball':
            leg, at = _apart(rig, _forward_toe(rig, body, list(body.legs[side])))
            foot = _chain(rig, leg, carriers=_last_joint(body, at[leg[-1]]))
            hip = foot.joints[0] + ((hips - foot.joints[0]) @ up) * up
            chain = Chain([root] + leg, np.vstack([hip, foot.joints]), foot.end)
            ankle, toe = _ankle(rig, body, chain.bones, chain.end)
            chain.anchors = [0, ankle, toe]
        else:
            z = rig.points @ up
            band = np.abs(z - hips @ up) <= 0.05 * np.ptp(z)
            across = (rig.points[band] - hips) @ body.left if band.any() else np.zeros(1)
            sign = 1 if side == 'left' else -1
            half = across.max() if sign > 0 else -across.min()
            hip = hips + sign * 0.5 * max(half, 0.0) * body.left
            ankle = hip - ((hip @ up) - z.min()) * up * 0.95
            toe = ankle + body.front * 0.1 * ((hip @ up) - z.min())
            chain = Chain([root, root], np.vstack([hip, ankle]), toe, [0, 1, None])
        out[f'{side} leg'] = chain
    return out


def arm_chains(rig, body):
    """The model's arm chains (correspond.model_chains) for a body with no arms (a Bob-omb's): each from the fighter's
    shoulder in the body, level with its chest joint and half way out to the side of the body there, out to the side
    of the body; its bones the chest's own (they carry nothing of the arms). Returns {'left arm': Chain, ...}."""
    from .correspond import Chain
    chest = rig.heads[body.chest]
    z = rig.points @ body.up
    band = np.abs(z - chest @ body.up) <= 0.05 * np.ptp(z)
    across = (rig.points[band] - chest) @ body.left if band.any() else np.zeros(1)
    out = {}
    for side, sign in (('left', 1), ('right', -1)):
        half = max(across.max() if sign > 0 else -across.min(), 0.0)
        shoulder = chest + sign * 0.5 * half * body.left
        hand = chest + sign * half * body.left
        out[f'{side} arm'] = Chain([body.chest, body.chest], np.vstack([shoulder, hand]),
                                   hand + sign * 0.1 * half * body.left, [0, 0, 1])
    return out


def region(rig, body, attach, carries):
    """The region (correspond._regions) a chain hanging from ``attach`` and carrying ``carries`` hangs from, where the
    plan says it (else None: the region of the fighter bone ``attach`` went to). A legless body's trunk is one piece,
    hips and chest: the dress off it (a chain coming down to the floor) hangs from the hips, its other chains (a
    collar, a bow at the chest) from the chest. A ball's body is its head too: what hangs from the face's bone (eyes,
    a mask) hangs from the head, not swung as a piece worn on the body."""
    if body.plan == 'legless' and attach == body.root:
        z = rig.points @ body.up
        own = rig.weights[:, carries].sum(axis=1) > 0.5
        return 'lower' if own.any() and z[own].min() <= z.min() + LIFTED * np.ptp(z) else 'upper'
    if body.plan == 'ball' and attach == body.skull:
        return 'head'
    return None


def fighter_chains(frig, by_role):
    """A ball fighter's chains (correspond.fighter_chains), read from its part table (``by_role``: role -> bones),
    which names a body, two stubs and two feet, and from its skeleton, which is a biped's: its game animates leg
    joints above the feet (a thigh and a shin, inside the body, carrying nothing) and a joint above each stub (from the
    body's middle out to it, which swings the stub round the body).
    - the spine: from the body's first bone down to the one carrying its mesh, which is its chest and its head (the
      body is the head); the model's torso and head go onto it whole, keeping their own size (correspond.match: the
      spine's last bone is an end);
    - each arm: the stub (its bones), with no hand: the model's arm along it, the stub's second bone at the elbow;
    - each leg: the leg joints from where the leg leaves the trunk down to the foot, and its toe.
    Along a limb, of the bones at one joint the deepest stands for it: the game turns that one (an adjust joint above
    it sits still)."""
    from .correspond import _chain
    depth = {b: len(B._ancestors(frig.parents, b)) for b in range(len(frig.names))}
    w = frig.carried()
    body = by_role['body']
    first = min(body, key=lambda b: (depth[b], b))
    heart = max((b for b in body if B._descends(frig.parents, b, first)), key=lambda b: w[b])
    spine = B._path(frig.parents, first, heart)
    chains = {'spine': _chain(frig, spine, anchors=[0, len(spine) - 1, len(spine) - 1], top=True)}
    feet = {side: sorted(by_role[f'{side}_foot'], key=lambda b: (depth[b], b)) for side in ('left', 'right')}
    hips = B._lca(frig.parents, feet['left'][0], feet['right'][0])
    for side in ('left', 'right'):
        stub = sorted(by_role[f'{side}_stub'], key=lambda b: (depth[b], b))
        arm = _joints(frig, B._path(frig.parents, stub[0], stub[-1]))
        chains[f'{side} arm'] = _chain(frig, arm, anchors=[0, 0, None])
        leg = _joints(frig, B._path(frig.parents, hips, feet[side][0])[1:])
        toe = [b for b in feet[side] if frig.parents[b] == leg[-1]]
        chains[f'{side} leg'] = _chain(frig, leg + toe[:1], anchors=[0, len(leg) - 1, len(leg) if toe else None])
    return chains


def _joints(rig, bones):
    """``bones`` (down a limb) with one bone per joint: of the bones whose heads sit at one place, the deepest."""
    from .correspond import SAME_PLACE
    near = SAME_PLACE * np.ptp(rig.points, axis=0).max()
    out = []
    for b in bones:
        if out and np.linalg.norm(rig.heads[b] - rig.heads[out[-1]]) <= near:
            out[-1] = b
        else:
            out.append(b)
    return out


def reach_hips(body, fch, mch):
    """A ball fighter's legs start inside its body, at its middle: where the model's hips are lower, each of its leg
    chains (correspond.model_chains) reaches up from its hip, straight above it, to the fighter's proportional hip height,
    through the trunk (its first joint, which the spine has). With no separate trunk landmarks, the model's body
    height sets this coordinate conversion; its knees fall where their share of its legs puts them. Changes
    ``mch`` in place."""
    from .correspond import Chain
    for side in ('left', 'right'):
        f, m = fch[f'{side} leg'], mch[f'{side} leg']
        height = mch['spine'].end[2] / fch['spine'].end[2]
        rise = (f.joints[0] * height - m.joints[0]) @ body.up
        if rise <= 0:
            continue
        mch[f'{side} leg'] = Chain([body.spine[0]] + list(m.bones), np.vstack([m.joints[0] + rise * body.up, m.joints]),
                                   m.end, [0] + [a + 1 if a is not None else None for a in m.anchors[1:]])


def ball_region(rig, mch, attach, onto):
    """The region (correspond._regions) a model chain hanging from ``attach`` hangs from on a ball fighter, whose body
    is its head too (its spine's last bone): the model's own says. From the model's head joint up, the head's;
    else that of the fighter bone it goes to (``onto``), the upper body's for the body."""
    spine = mch['spine']
    if B._descends(rig.parents, attach, spine.bones[spine.anchors[2]]):
        return 'head'
    return 'upper' if onto == 'head' else onto
