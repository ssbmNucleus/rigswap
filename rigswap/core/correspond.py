"""Which part of the model each fighter bone plays, and where each fighter joint sits inside the model.

Both sides are read as chains: the spine (hips to head), each arm (shoulder to hand) and each leg (hip to toe), each a
polyline through its joints to a far end (the farthest vertex its last bone carries). The fighter's come from the
game's own part table; the model's from its skeleton (body.py). A fighter joint goes to the model joint at the same
fraction along the chain, or, where the model has no joint near it, to the point at that fraction. The spine is
anchored at the chest (where the arms branch) and the skull. The model between two fighter joints' places rides the
fighter bone between them: a model bone belongs to the stretch its own joint lies in, and one a fighter joint falls
inside is cut there (repose.split). A bone off the chains rides the fighter bone carrying its nearest ancestor on them
at the point it hangs from. So neighbours stay neighbours: a model bone and the one it hangs from end up on one fighter
bone or on two next to each other at that joint, which match() checks (Match.apart).
"""
import json
from dataclasses import dataclass, field, replace

import numpy as np

from . import body as body_reading, plans

SPINE = ('hips', 'spine', 'chest', 'neck', 'head')
ARM = ('shoulder', 'upper_arm', 'forearm', 'hand')
LEG = ('thigh', 'shin', 'foot')
INTERPOLATE = 0.15      # a joint snaps to a model joint within this share of the chain, else sits between
BESIDE = 1.6            # a left-over chain this alike in start and hang (of 2) swings with that fighter chain
BELOW = {'head': ('upper', 'lower'), 'upper': ('lower',)}  # the regions down the body from one a chain hangs from
SKINNED = 0.01          # a model whose skin blends bones on this share of its vertices or more is skinned, not built of
                        # rigid pieces throughout (a segmented rip: 0 on the core loop's; skinned ones 0.03 and more)
RING = 3                # a fighter whose game swings this many chains or more from one lower bone wears a skirt
FITS = {'hair': ('hair', 'other'), 'cloth': ('cape', 'other'), 'tail': ('tail',), 'ear': ('ear',)}
                        # the fighter's chains (its part table's kinds) a model chain of each kind may swing on: its
                        # 'other' are a hat's tail, a tie, coat tails and skirt strands
HANG = ('hair', 'cloth')    # the kinds that hang (labels.HANG)
OWN = ('tail',)             # the kinds that swing on joints of their own where no chain of the fighter's carries them
RIGID = 0.999           # a vertex rides a held prop alone when the prop's bones carry this much of it (held_props)
ALONE = 0.9             # a held prop's own mesh rides it alone, this share of it or more: the rest a strap or grip
                        # its author tied to the arm (u-link's shield, 451 of 463)
HELD = 2.0             # a held prop is this many times the size of the rest of the hand or more: an object reaching
                        # well out of the fist, not a piece of a hand cut in rigid parts
WELD = 1e-3            # two pieces are one object where they share a point, this share of the hand's size apart or less


@dataclass
class Chain:
    bones: list                     # bones whose heads are the chain's joints, in order
    joints: np.ndarray              # (n, 3)
    end: np.ndarray                 # the chain's far end
    anchors: list = field(default_factory=list)   # indices into bones fixed to the other side's same indices
    level: bool = False             # a model's spine on four legs: carried level, its levels may run along it
    kind: str = None                # what it is: the fighter's part table's kind, or a model chain's the labels ask
                                    # named (labels.chains); None when nobody said
    held_out: bool = False          # a model's hair or cloth the bind pose holds out where it would not rest

    def fractions(self):
        pts = np.vstack([self.joints, self.end])
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = steps.sum() or 1.0
        return np.concatenate([[0.0], np.cumsum(steps) / total]), pts


@dataclass
class Match:
    owner: dict                     # model bone -> fighter bone
    embedded: dict                  # fighter bone -> (head, tail) inside the model
    rest: dict                      # fighter bone -> (head, tail) at the fighter's rest
    notes: list = field(default_factory=list)
    ends: set = field(default_factory=set)      # fighter bones that keep the model's own size: head, hands, feet
    placed: dict = field(default_factory=dict)  # fighter bone -> 3x4 given outright: a prop in a weapon slot, or
                                                # one from outside the body on a hand's item joint (_match_props)
    replaced: list = field(default_factory=list)    # the weapon slots a prop took (weapons_replaced.json)
    whole: dict = field(default_factory=dict)   # fighter bone -> the model vertices riding it alone: a prop a weapon
                                                # slot took, with the strap tying it to the arm (repose.alone)
    loose: list = field(default_factory=list)   # (bone it hangs from, Chain, bones it carries): the model's hair and
                                                # cloth with no partner on the fighter, riding the bone they hang from
    splits: list = field(default_factory=list)  # (model bone, its Chain, [(share along it, fighter bone from there)]):
                                                # fighter joints inside a model bone, cutting it (_cuts, repose.split)
    paired: dict = field(default_factory=dict)  # model bone -> the fighter bone whose joint is paired with its own
                                                # (_place, before the floor moves the ankle): a rigid piece rides it
                                                # whole where that keeps it on the next piece (_whole)
    fitted: set = field(default_factory=set)    # the fighter's skirt strands a model strand is laid on one to one
                                                # (_fit): cloth squashed onto them, no part of the model's own size
    end_parents: dict = field(default_factory=dict)  # end bone -> nearest matched ancestor, also an end
    feet: set = field(default_factory=set)      # ankle bones: their placed soles must be checked against the floor
    ankles: dict = field(default_factory=dict)  # ankle placement evidence, in the scaled model's units
    apart: list = field(default_factory=list)   # (model bone, the one it hangs from, their fighter bones there): on
                                                # fighter bones not next to each other, so they part (_apart_at)
    meets: list = field(default_factory=list)   # where each pair of ``apart`` meets in the model, in its order
    trunk: list = field(default_factory=list)   # the fighter's spine bones below its head: they thicken as one
                                                # (repose.transforms)
    widen: dict = field(default_factory=dict)   # chest bone -> (times, the fighter's midline x): it widens across
                                                # to shoulders the model's arms have no clavicle to reach (_widen)
    head: int = None                            # rec: the fighter's head bone (repose.transforms sizes it by a dial)

    def carrier(self, mb, p):
        """The fighter bone carrying model bone ``mb`` at the point ``p`` (_carrier)."""
        return _carrier(self.owner, {s[0]: s for s in self.splits}, mb, p)


def levels(chains, fighter=False):
    """The trunk's landmarks as heights up it: the hips, the shoulders (the arm roots, or the spine joint they hang
    from when that is higher: arms set low on the sides), the head joint and the top (the head's own height above its
    joint: a leaning trunk's head still stands up off it). A ``fighter``'s at its levels (_level), a model's at its own
    (_model_level)."""
    spine, arms = chains['spine'], [chains['left arm'], chains['right arm']]

    def level(p):
        return float(_level(spine, arms, p)) if fighter else float(_model_level(spine, arms, p))
    shoulders = max(np.mean([level(c.joints[0]) for c in arms]), level(spine.joints[spine.anchors[1]]))
    head = level(spine.joints[spine.anchors[2]])
    return [float(spine.joints[0][2]), shoulders, head, float(spine.end[2] + (head - spine.joints[spine.anchors[2]][2]))]


def _level(spine, arms, p):
    """How high ``p`` stands on a fighter's trunk: its height, or, on a trunk its game carries leaning (its
    shoulders, the arm roots, farther ahead of its hips than above them: a body carried level, as a dinosaur's), its
    height above the hips along the trunk, from the hips towards the shoulders. By height alone, the joints from its
    hips to its chest would stand at one height, and the model's trunk between them would stretch without bound; the
    model's is read by height, and the re-pose leans it as the fighter's."""
    hips = spine.joints[0]
    d = np.mean([c.joints[0] for c in arms], axis=0) - hips
    if -d[1] <= d[2]:                       # it stands facing -y
        return p[2]
    return hips[2] + (p - hips) @ (np.array([0., d[1], d[2]]) / np.hypot(d[1], d[2]))


def _model_level(spine, arms, p):
    """How high ``p`` stands on a model's trunk: its height, or, on a trunk carried level (its shoulders farther ahead
    of its hips than above them: a four-legged body's), how far along the trunk it lies: along the spine from the hips
    through the chest (``spine``'s anchor 1) to the head joint (anchor 2), its point nearest there; past the head
    joint, the head stands up off it. Matched level to level with an upright fighter's trunk, the model goes onto it
    standing up: a quadruped on its hind legs, its forelegs the fighter's arms. (A fighter's trunk carried level is
    measured along its hips to shoulders, _level: its head reaches on ahead.)"""
    line = _trunk_line(spine, arms)
    if line is None:
        return p[2]
    return _along(line, p)


def _trunk_line(spine, arms):
    """A four-legged model's trunk carried level (``spine.level``, model_chains): its hips, chest and head joints;
    None when it stands upright (_model_level). A biped's trunk carried leaning (a bird's, a mantis') keeps its levels
    by height: the fighter's legs and arms take it as it stands."""
    hips = spine.joints[0]
    d = np.mean([c.joints[0] for c in arms], axis=0) - hips
    if not spine.level or -d[1] <= d[2]:    # it stands facing -y
        return None
    return np.vstack([hips, spine.joints[spine.anchors[1]], spine.joints[spine.anchors[2]]])


def _along(line, p):
    """``p``'s level along the polyline ``line``: its nearest point on it, its distance along it from the first
    point's height; past the last point, its height above that point added (the head stands up off it)."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    best = None
    for k in range(len(line) - 1):
        a, d = line[k], line[k + 1] - line[k]
        t = float(np.clip((p - a) @ d / ((d @ d) or 1.0), 0, 1))
        dist = np.linalg.norm(a + t * d - p)
        if best is None or dist < best[0]:
            best = (dist, k, t)
    _, k, t = best
    level = line[0][2] + steps[:k].sum() + t * steps[k]
    if k == len(line) - 2 and t >= 1:
        level += max(p[2] - line[-1][2], 0.0)
    return level


def _at_level(line, level):
    """The point of the polyline ``line`` at ``level`` (_along's), standing up off its last point past it and down
    off its first below it."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    x = level - line[0][2]
    if x < 0:
        return line[0] + x * np.array([0., 0., 1.])
    for k, n in enumerate(steps):
        if x <= n:
            return line[k] + (x / (n or 1.0)) * (line[k + 1] - line[k])
        x -= n
    return line[-1] + x * np.array([0., 0., 1.])


def fighter_plan(description):
    """The fighter's body plan: 'ball' when its part table names a body (with stubs and feet off it), else 'biped'."""
    return 'ball' if 'body' in description['bones']['roles'].values() else 'biped'


def fighter_chains(frig, description):
    """The fighter's chains from its part table (description.json bones.roles: bone n = JOBJ_(n-1))."""
    roles = {f'JOBJ_{int(n) - 1}': r for n, r in description['bones']['roles'].items()}
    depth = _depths(frig.parents)
    by_role = {}
    for b, name in enumerate(frig.names):
        r = roles.get(name)
        if isinstance(r, str):
            by_role.setdefault(r, []).append(b)
    if fighter_plan(description) == 'ball':     # a body, two stubs and two feet (plans.py)
        return plans.fighter_chains(frig, by_role)
    primary = {r: min(bs, key=lambda b: (depth[b], b)) for r, bs in by_role.items()}
    spine = [primary[p] for p in SPINE if p in primary]
    shoulder = primary.get('left_shoulder', primary.get('left_upper_arm'))
    chest = next((x for x in _ancestors(frig.parents, shoulder)[1:] if x in spine), spine[-2])
    # anchors, in the order both sides list them: spine (hips, chest, head joint), arm (start, shoulder joint, wrist),
    # leg (hip joint, ankle, toe); None where the side has no such joint
    chains = {'spine': _chain(frig, spine, anchors=[0, spine.index(chest), len(spine) - 1], top=True)}
    for side in ('left', 'right'):
        arm = [primary[f'{side}_{p}'] for p in ARM if f'{side}_{p}' in primary]
        leg = [primary[f'{side}_{p}'] for p in LEG if f'{side}_{p}' in primary]
        toe = [b for b in by_role.get(f'{side}_foot', []) if frig.parents[b] == leg[-1]]
        has_clavicle = f'{side}_shoulder' in primary
        # rec: a fourth anchor, the elbow and the knee, pairs with a hand record's (_place); nothing else reads it
        elbow = arm.index(primary[f'{side}_forearm']) if f'{side}_forearm' in primary else None
        knee = leg.index(primary[f'{side}_shin']) if f'{side}_shin' in primary else None
        chains[f'{side} arm'] = _chain(frig, arm, anchors=[0, 1 if has_clavicle else 0, len(arm) - 1, elbow])
        chains[f'{side} leg'] = _chain(frig, leg + toe[:1], anchors=[0, len(leg) - 1, len(leg) if toe else None, knee])
    return chains


def model_chains(rig, body):
    """The model's chains with the same anchors, read from its geometry: the head joint where the skull's own mesh
    begins; the shoulder joint after a clavicle (a first segment starting near the midline or short beside the arm);
    the hand where the arm's path ends (body reading: where its hand is, forks.py); the ankle where the leg turns
    forward into the foot. A hand has an upper arm
    and a forearm before it, an ankle a thigh and a shin: on a limb with fewer joints (Olimar's arms and legs, one
    bone each) the fighter's fall along it by their own shares, inside its bones (_cuts), rather than all at its root.
    A limb is read by where its joints lie along it, so it counts each place once (_apart); the spine is read by
    heights. A one-piece trunk spans the places where legs and arms leave it,
    regardless of its bone's pivot. Where both pairs hang directly from that
    bone, its hips are the midpoint of the legs' root joints. A ball's body is its head: its top is the highest skin
    carried by its spine's joints, including the coincident bones body reading merged into them. Skin on other
    chains (a hat or antenna) does not set that top."""
    spine = list(body.spine)
    carriers = None
    if body.plan == 'ball':
        mass = body_reading._Mass(rig)
        carriers = [k for b in spine for k in mass.groups[b]]
    marks = getattr(body, 'record', None)     # rec: the anchors a hand record names, in place of the ones read from geometry below
    head = spine.index(marks['head']) if marks else _head_joint(rig, body, spine)
    chains = {'spine': _chain(rig, spine, anchors=[0, spine.index(body.chest), head], top=True, carriers=carriers)}
    if body.plan == 'quadruped':
        # a four-legged body's hips are where its hind legs leave it, on the midline between them (a segmented rip's
        # hips piece hangs from the chest: its own joint is at the chest)
        roots = [next(b for b in body.legs[side] if b not in spine) for side in ('left', 'right')]
        mid = rig.heads[roots].mean(axis=0)
        chains['spine'].joints[0] = mid - ((mid - rig.heads[spine[0]]) @ body.left) * body.left
        chains['spine'].level = True
    # a body with no thigh or shin of its own (a ball's feet, a dress) places the fighter's legs by its plan (plans.py)
    planned = {} if body.plan in ('biped', 'quadruped') else plans.leg_chains(rig, body)
    shared = [] if planned else _shared_prefix(body.legs['left'], body.legs['right'])
    if body.plan != 'quadruped' and body.chest == body.spine[0] and body.legs:
        roots = [body.legs[side][len(shared)] for side in ('left', 'right')]
        chains['spine'].joints[0] = (rig.heads[shared[-1]] if len(shared) > 1
                                     else rig.heads[roots].mean(axis=0))
    # a body with no arms (a Bob-omb) has the fighter's lie in it, carrying nothing (plans.arm_chains)
    armless = {} if body.arms['left'] else plans.arm_chains(rig, body)
    arms = {side: _apart(rig, list(body.arms[side])) for side in ('left', 'right') if not armless}
    # a skeleton is symmetric: a clavicle read on one arm is on both (one arm bent forward can hide it)
    clavicle = max((_shoulder_joint(rig, body, arm) for arm, _ in arms.values()), default=0)
    for side in ('left', 'right'):
        if armless:
            chains[f'{side} arm'] = armless[f'{side} arm']
        else:
            arm, at = arms[side]
            shoulder = min(clavicle, len(arm) - 2) if len(arm) > 2 else 0
            hand = len(arm) - 1 if len(arm) >= 3 else None           # past an upper arm and a forearm
            extra = []
            if marks and marks['arms']:
                named = marks['arms'][side]
                shoulder = _joint_of(arm, at, named['shoulder'])
                hand = None if named['hand'] is None else _joint_of(arm, at, named['hand'])
                extra = [None if named['elbow'] is None else _joint_of(arm, at, named['elbow'])]
            chains[f'{side} arm'] = _chain(rig, arm, anchors=[0, shoulder, hand] + extra, carriers=at[arm[-1]])
        if planned:
            chains[f'{side} leg'] = planned[f'{side} leg']
            continue
        if marks and marks['legs']:
            named = marks['legs'][side]
            leg, at = _apart(rig, named['chain'] if named['chain'] is not None else body.legs[side][len(shared):])
            chain = _chain(rig, leg, carriers=_last_joint(body, at[leg[-1]]))
            chain.anchors = [0] + [None if named[k] is None else _joint_of(leg, at, named[k])
                                   for k in ('ankle', 'toe', 'knee')]
            chains[f'{side} leg'] = chain
            continue
        leg, at = _apart(rig, _forward_toe(rig, body, body.legs[side][len(shared):]))
        chain = _chain(rig, leg, carriers=_last_joint(body, at[leg[-1]]))
        ankle, toe = _ankle(rig, body, leg, chain.end)
        chain.anchors = [0, ankle, toe] if ankle >= 2 else [0, None, None]
        chains[f'{side} leg'] = chain
    return chains, shared


def _joint_of(chain, at, bone):
    """rec: the place along ``chain`` (one bone per joint, ``at`` its bones there: _apart) of the joint ``bone`` is at."""
    return next(i for i, b in enumerate(chain) if bone in at[b])


def _last_joint(body, bones):
    """All the bones of a leg's last joint (body.groups), given the ones on its path. A file can carry a joint's skin on
    a second bone at the same place (every joint of a Colosseum rip): the first alone carries none, and says only the
    tail an importer made up for it, back up the shin. No arm's: the second bone at a hand can be what it holds (Kamek's
    staff, at his attach_R_hand), and a chain ends on its own vertices, not a wand it carries (_chain)."""
    return [k for b in bones for k in body.groups.get(b, [b])]


def _head_joint(rig, body, spine):
    """The spine joint nearest where the skull's own mesh begins (low in it, past stray strands)."""
    on = rig.weights[:, body.skull] > 0.5
    if not on.any():
        return len(spine) - 1
    bottom = np.percentile(rig.points[on] @ body.up, 10)
    above = range(spine.index(body.chest) + 1, len(spine))
    return min(above, key=lambda i: abs(rig.heads[spine[i]] @ body.up - bottom), default=len(spine) - 1)


def _shoulder_joint(rig, body, arm):
    """1 where the arm's first segment is a clavicle, else 0. Only an arm with an upper arm and a forearm still after
    it before the hand can show one: on three joints the first is the shoulder, however near the midline a narrow
    chest puts it beside a hand held out (a carnival skeleton's, a fifth of its reach: read as a clavicle, the
    fighter's elbow bent its forearm in the middle). An arm too short to show it (Samus's cannon) takes its twin's."""
    if len(arm) < 4:
        return 0
    p = rig.heads[arm]
    side = np.abs((p - rig.heads[body.chest]) @ body.left)
    first = np.linalg.norm(p[1] - p[0])
    rest = np.linalg.norm(p[-1] - p[1])
    clavicle = side[0] < 0.25 * side[-1] or first < 0.25 * rest
    return 1 if clavicle else 0


FOOT_TURN = np.cos(np.radians(40))    # a leg runs down within this of the vertical; the foot leaves it
ANKLE_LOW = 0.4                        # an ankle lies below this share of the hip's height; a knee about half


def _ankle(rig, body, leg, end):
    """(ankle index, toe index or None): the first joint low in the leg after which the chain leaves the downward
    vertical: the foot turning off the shin. Low: below ANKLE_LOW of the hip's height off the floor; a knee bent in a
    stance turns the shin off the vertical too, but it sits about halfway up."""
    p = np.vstack([rig.heads[leg], end])
    h = (p - p[-1]) @ body.up
    for j in range(1, len(leg)):
        d = p[j + 1] - p[j]
        n = np.linalg.norm(d)
        if n > 1e-9 and (d @ -body.up) / n < FOOT_TURN and h[j] < ANKLE_LOW * h[0]:
            return j, (j + 1 if j + 1 < len(leg) else None)
    return len(leg) - 1, None


def match(rig, body, frig, description, items=None, slots=(), named=None, kinds=None, digits=()):
    """Match the fighter's chains to the model's. ``items``: the fighter's item joints, {'right_hand': 'JOBJ_42'},
    where a prop the model carries outside its body goes; ``slots``: its weapon slots (<Name>.weapons.json), which a
    prop the model holds, or else one it carries, takes instead (_match_props); ``named``: what each chain off the
    body is ({bone name: {kind, held_out}}, labels.chains); ``kinds``: what each prop is, where the slots are of more
    than one kind ({root bone name: kind}, labels.equipment); ``digits``: the bones the forks named fingers, a thumb,
    toes or claws (forks.py), whose fingers take the fighter's (_match_fingers). The ankles are placed after scaling,
    with the model's soles on the floor."""
    fch = fighter_chains(frig, description)
    mch, pelvis = model_chains(rig, body)
    rolls = fighter_plan(description) == 'ball'
    if rolls:                                       # its legs start inside its body, at its middle (plans.py)
        plans.reach_hips(body, fch, mch)
    owner, embedded, rest, notes, splits, ankles, paired, bare = {}, {}, {}, [], [], {}, {}, {}
    # A ball has no separate trunk landmarks: embed the fighter's joints at proportional body heights.
    cuts = _match_spine(rig, frig, fch, mch, owner, embedded, rest, notes, same=rolls or body.plan == 'ball',
                        groups=body.groups)
    if body.plan == 'ball':
        # a ball's body is one piece: the fighter's chest, where the arms hang, carries it whole, keeping its own size
        # as a head does (in the fighter's hips or head, the stubs part from it as the chest bends or the head turns)
        ball = fch['spine'].bones[fch['spine'].anchors[1]]
        for b in mch['spine'].bones:
            owner[b] = ball
        notes.append(f'ball: {", ".join(rig.names[b] for b in mch["spine"].bones)} -> {frig.names[ball]}, whole')
    else:
        splits.extend(cuts)
    for name, f in fch.items():
        if name == 'spine':
            continue
        m = mch[name]
        placed = _place(f, m)                       # per fighter joint: (model joint index or None, position)
        if name.endswith(' arm') and f.anchors[1] and placed[f.anchors[1]][0] == 0:
            bare[name] = f.joints[f.anchors[1]]     # the fighter's shoulder: its clavicle has no length in the model
        # a chain the plan drew on one bone of the trunk carries nothing of the limb (plans.arm_chains, leg_chains): the
        # bone stays the trunk's, and no joint of the limb pairs with it or cuts it
        own = len(set(m.bones)) == len(m.bones)
        # the model's ankle stays paired with the fighter's wherever the floor then puts that (_place_ankle)
        if own:
            paired.update({k: fb for fb, (j, _) in zip(f.bones, placed) if j is not None
                           for k in body.groups.get(m.bones[j], [m.bones[j]])})
        if name.endswith(' leg'):
            ankles[f.bones[f.anchors[1]]] = _place_ankle(rig, f, m, placed)
        ends = [p for _, p in placed[1:]] + [m.end]
        f_ends = list(f.joints[1:]) + [f.end]
        for i, fb in enumerate(f.bones):
            embedded[fb] = (placed[i][1], ends[i])
            rest[fb] = (f.joints[i], f_ends[i])
        # model bones along the chain: each belongs to the last fighter joint at or before its own joint; one before
        # the first (a clavicle where the fighter has none: a ball's stub) stays with the trunk (_rest_of_tree)
        t_f = [_fraction_of(m, p) for _, p in placed]
        t_m, _ = m.fractions()
        for j, mb in enumerate(m.bones):
            i = max((k for k in range(len(f.bones)) if t_f[k] <= t_m[j] + 1e-9), default=None)
            if i is not None:
                owner.setdefault(mb, f.bones[i])
        snapped = [m.bones[j] if j is not None else None for j, _ in placed]
        notes.append(f'{name}: ' + ', '.join(f'{frig.names[fb]}->{rig.names[s] if s is not None else "between"}'
                                             for fb, s in zip(f.bones, snapped)))
        parts = _parts(rig, m, body.groups, mch['spine'].bones) if own else None
        if parts:
            splits.extend(_cuts(parts, [embedded[fb][0] for fb in f.bones[1:]], f.bones[1:]))
    hips = fch['spine'].bones[0]
    for b in pelvis:
        owner.setdefault(b, hips)
    splits.extend(_at_joint(rig, body, owner, splits))
    splits = _whole(rig, frig, owner, paired, splits, notes)
    cut_at = {s[0]: s for s in splits}

    def at(mb, p):
        return _carrier(owner, cut_at, mb, p)
    placed, replaced, whole = {}, [], {}        # props first: one in a slot is no chain of the arm it was held in
    held = _match_props(rig, body, frig, items or {}, slots, owner, embedded, rest, placed, replaced, notes, kinds,
                        whole)
    tips = _match_fingers(rig, frig, description, fch, mch, owner, embedded, rest, notes, digits)
    loose, swung, fitted = _match_extra(rig, frig, description, fch, mch, owner, embedded, rest, notes, body, at,
                                        named)
    _rest_of_tree(rig, owner, hips, at)
    ends = {fch['spine'].bones[-1]} | held | swung | tips
    if body.plan == 'ball':
        ends.add(ball)
    for side in ('left', 'right'):
        ends.add(fch[f'{side} arm'].bones[-1])
        leg = fch[f'{side} leg']
        ends.update(leg.bones[leg.anchors[1]:])
    end_parents = _end_parents(frig.parents, embedded, ends)
    through = {b: a for b, a in end_parents.items() if _no_length(embedded[a])}
    meets = []
    match = Match(owner, embedded, rest, notes, ends, placed=placed, replaced=replaced, whole=whole, loose=loose,
                  splits=splits,
                  paired=paired, fitted=fitted, end_parents=end_parents,
                  feet={fch[f'{s} leg'].bones[fch[f'{s} leg'].anchors[1]] for s in ('left', 'right')}, ankles=ankles,
                  apart=_apart_at(rig, frig, owner, at, placed, through, meets), meets=meets,
                  trunk=[b for b in fch['spine'].bones if b not in ends], head=fch['spine'].bones[-1])
    if body.arms['left'] and body.plan != 'ball':
        match.widen = _widen(rig, frig, fch, owner, bare, match, notes)
    return match


def _widen(rig, frig, fch, owner, bare, match, notes):
    """The chest widened across the fighter's midline to its shoulders, where the model's arms start at their own
    shoulder joints with no clavicle for the fighter's to take (``bare``: per arm, the fighter's shoulder, _place
    giving the fighter's clavicle no length in the model). The upper arm rides the fighter's shoulder joint, so the
    clavicle's length would open between it and the chest: a carnival skeleton's upper arms, tucked into its ribcage's
    corners 2 from its middle, stood at Ganondorf's 2.9, off a ribcage still 1.8 wide. The chest bones from above the
    hips to the one the arms hang from widen until the chest's own surface, re-posed (the skin they carry once cut at
    the fighter's joints, repose.split: a skeleton's ribcage hangs from a bone at its pelvis; a stretched trunk has
    thickened already), is as wide as the shoulders; the hips keep the legs' own. Only wider: a chest already that wide
    reaches them, its arms starting inside it (Eevee's ruff, widened by its legs' spread 3.2 times onto Pikachu's
    shoulders, stood out as a plate). {chest bone: (times, the fighter's midline x)}."""
    if len(bare) < 2:
        return {}
    from . import repose
    spine = fch['spine']
    chest = spine.bones[1:spine.anchors[1] + 1]
    bones = sorted(match.embedded)
    w = repose.split(rig, repose.fighter_weights(rig, owner, bones), bones, match.splits, owner)
    on = w[:, [bones.index(b) for b in chest if b in bones]].sum(axis=1) > .5
    x = repose.deform(rig.points[on], w[on], bones, repose.transforms(match, bones))[:, 0]
    fl, fr = bare['left arm'], bare['right arm']
    times = abs(fl[0] - fr[0]) / max(float(np.ptp(x)) if len(x) else 0., 1e-6)
    if times <= 1 or not len(x):
        return {}
    notes.append(f'chest widens {times:.2f} to the shoulders: {", ".join(frig.names[b] for b in chest)}')
    return {b: (float(times), float(fl[0] + fr[0]) / 2) for b in chest}


def _place_ankle(rig, f, m, placed):
    """The floor decides the ankle: the fighter's ankle height above the scaled model's sole.

    Keep its horizontal place; move it vertically before ownership and cuts, so the shin reaches the new ankle
    and a tall boot gives its upper portion to the shin. The foot below keeps its size. The authored ankle (_ankle)
    identifies the foot's skin, including its children, but does not dictate the fighter joint's height in it.
    An unweighted foot has no measurable sole; record that absence instead of inventing a surface.
    """
    a = f.anchors[1]
    j, before = placed[a]
    if j is None:
        t, _ = m.fractions()
        j = min(int(np.searchsorted(t, _fraction_of(m, before), side='right')) - 1, len(m.bones) - 1)
    b = m.bones[j]
    on = rig.weights[:, [b] + _below(rig.children(), b)].sum(axis=1) > 0.5
    sole = float(rig.points[on, 2].min()) if on.any() else None
    after = np.array(before, float)
    if sole is not None:
        after[2] = sole + f.joints[a][2]             # the fighter's floor is z=0
        placed[a] = (None, after)                  # it is a chosen joint, not an authored anchor
    return {'embedded_before': np.asarray(before).tolist(), 'embedded_after': after.tolist(),
            'source_sole_z': sole, 'fighter_ankle_height': float(f.joints[a][2])}


def _end_parents(parents, embedded, ends):
    """Parts within an end inherit its placement, including across unrepresented helper joints.

    The nearest matched ancestor is the parent of the placed piece. A segment between joints breaks the end:
    an earlier end elsewhere up the skeleton does not make this segment or its children part of that end. A bone with
    no length inside the model is no segment: an end below it is placed by it, so a neckless model's head (the
    fighter's neck on its head joint: _snap) sits on its torso where the fighter's neck starts, as the model is built,
    not a neck's length above it.
    """
    nearest = {b: next((a for a in _ancestors(parents, b)[1:] if a in embedded), None) for b in ends}
    return {b: a for b, a in nearest.items() if a in ends or (a is not None and _no_length(embedded[a]))}


def _no_length(segment):
    """Whether a fighter bone's (head, tail) inside the model is one point (repose._bone then only moves it)."""
    return np.linalg.norm(np.subtract(segment[1], segment[0])) < 1e-4


def _match_spine(rig, frig, fch, mch, owner, embedded, rest, notes, same=False, groups=None):
    """The spine by height between its landmarks on both sides: the hips, the shoulders (where the arms start), the
    head joint and the top. Which bone the arms hang from says nothing about height (Mario's hangs them from his waist
    joint, a segmented rig from a node at shoulder height), so the levels place the fighter's joints. As along a limb,
    a model spine bone goes to the fighter bone whose stretch its own joint lies in, and one a fighter joint falls
    inside is cut there: a fighter's neck inside a model's long neck bone, or inside a segmented trunk's one torso
    piece, carries the stretch between its joints, so the head stays on. Returns those cuts (_cuts). ``same``: a ball's body
    has no separate trunk landmarks; the fighter's relative heights place its joints inside the model's height.
    Bulk need not make their heights equal. ``groups``: the bones at each joint (body.groups).
    No fighter joint is placed past the model's head joint. A head joint under its shoulders (arms rooted a hair above
    it) has its level lifted to theirs, as a neckless head starts there; a place at that lifted level stood that hair
    above the joint, and the hair was the fighter's neck: a skeleton's neck bone, its joint level with its arm roots,
    was stretched fourteen times up Ganondorf's (the app's import of 2026-11-10). The place is the joint."""
    f, m = fch['spine'], mch['spine']

    farms = [fch['left arm'], fch['right arm']]
    marms = [mch['left arm'], mch['right arm']]
    lf = levels(fch, fighter=True)
    lm = [h * m.end[2] / f.end[2] for h in lf] if same else levels(mch)
    for k in range(1, 4):                                # in order up the body; a neckless head starts at the shoulders
        for lv in (lm, lf):                              # (a ball fighter's body is its head: it has no head joint)
            lv[k] = max(lv[k], lv[k - 1] + (1e-6 if k != 2 else 0.0))
    root, head_joint = m.joints[0], m.joints[m.anchors[2]]
    line = None if same else _trunk_line(m, marms)

    def inside(h_fighter):
        z = np.interp(h_fighter, lf, lm)
        if line is not None:                    # a trunk carried level: the point along it at that level
            return _at_level(line, z)
        if not same and z >= head_joint[2]:     # at the head joint or, its level lifted to the shoulders', past it
            return np.array(head_joint, float)
        u = (z - root[2]) / ((head_joint[2] - root[2]) or 1.0)
        xy = root[:2] + np.clip(u, 0, 1) * (head_joint[:2] - root[:2])
        return np.array([xy[0], xy[1], z])
    neck = len(f.joints) - 2 if len(f.joints) - 2 > f.anchors[1] else None
    places, on = _snap(rig, m, [inside(_level(f, farms, j)) for j in f.joints], m.anchors[2], neck)
    top = inside(lf[3]) if line is not None else np.array([head_joint[0], head_joint[1], lm[3]])
    for i, fb in enumerate(f.bones):
        embedded[fb] = (places[i], places[i + 1] if i + 1 < len(places) else top)
        rest[fb] = (f.joints[i], f.joints[i + 1] if i + 1 < len(f.joints) else f.end)
    t_f = [_fraction_of(m, p) for p in places]
    t_m, _ = m.fractions()
    # the model's hips are the fighter's (the first anchors pair), though the fighter's spine may start there too
    owner.setdefault(m.bones[0], f.bones[0])
    for j, mb in enumerate(m.bones):
        owner.setdefault(mb, f.bones[max((k for k in range(len(f.bones)) if t_f[k] <= t_m[j] + 1e-9), default=0)])
    notes.append('spine levels (hips, shoulders, head, top): fighter ' + ', '.join(f'{x:.2f}' for x in lf)
                 + '; model ' + ', '.join(f'{x:.2f}' for x in lm) + '; '
                 + ', '.join(f'{rig.names[mb]}->{frig.names[owner[mb]]}' for mb in m.bones) + '; joints '
                 + ', '.join(f'{frig.names[fb]}->{rig.names[m.bones[j]] if j is not None else f"{t:.3f}"}'
                             for fb, j, t in zip(f.bones[1:-1], on[1:-1], t_f[1:-1]))
                 + f' of {", ".join(f"{x:.3f}" for x in t_m[1:m.anchors[2] + 1])}')
    parts = _parts(rig, m, groups or {})
    return _cuts(parts, places[1:], f.bones[1:]) if parts else []


def _snap(rig, m, places, head, neck=None):
    """The spine's places with a fighter joint between the hips and the head on a model joint near it (within
    INTERPOLATE of the chain) and nearer it than any other fighter joint is, the hips' and head's among them; else
    where its level put it. Returns (places, for each the model joint index it is on, or None). A fighter joint a
    model joint is near is that joint: the model bends there, and a place a hair from it would cut a sliver off the
    bone there, stretched to the fighter bone's whole length (Lucario's chest fur, cut just under his neck joint, rode
    Falco's neck; Leon's shoulder pads, cut just under his head piece, stretched six times up Falcon's neck into a
    sheet). The joint goes to the fighter joint nearest it, not the first that reaches it: given to Falcon's spine
    joint, Leon's torso joint (by Falcon's chest joint) left Falcon's spine bone a sliver of torso. One at the hips'
    place is the hips', wherever the chain reaches it: a fighter bone of no length above them would leave a gap its
    length between the pieces either side (Gardevoir's pelvis and spine both hang at her hips from her root, which
    sits above them: the chain climbs to it and comes back down, and along it her spine's joint lay as far from her
    hips as her spine's place did). The model's head joint (``head``, its index) may be the neck's (``neck``, the
    fighter's last joint before the head, where one comes after the joint its arms hang from: Pikachu's head hangs
    from the joint carrying his arms): a neck there has no length in the model, which has none (Leon's is inside his
    head piece), and the head is placed by it, on the torso where the fighter's neck starts (_end_parents). A place
    snaps only between the places either side of it, so the chain's places stay in order and no fighter bone runs
    backwards along the model (Meta Knight's head joint lies below his arms, under the fighter's chest joint's place:
    snapped onto it, the neck would have come before the chest, and the chest bone, running down the model, turned his
    ball upside down onto it). The nearest joint always lies between; the neck's is the one that may not."""
    t_m, _ = m.fractions()
    t = [_fraction_of(m, np.asarray(p, float)) for p in places]
    hips = SAME_PLACE * np.ptp(rig.points, axis=0).max()
    chosen = {}
    for j in range(1, head + 1):
        if np.linalg.norm(m.joints[j] - np.asarray(places[0], float)) <= hips:
            continue
        i = neck if j == head and neck is not None else min(range(len(places)), key=lambda k: abs(t[k] - t_m[j]))
        d = abs(t[i] - t_m[j])
        if 0 < i < len(places) - 1 and d <= INTERPOLATE and t[i - 1] < t_m[j] <= t[i + 1] and \
                (i not in chosen or d < abs(t[i] - t_m[chosen[i]])):
            chosen[i] = j
    out = [m.joints[chosen[i]] if i in chosen else p for i, p in enumerate(places)]
    return out, [chosen.get(i) for i in range(len(places))]


def extra_chains(frig, description, fch):
    """The chains the fighter's game swings besides the body (description.json chains, fingers aside: the model's
    take them, _match_fingers): Peach's and Zelda's skirt strands, capes, hat tails, tails, ears: [(region, the main
    bone it hangs from, Chain)]. A strand of
    its skirt (a ring of RING or more off one lower bone) ends where its own mesh does: its last bone is a tip carrying
    none, lying out flat along the floor, whose tail is no hem. Read to it, a gown fitted to the strand reached through
    the floor."""
    region, fmain = _regions(fch), {b for c in fch.values() for b in c.bones}
    index = {n: i for i, n in enumerate(frig.names)}
    found = []
    for c in description.get('chains') or []:
        bones = [index[f'JOBJ_{n - 1}'] for n in c['bones'] if f'JOBJ_{n - 1}' in index]
        if c.get('kind') == 'finger' or not bones:
            continue
        anchor = next((x for x in _ancestors(frig.parents, bones[0])[1:] if x in fmain), None)
        if anchor is not None:
            found.append((region.get(anchor), anchor, bones, c.get('kind')))
    off = [a for r, a, _, _ in found if r == 'lower']
    carried = frig.carried()
    out = []
    for r, a, bones, kind in found:
        if r == 'lower' and off.count(a) >= RING:
            while len(bones) > 1 and carried[bones[-1]] <= 0:
                bones = bones[:-1]
        chain = _chain(frig, bones)
        chain.kind = kind
        out.append((r, a, chain))
    return out


def _match_fingers(rig, frig, description, fch, mch, owner, embedded, rest, notes, digits):
    """The model's fingers on the fighter's (the game closes a fist by turning them). Per hand, the digits off the
    model's hand (``digits``: the forks' branches so named; one splitting into several fingers gives each, its palm
    riding the hand) pair one to one with the fighter's finger chains (description.json chains of kind finger) by
    where each ends and which way it points, in the hand's own frame (_hand_frame), thumb to thumb. A paired finger is a
    limb (_place): the fighter's joints at the same shares along the model's finger, the stretches between them to
    the fighter's lengths, its last joint an end keeping its own size. The hand keeps its own move. Digits left over
    ride the hand (_rest_of_tree). Returns the fighter's paired fingers' last joints."""
    if not digits:
        return set()
    index = {n: i for i, n in enumerate(frig.names)}
    fingers = [[index[f'JOBJ_{n - 1}'] for n in c['bones'] if f'JOBJ_{n - 1}' in index]
               for c in description.get('chains') or [] if c.get('kind') == 'finger']
    kids, carried = rig.children(), rig.carried()
    tips = set()
    for side in ('left', 'right'):
        f_arm, m_arm = fch.get(f'{side} arm'), mch.get(f'{side} arm')
        if f_arm is None or m_arm is None or len(f_arm.bones) < 2 or len(m_arm.bones) < 2:
            continue
        fhand, mhand = f_arm.bones[-1], m_arm.bones[-1]
        fc = [_chain(frig, bones) for bones in fingers if bones and fhand in _ancestors(frig.parents, bones[0])[1:]]
        roots = [b for b in range(len(rig.names)) if rig.names[b] in digits and b not in owner
                 and mhand in _ancestors(rig.parents, b)[1:]]
        roots = [b for b in roots if not set(_ancestors(rig.parents, b)[1:]) & set(roots)]
        mc = [_chain(rig, _apart(rig, path)[0]) for b in roots for path in _finger_paths(kids, carried, b)]
        if len(fc) < 2 or len(mc) < 2:     # no thumb apart from the rest: no frame of the hand's own
            continue
        (fx, f_thumb), (mx, m_thumb) = _hand_frame(f_arm.joints[-1], fc), _hand_frame(m_arm.joints[-1], mc)
        cost = np.array([[sum(np.linalg.norm(a - b) for a, b in zip(mx(m), fx(f))) for f in fc] for m in mc])
        ms, fs = [i for i in range(len(mc)) if i != m_thumb], [i for i in range(len(fc)) if i != f_thumb]
        for mi, fi in [(m_thumb, f_thumb)] + [(ms[i], fs[j]) for i, j in _assign(cost[np.ix_(ms, fs)])]:
            f, m = fc[fi], mc[mi]
            placed = _place(f, m)
            ends = [p for _, p in placed[1:]] + [m.end]
            f_ends = list(f.joints[1:]) + [f.end]
            for i, fb in enumerate(f.bones):
                embedded[fb] = (placed[i][1], ends[i])
                rest[fb] = (f.joints[i], f_ends[i])
            t_f = [_fraction_of(m, p) for _, p in placed]
            t_m, _ = m.fractions()
            for j, mb in enumerate(m.bones):
                owner[mb] = f.bones[max(k for k in range(len(f.bones)) if t_f[k] <= t_m[j] + 1e-9)]
            tips.add(f.bones[-1])
            notes.append(f'finger {rig.names[m.bones[0]]}..{rig.names[m.bones[-1]]} -> '
                         f'{frig.names[f.bones[0]]}..{frig.names[f.bones[-1]]} ({cost[mi, fi]:.2f})')
    return tips


def _finger_paths(kids, carried, b):
    """The fingers a digits branch holds: its path down while it doesn't split, then each branch's where it does (a
    palm bone carrying four)."""
    path = [b]
    while True:
        nxt = [k for k in kids[path[-1]] if carried[[k] + _below(kids, k)].sum() > 0]
        if len(nxt) != 1:
            break
        path.append(nxt[0])
    if len(nxt) < 2:
        return [path]
    return [p for k in nxt for p in _finger_paths(kids, carried, k)]


def _hand_frame(wrist, chains):
    """The hand's own frame, and which of ``chains`` is its thumb: (a finger chain -> (where it ends, in the hand's
    reach; which way it points), the thumb's index). The thumb is
    the digit ending nearest the wrist (on every rig read; its root need not be: a pinky may begin at its palm bone).
    The frame runs from the wrist to the other digits' ends, across toward the thumb, then out of the palm: the hand's
    own, however the arm holds it (a model's rest hand turns about its arm unlike the fighter's) and wherever its last
    bone points (a short hand bone may lie across the fingers)."""
    wrist = np.asarray(wrist, float)
    ends = np.array([c.end for c in chains], float) - wrist
    reach = np.linalg.norm(ends, axis=1)
    thumb = int(np.argmin(reach))
    x = np.delete(ends, thumb, 0).mean(0)
    x /= np.linalg.norm(x) or 1.0
    y = ends[thumb] - (ends[thumb] @ x) * x
    y /= np.linalg.norm(y) or 1.0
    frame = np.array([x, y, np.cross(x, y)])
    scale = reach.mean() or 1.0

    def at(c):
        d = c.end - c.joints[0]
        return frame @ (c.end - wrist) / scale, frame @ (d / (np.linalg.norm(d) or 1.0))
    return at, thumb


def _assign(cost):
    """Rows to columns one to one, as many as the fewer, at the least total cost: [(row, column)]."""
    from itertools import permutations
    rows, cols = cost.shape
    if rows <= cols:
        best = min(permutations(range(cols), rows), key=lambda p: sum(cost[i, j] for i, j in enumerate(p)))
        return list(enumerate(best))
    best = min(permutations(range(rows), cols), key=lambda p: sum(cost[i, j] for j, i in enumerate(p)))
    return [(i, j) for j, i in enumerate(best)]


def _match_extra(rig, frig, description, fch, mch, owner, embedded, rest, notes, body=None, at=None, named=None):
    """The model's other chains (a skirt, hair, a cape, a tail) onto the fighter's own (the game's part table lists
    Peach's and Zelda's skirt strands, capes, hat tails, tails and ears), one to one: a chain of a kind that fits the
    fighter's (FITS: what each is, ``named``, labels.chains) hanging from the same region (lower body, upper body,
    head), starting on the same side of the body and hanging the same way. A paired
    chain keeps the model's own length: its joints are only where the game swings it (the fighter's joints at their own
    distances along it), so a long cape stays long and a short one short, unstretched; a strand of the fighter's
    skirt (its ring of them) is laid on as on the ring (cloth.gown, _fit): held as the legs hold the model, squashed
    with them, and fitted to the strand where it reaches past its end. Hair or cloth left over rides beside a
    fighter chain it hangs alongside (a tabard over a skirt, a mantle down over its back); an object (a scabbard, a
    device on a belt: no hair or cloth) takes no chain and rides the fighter's own bone for it, else the bone it hangs
    from. Any other model chain with no partner keeps riding the bone it hangs from (a skirt's strands are then laid
    on the fighter's skirt or its legs: cloth.py). Hair and cloth hang because of what they are: each, off the body,
    the head or a limb (Galahad's sleeves off her forearms), is returned as (the bone it hangs from, the chain, the
    bones it carries), for export to make it swing where the game can; so is a tail no chain of the fighter's carries
    (OWN: none there, or one too short for it), which keeps the shape it was drawn in; everything else only rides. A
    chain nobody
    named (no answer, or no surface of its own to show) rides. A chain's region is that of the bone it hangs from, but
    where the body's plan says otherwise (plans.region: a dress off a legless body's one-piece trunk hangs from the
    hips, its other chains from the chest; a ball's face things from its head). Returns (those, the fighter bones
    paired that keep the model's own size, the fighter's skirt strands fitted)."""
    at = at or (lambda mb, p: owner[mb])
    # a bone at a joint of the body's is that joint (body.read's groups: Lucario's Hips at his Waist, his tail off it)
    joint = {k: b for b, group in (getattr(body, 'groups', None) or {}).items() if b in owner for k in group}
    main = set(owner) | set(joint)
    fmain = {b for c in fch.values() for b in c.bones}
    region = _regions(fch)
    rolls = fighter_plan(description) == 'ball'
    kids, carried = rig.children(), rig.carried()
    model = []
    for b in range(len(rig.names)):
        attach = joint.get(rig.parents[b], rig.parents[b])
        if b in main or attach < 0 or attach not in owner:
            continue
        # a branch with the body's own bones below it joins them (a clavicle root both arms hang from), it doesn't hang
        if carried[[b] + _below(kids, b)].sum() <= 0 or main & set(_below(kids, b)):
            continue
        path = [b]
        while True:
            nxt = [k for k in kids[path[-1]] if carried[[k] + _below(kids, k)].sum() > 0]
            if not nxt:
                break
            path.append(max(nxt, key=lambda k: carried[[k] + _below(kids, k)].sum()))
        hangs = plans.region(rig, body, attach, [b] + _below(kids, b)) if body is not None else None
        hung = region.get(at(attach, rig.heads[b]))     # the fighter bone carrying it where it hangs
        if hangs is None and rolls:         # a ball fighter's body is its head too: the model's own head says
            hangs = plans.ball_region(rig, mch, attach, hung)
        chain = _chain(rig, path)
        said = (named or {}).get(rig.names[b]) or {}
        chain.kind, chain.held_out = said.get('kind'), bool(said.get('held_out'))
        model.append((hangs or hung, attach, chain, [b] + _below(kids, b)))
    fighter = extra_chains(frig, description, fch)

    def heading(chain, spine):
        """Which way round the body the chain starts (from the spine where it is level with the chain's root: a cape
        hung from each clavicle starts left, middle and right of it) and which way it hangs."""
        at = spine.joints[np.argmin(np.abs(spine.joints[:, 2] - chain.joints[0][2]))]
        r = chain.joints[0] - at
        r = r - r[2] * np.array([0, 0, 1.])
        d = chain.end - chain.joints[0]
        return (r / (np.linalg.norm(r) or 1.0), d / (np.linalg.norm(d) or 1.0))
    # an object (a scabbard, a device on the belt, an earring): its author bound its own mesh to bones of its own, each
    # vertex wholly to one, apart from the skin the model's cloth and hair bend in (a model of rigid pieces throughout
    # says nothing by it), and no surface of it runs on from one of its bones to the next. It doesn't bend, so it takes
    # no chain the game swings; it may ride beside one. A low-poly cloth bound one vertex to one bone still bends: its
    # surface runs on across its joints (a gown's flap, a tail)
    skinned = (rig.weights.max(axis=1)[rig.weights.sum(axis=1) > 0] < RIGID).mean() > SKINNED
    objects = set()
    for mi, (_, _, mc, carries) in enumerate(model):
        if mc.kind in HANG:
            continue
        w = rig.weights[:, carries]
        own = w.sum(axis=1) > 0.5
        if skinned and own.any() and w[own].max(axis=1).min() >= RIGID and not _bends(rig, own, carries):
            objects.add(mi)
    pairs = []
    for mi, (mr, ma, mc, _) in enumerate(model):
        rm, dm = heading(mc, mch['spine'])
        for fi, (fr, fa, fc) in enumerate(fighter):
            if mr is None or mr != fr or fc.kind not in FITS.get(mc.kind, ()):
                continue
            rf, df = heading(fc, fch['spine'])
            pairs.append((float(rm @ rf + dm @ df), mi, fi))
    # the fighter's skirt: a ring of strands off one lower bone (Peach's and Zelda's eight)
    off = [fa for fr, fa, _ in fighter if fr == 'lower']
    ring = {fi for fi, (fr, fa, _) in enumerate(fighter) if fr == 'lower' and off.count(fa) >= RING}
    squash = _squash(fch, embedded, rest) if ring else 1.0

    def skirt(fc, mc):
        """A model strand laid on a strand of the fighter's skirt: its joints' places in it, its last bone's tail."""
        joints, tail = _fit(fc, np.vstack([mc.joints, mc.end]), squash)
        return [(None, p) for p in joints], tail

    used_m, used_f, swung, fitted, rooted, hung = set(), set(), set(), set(), set(), []
    for score, mi, fi in sorted(pairs, reverse=True):
        if score < 1.0 or mi in used_m or fi in used_f or mi in objects:
            continue
        mc, fc = model[mi][2], fighter[fi][2]
        used_m.add(mi)
        used_f.add(fi)
        # A tail the fighter's chain cannot carry to its end. A chain keeps its own length, the fighter's joints at
        # their own distances along it, so all of it past the fighter's last joint rides that bone stiff: Diddy's
        # tail, 15.5 long on Fox's 11.1, rode Fox's last tail bone for 41 % of its length, a rod (size-3, the owner:
        # "diddy tail on fox looks really strange"). Past the fighter's chain by more than its last bone's own
        # length, it swings on joints of its own, at its own length (OWN, with the loose chains below). Drawing it
        # in to the fighter's length was tried: it halved Baragon's tail on Bowser.
        # It still hangs where the fighter's tail hangs: the whole of it rides the fighter's first tail bone, its
        # root at that bone's level and as far behind the hips as the model has it (_rooted). Left to the bone it
        # hung from in the model it rode Fox's waist: lifted with the hips' own move it sat half a unit above Fox's
        # tail, and it tipped up with his back as he leant (limbs-1, the owner: "higher up on his body than fox")
        if mc.kind in OWN and fi not in ring and _long(mc) - _long(fc) > _long(fc) * (1 - fc.fractions()[0][-2]):
            fb = fc.bones[0]
            embedded[fb], rest[fb] = _rooted(mc, fc, mch['spine'].joints[0], fch['spine'].joints[0])
            for b in model[mi][3]:
                owner[b] = fb
            rooted.add(fi)
            hung.append((mc.bones[0], mc, model[mi][3]))       # it hangs from the fighter bone carrying its own root
            notes.append(f'chain {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} on joints of its own from '
                         f'{frig.names[fb]} ({score:.2f})')
            continue
        if fi in ring:      # a skirt's: laid on as on the ring, squashed onto the strand (no end of its own size)
            fitted.update(fc.bones)
            placed, tail = skirt(fc, mc)
            f_ends = list(fc.joints[1:]) + [_onward(fc)]
        else:
            swung.update(fc.bones)
            placed, tail = _place_along(fc, mc), mc.end
            f_ends = list(fc.joints[1:]) + [fc.end]
        ends = [p for _, p in placed[1:]] + [tail]
        for i, fb in enumerate(fc.bones):
            embedded[fb] = (placed[i][1], ends[i])
            rest[fb] = (fc.joints[i], f_ends[i])
        t_f = [_fraction_of(mc, p) for _, p in placed]
        t_m, _ = mc.fractions()
        for j, mb in enumerate(mc.bones):
            i = max(k for k in range(len(fc.bones)) if t_f[k] <= t_m[j] + 1e-9)
            owner[mb] = fc.bones[i]
        notes.append(f'chain {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} -> '
                     f'{frig.names[fc.bones[0]]}..{frig.names[fc.bones[-1]]} ({score:.2f})')
    # a chain left over that starts beside a fighter chain and hangs the same way (a tabard over the front of a skirt)
    # swings with it: its bones ride that chain's at their own distances along it (a chain that was placed: one no model
    # chain took has no place inside the model to ride from)
    for score, mi, fi in sorted(pairs, reverse=True):
        if score < BESIDE or mi in used_m or fi not in used_f or fi in rooted:
            continue
        used_m.add(mi)
        mc, fc = model[mi][2], fighter[fi][2]
        t_f = [_fraction_of(mc, p) for _, p in (skirt(fc, mc)[0] if fi in ring else _place_along(fc, mc))]
        t_m, _ = mc.fractions()
        for j, mb in enumerate(mc.bones):
            owner[mb] = fc.bones[max(k for k in range(len(fc.bones)) if t_f[k] <= t_m[j] + 1e-9)]
        notes.append(f'chain {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} rides beside '
                     f'{frig.names[fc.bones[0]]}..{frig.names[fc.bones[-1]]} ({score:.2f})')
    # a chain left over that hangs from higher up the body past where a fighter chain starts, beside it and hanging
    # the same way from there (a mantle over the back of a skirt), swings with it from there: its bones lying mostly
    # past that point ride the chain's at their own distances along it from there; the ones above keep riding the bone
    # it hangs from
    passing = []
    for mi, (mr, ma, mc, _) in enumerate(model):
        for fi, (fr, fa, fc) in enumerate(fighter):
            if (mi in used_m or mc.kind not in HANG or fc.kind not in FITS[mc.kind] or fr not in BELOW.get(mr, ())
                    or fc.bones[0] not in embedded):
                continue
            cut = _passing(mc, embedded[fc.bones[0]][0][2])
            if cut is not None:
                (rm, dm), (rf, df) = heading(cut[1], mch['spine']), heading(fc, fch['spine'])
                passing.append((float(rm @ rf + dm @ df), mi, fi, cut))
    for score, mi, fi, (start, cut) in sorted(passing, key=lambda p: p[0], reverse=True):
        if score < BESIDE or mi in used_m:
            continue
        used_m.add(mi)
        mc, fc = model[mi][2], fighter[fi][2]
        (t_m, pts), (t_f, f_pts) = mc.fractions(), fc.fractions()
        s = t_m * np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()               # along each, in its own length
        d_f = t_f * np.linalg.norm(np.diff(f_pts, axis=0), axis=1).sum()
        for j, mb in enumerate(mc.bones):
            if (s[j] + s[j + 1]) / 2 > start:
                owner[mb] = fc.bones[max(k for k in range(len(fc.bones)) if d_f[k] <= max(s[j] - start, 0) + 1e-9)]
        notes.append(f'chain {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} rides beside '
                     f'{frig.names[fc.bones[0]]}..{frig.names[fc.bones[-1]]} from {start:.2f} along it ({score:.2f})')
    # an object left over is never swung: it rides the fighter's own bone for such a piece where one hangs from the
    # same bone near its pivot (the game's sheath, which its animations move), else the bone it hangs from
    objects -= used_m
    taken, free = set(), _free_bones(frig, description, fmain)
    for mi, (mr, ma, mc, carries) in enumerate(model):
        if mi not in objects or mr not in ('upper', 'lower'):
            continue
        fa, own = at(ma, rig.heads[mc.bones[0]]), rig.weights[:, carries].sum(axis=1) > 0.5
        pivot = rig.heads[mc.bones[0]] - embedded[fa][0]
        near = [(np.linalg.norm(pivot - (frig.heads[b] - rest[fa][0])), b) for b in free
                if frig.parents[b] == fa and b not in taken]
        d, b = min(near, default=(np.inf, None))
        if d > _long_side(rig.points[own]) / 2:
            continue
        taken.add(b)
        embedded[b], rest[b] = embedded[fa], rest[fa]      # placed as the bone it hangs from places it
        for x in carries:
            owner[x] = b
        notes.append(f'object {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} rides {frig.names[b]}, '
                     f'the fighter\'s own off {frig.names[fa]} ({d:.2f} from its pivot)')
    loose = [(ma, mc, carries) for mi, (mr, ma, mc, carries) in enumerate(model)
             if mi not in used_m | objects and mc.kind in HANG + OWN] + hung
    for ma, mc, _ in loose:
        rides = sorted({frig.names[owner.get(b, at(ma, mc.joints[0]))] for b in mc.bones})
        notes.append(f'loose {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} rides {", ".join(rides)}')
    for mi in sorted(objects):
        mc = model[mi][2]
        if mc.bones[0] not in owner and model[mi][0] not in ('arm', 'leg'):
            notes.append(f'object {rig.names[mc.bones[0]]}..{rig.names[mc.bones[-1]]} rides '
                         f'{frig.names[at(model[mi][1], mc.joints[0])]}')
    return loose, swung, fitted


def _bends(rig, own, carries):
    """Whether a surface runs on from one of ``carries`` (bones) to another: an edge between two of its points (``own``)
    riding different ones. False when the rig keeps no edges."""
    edges = getattr(rig, 'edges', None)
    if edges is None or not len(edges):
        return False
    inside = edges[own[edges].all(axis=1)]
    bone = np.asarray(carries)[np.argmax(rig.weights[:, carries], axis=1)]
    return bool((bone[inside[:, 0]] != bone[inside[:, 1]]).any())


def _passing(mc, z):
    """(how far along ``mc`` it first comes down to the height ``z``, the chain from there on), or None when it never
    comes down to it."""
    t, pts = mc.fractions()
    total = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        if a[2] > z >= b[2] or (k == 0 and a[2] <= z):
            u = 0.0 if a[2] <= z else (a[2] - z) / (a[2] - b[2])
            return (t[k] + u * (t[k + 1] - t[k])) * total, Chain(mc.bones[k:], np.vstack(
                [a + u * (b - a)] + list(mc.joints[k + 1:])), mc.end)
    return None


def _free_bones(frig, description, fmain):
    """The fighter's own bones for a piece worn on it: a branch off one of its main bones that is no part of the body
    (no role in the part table, nothing of the body below it) and no chain its game swings, carrying a piece of the
    fighter's own costume in play (Marth's sheath, which his animations move as it hangs; Roy's game hides his, and
    moves its bone as it likes)."""
    roles = {f'JOBJ_{int(n) - 1}': r for n, r in description['bones']['roles'].items()}
    index = {n: i for i, n in enumerate(frig.names)}
    chained = {index[f'JOBJ_{n - 1}'] for c in description.get('chains') or [] for n in c['bones']
               if f'JOBJ_{n - 1}' in index}
    kids = [[] for _ in frig.names]
    for b, p in enumerate(frig.parents):
        if p >= 0:
            kids[p].append(b)
    carried = frig.carried()
    return [b for b in range(len(frig.names)) if frig.parents[b] in fmain and b not in fmain and carried[b] > 0
            and not any(x in fmain or x in chained or roles.get(frig.names[x], 'none') != 'none'
                        for x in [b] + _below(kids, b))]


def props(rig, body):
    """The model's props: [(root bone, its bones, its vertices, hand, held)], those it carries outside the body (a
    branch off the root above the hips: a sword its game puts in the hand by animation), each on the side its vertices
    are on, then those it holds (held_props). A branch carrying a limb the body reading took is the body, however it
    hangs (Swampert's feet, off a floor control beside its hips)."""
    kids, carried = rig.children(), rig.carried()
    inside = {body.root, *_below(kids, body.root)}
    limbs = {b for limb in (*body.legs.values(), *body.arms.values()) for b in limb}
    above = set(_ancestors(rig.parents, body.root))
    out = []
    for b in range(len(rig.names)):
        sub = [b] + _below(kids, b)
        if b in inside or b in above or rig.parents[b] not in above | {-1} or carried[sub].sum() <= 0 \
                or limbs.intersection(sub):
            continue
        sel = rig.weights[:, sub].sum(axis=1) > 0.5
        x = rig.points[sel].mean(axis=0) @ body.left if sel.any() else rig.heads[b] @ body.left
        out.append((b, sub, sel, 'left_hand' if x > rig.heads[body.root] @ body.left else 'right_hand', False))
    return out + [p + (True,) for p in held_props(rig, body)]


def armed(slots):
    """The weapon slots a prop can take: each alone in its hand (Game & Watch swaps several in one hand by move, so
    none of them is the weapon)."""
    return [s for s in slots if sum(t['hand'] == s['hand'] for t in slots) == 1]


def _match_props(rig, body, frig, items, slots, owner, embedded, rest, placed, replaced, notes, kinds=None,
                 whole=None):
    """The model's props (``props``). Each hand the fighter fights with a weapon in (``armed``) takes one prop, and the
    slot is listed as replaced, so the vanilla weapon and its other forms go. Where the hands' slots are of more than
    one kind (Link's sword and shield), what each prop is decides (``kinds``: {root bone name: kind},
    labels.equipment): a slot takes only a prop named its kind; with one kind, any. Of those, the longest held in
    either hand (the game swings its weapon in its own hand, whichever hand the model holds it in), else one outside
    the body, on that side first. A swung weapon's slot fits its prop to the vanilla weapon (_into_slot): the trails
    and hitboxes are sized to it. A shield's takes a prop held in either hand, its own hand's first, and fits it into
    the vanilla shield's place at its own size (_into_shield): from either hand alike, never mirrored. A prop a slot
    took rides it alone, a strap or a grip tying it to the arm with it (``whole``: {joint: its vertices}): moved into
    a slot, a share left on the arm would pull it back. Any other prop outside the body goes to the item joint of the
    nearer hand, its root, the grip, on the joint, keeping its own orientation and size, placed there outright: a
    weapon its game puts in the hand by animation, with no slot of its kind to say which it is. Where the props were
    asked what they are, the game puts in a hand only the one its slot took, and any other is no weapon: it rides
    where it lies on the model
    (_rest_of_tree; Toon Link's rod, named none). Any other held prop stays in its hand. A prop the file drew on the
    hand's own bone (no bones of its own: _on_the_hand) is taken only by a slot of the kind it is named; where
    nothing is asked, by none. Returns the joints used."""
    index = {n: i for i, n in enumerate(frig.names)}
    # pokecast: a bone carrying a vertex or two has no shape to fit a slot (Gardevoir-Peach: _into_slot had one point)
    found, slots = [p for p in props(rig, body) if p[2].sum() > 2], armed(slots)
    by_kind = len({s['kind'] for s in slots}) > 1
    takes = _takes(rig, found, slots, kinds)
    used, taken = set(), {p[0] for _, p in takes}
    for slot, (b, sub, sel, at, held) in takes:
        hand, joint, swung = slot['hand'], index[slot['bone']], slot.get('swung', True)
        if swung:
            placed[joint], (grip, tip, length) = _into_slot(rig.points[sel], rig.heads[b], slot)
            embedded[joint] = (grip, grip + tip * length)
            rest[joint] = (np.asarray(slot['grip']), np.asarray(slot['grip']) + slot['length'] *
                           np.asarray(slot['tip_direction']))
            how = f"length {length:.2f} -> {slot['length']:.2f}"
        else:
            placed[joint], (middle, long) = _into_shield(rig.points[sel], slot)
            embedded[joint] = (middle, middle + long)
            rest[joint] = tuple(placed[joint][:, :3] @ x + placed[joint][:, 3] for x in embedded[joint])
            turned = np.degrees(np.arccos(np.clip((np.trace(placed[joint][:, :3]) - 1) / 2, -1, 1)))
            how = f"{'its own hand' if at == hand else 'from the other hand'}, its own size, turned {turned:.0f} degrees"
        for x_ in sub:
            owner[x_] = joint
        if whole is not None:
            whole[joint] = sel
        used.add(joint)
        replaced.append(dict(slot, item=rig.names[b], item_hand=at, held=held))
        notes.append(f"prop {rig.names[b]} ({'held' if held else 'outside the body'}, {at}) -> {frig.names[joint]} "
                     f"({hand}): {slot['part']}'s slot, {how}")
    for b, sub, sel, hand, held in found:
        if held or b in taken or by_kind:
            continue
        joint = index.get(items.get(hand, ''))
        if joint is None:
            continue
        grip = rig.heads[b]
        embedded[joint] = (grip, grip + (frig.tails[joint] - frig.heads[joint]))
        rest[joint] = (frig.heads[joint], frig.tails[joint])
        # placed outright: the item joint hangs under the hand, an end, and a part of an end takes the end's move
        # (_end_parents), which carries the prop at its distance from the model's hand, where it lies off the body,
        # never in the fist (the Barbarian's sword, stood on the floor beside him: beside Donkey Kong's fist)
        placed[joint] = np.column_stack([np.eye(3), frig.heads[joint] - grip])
        for x_ in sub:
            owner[x_] = joint
        used.add(joint)
        notes.append(f'prop {rig.names[b]} -> {frig.names[joint]} ({hand})')
    return used


def _takes(rig, found, slots, kinds):
    """The prop each armed slot takes, of ``found`` (props): [(slot, prop)]. Where the slots are of more than one kind,
    only a prop named its kind; a held prop first, the longest (the weapon, not a shield or a dagger beside it), a
    shield in the slot's own hand before one in the other; else one outside the body, on that side first."""
    by_kind = len({s['kind'] for s in slots}) > 1
    out, taken = [], set()
    for slot in sorted(slots, key=lambda s: s['hand']):
        hand, swung = slot['hand'], slot.get('swung', True)
        ready = [p for p in found if p[2].any() and p[0] not in taken
                 and (not by_kind or (kinds or {}).get(rig.names[p[0]]) == slot['kind'])
                 and (swung or p[4]) and (by_kind or len(p[1]))]
        if ready:
            prop = min(ready, key=lambda p: (not p[4], (not swung or not p[4]) and p[3] != hand,
                                             -_long_side(rig.points[p[2]])))
            taken.add(prop[0])
            out.append((slot, prop))
    return out


def held_props(rig, body):
    """The props the model holds: [(root bone, its bones, its vertices, 'left_hand' or 'right_hand')]. A held prop is a
    branch off an arm, up to its hand, whose own mesh (the vertices it carries most of) rides that branch alone, but
    for a strap or a grip tied to the arm (ALONE of it or more): its author bound it to bones of its own, apart from
    the skin, as an object in the fist (a finger, a sleeve, a shield strapped to the forearm are skinned into the arm;
    a trace of the branch's weight on the hand's skin is not its mesh). Its grip, the head of its bone, is within the hand's reach, and it is HELD times the hand's size or more: a
    pad on the shoulder is not in the hand; a finger, a thumb or a claw cut as a piece of its own is about the size of
    the rest of the hand or smaller (on the core loop's rips, 2026-10-25: swords, a wand, a shield and a boomerang 2.4
    to 7.6 times it; fingers, thumbs, claws, sleeves and pads 1.05 or less). The hand is what its bone and the bones
    below it carry, the prop left out; a size is the long side of the points' own box (_long_side). Then, after a
    hand's branches, an object the file drew on the hand's own bone (_on_the_hand: Ocarina of Time's display lists,
    every hand form and every object in the fist a mesh on that bone)."""
    kids, out = rig.children(), []
    for side, arm in body.arms.items():
        if not arm:                                 # no arms (plans.arm_chains): nothing held
            continue
        path = list(arm)
        fist, hand_bones = rig.heads[path[-1]], [path[-1]] + _below(kids, path[-1])
        branches = []
        for b in (k for a in path for k in kids[a] if k not in path):
            sub = [b] + _below(kids, b)
            w = rig.weights[:, sub].sum(axis=1)
            hand = rig.weights[:, [x for x in hand_bones if x not in sub]].sum(axis=1) > 0.5
            if not (w > 0.5).any() or (w[w > 0.5] >= RIGID).mean() < ALONE or not hand.any():
                continue
            reach = np.linalg.norm(rig.points[hand] - fist, axis=1).max()
            if (np.linalg.norm(rig.heads[b] - fist) <= reach
                    and _long_side(rig.points[w > 0.5]) >= HELD * _long_side(rig.points[hand])):
                branches.append((b, sub, w > 0.5, f'{side}_hand'))
        out += branches
        drawn = _on_the_hand(rig, path[-1], hand_bones, [p[2] for p in branches])
        if drawn is not None:
            out.append((path[-1], [], drawn, f'{side}_hand'))
    return out


def _on_the_hand(rig, hand, hand_bones, held=()):
    """The vertices of the meshes of the file riding the hand's own bone ``hand`` alone (every vertex, RIGID) that are
    HELD times the hand or more, together: an object in the fist the file drew on the hand's bone, one prop (a
    shield's face and its back plate); None when there is none. The hand here is its skin: what that bone and those
    below it (``hand_bones``) carry outside such meshes and outside the props ``held`` off it. A hand with no skin of
    its own holds none. On tlink, after the everyday choice: his shield 3.1 times his right glove, his fists 0.6, his
    sword's hilt 1.1. The pieces welded to those are the object's too (a blade's guard: WELD). Such a prop has no bones
    of its own, so the hand's
    bone stays the hand's and its vertices ride the slot that takes it alone (Match.whole); and the size can't tell a
    fist from a bracelet half its size, so only a slot of the kind it is named takes it (_match_props)."""
    if rig.counts is None:
        return None
    pieces = _alone_on(rig, hand)
    alone = np.zeros(len(rig.points), bool)
    for s, e in pieces:
        alone[s:e] = True
    skin = (rig.weights[:, hand_bones].sum(axis=1) > 0.5) & ~alone
    for sel in held:
        skin &= ~sel
    if not pieces or not skin.any():
        return None
    size, drawn = _long_side(rig.points[skin]), np.zeros(len(rig.points), bool)
    for s, e in pieces:
        if _long_side(rig.points[s:e]) >= HELD * size:
            drawn[s:e] = True
    grown = drawn.any()
    while grown:                                    # with the pieces welded to it: tlink's blade's guard
        grown = False
        for s, e in pieces:
            if not drawn[s] and _welded(rig.points[s:e], rig.points[drawn], WELD * size):
                drawn[s:e] = grown = True
    return drawn if drawn.any() else None


def _alone_on(rig, bone):
    """The meshes of the file riding ``bone`` alone (every vertex, RIGID), as (start, end) rows."""
    starts = np.cumsum([0] + list(rig.counts))
    return [(s, e) for s, e in zip(starts[:-1], starts[1:]) if e > s and (rig.weights[s:e, bone] >= RIGID).all()]


def _welded(a, b, tol):
    """Whether points ``a`` and ``b`` share a point (within ``tol``)."""
    for i in range(0, len(a), 256):
        if (np.linalg.norm(a[i:i + 256, None] - b[None], axis=2) <= tol).any():
            return True
    return False


def unheld(rig, body, kinds, slots=()):
    """The meshes, by name, a fighter holding a weapon of each kind doesn't show once its props were asked what they
    are (``kinds``: {root bone name: kind, or None for no weapon}, labels.equipment). A hand whose slot takes an object
    the file drew on the hand's own bone (_on_the_hand: its big pieces and those welded to them) holds it alone: the
    file drew every other move's object there too (Ocarina of Time, a display list each: tlink's hammer, bomb and
    hilt beside his blade, his ocarina beside his shield), and they go, as the vanilla weapon's other forms go. The
    hand itself, skinned off that bone too, stays. A prop no slot of the fighter's (``slots``) takes is no weapon,
    whatever it was named (the game puts in a hand only the one its slot takes, _takes): it rides where the file lays
    it (_match_props); one on bones all left at the file's origin, its root's place, is laid nowhere, the game's to
    place by animation (tlink's pole, from the floor between his feet up through him, named a rod once and a sword
    once), and goes."""
    if rig.counts is None or not kinds:
        return []
    gone = np.zeros(len(rig.points), bool)
    found = props(rig, body)
    taken = {p[0] for _, p in _takes(rig, found, armed(slots), kinds)}
    for b, sub, sel, _, held in found:
        name = rig.names[b]
        if held and not sub and kinds.get(name):
            for s, e in _alone_on(rig, b):
                gone[s:e] |= ~sel[s:e]
        elif not held and name in kinds and b not in taken:
            top = b
            while rig.parents[top] >= 0:
                top = rig.parents[top]
            if np.allclose(rig.heads[sub], rig.heads[top], atol=WELD * _long_side(rig.points)):
                gone |= sel
    starts = np.cumsum([0] + list(rig.counts))
    return [n for n, s, e in zip(rig.meshes, starts[:-1], starts[1:]) if e > s and gone[s:e].all()]


def empty_hands(rig, body):
    """The meshes, by name, a fighter that holds nothing through play doesn't show: those all on a prop it holds, on
    bones of its own (held_props). It would be in its hand in every move (tmario's blasters, three bones each off his
    hands). A mesh only partly on the prop stays. The run no longer drops them by this: the look is told the fighter
    holds nothing and says what goes (look.EMPTY, 2026-11-18)."""
    if rig.counts is None:
        return []
    gone = np.zeros(len(rig.points), bool)
    for _, sub, sel, _ in held_props(rig, body):
        if sub:
            gone |= sel
    starts = np.cumsum([0] + list(rig.counts))
    return [n for n, s, e in zip(rig.meshes, starts[:-1], starts[1:]) if e > s and gone[s:e].all()]


def _long_side(points):
    """The long side of the points' own box (along their principal directions)."""
    if len(points) < 2:
        return 0.0
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return float(np.ptp((points - points.mean(axis=0)) @ vt[0]))


def _into_shield(points, slot):
    """The 3x4 taking a shield (its vertices ``points``) into a shield slot at its own size, and its (middle, long
    direction). Both are measured by their own boxes: the shield's is turned the least way onto the slot's (long side
    onto long side, face onto face: of the four turns between them, the smallest), and its middle lands on the
    slot's. Both rests are the fighter's frame, and a shield lies on its forearm there the way its game holds it
    (Link's, u-link's and BB Link's face up, the wide top toward the front), so the least turn keeps its top up; a
    fit by a grip end, as a sword's, would guess at a thing held at its middle."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    p = (points - c) @ vt.T
    lo, hi = p.min(axis=0), p.max(axis=0)
    order = np.argsort(lo - hi, kind='stable')                  # the longest side first
    axes, lo, hi = vt[order], lo[order], hi[order]
    centre = c + ((lo + hi) / 2) @ axes
    tip, flat = (np.asarray(slot[k], float) for k in ('tip_direction', 'flat'))
    to = np.array([tip, flat, np.cross(tip, flat)])
    turns = [to.T @ np.diag(signs) @ axes for signs in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
    turn = max((t for t in turns if np.linalg.det(t) > 0), key=np.trace)
    middle = np.asarray(slot['grip'], float) + tip * slot['length'] / 2
    return np.hstack([turn, (middle - turn @ centre)[:, None]]), (centre, axes[0])


def _into_slot(points, pivot, slot):
    """The 3x4 taking a prop (its vertices ``points``) into a weapon slot, and the prop's (grip, tip direction,
    length). Both are measured alike: the long side of their own box is the length, the held end is the end of it
    nearer the hand (for the prop, nearer ``pivot``, the head of the bone its author turns it about), the grip is the
    middle of that end, and the flat is the box's second side. The prop is scaled uniformly to the slot's length (the
    trails and hitboxes are sized to the vanilla weapon), turned so its tip points along the slot's and its flat lies
    on the slot's (whichever way along it is the smaller turn), and its grip lands on the slot's."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    p = (points - c) @ vt.T
    lo, hi = p.min(axis=0), p.max(axis=0)
    order = np.argsort(lo - hi, kind='stable')                  # the longest side first
    axes, lo, hi = vt[order], lo[order], hi[order]
    centre, length = c + ((lo + hi) / 2) @ axes, hi[0] - lo[0]
    held = 1.0 if (pivot - centre) @ axes[0] > 0 else -1.0
    grip, tip = centre + held * axes[0] * length / 2, -held * axes[0]
    to_tip, to_flat = (np.asarray(slot[k], float) for k in ('tip_direction', 'flat'))

    def frame(x, y):
        x = x / np.linalg.norm(x)
        y = y - (y @ x) * x
        y = y / np.linalg.norm(y)
        return np.column_stack([x, y, np.cross(x, y)])
    turn = max((frame(to_tip, to_flat) @ frame(tip, s * axes[1]).T for s in (1, -1)), key=np.trace)
    m = slot['length'] / length * turn
    return np.hstack([m, (np.asarray(slot['grip']) - m @ grip)[:, None]]), (grip, tip, length)


def _regions(fch):
    """Fighter main bones by region: the head, the lower body (the first two spine bones), the upper body (the rest of
    the spine, and the clavicles: an arm's bones before its shoulder joint, which a cape or a strap hangs from), the
    arms, the legs."""
    spine = fch['spine'].bones
    out = {b: 'lower' for b in spine[:2]}
    out.update({b: 'upper' for b in spine[2:-1]})
    out[spine[-1]] = 'head'
    for side in ('left', 'right'):
        arm = fch[f'{side} arm']
        out.update({b: 'upper' if i < arm.anchors[1] else 'arm' for i, b in enumerate(arm.bones)})
        out.update({b: 'leg' for b in fch[f'{side} leg'].bones})
    return out


def _place(f, m):
    """Each fighter joint's place in the model chain: (model joint index or None, position)."""
    t_f, _ = f.fractions()
    t_m, pts = m.fractions()
    n = len(f.bones)
    out = [None] * n
    fixed = {0: 0}
    for fa, ma in zip(f.anchors, m.anchors):        # a later pair for the same fighter joint wins (no clavicle)
        if fa is not None and ma is not None:
            fixed[fa] = ma
    # keep the pairs in order along both chains; two fighter joints may share one model joint (a fighter clavicle
    # where the model has none: it lies at the model's shoulder, its length nothing, and the upper arm is the model's)
    for fa in sorted(fixed):
        if any(fb < fa and fixed[fb] > fixed[fa] for fb in fixed if fb in fixed) and fa != 0:
            fixed.pop(fa)
    for i, j in fixed.items():
        out[i] = (j, m.joints[j])
    keys = sorted(fixed)
    bounds = keys + [n]
    for a, b in zip(bounds, bounds[1:]):
        ja = fixed[a]
        jb = fixed[b] if b in fixed else len(m.bones)
        fa, fb_ = t_f[a], t_f[b] if b < n else 1.0
        ma, mb_ = t_m[ja], t_m[jb] if jb < len(m.bones) else 1.0
        for i in range(a + 1, b):
            # where fighter joint i falls between its anchors, mapped onto the model's same stretch
            u = (t_f[i] - fa) / ((fb_ - fa) or 1.0)
            target = ma + u * (mb_ - ma)
            prev = out[i - 1][0] if out[i - 1][0] is not None else ja
            cands = [j for j in range(prev + 1, jb) if j not in fixed.values()]
            best = min(cands, key=lambda j: abs(t_m[j] - target), default=None)
            if best is not None and abs(t_m[best] - target) <= INTERPOLATE:
                out[i] = (best, m.joints[best])
            else:
                out[i] = (None, _at(t_m, pts, target))
    return out


CUT = 1e-3              # a fighter joint this near a model joint (a share of the chain) is at it


def _at_joint(rig, body, owner, splits):
    """The cuts of the other bones at a cut bone's joint (body.groups): a joint carries their skin together
    (body._Mass), so each is cut where that bone is, riding as it does, on its chain with itself at that joint. Left to
    hang below it (_rest_of_tree), a carnival skeleton's ribcage, skinned to a torso bone whose joint is its pelvis's,
    rode the fighter's hips whole, and its arms came away from it as the fighter's chest leaned."""
    carried = rig.carried()
    out = []
    for mb, chain, cuts in splits:
        for k in body.groups.get(mb, ()):
            if k != mb and k not in owner and carried[k] > 0:
                owner[k] = owner[mb]
                out.append((k, replace(chain, bones=[k if b == mb else b for b in chain.bones]), cuts))
    return out


def _whole(rig, frig, owner, paired, splits, notes):
    """``splits`` without the rigid pieces that ride whole. A rigid piece (every vertex it carries, it carries alone:
    a blade an arm ends in, a limb built of one stiff part) keeps the shape its author built: it rides whole the
    fighter bone paired with its own joint (``paired``), or else the one its joint lies in, unless that parts it from
    the next piece on its chain. Two pieces stay together only where both turn about the place they meet, the next
    piece's own joint. So it parts them when the next piece's fighter bone is not that bone or one next to it
    (_next_to: Leon's torso, with the fighter's chest and neck inside it, and his head beyond), and when it is the one
    next to it but turns about a joint inside this piece, not at its end (one of its ``cuts``, _turns_inside): the
    floor puts Mewtwo's ankle, 3.6 up his long foot, 2.6 up a shin whose own ankle is at 1.0 (_place_ankle), and
    Frieza's shin, whole on the fighter's, hung that far past the joint his foot turned about; the two came 3.5 apart
    at the ankle in the idle (the app's import of 2026-11-18). Then the piece bends at the fighter joints inside it
    as a skinned bone does, so the next piece stays on it: the shin below the fighter's ankle goes with the foot. The
    last piece of a chain has none beyond it to keep (Scyther's blades, Kritter's one-piece arms)."""
    kids, carried = rig.children(), rig.carried()
    out = []
    for mb, chain, cuts in reversed(splits):        # down each chain from its far end: the next piece's bone is settled
        on = rig.weights[:, mb] > 0
        if not on.any() or not (np.count_nonzero(rig.weights[on], axis=1) == 1).all():
            out.append((mb, chain, cuts))
            continue
        whole = paired.get(mb, owner[mb])
        j = chain.bones.index(mb)
        nxt = next((b for b in chain.bones[j + 1:] if carried[[b] + _below(kids, b)].sum() > 0), None)
        if nxt is None or nxt not in owner or (_next_to(frig, whole, owner[nxt])
                                               and not _turns_inside(frig, whole, owner[nxt], cuts)):
            owner[mb] = whole
            continue
        out.append((mb, chain, cuts))
        notes.append(f'rigid {rig.names[mb]} bends at {", ".join(frig.names[fb] for _, fb in cuts)}: '
                     f'{rig.names[nxt]} beyond it rides {frig.names[owner[nxt]]}')
    return out[::-1]


def _next_to(frig, a, b):
    """Whether fighter bone ``b`` is ``a`` or hangs right below it: its child, or below it only through helpers
    turning at ``b``'s own joint (Melee's leg helpers), which move nothing there."""
    near = SAME_PLACE * np.ptp(frig.heads, axis=0).max()
    if b == a:
        return True
    p = frig.parents[b]
    while p >= 0 and p != a:
        if np.linalg.norm(frig.heads[p] - frig.heads[b]) > near:
            return False
        p = frig.parents[p]
    return p == a


def _turns_inside(frig, a, b, cuts):
    """Whether fighter bone ``b``, next to ``a`` (_next_to), turns about a joint inside the piece riding ``a``: one of
    the piece's ``cuts`` ([(share, fighter bone)], _cuts) is at ``b``'s joint, ``b``'s own or a helper's between the
    two there. A joint at the piece's end is no cut (CUT), and ``a`` itself turns about none of them."""
    at = set()
    while b >= 0 and b != a:
        at.add(b)
        b = frig.parents[b]
    return any(fb in at for _, fb in cuts)


def _carrier(owner, cut_at, mb, p):
    """The fighter bone carrying model bone ``mb`` at the point ``p``: where fighter joints cut it (``cut_at``, its
    splits entry), the one whose stretch of its chain ``p`` lies on, within the bone's own stretch (repose.split)."""
    if mb not in cut_at:
        return owner[mb]
    _, chain, cuts = cut_at[mb]
    t_m, _ = chain.fractions()
    j = chain.bones.index(mb)
    t = float(np.clip(_fraction_of(chain, np.asarray(p, float)), t_m[j], t_m[j + 1]))
    return next((fb for at, fb in reversed(cuts) if at <= t), owner[mb])


def _apart_at(rig, frig, owner, at, placed=(), through=None, meets=None):
    """The neighbours that part: two model bones whose mesh meets, on fighter bones where they meet (``at``:
    _carrier) that are neither one bone nor next to each other either way round (a model may hang its trunk the other
    way up from the fighter's: Eevee's hips from its chest), unless the model spans that seam itself (_spanned: a
    clavicle carrying no skin, the chest's running on to the upper arm). Bones meet where the surface runs on from the
    one's skin to the other's (an edge between vertices each carries most of, met at those edges' middle), whatever the tree says
    (a flat rig hangs every bone off its root); a bone whose mesh runs on to no other's is a piece, meeting the
    nearest bone above it that carries mesh at its own joint. A prop given its place outright (``placed``: a weapon
    slot, an item joint) parts by design. An end placed by a bone with no length inside the model (``through``: a
    neckless head, by the fighter's neck) meets what it sits on there, parting only as far as it turns about its own
    joint, a neck's length on. [(model bone, model bone it meets, fighter bone, fighter bone)]; ``meets``, a list
    given, gets where each pair meets, in order."""
    carried = rig.carried()
    pairs = {}
    edges = np.asarray(getattr(rig, 'edges', None) if getattr(rig, 'edges', None) is not None else [], int)
    if edges.size:
        weighted = rig.weights.sum(axis=1) > 0
        most = np.argmax(rig.weights, axis=1)
        e = edges.reshape(-1, 2)
        e = e[weighted[e].all(axis=1) & (most[e[:, 0]] != most[e[:, 1]])]
        ends = np.sort(most[e], axis=1)
        keys, inverse = np.unique(ends, axis=0, return_inverse=True)
        middles = rig.points[e].mean(axis=1)
        for k, (x, y) in enumerate(keys):
            pairs[(int(x), int(y))] = middles[inverse.ravel() == k].mean(axis=0)
    met = {b for pair in pairs for b in pair}
    depth = _depths(rig.parents)
    for b in range(len(rig.names)):
        if carried[b] <= 0 or b in met:
            continue
        a = next((x for x in _ancestors(rig.parents, b)[1:] if carried[x] > 0), None)
        if a is not None:
            pairs[(a, b)] = rig.heads[b]
    out = []
    for (x, y), p in sorted(pairs.items()):
        if x not in owner or y not in owner:
            continue
        b, a = (x, y) if depth[x] > depth[y] else (y, x)
        fa, fb = at(a, p), at(b, p)
        if fa in placed or fb in placed:
            continue
        fa, fb = (through or {}).get(fa, fa), (through or {}).get(fb, fb)
        if not (_next_to(frig, fa, fb) or _next_to(frig, fb, fa) or _spanned(rig, frig, owner, a, b, fa, fb)):
            out.append((rig.names[b], rig.names[a], frig.names[fb], frig.names[fa]))
            if meets is not None:
                meets.append(np.asarray(p, float))
    return out


def _spanned(rig, frig, owner, a, b, fa, fb):
    """Whether model bones ``a`` and ``b``, whose skins meet on fighter bones ``fa`` and ``fb``, span that seam in the
    model's own tree: every fighter bone between ``fa`` and ``fb`` (a helper at its neighbour's joint aside) carries a
    model bone between ``a`` and ``b``. Then the seam bends as the model's own does, across the bones between: a
    clavicle carrying no skin, the chest's running on to the upper arm (u-link, Marth Ultimate, Inkling), is how the
    game's own costumes are built (vanilla Falcon's clavicle carries 12 weight units of skin to his chest's 387). A
    model with no bone there (Kritter's arm off his chest) leaves the fighter's clavicle carrying nothing: that seam
    parts."""
    try:
        model = body_reading._between(rig.parents, a, b)[1:-1]
        fighter = body_reading._between(frig.parents, fa, fb)
    except StopIteration:               # in trees of their own: nothing spans them
        return False
    near = SAME_PLACE * np.ptp(frig.heads, axis=0).max()
    joints = [f for i, f in enumerate(fighter[1:-1], 1)
              if min(np.linalg.norm(frig.heads[f] - frig.heads[fighter[i - 1]]),
                     np.linalg.norm(frig.heads[f] - frig.heads[fighter[i + 1]])) > near]
    carried = {owner.get(x) for x in model}
    return bool(joints) and all(f in carried for f in joints)


def _parts(rig, m, groups, trunk=()):
    """The chain ``m`` as its parts: the joints that carry mesh, each from its own joint to the next one's. A joint
    carries what all its bones do (``groups``, body.groups; _at_joint cuts them alike): a Colosseum rip's skin is on the
    second bone at each joint, and read off the first, the fighter's waist and chest cut nothing and its torso rode the
    hips whole. A joint carrying none is no part; the stretch it spans is the one before it's, whose skin reaches over
    it (Spider-Man's two neck bones carry nothing: his neck is skinned to his chest and head, so the fighter's neck cuts
    the chest's). A bone of the ``trunk`` is no part of a limb reaching through it (plans.reach_hips): the body is the
    trunk's, not a leg's to cut. None when no joint of it carries mesh."""
    carried = rig.carried()
    keep = [j for j, b in enumerate(m.bones) if carried[groups.get(b, [b])].sum() > 0 and b not in trunk]
    if not keep:
        return None
    return Chain([m.bones[j] for j in keep], m.joints[keep], m.end, level=m.level)


def _cuts(m, heads, fbones):
    """Where fighter joints (``heads``, inside the model, of ``fbones``) fall inside the model chain ``m``'s bones, not
    at its joints: one placed between them. There the model's skin has no
    seam, and a bone of it would bend whole about a joint inside it: the fighter's joint bends the limb where its mesh
    has none, and its seam bends where the fighter has no joint. Returns [(model bone, m, [(share along m, fighter
    bone from there on), ...])] for repose.split, which cuts the bone's mesh at the fighter's joints."""
    t_m, _ = m.fractions()
    inside = {}
    for fb, p in zip(fbones, heads):
        t = _fraction_of(m, np.asarray(p, float))
        j = int(np.searchsorted(t_m, t, side='right')) - 1
        if j < len(m.bones) and t - t_m[j] > CUT and t_m[j + 1] - t > CUT:
            inside.setdefault(m.bones[j], []).append((t, fb))
    return [(mb, m, sorted(cuts)) for mb, cuts in inside.items()]


def _onward(chain):
    """A skirt's strand's last bone at rest goes on the way the strand comes down to it (as far as its own end lies,
    where its own hem's mesh reaches: extra_chains): the fighter's own hem flares out towards the floor (Peach's),
    which a gown's own hem would be turned onto."""
    if len(chain.joints) < 2:
        return chain.end
    d = chain.joints[-1] - chain.joints[-2]
    return chain.joints[-1] + d / (np.linalg.norm(d) or 1.0) * np.linalg.norm(chain.end - chain.joints[-1])


def _squash(fch, embedded, rest):
    """How the fighter's legs squash the model's: the height of its hip joints over the model's (the legs' places,
    match()), at most 1: a long-legged fighter stretches nothing."""
    legs = [fch[f'{side} leg'].bones[0] for side in ('left', 'right')]
    m, f = (np.mean([at[b][0][2] for b in legs]) for at in (embedded, rest))
    return float(min(1.0, f / m)) if m > 0 else 1.0


def _fit(chain, curve, squash=1.0):
    """A skirt's strand (the fighter's ``chain``) laid along the model's skirt as it hangs (``curve``, from the
    strand's root to its hem): its joints and the tail of its last bone, the skirt held as the fighter's legs hold the
    model, squashed as they are (``squash``: _squash) and never stretched. At its own distances along the curve over
    ``squash``, as far as the curve reaches (past its end, along its last stretch); a skirt that, so squashed, reaches
    past the strand's end is fitted to it: the joints at their shares of the curve, its hem at the strand's end. So a
    dress keeps its place on the legs (one to the model's mid-shin hangs to the fighter's: taerithcitric on Zelda,
    whose legs are shorter, its hem had lain at her floor) and never runs past the fighter's skirt along the floor."""
    t, pts = chain.fractions()
    own = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    length = steps.sum()
    arc = np.concatenate([[0.0], np.cumsum(steps)]) / (length or 1.0)
    at = t * max(1.0, own / ((length * squash) or 1.0))
    placed = np.array([_at(arc, curve, x) for x in at])
    return placed[:-1], placed[-1]


def _long(chain):
    """A chain's length along its joints to its end."""
    return float(np.linalg.norm(np.diff(chain.fractions()[1], axis=0), axis=1).sum())


def _rooted(mc, fc, model_hips, fighter_hips):
    """Where the fighter's first bone of the chain ``fc`` lies inside the model, and at its rest, for a model chain
    ``mc`` that rides it whole: ((head, tail) in the model, (head, tail) at rest), the two of one length, so the chain
    is only moved and turned, to start along the fighter's own. The fighter's joint lies inside its body, nearer its
    hips than the model's chain starts where the model is deeper there; the model's root then lies that much farther
    out along the fighter's bone, as far behind the hips as the model has it. (The joints given are the hips': each
    side's distance from them is read across the floor.)"""
    pts_m, pts_f = np.vstack([mc.joints, mc.end]), np.vstack([fc.joints, fc.end])
    way = pts_m[1] - pts_m[0]
    way = way / (np.linalg.norm(way) or 1.0)
    first = float(np.linalg.norm(pts_f[1] - pts_f[0]))
    out = max(0.0, float(np.linalg.norm((pts_m[0] - model_hips)[:2]) - np.linalg.norm((pts_f[0] - fighter_hips)[:2])))
    head = pts_m[0] - way * out
    return (head, head + way * first), (pts_f[0], pts_f[1])


def _place_along(f, m):
    """Each fighter joint's place in the model chain at its own distance along it, the model chain keeping its length:
    (model joint index or None, position). A joint past the model's end lies on the line of its last segment (the
    fighter bone there carries nothing)."""
    t_f, pf = f.fractions()
    t_m, pm = m.fractions()
    scale = _long(f) / (_long(m) or 1)
    out, prev = [(0, m.joints[0])], 0
    for i in range(1, len(f.bones)):
        x = t_f[i] * scale
        best = min(range(prev + 1, len(m.bones)), key=lambda j: abs(t_m[j] - x), default=None)
        if best is not None and abs(t_m[best] - x) <= INTERPOLATE:
            out.append((best, m.joints[best]))
            prev = best
        else:
            out.append((None, _at(t_m, pm, x)))
    return out


def _at(t, pts, x):
    k = int(np.clip(np.searchsorted(t, x) - 1, 0, len(t) - 2))
    u = (x - t[k]) / ((t[k + 1] - t[k]) or 1.0)
    return pts[k] + u * (pts[k + 1] - pts[k])


def _fraction_of(m, p):
    t, pts = m.fractions()
    best, at = None, 0.0
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        d = b - a
        u = float(np.clip((p - a) @ d / ((d @ d) or 1.0), 0, 1))
        dist = np.linalg.norm(a + u * d - p)
        if best is None or dist < best:
            best, at = dist, t[k] + u * (t[k + 1] - t[k])
    return at


def _chain(rig, bones, anchors=None, top=False, carriers=None):
    """A chain through ``bones``' heads to its far end: the vertex its last joint's bones (``carriers``, default the
    last bone) carry farthest from the chain's start, or, for the spine (``top``), the highest one: a hair tip must not
    tilt the head."""
    joints = rig.heads[bones]
    carriers = carriers or [bones[-1]]
    on = rig.weights[:, carriers].sum(axis=1) > 0.5     # its own vertices: not a shield, wand or finger it carries
    pts = rig.points[on] if on.any() else rig.tails[[carriers[-1]]]
    if top:
        end = pts[np.argmax(pts[:, 2])]
        end = np.array([joints[-1][0], joints[-1][1], end[2]])
    else:
        # along where its own mesh lies on average, as far as most of it reaches: spare copies far off can't bend it
        d = pts.mean(axis=0) - joints[-1]
        d = d / (np.linalg.norm(d) or 1.0)
        end = joints[-1] + d * max(np.percentile((pts - joints[-1]) @ d, 90), 1e-6)
    return Chain(list(bones), joints, end, anchors or [])


SAME_PLACE = 1e-3       # bone heads closer than this share of the model's size sit at one joint


def _apart(rig, bones):
    """``bones`` with one bone per joint, and the bones at each joint. A bone whose head sits on the head before it (a
    helper turning in place at its parent's joint, of no length) adds no joint to a chain. Counted as one, it reads as
    a clavicle (a short first segment) or as the knee (so the ankle is found at the hip), the fighter bones anchored
    there get no length inside the model, and the re-pose stretches what they carry without bound. The first bone at
    a joint stands for it; the others are below it, so they go with it (_rest_of_tree)."""
    near = SAME_PLACE * np.ptp(rig.points, axis=0).max()
    kept, at = [], {}
    for b in bones:
        if kept and np.linalg.norm(rig.heads[b] - rig.heads[kept[-1]]) <= near:
            at[kept[-1]].append(b)
        else:
            kept.append(b)
            at[b] = [b]
    return kept, at


def _forward_toe(rig, body, leg):
    """The leg down to its lowest bone, except where the foot branches (a heel and a toe): there, the branch that
    reaches farthest forward. The lowest bone can be a heel. A toe stands on the floor as the heel does: a branch whose
    mesh stops short of where the leg's own comes down to (a plate off the knee, Aigis's) is no toe; where none comes
    down as low (Pidgeot's curled foot, its pad the lowest), the leg ends at that joint. A fork the leg goes on through
    (body.forks: a skirt panel off the shin) is above its foot: the foot below it doesn't branch. A joint branches
    from all its bones (body.groups: tmario's ankle is two bones at one place, the second carrying the foot's four)."""
    kids = rig.children()
    carried = rig.carried()
    z = rig.points @ body.up

    def low(c):                 # how far down a branch's mesh comes
        on = rig.weights[:, [c] + _below(kids, c)].sum(axis=1) > 0
        return z[on].min() if on.any() else np.inf
    for k in range(len(leg) - 1, 1, -1):
        joint = body.groups.get(leg[k], [leg[k]])
        branches = [c for b in joint for c in kids[b] if c not in joint and carried[[c] + _below(kids, c)].sum() > 0]
        if len(branches) > 1 and body.forks.get(leg[k]) is not None:
            return leg
        if len(branches) > 1:
            down = low(leg[min(k + 1, len(leg) - 1)]) + SAME_PLACE * np.ptp(z)
            tips = [b for c in branches if low(c) <= down for b in [c] + _below(kids, c)]
            if not tips:
                return leg[:k + 1]
            tip = max(tips, key=lambda b: rig.heads[b] @ body.front)
            return leg[:k + 1] + _path_down(rig.parents, leg[k], tip)
    return leg


def _rest_of_tree(rig, owner, fallback, at=None):
    """Every other weighted bone rides the fighter bone carrying its nearest ancestor that has an owner at the point it
    hangs from (``at``: _carrier; a cape off a torso cut at the fighter's chest joints rides the stretch it hangs on);
    one with none, the bone carrying the nearest owned bone where its vertices are."""
    at = at or (lambda mb, p: owner[mb])
    carried = rig.carried()
    owned = list(owner)
    centres = {}
    for b in range(len(rig.names)):
        if b in owner or carried[b] <= 0:
            continue
        a = next((x for x in _ancestors(rig.parents, b)[1:] if x in owner), None)
        if a is not None:
            owner[b] = at(a, rig.heads[b])
            continue
        sel = rig.weights[:, b] > 0
        c = (rig.weights[sel, b:b + 1] * rig.points[sel]).sum(0) / rig.weights[sel, b].sum()
        near = min(owned, key=lambda x: np.linalg.norm(rig.heads[x] - c)) if owned else None
        owner[b] = at(near, c) if near is not None else fallback


def _depths(parents):
    return [len(_ancestors(parents, b)) for b in range(len(parents))]


def _ancestors(parents, b):
    out = []
    while b >= 0:
        out.append(b)
        b = parents[b]
    return out


def _below(kids, b):
    out, todo = [], list(kids[b])
    while todo:
        x = todo.pop(0)
        out.append(x)
        todo.extend(kids[x])
    return out


def _path_down(parents, top, b):
    chain = _ancestors(parents, b)
    return chain[:chain.index(top)][::-1]


def _shared_prefix(a, b):
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def read_description(path):
    return json.loads(open(path, encoding='utf-8').read())
