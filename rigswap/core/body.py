"""Read one body: mass, mirror plane, twins, connected trunk and chains; its limbs from the roles labels.py asks for.

The plan names the structure found by that reading. It never selects another reader.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import json

import numpy as np


@dataclass
class Body:
    up: np.ndarray
    front: np.ndarray
    left: np.ndarray
    root: int           # the body's top: the legs and the head both hang from it
    spine: list         # from the hips, the joint the legs part from, to the head
    skull: int
    legs: dict
    arms: dict
    chest: int
    floor: float
    plan: str = 'biped'
    forks: dict = field(default_factory=dict)   # fork bone: the branch its limb goes on along, None where it ends
    groups: dict = field(default_factory=dict)
    record: dict = None     # rec: the anchors a hand record names, as bones ({'head': b, 'legs': {side: {...}}, ...})


EXACT = 1e-4           # file precision for positions and edge lengths, as a share of model size
FLOOR = 0.02            # surface contact, as a share of the body's height
ORIGIN = 0.1            # a file stands on its origin: its lowest point this share of its height from it, or less
ROLES = ('legs', 'front legs', 'arms')      # what a twin pair can be to the body; anything else is an attachment
RECORD = None   # rec: this model's body read by hand (record.py, <load>/record.json); run.swap sets it


class _Mass:
    """The heaviest root's tree is the body; empty subtrees are controls.

    Adjacent heads within EXACT are one joint, represented by their highest bone.
    Their skin is counted together without changing the caller's rig. The body's
    middle is where it first branches: bones above that only carry it whole (a
    rip's origin empty, a master control), and a pose can leave them beside it.
    It is a joint carrying skin of its own: one carrying none has nothing on the
    body saying where it is (Meta Knight's hip, below his feet and a third of
    his size to one side), so past it the trunk goes on into its heaviest branch.
    """

    def __init__(self, rig):
        self.rig = rig
        own, kids = rig.carried(), rig.children()
        descendants = {}

        def visit(b):
            descendants[b] = [b]
            for k in kids[b]:
                descendants[b].extend(visit(k))
            return descendants[b]

        roots = np.flatnonzero(rig.parents < 0)
        for b in roots:
            visit(int(b))
        self.top = int(max(roots, key=lambda b: own[descendants[b]].sum()))
        active = [b for b in descendants[self.top] if own[descendants[b]].sum() > 0]
        if not active:
            raise ValueError('no tree carries mesh')
        surface = rig.weights[:, descendants[self.top]].sum(axis=1) > 0
        self.size = float(np.ptp(rig.points[surface], axis=0).max()) or 1.0
        self.parents, self.groups = rig.parents.copy(), {}
        representatives = {}
        for b in active:  # parents precede their children in the traversal
            p = int(rig.parents[b])
            same = p in representatives and np.linalg.norm(rig.heads[b] - rig.heads[p]) <= (
                EXACT * self.size)
            representative = representatives[p] if same else b
            representatives[b] = representative
            self.groups.setdefault(representative, []).append(b)
            if representative == b:
                self.parents[b] = representatives.get(p, -1)
        self.bones = set(self.groups)
        self.kids = {b: [] for b in self.bones}
        for b in self.bones:
            if self.parents[b] in self.bones:
                self.kids[self.parents[b]].append(b)
        self.own = np.zeros(len(own))
        self.total, self.sub = {}, {}
        for b, group in self.groups.items():
            self.own[b] = own[group].sum()
            self.sub[b] = [k for k in self.bones if _descends(self.parents, k, b)]
            self.total[b] = float(own[[k for j in self.sub[b] for k in self.groups[j]]].sum())
        middle = self.top
        while self.kids[middle] and (len(self.kids[middle]) == 1 or not self.own[middle]):
            middle = max(self.kids[middle], key=lambda k: self.total[k])
        self.middle = rig.heads[middle]
        self.centres = {b: self.weights([b]) @ rig.points / self.own[b] if self.own[b] else rig.heads[b]
                        for b in self.bones}

    def weights(self, bones):
        """A joint carries the combined skin of the bones represented by it."""
        return self.rig.weights[:, [k for b in bones for k in self.groups[b]]].sum(axis=1)


def _mirror_distances(heads, origin, axis):
    """Positional axis tie votes use mutual nearest mirror images.

    Both bones must satisfy the strict inequality. A midline bone is its own
    mirror, so it cannot pair; exact narrow twins need no distance tolerance.
    """
    side = (heads - origin) @ axis
    reflected = heads - 2 * side[:, None] * axis
    distance = np.linalg.norm(reflected[:, None, :] - heads[None, :, :], axis=2)
    nearest = distance.argmin(axis=1)
    mutual = (nearest[:, None] == np.arange(len(heads))) & (nearest[None, :] == np.arange(len(heads))[:, None])
    own_mirror = 2 * np.abs(side)
    return np.where(mutual & (distance < np.minimum(own_mirror[:, None], own_mirror[None, :])), distance, np.inf)


def _assignment(cost):
    """Minimum-cost one-to-one child assignment, by augmenting shortest paths.

    Rectangular Hungarian assignment keeps forks polynomial, including hands
    with many fingers. The smaller side supplies the rows.
    """
    if cost.shape[0] > cost.shape[1]:
        return [(j, i) for i, j in _assignment(cost.T)]
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, via = np.zeros(m + 1, int), np.zeros(m + 1, int)
    for row in range(1, n + 1):
        owner[0], column = row, 0
        distance, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
        while True:
            used[column] = True
            current = owner[column]
            for j in range(1, m + 1):
                if not used[j]:
                    step = cost[current - 1, j - 1] - u[current] - v[j]
                    if step < distance[j]:
                        distance[j], via[j] = step, column
            next_column = min((j for j in range(1, m + 1) if not used[j]), key=lambda j: distance[j])
            delta = distance[next_column]
            u[owner[used]] += delta
            v[used] -= delta
            distance[~used] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = via[column]
            owner[column] = owner[previous]
            column = previous
    return [(int(owner[j] - 1), j - 1) for j in range(1, m + 1) if owner[j]]


def _shared(mass, axis=None):
    """Copies share parent-to-head edges within the file's precision, through forks.

    Children match one-to-one to maximize the total shared edges. Unmatched
    props do not break a copy; unequal roots can still share a child's edge.
    Only position breaks equal counts, never sub-EXACT length differences.
    """
    heads = mass.rig.heads
    edges = {b: np.linalg.norm(heads[b] - heads[mass.parents[b]])
             for b in mass.bones if mass.parents[b] in mass.bones}
    images = heads if axis is None else heads - 2 * ((heads - heads[mass.top]) @ axis)[:, None] * axis
    distances = np.linalg.norm(images[:, None] - heads[None], axis=2)

    @lru_cache(maxsize=None)
    def shared(a, b):
        count = int(a in edges and b in edges and abs(edges[a] - edges[b]) <= EXACT * mass.size)
        pairs = []
        left, right = mass.kids[a], mass.kids[b]
        if left and right:
            counts = np.array([[shared(x, y)[0] for y in right] for x in left])
            places = distances[np.ix_(left, right)]
            # Across a whole assignment, position costs less than one edge.
            cost = -counts + places / ((places.max() + mass.size) * (min(len(left), len(right)) + 1))
            for i, j in _assignment(cost):
                c, matched = shared(left[i], right[j])
                if c:
                    count += c
                    pairs.extend(matched)
        return count, ((a, b), *pairs) if count else ()
    return shared, distances


def _copies(mass, axis=None, matching=None):
    """Outermost subtrees sharing the most edges pair; equal copies use position.

    Across a known plane, EXACT separates either side from its midline, which
    runs through the body's middle (where it first branches), not the
    armature's root: a rip can leave that an empty at the world origin, beside
    the body (Tauros, posed charging, his pelvis 1.6 to one side of it). Before
    the axis is known, candidates need only be on different tree paths.
    A matched descendant is part of its outer copy, not another limb vote.
    """
    shared, distances = matching or _shared(mass, axis)
    heads, parents = mass.rig.heads, mass.parents
    bones = sorted(mass.bones - {mass.top})
    side = (heads - mass.middle) @ axis if axis is not None else None
    candidates, nested = [], set()
    for i, a in enumerate(bones):
        for b in bones[i + 1:]:
            if (_descends(parents, a, b) or _descends(parents, b, a)
                    or (side is not None and not (min(side[a], side[b]) < -EXACT * mass.size
                                                 and max(side[a], side[b]) > EXACT * mass.size))):
                continue
            count, pairs = shared(a, b)
            if count:
                candidates.append((-count, distances[a, b], a, b))
                nested.update(tuple(sorted(p)) for p in pairs[1:])
    roots, inside = [], set()
    for _, _, a, b in sorted(candidates):
        if ((a, b) not in nested and not inside.intersection(mass.sub[a])
                and not inside.intersection(mass.sub[b])):
            roots.append((a, b))
            inside.update(mass.sub[a])
            inside.update(mass.sub[b])
    return roots


def _copy_steps(mass):
    """A copy votes at its outermost matched pair whose own edges agree.

    Unequal roots only carry a descendant copy; their step says nothing about
    the mirror. A step within EXACT uses the copy's carried-mesh centres instead.
    Twins lie on either side of the body's middle: a pair on one side of it
    along their own step (Bart's eyelids, top and bottom, both above his
    middle) are copies but no mirror's, and cast no vote.
    """
    shared, _ = matching = _shared(mass)
    heads, parents = mass.rig.heads, mass.parents
    edges = {b: np.linalg.norm(heads[b] - heads[parents[b]])
             for b in mass.bones if parents[b] in mass.bones}
    steps = []
    for a, b in _copies(mass, matching=matching):
        equal = [(x, y) for x, y in shared(a, b)[1]
                 if x in edges and y in edges and abs(edges[x] - edges[y]) <= EXACT * mass.size]
        x, y = min(equal, key=lambda pair: len(_path(parents, a, pair[0])))
        ends = heads[[x, y]]
        if np.linalg.norm(ends[0] - ends[1]) <= EXACT * mass.size:
            ends = np.array([mass.weights(mass.sub[a]) @ mass.rig.points / mass.total[a],
                             mass.weights(mass.sub[b]) @ mass.rig.points / mass.total[b]])
        step = ends[0] - ends[1]
        length = np.linalg.norm(step)
        side = (ends - mass.middle) @ step / length if length else np.zeros(2)
        if side[0] > 0 > side[1]:
            steps.append((mass.total[a] + mass.total[b]) * np.abs(step) / length)
    return np.sum(steps, axis=0) if steps else np.zeros(3)


def _mirror(mass, up=None):
    """The mirror axis runs across copied limbs, weighted by their carried mesh.

    Bone-head positional pairing only breaks ties between axes the copies do
    not separate. With up given, only the other world axes can be lateral.
    """
    axes = np.eye(3)
    if up is not None:
        axes = np.delete(axes, np.argmax(np.abs(up)), axis=0)
    bones = sorted(mass.bones)
    heads = mass.rig.heads[bones]

    def paired(axis):
        distance = _mirror_distances(heads, mass.rig.heads[mass.top], axis)
        near = distance.argmin(axis=1)
        return sum(mass.total[b] for i, b in enumerate(bones)
                   if near[near[i]] == i and np.isfinite(distance[i, near[i]]))

    across = _copy_steps(mass)
    return max(axes, key=lambda axis: (across @ axis, paired(axis)))


def symmetry_axis(rig):
    """Copied limbs establish left-to-right before up is known."""
    return _mirror(_Mass(rig))


def mirror_axis(rig, up):
    """With up given, the mirror normal is one of the other world axes."""
    return _mirror(_Mass(rig), np.asarray(up, float))


def _twins(mass, axis):
    """A built skeleton states its twins; otherwise every matched bone is a twin.

    Stated links follow coincident joints onto their representatives. A piece
    that is its own twin belongs to the midline, not to a limb pair.
    Across the known plane, unpaired heads mirrored within EXACT are twins
    even when their parents' positions make the incoming edges unequal.
    A rig placed by hand is only near-symmetric, so twins then grow outward along the tree: two joints hanging from one
    joint, or from twins, are twins when each is the other's nearest mirror image among all the body's joints
    (_mirror_distances), missing it by less than either lies from any other joint. A carnival skeleton's arm bones
    differ by up to 0.6% of its height from side to side; paired within file precision only, its arms stayed in the
    trunk and a hand, carrying more skin than the skull, was read as the head. Hanging from unrelated joints, a
    skirt strand and a brooch near each other's mirror images are no pair.
    """
    if mass.rig.twins is not None:
        representative = {k: b for b, group in mass.groups.items() for k in group}
        return {representative[a]: representative[b] for a, b in mass.rig.twins.items()
                if a in representative and b in representative and representative[a] != representative[b]}
    matching = _shared(mass, axis)
    twins = {}
    for a, b in _copies(mass, axis, matching):
        for x, y in matching[0](a, b)[1]:
            twins[x], twins[y] = y, x
    heads = mass.rig.heads
    bones = sorted(mass.bones - set(twins))
    side = (heads[bones] - heads[mass.top]) @ axis
    images = heads[bones] - 2 * side[:, None] * axis
    near = np.linalg.norm(images[:, None] - heads[bones][None], axis=2) <= EXACT * mass.size
    for i, j in zip(*np.nonzero(near)):
        if (side[i] < -EXACT * mass.size and side[j] > EXACT * mass.size
                and bones[i] not in twins and bones[j] not in twins):
            twins[bones[i]], twins[bones[j]] = bones[j], bones[i]
    bones = sorted(mass.bones)
    side = (heads[bones] - heads[mass.top]) @ axis
    miss = _mirror_distances(heads[bones], heads[mass.top], axis)
    apart = np.linalg.norm(heads[bones][:, None] - heads[bones][None], axis=2)
    np.fill_diagonal(apart, np.inf)
    gap = apart.min(axis=1)
    near = [(bones[i], bones[j]) for i, j in zip(*np.nonzero(np.isfinite(miss)))
            if side[i] < -EXACT * mass.size and side[j] > EXACT * mass.size and miss[i, j] < min(gap[i], gap[j])]
    grown = True
    while grown:
        grown = False
        for a, b in near:
            p, q = mass.parents[a], mass.parents[b]
            if a not in twins and b not in twins and (p == q or twins.get(p, -1) == q):
                twins[a], twins[b] = b, a
                grown = True
    return twins


def _trunk(mass, twins):
    """The trunk is the connected, mass-carrying subtree with no twin bones.

    A branch cannot return to the trunk through a twin limb's bone. Empty
    connecting joints remain because the mass graph includes their descendants.
    """
    trunk, todo = set(), [mass.top]
    while todo:
        b = todo.pop()
        if b not in twins:
            trunk.add(b)
            todo.extend(mass.kids[b])
    return trunk


def _limbs(mass, trunk, forks=None):
    """A limb runs out from the trunk to its hand or foot. Where its bones branch, ``forks`` says which way it goes
    (forks.ways: toward where its digits spread, by what the AI says each fork's branches are): on along one branch, or
    no farther (None: its hand or foot is there, and the fingers or toes, what it holds or wears, branch from it).
    Where there is no answer, it goes on along its heaviest branch: a limb nothing it carries outweighs.

    The other branches are attachments; neither their lengths nor their number
    chooses the limb's chain or supplies extra pairing evidence.
    """
    forks = forks or {}
    chains = {}
    for b in sorted(mass.bones - trunk):
        if mass.parents[b] not in trunk:
            continue
        chain = [b]
        while mass.kids[chain[-1]]:
            here = chain[-1]
            if len(mass.kids[here]) > 1 and here in forks:
                if forks[here] is None:
                    break
                chain.append(forks[here])
            else:
                chain.append(max(mass.kids[here], key=mass.total.get))
        chains[b] = chain
    return chains


def _pairs(mass, twins, chains):
    """Copied limb roots keep their full, independently mass-selected paths."""
    return [(a, twins[a]) for a in chains if a in twins and a < twins[a] and twins[a] in chains]


def _support(mass, chains, up):
    """Floor contact belongs to a chain's own mesh, not to a hanging attachment."""
    h = mass.rig.points @ up
    surface = mass.weights(mass.bones) > 0
    floor = float(h[surface].min())
    near = floor + FLOOR * np.ptp(h[surface])
    contact = {b: int(np.count_nonzero((mass.weights(chain) > 0) & (h <= near)))
               for b, chain in chains.items()}
    return floor, near, contact


def on_origin(rig, side):
    """The way up a model file stands, when it stands on its origin: a model is made standing on the floor at its
    origin, as a game's characters are, its lowest point at the origin's height (within ORIGIN of its height) and the
    origin inside its footprint. Of the ways across its sides (``side``), the one way it stands so; None when none
    does, or more than one (a file posed in the air, Pidgeot; one centred on its middle). 112 runs stood on their
    origin, every one right that way; the three that didn't stood wrong (Boldore on its head, auntieheart on her back,
    Grace upside down)."""
    p = np.asarray(rig.points, float)
    lateral = int(np.argmax(np.abs(side)))
    ways = []
    for axis in (a for a in range(3) if a != lateral):
        inside = all(p[:, a].min() < 0 < p[:, a].max() for a in range(3) if a != axis)
        for sign in (1, -1):
            h = sign * p[:, axis]
            if inside and abs(h.min()) <= ORIGIN * np.ptp(h):
                ways.append(sign * np.eye(3)[axis])
    return ways[0] if len(ways) == 1 else None


def standing(rig, side, legs):
    """A way up stands on the carried surfaces of twin chains at the floor: the pairs the labels ask named legs
    (``legs``, their root bones' names: legs, a four-legged body's front legs). What a pair is, its contact doesn't
    say: a crest's plumes (Pidgeot), back ornaments (auntieheart) and shoulders (Grace) reaching the floor turned those
    three upside down or on their backs over a right picture, as Waddle Dee's one-bone feet and Sceptile's legs turn a
    wrong one right."""
    mass = _Mass(rig)
    twins = _twins(mass, np.asarray(side, float))
    chains = _limbs(mass, _trunk(mass, twins))
    pairs = [(a, b) for a, b in _pairs(mass, twins, chains) if {rig.names[a], rig.names[b]} & set(legs)]
    axes = np.delete(np.eye(3), np.argmax(np.abs(side)), axis=0)
    out = []
    for axis in axes:
        for sign in (1, -1):
            up = sign * axis
            _, _, contact = _support(mass, chains, up)
            out.append((up, float(sum(min(contact[a], contact[b]) for a, b in pairs))))
    return out


def _head(mass, trunk, departures, up, feet):
    """The head carries its own skin, strictly above the arms and root.

    Nil skin is measured at float32 precision against the body's mass. If no
    skinned trunk bone is above the arms, the heaviest trunk bone is the head.
    Without arms, only a trunk with no skinned bone above root is its own head.
    """
    skinned = [b for b in trunk if mass.own[b] > np.finfo(np.float32).eps * mass.total[mass.top]]
    height = max((mass.rig.heads[b] @ up for b in departures), default=-np.inf)
    above = [b for b in skinned if mass.rig.heads[b] @ up >
             mass.rig.heads[_lca(mass.parents, feet, b)] @ up]
    candidates = [b for b in above if mass.rig.heads[b] @ up > height]
    if departures and not any(mass.rig.heads[b] @ up > height for b in skinned):
        return max(skinned, key=lambda b: mass.own[b], default=None)
    if not above and feet in skinned and mass.rig.heads[feet] @ up >= height:
        return feet
    return max(candidates, key=lambda b: mass.own[b], default=None)


def _chest(mass, spine, arms, up):
    """A one-piece trunk is both hips and chest; its span is between limb roots.

    With a spine bone between root and head, the chest is where the arms hang from: the spine joint their roots
    branch off, as the fighter's is read (correspond.fighter_chains). The joint nearest their height is often a neck
    just above them (Mario's, Link's) or, on a body with no neck, the head (Kritter's), which lifts the shoulders'
    level (correspond.levels) over the arms. Arms hung from the root itself (a rip parenting its pieces to its hips)
    say nothing of it: then it is the spine joint above the root nearest their height.
    Without arms it is just below the head, or the head when that would be root.
    """
    if arms:
        if len(spine) <= 2:
            return spine[0]
        hung = []
        for b in arms:
            while b >= 0 and b not in spine:
                b = mass.parents[b]
            hung.append(spine.index(b) if b in spine else 0)
        if max(hung):
            return spine[max(hung)]
        height = np.mean([mass.rig.heads[b] @ up for b in arms])
        return min(spine[1:], key=lambda b: abs(mass.rig.heads[b] @ up - height))
    below = spine[-2] if len(spine) > 1 else spine[0]
    return spine[-1] if below == spine[0] else below


def limb_pairs(rig, up):
    """The twin chains leaving the trunk, as ``read`` finds them: (mass, chains, pairs, twins). What each pair is,
    legs, arms or an attachment, the skeleton doesn't say: labels.py asks; which way each goes where it branches,
    forks.py."""
    mass = _Mass(rig)
    axis = _mirror(mass, np.asarray(up, float)) if rig.mirror is None else np.asarray(rig.mirror, float)
    twins = _twins(mass, axis)
    chains = _limbs(mass, _trunk(mass, twins))
    return mass, chains, _pairs(mass, twins, chains), twins


def read(rig, up=None, front=None, roles=None, forks=None):
    """Read the ordered body facts once, then name the plan the limbs describe.

    ``roles`` says what each twin pair is, by its bones' names (labels.py: the AI's answer from a picture of the
    pairs): 'legs', 'front legs' (a four-legged body's) or 'arms'; several pairs given one role are pieces of that
    limb, joined in order down it. A pair given none is an attachment, and with no roles there are no legs and no
    arms: what a pair is, the skeleton doesn't say. ``forks`` says, by name, which way a limb goes where its bones
    branch (forks.py: {fork: branch, or None where its hand or foot is}).
    """
    up = np.array([0., 0., 1.]) if up is None else np.asarray(up, float)
    mass = _Mass(rig)
    parents = mass.parents
    if RECORD is not None:      # rec: read by hand; the front is the run's
        return _from_record(rig, mass, up, np.asarray(front, float), RECORD)
    axis = _mirror(mass, up) if rig.mirror is None else np.asarray(rig.mirror, float)
    front = np.cross(axis, up) if front is None else np.asarray(front, float)
    left = np.cross(up, front)
    twins = _twins(mass, axis)
    trunk = _trunk(mass, twins)
    index = {n: b for b, n in enumerate(rig.names)}
    answered ={index[f]: None if c is None else index[c] for f, c in (forks or {}).items()
                if index.get(f) in mass.bones and (c is None or index.get(c) in mass.kids[index[f]])}
    chains = _limbs(mass, trunk, answered)
    pairs = _pairs(mass, twins, chains)
    floor, near, _ = _support(mass, chains, up)
    roles = roles or {}
    # Every pair given a role is a piece of that limb: a rig may hang each piece of it off the trunk, a pair each
    # (Boomerang Bro's thighs, shins and feet). In order out from the trunk, down a leg and out along an arm; the
    # first is the limb's root, read for the body's own landmarks, and the others go on from it.
    pieces = {role: sorted((p for p in pairs if role in (roles.get(rig.names[p[0]]), roles.get(rig.names[p[1]]))),
                           key=lambda p: abs((rig.heads[p[0]] - mass.middle) @ left) if role == 'arms'
                           else -(rig.heads[p[0]] @ up)) for role in ROLES}
    given = {role: ps[0] if ps else None for role, ps in pieces.items()}
    legs_pair, fore_pair = given['legs'], given['front legs']
    def seen(b):
        """The limb from ``b`` on from its first joint carrying skin of its own (a vertex over half on it): a bone
        that only blends the body around it (Psyduck's thighs, inside his body) is no part of the leg one sees."""
        return chains[b][next((i for i, j in enumerate(chains[b]) if (mass.weights([j]) > 0.5).any()), 0):]

    # A foot begins at the first joint carrying the floor surface. If that is
    # the first joint of the leg one sees, there is no intervening leg joint.
    # Carrying is as seen() reads it, over half: a smooth skin's tail is not
    # (Richard's leg reaches his soles at 0.0055 from a fifth of his height).
    foot_only = legs_pair is not None and all(
        np.any((mass.weights([seen(b)[0]]) > 0.5) & (rig.points @ up <= near)) for b in legs_pair)
    plan = 'quadruped' if fore_pair and legs_pair else 'legless' if legs_pair is None else \
        'ball' if foot_only else 'biped'
    feet = _lca(parents, *(chains[b][-1] for b in legs_pair)) if legs_pair else mass.top
    # a four-legged body's front legs do what arms do
    arms_role = 'front legs' if fore_pair else 'arms'
    arms_pair = given[arms_role]
    skull = _head(mass, trunk, [parents[b] for b in arms_pair] if arms_pair else [], up, feet)
    if skull is None:
        raise ValueError('no midline head above root and arm junction')
    # The spine runs from the hips, the joint the legs part from (``feet``), to the head, along the tree whichever
    # way the file's parents point: a rip roots its tree where its author did, and a segmented one can root it at the
    # chest and hang the hips piece from it (Eevee's, Leon's). The root, the body's top, joins both; coincident joints
    # already use the highest.
    root = _lca(parents, feet, skull)
    spine = _between(parents, feet, skull)
    chest = _chest(mass, spine, arms_pair, up)

    def named(role, include_hips, chain=lambda b: chains[b]):
        """The limbs keep their independently chosen chains, including unequal tips; a limb in pieces, each piece's
        after the one before it."""
        if given[role] is None:
            return {'left': [], 'right': []}
        out = {}
        for pair in pieces[role]:
            ordered = sorted(pair, key=lambda b: mass.centres[b] @ left, reverse=True)
            for side, b in zip(('left', 'right'), ordered):
                out[side] = out.get(side, _between(parents, feet, b)[:-1] if include_hips else []) + chain(b)
        return out

    # a ball stands on its feet: its legs are what one sees of them
    legs = (named('legs', False, seen) if plan == 'ball' else named('legs', True)) if legs_pair else {}
    result = Body(up, front, left, root, spine, skull, legs, named(arms_role, False), chest, floor, plan, answered,
                  mass.groups)
    # The scope report needs the actual decisions, not reconstructed readings
    # from an earlier run's input. Keep those small outputs in the normal log.
    print('body reading: ' + json.dumps(dict(
        root=rig.names[root], skull=rig.names[skull], plan=plan, chest=rig.names[chest],
        spine=[rig.names[b] for b in spine],
        legs={side: [rig.names[b] for b in chain] for side, chain in legs.items()},
        arms={side: [rig.names[b] for b in chain] for side, chain in result.arms.items()})))
    return result


def _from_record(rig, mass, up, front, rec):
    """The body a record names (record.py): a full one lists the body's chains and its marks as a run read them; a
    hand one names joints, and a chain is then the tree's path between the joints named for its two ends (a named
    bone standing for its joint, the bones at one place). The sides are the record's: no check that 'left' lies to
    the left. Body.record holds the marks as bones, for correspond.model_chains:

        {'head': the joint the head turns on,
         'arms': None (the plan draws them) or {side: {'shoulder', 'hand', 'elbow'}},
         'legs': None or {side: {'chain': its bones down to the toe, or None for the body's own, 'ankle', 'toe', 'knee'}}}
    """
    index = {n: b for b, n in enumerate(rig.names)}
    joint = {k: b for b, group in mass.groups.items() for k in group}

    def raw(name):
        if name not in index:
            raise ValueError(f'record.json names {name!r}, which is no bone of this model')
        return index[name]

    def maybe(name):
        return None if name is None else raw(name)

    def bone(name):
        if raw(name) not in joint:
            raise ValueError(f'record.json names {name!r}, which is no bone carrying any of the body')
        return joint[raw(name)]
    left = np.cross(up, front)
    surface = mass.weights(mass.bones) > 0
    floor = float((rig.points @ up)[surface].min())
    full = rec.get('body')
    if full:        # as a run wrote it: the bones themselves, in the order it read them
        said = full['marks']
        marks = {'head': raw(said['neck']),
                 'arms': None if said['arms'] is None else {
                     side: {'shoulder': raw(a['shoulder']), 'hand': maybe(a['hand']), 'elbow': maybe(a.get('elbow'))}
                     for side, a in said['arms'].items()},
                 'legs': None if said['legs'] is None else {
                     side: {'chain': [raw(n) for n in leg['chain']], 'ankle': maybe(leg['ankle']),
                            'toe': maybe(leg['toe']), 'knee': maybe(leg.get('knee'))}
                     for side, leg in said['legs'].items()}}
        return Body(up, front, left, raw(full['root']), [raw(n) for n in full['spine']], raw(full['skull']),
                    {side: [raw(n) for n in chain] for side, chain in full['legs'].items()},
                    {side: [raw(n) for n in chain] for side, chain in full['arms'].items()},
                    raw(full['chest']), floor, full['plan'],
                    {raw(f): maybe(c) for f, c in (full.get('forks') or {}).items()}, mass.groups, marks)

    def path(a, b, named=None):
        """The chain from ``a`` to ``b``: the tree's path, or the joints the record lists (``named``) where the file
        hangs a piece of a limb off another bone (a foot on a floor control beside the hips)."""
        if named:
            out = []
            for n in named:
                if not out or bone(n) != out[-1]:
                    out.append(bone(n))
            if a not in out or b not in out:
                raise ValueError(f'record.json: {named} leaves out one of its own joints')
            return out
        if not _descends(mass.parents, b, a):
            raise ValueError(f'record.json: {rig.names[b]} is not below {rig.names[a]}; list its "bones"')
        return _path(mass.parents, a, b)
    hips, skull = bone(rec['hips']), bone(rec['skull'])
    spine = _between(mass.parents, hips, skull)
    chest, head = bone(rec['chest']), bone(rec.get('neck', rec['skull']))
    if chest not in spine or head not in spine:
        raise ValueError('record.json: the chest and the head joint lie on the spine, between the hips and the skull')
    marks = {'head': head, 'legs': {}, 'arms': {}}
    legs, arms = {}, {'left': [], 'right': []}
    for side, leg in (rec.get('legs') or {}).items():
        hip, ankle = bone(leg['hip']), bone(leg['ankle'])
        end = bone(leg['toe']) if leg.get('toe') else ankle
        legs[side] = [hips] + path(hip, end, leg.get('bones'))
        marks['legs'][side] = {'chain': None, 'knee': bone(leg['knee']) if leg.get('knee') else None, 'ankle': ankle,
                               'toe': end if leg.get('toe') else None}
    for side, arm in (rec.get('arms') or {}).items():
        shoulder, hand = bone(arm['shoulder']), bone(arm['hand'])
        start = bone(arm['clavicle']) if arm.get('clavicle') else shoulder
        arms[side] = path(start, hand, arm.get('bones'))
        marks['arms'][side] = {'shoulder': shoulder, 'elbow': bone(arm['elbow']) if arm.get('elbow') else None,
                               'hand': hand}
    marks['legs'] = marks['legs'] or None
    marks['arms'] = marks['arms'] or None
    result = Body(up, front, left, _lca(mass.parents, hips, skull), spine, skull, legs, arms, chest, floor,
                  rec.get('plan', 'biped'), {}, mass.groups, marks)
    print('body record: ' + json.dumps(dict(
        root=rig.names[result.root], skull=rig.names[skull], plan=result.plan, chest=rig.names[chest],
        head=rig.names[head], spine=[rig.names[b] for b in spine],
        legs={side: [rig.names[b] for b in chain] for side, chain in legs.items()},
        arms={side: [rig.names[b] for b in chain] for side, chain in arms.items()})))
    return result


def _below_all(kids, b):
    out, todo = [], list(kids[b])
    while todo:
        x = todo.pop()
        out.append(x)
        todo.extend(kids[x])
    return out


def _ancestors(parents, b):
    out = []
    while b >= 0:
        out.append(b)
        b = parents[b]
    return out


def _lca(parents, a, b):
    seen = set(_ancestors(parents, a))
    return next(x for x in _ancestors(parents, b) if x in seen)


def _descends(parents, b, top):
    return top in _ancestors(parents, b)


def _path(parents, top, b):
    """Bones from ``top`` down to ``b`` (both included)."""
    chain = _ancestors(parents, b)
    return chain[:chain.index(top) + 1][::-1]


def _between(parents, a, b):
    """The unique tree path from ``a`` to ``b``, through their common ancestor."""
    top = _lca(parents, a, b)
    return _path(parents, top, a)[::-1] + _path(parents, top, b)[1:]


def _branch_on(parents, b, on):
    """The first bone of ``on`` above ``b``."""
    return next((x for x in _ancestors(parents, b)[1:] if x in on), None)
