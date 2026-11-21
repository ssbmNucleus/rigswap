"""A gown on the fighter's ring of skirt strands, or on its two legs.

A fighter whose game swings a skirt hangs a ring of strands from one bone (Peach's and Zelda's eight from the hips).
A model whose skirt has fewer strands than that ring (a gown on a left and a right bone and a train) pairs each of its
strands with one of the ring's (correspond._match_extra), and then each carries a whole side of the dress through that
one strand's flare, and what the hips carry rides them rigidly: the dress sweeps out as a board. Instead the gown is
laid on the whole ring by angle round the body: below the hip joint, the share of each vertex the model's skirt bones
and its hips carry goes to the two strands either side of it, by its angle, and along each to the joints either side
of it, by how far from the strand's root it lies; the rest of its weight stays where the model's own skin put it (the
legs). Each strand of the ring is placed in the model where the hips carry its root from, down the gown at its angle
as the gown lies there (bending where a train lies out along the floor), and each of its bones turns its own stretch
of that onto the strand's rest (the game's animation brings it down from there): the dress opens and swings round the
body as the fighter's own skirt does. The gown is held as the legs hold the model, squashed with them and never
stretched (correspond._fit), and one reaching past the ring so is fitted to it (its hem at the strand's end, not along
the floor).

A fighter whose game swings no skirt holds the gown at rest on its two legs, squashed with them and never stretched
(legs()). In play, averaging legs that swing apart collapses the surface between them. The front and back therefore
hang from the hips, and each side follows its own leg, easing between that leg and the hips over repose.BLEND of the
angular interval between the legs. Rest fitting and motion are separate: the hem still fits the fighter's floor.
"""
from dataclasses import dataclass

import numpy as np

from . import correspond
from . import repose

STEPS = 8               # a strand's place down the gown is read in this many steps of distance from its root


@dataclass
class Gown:
    bones: list             # the model's skirt bones, whose share of each vertex goes onto the ring by ``fall``
    hips: list              # the model bones riding the ring's bone, whose share goes onto it by ``lift``
    lift: np.ndarray        # (V,) how much of those bones' share goes onto the ring: 1 below the hip joint
    ring: np.ndarray        # (V, len(columns)) each vertex's share on the ring's bones, rows summing to 1 (or 0)
    columns: list           # the ring's fighter bones, or the two legs and their hips
    fall: object = 1.0      # (V,) how much of the skirt bones' share goes onto the ring: all of it on a ring of
                            # strands, which hangs from the hips; on the legs, what lies below the hip joint
    held: np.ndarray = None  # (V, len(columns)) the shares that place it at the fighter's rest (rest()), when they
                             # are not those its game moves it by (``ring``)
    moves: dict = None      # fighter bone -> the repose.Move placing the gown on it at rest, when it is not the body's
                            # (rest())


def gown(rig, m, frig, description):
    """Lay the model's skirt on the fighter's ring when the ring has strands no model strand took: places each of the
    ring's strands in the model (``m.embedded``, ``m.rest``; fitted, so out of ``m.ends``), takes the skirt's loose
    strands off ``m.loose``, and returns the Gown for weights(), or None (no skirt, or one model strand on each of the
    ring's). A fighter with no ring lays it on its legs (legs())."""
    fch = correspond.fighter_chains(frig, description)
    groups = {}
    for region, anchor, c in correspond.extra_chains(frig, description, fch):
        if region == 'lower':
            groups.setdefault(anchor, []).append(c)
    anchor, ring = max(groups.items(), key=lambda kv: len(kv[1]), default=(None, []))
    if len(ring) < correspond.RING:
        return legs(rig, m, fch)
    strand = {fb: k for k, c in enumerate(ring) for fb in c.bones}
    skirt = {mb for mb, fb in m.owner.items() if fb in strand}
    left = [x for x in m.loose if m.owner.get(x[0]) == anchor and x[1].kind == 'cloth']
    skirt |= {b for _, _, carries in left for b in carries}
    if not skirt or len({strand[fb] for fb in m.owner.values() if fb in strand}) == len(ring):
        return None
    # the gown: what the skirt's bones carry, and below the hip joint what rides the ring's bone (an author may hang
    # the back of a dress on the hips and only its hem on a bone of its own: the ring swings the whole of it, from the
    # waist), easing in over a tenth of its height
    bones, hips = sorted(skirt), [mb for mb, fb in m.owner.items() if fb == anchor and mb not in skirt]
    own = rig.weights[:, bones].sum(axis=1)
    z = rig.points[:, 2]
    top, low = m.embedded[anchor][0][2], z[own > 0].min()
    band = 0.1 * (top - low) or 1e-6
    lift = np.clip((top - z) / band, 0, 1)
    share = own + lift * rig.weights[:, hips].sum(axis=1)
    centre, f_centre = np.asarray(m.embedded[anchor][0]), np.asarray(m.rest[anchor][0])
    # where the ring's bone carries each strand's root from: there the gown leaves the waist as the body does
    hip = repose.transforms(m, [anchor])[anchor].m
    back = np.linalg.inv(hip[:, :3])
    theta = np.array([_angle(c.joints[0] - f_centre) for c in ring])
    order = np.argsort(theta)
    phi = _angle(rig.points - centre)
    near = _between(phi, theta[order])                      # per vertex: (strand, its share) for the two either side
    columns = [fb for c in ring for fb in c.bones]
    col = {fb: i for i, fb in enumerate(columns)}
    out = np.zeros((len(rig.points), len(columns)))
    on = share > 0
    squash = correspond._squash(fch, m.embedded, m.rest)       # the gown held as the legs hold the model
    for k, c in enumerate(ring):
        mine = sum(np.where(order[a] == k, u, 0.0) for a, u in near)
        root = back @ (c.joints[0] - hip[:, 3])
        joints, tail = correspond._fit(c, _curve(rig.points, share, mine, root, back @ (c.end - c.joints[0])), squash)
        ends = list(joints[1:]) + [tail]
        f_ends = list(c.joints[1:]) + [correspond._onward(c)]
        for i, fb in enumerate(c.bones):
            m.embedded[fb] = (joints[i], ends[i])
            m.rest[fb] = (c.joints[i], f_ends[i])
            m.ends.discard(fb)
            m.fitted.add(fb)
        reach = np.linalg.norm(np.vstack([joints, tail]) - root, axis=1)
        along = _along(np.linalg.norm(rig.points[on] - root, axis=1), np.maximum.accumulate(reach))
        for i, fb in enumerate(c.bones):
            out[on, col[fb]] += mine[on] * along[:, i]
    m.loose[:] = [x for x in m.loose if all(x is not y for y in left)]
    m.notes.append(f'gown: {len(bones)} skirt bones of the model ({", ".join(rig.names[b] for b in bones)}) laid on '
                   f'the ring of {len(ring)} strands off {frig.names[anchor]} by angle')
    return Gown(bones, hips, lift, out, columns)


def legs(rig, m, fch):
    """Lay the model's skirt on the fighter's two legs, its ring of two, when its game swings no skirt: below the hip
    joint, the share of each vertex the skirt's strands carry (its cloth off the lower body, labels.chains, some of
    whose own surface lies there: a ball's bandana is none), and the hips' share on the skirt's own surface there (an author may hang the top of a dress on the hips), goes to its own
    leg and the hips by angle round the body. Opposite leg rotations collapse a surface averaged between them: at each angular bisector the hips
    carry it all, and towards either side that leg's share rises to one over repose.BLEND of the interval between
    the legs. No gown share blends the two legs in play. Above the hip joint it rides the hips, as the legs turn
    about it. Down each leg it is held at rest by its thigh and shin by its height (from the hip joint to the
    knee, shared between them by how near it lies to each; below the knee, the shin), as the fighter's leg lies,
    squashed with it and never stretched (Gown.held, Gown.moves): a gown as long as the model's legs keeps its hem at
    the fighter's floor, a short skirt on a long-legged fighter keeps its own size. In play, what the shin holds below
    the knee hangs from the knee, carried by thigh and shin alike: on the shin alone what lies out beyond the leg (a
    train along the floor, a flare) is a lever the knee flings up as it bends, on the thigh alone one the thigh flings
    up as it lifts. Takes those strands off ``m.loose`` (they move with the legs, not on their own) and returns the
    Gown for weights(), or None when the model wears no skirt."""
    region = correspond._regions(fch)
    hips = fch['spine'].bones[0]
    leg = [fch[f'{side} leg'] for side in ('left', 'right')]
    strands = [c.bones[:max(c.anchors[1], 1)] for c in leg]            # thigh and shin, not the foot
    top = float(np.mean([m.embedded[s[0]][0][2] for s in strands]))     # the hip joints
    z = rig.points[:, 2]
    skirt = [x for x in m.loose if region.get(m.owner.get(x[0])) == 'lower' and x[1].kind == 'cloth'
             and (z[rig.weights[:, x[2]].sum(axis=1) > 0.5] < top).any()]
    if not skirt:
        return None
    bones = sorted({b for _, _, carries in skirt for b in carries})
    lower = [mb for mb, fb in m.owner.items() if region.get(fb) == 'lower' and mb not in bones]
    own = rig.weights[:, bones].sum(axis=1)
    band = 0.1 * (top - z[own > 0].min()) or 1e-6         # easing in over a tenth of the skirt's height below the joint
    lift = np.clip((top - z) / band, 0, 1)
    # the skirt's own surface below the hip joint: its vertices and those joined to them there, through none the legs
    # or the rest of the body carry most of (the crotch's skin meets a dress only up at the waist)
    taken = {b for _, _, cs in m.loose for b in cs}
    body = rig.weights[:, [mb for mb, fb in m.owner.items() if region.get(fb) != 'lower' and mb not in taken]]
    free = (z < top) & (body.sum(axis=1) <= 0.5)
    edges = rig.edges if rig.edges is not None else np.zeros((0, 2), int)
    piece = _pieces(len(z), edges[free[edges].all(axis=1)])
    on = np.isin(piece, piece[own > 0.5])
    centre, f_centre = np.asarray(m.embedded[hips][0]), np.asarray(m.rest[hips][0])
    theta = np.array([_angle(c.joints[0] - f_centre) for c in leg])
    order = np.argsort(theta)
    near = _between(_angle(rig.points - centre), theta[order])
    columns = [fb for s in strands for fb in s] + [hips]
    held, out = np.zeros((len(z), len(columns))), np.zeros((len(z), len(columns)))
    # held down each leg as the fighter's leg lies, squashed with it and never stretched: a gown longer than the leg
    # is fitted to it, a shorter one keeps its length (as on a ring of strands, correspond._fit); stretched, a kilt on
    # a model's short legs grew into a long apron round the fighter's
    moves = repose.transforms(m, [hips])                    # the hips' held column is zero; rest fitting stays on legs
    for s in strands:
        at = np.asarray(m.rest[s[0]][0], float)
        for fb in s:
            (he, te), (hr, tr) = (tuple(np.asarray(x, float) for x in m.embedded[fb]),
                                  tuple(np.asarray(x, float) for x in m.rest[fb]))
            lr = np.linalg.norm(tr - hr)
            to = at + (tr - hr) / (lr or 1.0) * min(lr, np.linalg.norm(te - he))
            moves[fb], at = repose._bone(he, te, at, to), to
    for k, s in enumerate(strands):
        mine = sum(np.where(order[a] == k, u, 0.0) for a, u in near)
        heights = [m.embedded[fb][0][2] for fb in s] + [m.embedded[s[-1]][1][2]]
        i = columns.index(s[0])
        along = mine[:, None] * _along(heights[0] - z, heights[0] - np.minimum.accumulate(heights))
        held[:, i:i + len(s)] += along
        # A leg takes only its own side of the bisector; the common support takes everything between the legs.
        swing = np.clip((mine - 0.5) / repose.BLEND, 0, 1)
        along = swing[:, None] * _along(heights[0] - z, heights[0] - np.minimum.accumulate(heights))
        out[:, i:i + len(s)] += along / 2                       # what hangs below the knee: thigh and shin alike
        out[:, i] += along[:, 0] / 2 + along[:, 1:].sum(axis=1) / 2
    out[:, -1] = np.maximum(0, 1 - out.sum(axis=1))
    m.loose[:] = [x for x in m.loose if all(x is not y for y in skirt)]
    m.notes.append(f'gown: {len(skirt)} skirt strands of the model ('
                   f'{", ".join(rig.names[c.bones[0]] for _, c, _ in skirt)}) laid on the two legs below the hip joint')
    return Gown(bones, lower, lift * on, out, columns, fall=lift, held=held, moves=moves)


def weights(rig, owner, fighter_bones, g):
    """repose.fighter_weights with the gown's share of each vertex on the ring instead of where its bones' owners put
    it: (V, len(fighter_bones)), at most MAX_INFLUENCES per vertex, rows summing to 1."""
    return _skin(rig, owner, fighter_bones, g, g.ring, g.columns)


def rest(rig, owner, fighter_bones, g, T):
    """What places the model at the fighter's rest, for repose.deform: (bones, weights, moves).
    On the legs the gown is held there by moves of its own (Gown.moves), its shares on them by height (Gown.held);
    the rest of the model as weights() and ``T`` (repose.transforms) have it."""
    if g.held is None:
        return fighter_bones, weights(rig, owner, fighter_bones, g), T
    keys = [('held', fb) for fb in g.columns]
    bones = list(fighter_bones) + keys
    moves = {**T, **{k: g.moves[fb] for k, fb in zip(keys, g.columns)}}
    return bones, _skin(rig, owner, bones, g, g.held, keys), moves


def _skin(rig, owner, bones, g, ring, columns):
    """The model's skin renamed onto ``bones``, with the gown's share of each vertex on ``columns`` (of ``bones``) by
    ``ring``, at most MAX_INFLUENCES per vertex, rows summing to 1."""
    col = {b: i for i, b in enumerate(bones)}
    skirt, hips = set(g.bones), set(g.hips)
    w = np.zeros((len(rig.points), len(bones)))
    for mb, fb in owner.items():
        w[:, col[fb]] += rig.weights[:, mb] * (1 - g.lift if mb in hips else 1 - g.fall if mb in skirt else 1)
    share = g.fall * rig.weights[:, g.bones].sum(axis=1) + g.lift * rig.weights[:, g.hips].sum(axis=1)
    for j, key in enumerate(columns):
        w[:, col[key]] += share * ring[:, j]
    return repose._capped(w)


def _angle(d):
    """The angle round the body (about +z) of offsets ``d``: 0 at the front (-y), growing towards the left (+x)."""
    d = np.atleast_2d(d)
    out = np.arctan2(d[:, 0], -d[:, 1])
    return out if len(out) > 1 else out[0]


def _between(phi, theta):
    """For angles ``phi`` and the ring's sorted angles ``theta``: [(index into theta, share)] for the strand before
    and the one after each angle, the shares by how near it is to each (summing to 1)."""
    n = len(theta)
    k = (np.searchsorted(theta, phi) - 1) % n
    a, b = theta[k], theta[(k + 1) % n]
    gap = (b - a) % (2 * np.pi)
    gap = np.where(gap > 0, gap, 2 * np.pi)
    u = ((phi - a) % (2 * np.pi)) / gap
    return [(k, 1 - u), ((k + 1) % n, u)]


def _curve(points, share, mine, root, hang):
    """The gown down from a strand's root (``root``, in the model) at its angle, as it lies: a polyline from the root
    through the middle (by the gown's share) of the gown's vertices the strand takes the larger part of, in steps of how
    far they lie from the root, its last stretch going on to reach the farthest of them (its hem on the strand's end,
    none past it). A dress whose train lies out along the floor bends there, and each of the strand's bones turns its
    own stretch onto the strand's rest. Where the gown has none, the strand's own hang (``hang``)."""
    sel = (mine >= 0.5) & (share > 0)
    if sel.sum() < 3:
        sel = (mine > 0) & (share > 0)
    if sel.sum() < 3:
        return np.vstack([root, root + hang])
    p, s = points[sel], share[sel]
    r = np.linalg.norm(p - root, axis=1)
    reach = float(r.max()) or 1e-6
    step = np.minimum((r / reach * STEPS).astype(int), STEPS - 1)
    out = [root]
    for k in range(STEPS):
        at = step == k
        if at.any():
            c = (s[at, None] * p[at]).sum(axis=0) / s[at].sum()
            if np.linalg.norm(c - root) > np.linalg.norm(out[-1] - root):
                out.append(c)
    if len(out) < 2:
        return np.vstack([root, root + hang])
    a, b = out[-2], out[-1]                 # on along the last stretch to the reach: |b + x d - root| = reach
    d = (b - a) / (np.linalg.norm(b - a) or 1.0)
    q = b - root
    out[-1] = b + d * max(0.0, np.sqrt(max((q @ d) ** 2 - q @ q + reach ** 2, 0.0)) - q @ d)
    return np.vstack(out)


def _along(r, at):
    """Per point, its share on each of a strand's n bones by how far along it lies (``r``; from the strand's root, or
    down a leg from its hip joint), its joints and the tail of its last bone lying at ``at`` (n + 1, rising): between
    two joints, on the bones at either end of that stretch by how near it is to each; nearer than the first, on the
    first; past the last joint, on the last."""
    at = np.asarray(at, float)
    n = len(at) - 1
    out = np.zeros((len(r), n))
    k = np.clip(np.searchsorted(at[:n], r, side='right') - 1, 0, n - 1)
    nxt = np.minimum(k + 1, n - 1)
    u = np.clip((r - at[k]) / np.maximum(at[k + 1] - at[k], 1e-9), 0, 1)
    u = np.where(k == n - 1, 0.0, u)
    rows = np.arange(len(r))
    np.add.at(out, (rows, k), 1 - u)
    np.add.at(out, (rows, nxt), u)
    return out


def _pieces(n, edges):
    """The piece of surface each of ``n`` vertices is on (joined by ``edges``), numbered from 0."""
    label = np.arange(n)
    edges = np.asarray(edges if edges is not None else np.zeros((0, 2)), int).reshape(-1, 2)
    while len(edges):                   # hook each edge's two labels onto the smaller, then shortcut
        a, b = label[edges[:, 0]], label[edges[:, 1]]
        new = label.copy()
        np.minimum.at(new, a, np.minimum(a, b))
        np.minimum.at(new, b, np.minimum(a, b))
        while not (new[new] == new).all():
            new = new[new]
        if (new == label).all():
            break
        label = new
    return np.unique(label, return_inverse=True)[1].ravel()
