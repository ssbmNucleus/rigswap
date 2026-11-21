"""Move the whole mesh into the fighter's rest with one skinning deformation.

Each fighter bone has a place inside the model (correspond.Match.embedded) and a place in the fighter's rest. Its
move turns the first into the second, stretching along the bone to the fighter's length. The model's skin is renamed
onto fighter bones, and hard boundaries across a connected surface blend over a band before either rest or play.
"""
import heapq

import numpy as np

MAX_INFLUENCES = 6
DIALS = {}              # rec: a hand record's dials, {'limbs': x, 'trunk': x, 'head': x} (record.py); run.swap sets it
THICKEN = 0.5           # a limb's segment stretched by s along its bone grows by s**THICKEN across it; the trunk's
                        # by the whole trunk's stretch (transforms)


def normalize(rig, up, front, k):
    """Turn the model so it stands on +z and faces -y (as the fighter does), scale it by ``k`` and stand it on the
    floor. Changes ``rig`` in place; returns the 4x4 (normals go through its 3x3 too)."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    f = np.asarray(front, float) - (np.asarray(front, float) @ up) * up
    f /= np.linalg.norm(f)
    rot = np.vstack([np.cross(up, f), -f, up])        # left -> +x, front -> -y, up -> +z
    lo = (rig.points @ rot.T).min(0)
    m = np.eye(4)
    m[:3, :3] = rot * k
    m[:3, 3] = -np.array([0, 0, lo[2]]) * k
    rig.points = rig.points @ m[:3, :3].T + m[:3, 3]
    rig.heads = rig.heads @ m[:3, :3].T + m[:3, 3]
    rig.tails = rig.tails @ m[:3, :3].T + m[:3, 3]
    return m


def fighter_weights(rig, owner, fighter_bones):
    """The model's skin renamed onto fighter bones: (V, len(fighter_bones)), at most MAX_INFLUENCES per vertex."""
    col = {b: i for i, b in enumerate(fighter_bones)}
    w = np.zeros((len(rig.points), len(fighter_bones)))
    for mb, fb in owner.items():
        w[:, col[fb]] += rig.weights[:, mb]
    return _capped(w)


def alone(w, fighter_bones, whole):
    """``w`` with the vertices each fighter bone of ``whole`` names riding that bone alone (Match.whole: a prop a weapon
    slot took, a strap tying it to the arm with it)."""
    col = {b: i for i, b in enumerate(fighter_bones)}
    w = w.copy()
    for fb, vertices in whole.items():
        if fb in col:
            w[vertices] = 0.0
            w[vertices, col[fb]] = 1.0
    return w


def _capped(w):
    """``w`` with at most MAX_INFLUENCES bones per vertex (its strongest), rows summing to 1 (or 0)."""
    if w.shape[1] > MAX_INFLUENCES:
        cut = np.argsort(-w, axis=1)[:, MAX_INFLUENCES:]
        np.put_along_axis(w, cut, 0.0, axis=1)
    s = w.sum(axis=1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


class Move:
    """A fighter bone's move from its place inside the model to its place at the fighter's rest: ``m``, the 3x4 that
    stretches along the bone over the segment between its joints. The stretch is the segment's: mesh the bone carries
    past either joint (a shin's below an ankle stood higher in the foot, a cape's hem past the bone it hangs from)
    keeps its own distance beyond that joint, as an end's mesh keeps its size, rather than being flung on by the
    stretch."""

    def __init__(self, m, head=None, axis=None, length=0.0, pull=None):
        self.m, self.head, self.axis, self.length, self.pull = np.asarray(m, float), head, axis, length, pull

    def __call__(self, pts):
        out = pts @ self.m[:, :3].T + self.m[:, 3]
        if self.pull is None:
            return out
        u = (pts - self.head) @ self.axis
        return out + np.outer(u - np.clip(u, 0, self.length), self.pull)


BLEND = 0.25            # a bend eases over this share of each adjacent bone's length


def split(rig, w, fighter_bones, splits, owner):
    """The weights ``w`` with each model bone a fighter joint falls inside (correspond._cuts) cut there: its share of
    a vertex goes to the fighter bone whose stretch of the chain the vertex lies on (by where its nearest point on the
    chain is, within the bone's own stretch), easing over BLEND of each adjacent segment. A rigid piece that rides
    whole is no split (correspond._whole). The rest of the skin is left as the model's own: a joint the model has, it
    bends at."""
    col = {b: i for i, b in enumerate(fighter_bones)}
    w = w.copy()
    total = rig.weights.sum(axis=1)
    for mb, chain, cuts in splits:
        on = np.flatnonzero(rig.weights[:, mb] > 0)
        if not len(on) or mb not in owner:
            continue
        a = rig.weights[on, mb] / np.maximum(total[on], 1e-12)
        t_m, pts = chain.fractions()
        j = chain.bones.index(mb)
        t = np.clip(_along(rig.points[on], t_m, pts), t_m[j], t_m[j + 1])
        at = np.array([c for c, _ in cuts])
        gaps = np.diff(np.concatenate([[t_m[j]], at, [t_m[j + 1]]]))
        offset = t[:, None] - at[None]
        band = BLEND * np.where(offset < 0, gaps[:-1], gaps[1:])
        past = np.clip(offset / np.maximum(2 * band, 1e-12) + 0.5, 0, 1)
        shares = np.diff(np.concatenate([np.ones((len(on), 1)), past, np.zeros((len(on), 1))], axis=1) * -1, axis=1)
        bones = [owner[mb]] + [fb for _, fb in cuts]
        w[on, col[owner[mb]]] -= a * (1 - shares[:, 0])
        for k, fb in enumerate(bones[1:], 1):
            w[on, col[fb]] += a * shares[:, k]
    np.clip(w, 0, None, out=w)
    return _capped(w)


class Surface:
    """Edges and coincident vertices join one surface, even across UV or normal splits.

    Distances follow its edges, never a shortcut through space to a nearby disconnected piece. Coincidence is exact:
    copies of a file vertex have the same position; a proximity radius would join overlapping armour and blades.
    """

    def __init__(self, rig, seam=()):
        """``seam``: rows of rig.points where a cut crossed a hard seam (refine.seam)."""
        self.seam = np.asarray(seam, int)
        self.points, self.vertex = np.unique(rig.points, axis=0, return_inverse=True)
        self.edges = np.asarray(rig.edges if rig.edges is not None else [], int).reshape(-1, 2)
        edges = np.unique(np.sort(self.vertex[self.edges], axis=1), axis=0)
        self.neighbours = [[] for _ in self.points]
        for (a, b), length in zip(edges, np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)):
            if a != b:
                self.neighbours[a].append((b, length))
                self.neighbours[b].append((a, length))

    def blend(self, w, bones, embedded):
        """A hard seam crosses between dominant bones with no existing blend between them.

        Its midpoint is the seam, or, on a surface that was cut, the point the cut put there (``seam``), between the two
        bones it holds most; shortest edge paths measure distance on either side. Within BLEND of that side's bone
        length, half its share at the seam falls linearly to zero at the band's edge. Authored blends are left alone.
        Overlapping bands share the available weight, and coincident copies receive one capped row.
        Returns the skin and counts of changed vertices and hard edges, including bands too narrow to reach a vertex.
        """
        if not len(w) or not w.shape[1]:
            return w.copy(), {'vertices': 0, 'pairs': []}
        dominant = w.argmax(axis=1)
        # At a split vertex, connect representatives of every distinct dominant bone, not just one arbitrary copy.
        order = np.lexsort((dominant, self.vertex))
        reps = order[np.r_[True, (np.diff(self.vertex[order]) != 0) | (np.diff(dominant[order]) != 0)]]
        groups = np.split(reps, np.flatnonzero(np.diff(self.vertex[reps])) + 1)
        joins = [(a, b) for group in groups if len(group) > 1 for i, a in enumerate(group) for b in group[i + 1:]]
        edges = np.concatenate([self.edges, np.asarray(joins, int).reshape(-1, 2)])
        u, v = edges.T
        a, b = dominant[u], dominant[v]
        hard = (a != b) & (w[u, b] == 0) & (w[v, a] == 0) & (w[u, a] > 0) & (w[v, b] > 0)
        edges, pairs = edges[hard], np.sort(np.column_stack([a[hard], b[hard]]), axis=1)
        top = np.argsort(w[self.seam], axis=1)[:, -2:]
        both = w[self.seam, top[:, 0]] > 0
        edges = np.concatenate([edges, np.column_stack([self.seam[both]] * 2)])
        pairs = np.concatenate([pairs, np.sort(top[both], axis=1)])
        if not len(edges):
            return w.copy(), {'vertices': 0, 'pairs': []}
        skin = np.zeros((len(self.points), w.shape[1]))
        np.add.at(skin, self.vertex, w)
        skin /= np.bincount(self.vertex)[:, None]
        incoming, outgoing = np.zeros_like(skin), np.zeros_like(skin)
        welded = np.zeros(len(skin), bool)
        report = []
        for a, b in np.unique(pairs, axis=0):
            seam = self.vertex[edges[(pairs == (a, b)).all(axis=1)]]
            half = np.linalg.norm(self.points[seam[:, 0]] - self.points[seam[:, 1]], axis=1) / 2
            welded[seam[half == 0].ravel()] = True
            widths = [BLEND * np.linalg.norm(np.subtract(*embedded[bones[k]])) for k in (a, b)]
            pure_a, pure_b = (skin[:, a] > 0) & (skin[:, b] == 0), (skin[:, b] > 0) & (skin[:, a] == 0)
            allowed = pure_a | pure_b
            allowed[seam.ravel()] = True
            distance = self._distance(seam, half, allowed, max(widths))
            changed = np.zeros(len(skin), bool)
            for own, other, pure, width in ((a, b, pure_a, widths[0]), (b, a, pure_b, widths[1])):
                on = np.flatnonzero(pure & (distance < width))
                amount = skin[on, own] * (1 - distance[on] / width) / 2
                outgoing[on, own] += amount
                incoming[on, other] += amount
                changed[on] = True
            changed[seam[half == 0].ravel()] = True
            report.append({'bones': [str(bones[a]), str(bones[b])], 'edges': len(seam),
                           'vertices': int(changed[self.vertex].sum()), 'band': widths})
        # At a junction several seams can draw on the same share: scale their transfers together, never go negative.
        fraction = np.divide(outgoing, skin, out=np.zeros_like(skin), where=skin > 0)
        divisor = np.maximum(1, fraction.max(axis=1))[:, None]
        changed = welded | (outgoing.sum(axis=1) > 0)
        blended = _capped(np.maximum(skin + (incoming - outgoing) / divisor, 0))
        out = w.copy()
        on = changed[self.vertex]
        out[on] = blended[self.vertex[on]]
        return out, {'vertices': int(np.any(out != w, axis=1).sum()), 'pairs': report}

    def _distance(self, seam, half, allowed, limit):
        """Distance from edge midpoints along this surface, stopping at the band or an already blended region."""
        distance = np.full(len(self.points), np.inf)
        np.minimum.at(distance, seam.ravel(), np.repeat(half, 2))
        queue = [(distance[v], v) for v in np.flatnonzero(distance < limit)]
        heapq.heapify(queue)
        while queue:
            d, v = heapq.heappop(queue)
            if d != distance[v]:
                continue
            for nxt, length in self.neighbours[v]:
                candidate = d + length
                if allowed[nxt] and candidate < min(distance[nxt], limit):
                    distance[nxt] = candidate
                    heapq.heappush(queue, (candidate, nxt))
        return distance


def _along(points, t, pts):
    """Each point's share along the polyline ``pts`` (shares ``t`` at its corners): at its nearest point on it."""
    a, d = pts[:-1], np.diff(pts, axis=0)
    u = np.clip(np.einsum('vkd,kd->vk', points[:, None] - a[None], d) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    gap = np.linalg.norm(a[None] + u[..., None] * d[None] - points[:, None], axis=2)
    k = np.argmin(gap, axis=1)
    return t[k] + u[np.arange(len(points)), k] * (t[k + 1] - t[k])


def transforms(match, fighter_bones):
    """Per fighter bone, its Move from its place inside the model to its place at the fighter's rest. A segment
    between joints stretches to the fighter's length (the joints must sit where the fighter's animations put them);
    the head, hands and feet keep the model's own size: oversized pokes past the fighter, which is fine; squashed is
    not. Within an end, every part takes its parent's whole move, preserving the assembled shape at rest.
    A limb's segment thickens by its own stretch (_bone): a stubby leg on a long-legged fighter would be a stilt. The
    trunk thickens as one, by its whole stretch: its landmarks place the fighter's joints, and one segment's
    stretch says where a joint sits, not how big the body is (Tauros's mane, 3 off a neck stretched 3.6 times between shoulders
    and head set close, was thrown 2.4 forward by thickening it alone).
    A foot stands flat: it turns only about vertical to face the fighter's way, keeping the sole's height relative
    to the embedded ankle. The child's game bone still animates its skin from there. Match.rest and the fighter
    skeleton stay fixed.
    A chest whose model has no clavicle where the fighter has one widens across the fighter's midline to meet the arms
    at the fighter's shoulders (Match.widen, correspond._widen).
    A prop in a weapon slot, or one from outside the body laid on a hand's item joint, is placed outright
    (Match.placed), never by the hand it hangs under."""
    out = {fb: Move(m) for fb, m in match.placed.items()}
    length = [sum(np.linalg.norm(np.subtract(*ends[fb])) for fb in match.trunk) for ends in (match.rest, match.embedded)]
    trunk = length[0] / length[1] if length[1] > 1e-4 else 1.

    def place(fb):
        if fb in out:
            return out[fb]
        if fb in match.end_parents:
            out[fb] = place(match.end_parents[fb])
            # rec: a head placed by its neck (a model with no neck: correspond._end_parents) takes the record's head
            # size as any other head does, about its own joint, where the neck's move leaves it. Without this the
            # entry changed nothing on such a model (Bart on Ness, 2026-11-09: "his head is too big" built the same file)
            if fb == match.head and DIALS.get('head', 1.) != 1.:
                out[fb] = _sized(out[fb], np.asarray(match.embedded[fb][0], float), DIALS['head'])
        else:
            he, te = match.embedded[fb]
            hr, tr = match.rest[fb]
            points = [np.asarray(p) for p in (he, te, hr, tr)]
            # rec: a hand record's dials (record.py): how wide the limbs' segments and the trunk's come out, as a
            # multiple of what the stretch gives them, and the head's size as a multiple of the model's own
            wide = DIALS.get('trunk', 1.) if fb in match.trunk else 1. if fb in match.ends else DIALS.get('limbs', 1.)
            size = DIALS.get('head', 1.) if fb == match.head else 1.
            out[fb] = _foot(*points) if fb in match.feet else _bone(*points, stretch=fb not in match.ends,
                                                                       across=trunk if fb in match.trunk else None,
                                                                       wide=wide, size=size)
            if fb in match.widen:
                out[fb] = _wider(out[fb], *match.widen[fb])
        return out[fb]

    for fb in fighter_bones:
        place(fb)
    return out


def _wider(move, times, x):
    """``move``, then widened ``times`` across the fighter's midline at ``x`` (a chest reaching the shoulders)."""
    w = np.diag([times, 1., 1.])
    m = w @ move.m
    m[0, 3] += (1 - times) * x
    return Move(m, move.head, move.axis, move.length, None if move.pull is None else w @ move.pull)


def end_report(match, frig, bones, w, points, moves):
    """An end's joints meet at rest; feet must also meet the game's floor (z=0).

    Record the actual placed skin's lowest foot vertex, including any error left by turning it after embedding
    the ankle above the source sole. The fighter's ankle is fixed; the embedded ankle's before/after is recorded too.
    """
    from .correspond import _ancestors

    joints = []
    for b, parent in sorted(match.end_parents.items()):
        joint = np.asarray([match.embedded[b][0]])
        own, carried = moves[b](joint)[0], moves[parent](joint)[0]
        joints.append({'bone': frig.names[b], 'parent': frig.names[parent],
                       'head_shift': (own - match.rest[b][0]).tolist(),
                       'joint_gap': float(np.linalg.norm(own - carried)),
                       'move_difference': float(np.max(np.abs(moves[b].m - moves[parent].m)))})
    feet = []
    for b in sorted(match.feet):
        columns = [i for i, bone in enumerate(bones) if b in _ancestors(frig.parents, bone)]
        on = w[:, columns].sum(axis=1) > 0.5
        ankle = moves[b](np.asarray([match.embedded[b][0]]))[0]
        sole = float(points[on, 2].min()) if on.any() else None
        feet.append({'bone': frig.names[b], **match.ankles.get(b, {}),
                     'vertices': int(on.sum()), 'ankle': ankle.tolist(),
                     'ankle_error': float(np.linalg.norm(ankle - match.rest[b][0])), 'sole_z': sole,
                     'grounding_translation_z': -sole if sole is not None else None,
                     'conflict': not bool(np.isclose(sole, 0)) if sole is not None else None})
    return {'joints': joints, 'feet': feet}


def deform(points, w, fighter_bones, T):
    out = np.zeros_like(points)
    for i, fb in enumerate(fighter_bones):
        sel = w[:, i] > 0
        if not sel.any():
            continue
        out[sel] += w[sel, i:i + 1] * T[fb](points[sel])
    unweighted = w.sum(axis=1) == 0
    out[unweighted] = points[unweighted]
    return out


def _foot(he, te, hr, tr):
    """A foot already stands on its sole: face its horizontal heading toward the fighter without tilting or scaling.

    The ankle's sole-relative height was set before cuts. A vertical bone has no horizontal heading to turn toward.
    """
    ae, ar = (te - he)[:2], (tr - hr)[:2]
    angle = np.arctan2(ae[0] * ar[1] - ae[1] * ar[0], ae @ ar)
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return Move(np.column_stack([m, hr - m @ he]))


def _sized(move, joint, size):
    """``move`` with what it carries scaled by ``size`` about ``joint``, which stays where the move puts it."""
    at = move(np.asarray(joint, float)[None])[0]
    m = move.m[:, :3] * size
    return Move(np.hstack([m, (at - m @ joint)[:, None]]))


def _bone(he, te, hr, tr, stretch=True, across=None, wide=1., size=1.):
    """The segment turned the least way onto its fighter bone (_swing), stretched along it and thickened across.

    ``across``: the stretch it thickens by, when not its own (the trunk's whole stretch). rec: ``wide``, its width as
    a multiple of that; ``size``, the whole of it scaled about its joint (a head: record.py's dials)."""
    ae, ar = te - he, tr - hr
    le, lr = np.linalg.norm(ae), np.linalg.norm(ar)
    if le < 1e-4 or lr < 1e-4:                     # fighter units; float32 noise in a file is ~1e-6 of its size
        m = np.eye(3) * size
        return Move(np.hstack([m, (hr - m @ he)[:, None]]))
    xe, xr = ae / le, ar / lr
    s = lr / le if stretch else 1.
    k = s if across is None else across
    t = k ** THICKEN if k > 1 else 1.               # a limb stretched to the fighter's length thickens with it,
    t *= wide
    along = np.outer(xe, xe)                        # partly: oversized reads as the model, spindly doesn't
    m = _swing(xe, xr) @ (s * along + t * (np.eye(3) - along)) * size
    return Move(np.hstack([m, (hr - m @ he)[:, None]]), he, xe, le, (1 - s) * xr if s != 1 else None)


def _swing(a, b):
    """The least turn carrying the unit ``a`` onto the unit ``b``: about the axis square to both, so the segment never
    rolls about itself and a small change in either gives a small change in the turn. Frames built from a fixed front
    (or up, near the front) rolled a segment half round whenever its turn passed that direction: Bowser Odyssey's
    back bone, turning from forward-up to up, crossed the switch to up and wore its shell in front.

    Turned right round there is no least axis: it turns about the front as the segment sees it (up, for one pointing
    front or back), so the front stays the front."""
    v, c = np.cross(a, b), a @ b
    if c < -1 + 1e-6:
        ref = np.array([0., -1., 0.]) if abs(a[1]) < 0.9 else np.array([0., 0., 1.])
        axis = ref - (ref @ a) * a
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)
