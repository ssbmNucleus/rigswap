"""A one-piece model's own skeleton, fitted inside it: a Rig like a rigged file's, so every later step runs as is.

Many downloads are one welded surface with no armature (an OBJ statue, posed or standing in a T): its trunk carries
its limbs in its own surface, so there are no pieces to read joints from (segments.OnePiece). The fit is made in two
steps, either side of the look that names the body's joints (fill.py). Neither says what a part is: that is meaning,
and the look's. (Until 2026-11-10 this file said it first, by sizes and places: which tips were feet, which branches
arms, where the neck narrowed into the head. The look could then only name the joints that left it: a ferret's head
began a quarter of the way up its neck.)

What the shape says, before anything is named (read()):
- the model is a volume, whatever its surface is made of (pieces worn over one another, faces twice, holes, a body
  cut in draw calls): the cells of a grid that its surface passes through or encloses;
- its mirror plane is the volume's (the trunk and the head mirror in any pose), and twins are apart wherever that
  plane between them is outside the body, however close they come: the surface's own section by the plane says where
  that is, finer than the grid (Volume.cut);
- going in from its top on the mirror plane, the distance travelled inside the volume rises along every part to its
  tip, so the cells at one distance are slices across the trunk and the limbs, each slice's middle inside its part;
  where a slice parts in several, the body branches (the Reeb graph of that distance). Starting on the plane, the
  slices cut both sides alike.
A joint is offered every DOTS slices along every part, a head and a neck as any other: a bone each, its head at its
slices' middle, the tree hanging from the top, each vertex on the slice just inside it.

What needs the names, once the look or a record has given them (named()):
- the tree hangs from the hips;
- the head is whole: all that lies past the skull is one bone with it, as an artist binds it;
- a bump (a nose, a knee cap, a fold) is no limb: a branch no longer than it is wide, no named joint on it, goes to
  what it grows from;
- a shoulder's joint lies inside the shoulder, in from the arm's first slice (shoulder_joints()), and the shoulders
  go with the arms (shoulders());
- the named joints of the left and the right are twins.
The skin is the slices' own: a vertex rides the slice just inside it, blended with the bones near it along the
skeleton over the narrower one's thickness. Blender's bone heat was tried first (2026-10-25): on a statue with an arm
hanging by its side it takes the side of the trunk with the arm, and the re-pose drags it out into a sheet; the slices
know which part is which.
"""
import copy
import heapq
import json
import re
from dataclasses import dataclass, field

import numpy as np

from . import segments

CELLS = 128         # grid cells along the model's longest side
BAND = 1.75         # a slice is this many cells deep: more than a diagonal step (sqrt 3), so none is skipped
DOTS = 2            # a joint is offered every this many slices along a part, from a fork out to the next or its tip
BUMP = 1.0          # a branch from a fork no longer than this many times its own width is a bump
MIDLINE = 0.1       # the top the slices start from is within this share of the model's width of the mirror plane
ONE_SIDE = 2 / 3    # a slice lies on one side of the mirror plane with this share of it or more out on that side
SHUT = 5            # a surface takes the room it walls in on this many of a cell's six sides or more
BLEND = 1.0         # joints blend over this many times the narrower side's thickness
SHOULDER = 1.5      # a shoulder lies within this many times the arm's thickness of its joint and bone
FLAT = 0.15         # a line's steps have a plane of their own when they spread this share of their length off it
OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
STEPS = np.linalg.norm(OFFSETS, axis=1)
NAME = 'slice'              # a fitted bone is named for its slices' node: slice<n>
ARMATURE = 'onepiece'       # the armature the open scene is given


class Volume:
    """The model's volume on a grid: ``solid`` (X, Y, Z) bools, cell (i, j, k)'s middle at origin + (ijk + .5) h."""

    def __init__(self, points, tris, cells=CELLS):
        wall, self.h, self.origin = _wall(points, tris, cells)
        shape = wall.shape
        self.solid = ~_flood(wall)
        for keep in (False, True):
            self.cells = np.argwhere(self.solid)                 # (N, 3) solid cells
            self.index = np.full(shape, -1, dtype=np.int64)
            self.index[self.solid] = np.arange(len(self.cells))
            nb = self.cells[:, None, :] + OFFSETS[None]
            self.neighbours = self.index[nb[..., 0], nb[..., 1], nb[..., 2]]    # (N, 26), -1 outside
            if keep:
                break
            # the body is its largest part: a piece apart from it (a floating eye, a claw) is skinned to it later
            i = np.repeat(np.arange(len(self.cells)), 26)
            j = self.neighbours.ravel()
            part = _label(len(self.cells), i[j >= 0], j[j >= 0])
            body = self.cells[part == np.argmax(np.bincount(part))]
            self.solid = np.zeros(shape, bool)
            self.solid[body[:, 0], body[:, 1], body[:, 2]] = True
        self.depth = _depth(self.solid)[self.solid]              # cells to the outside, face to face

    def middles(self, cells=None):
        c = self.cells if cells is None else self.cells[cells]
        return self.origin + (c + 0.5) * self.h

    def cut(self, side, centre, points, tris):
        """Twins are apart wherever the mirror plane (across ``side`` through ``centre``) between them is outside the
        body: legs pressed within a cell of each other (Richard's, 2026-10-28) fill the grid between them as one, and
        the slices never part at the crotch. Where the plane is inside the body, its section of the surface says (the
        plane's cells flooded in from its border up to it, as the volume is in space); a link between two cells across
        the plane holds only where it crosses inside. No cell is taken out: a crest thinner than a cell on the plane
        stays on its side. The plane is the volume's (mirror), to a quarter of a cell: twins less than half a cell
        apart can have it inside one of them, and stay one (Richard's feet are 0.8 of a cell apart, his plane 0.02 of
        one off the midpoints of his twin vertices)."""
        k = int(np.argmax(np.abs(side)))
        at = float(centre[k])
        ab = [d for d in range(3) if d != k]
        wall = np.zeros([self.solid.shape[d] for d in ab], bool)
        for p in _section(points, tris, k, at, self.h / 3):
            c = np.floor((p[:, ab] - self.origin[ab]) / self.h).astype(np.int64)
            wall[c[:, 0], c[:, 1]] = True
        outside = _flood(wall[..., None])[..., 0]
        mid = self.middles()
        over = mid[:, k] >= at
        slab = np.flatnonzero(np.abs(mid[:, k] - at) <= self.h)      # only these have links across the plane
        i, o = np.nonzero(self.neighbours[slab] >= 0)
        i = slab[i]
        j = self.neighbours[i, o]
        across = over[i] != over[j]
        i, o, j = i[across], o[across], j[across]
        t = (at - mid[i, k]) / (mid[j, k] - mid[i, k])
        c = np.floor((mid[i][:, ab] + t[:, None] * (mid[j][:, ab] - mid[i][:, ab]) - self.origin[ab]) / self.h)
        c = c.astype(np.int64)
        gone = outside[c[:, 0], c[:, 1]]
        self.neighbours[i[gone], o[gone]] = -1


def _wall(points, tris, cells):
    """The cells the surface passes through, on a grid of ``cells`` along its longest side with two spare around:
    (bools, the cell's size, the grid's origin)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    h = float((hi - lo).max()) / cells
    origin = lo - 2 * h
    wall = np.zeros(tuple(np.ceil((hi - lo) / h).astype(int) + 4), bool)
    for p in _samples(points, tris, h / 3):
        c = np.floor((p - origin) / h).astype(np.int64)
        wall[c[:, 0], c[:, 1], c[:, 2]] = True
    return wall, h, origin


def room(points, tris, cells=CELLS):
    """The room a surface takes, in its units cubed: its own cells and those it walls in on five of their six sides
    or more. A flood from outside fills a dress through its open hem, and a body whose pieces leave gaps at their
    joints (Melee's fighters); a hem or a gap is open on one side."""
    wall, h, _ = _wall(points, tris, cells)
    shut = np.zeros(wall.shape, np.int8)
    for a in range(3):
        shut += np.maximum.accumulate(wall, axis=a)
        shut += np.flip(np.maximum.accumulate(np.flip(wall, a), axis=a), a)
    return float((wall | (shut >= SHUT)).sum()) * h ** 3


def outline(points, tris, along, cells=CELLS):
    """The area a surface covers seen along axis ``along``, in its units squared: its cells with that axis laid flat.
    A hole seen right through stays a hole."""
    flat = np.array(points, float)
    flat[:, along] = 0
    wall, h, _ = _wall(flat, tris, cells)
    return float(wall.sum()) * h ** 2


def _flood(wall):
    """The outside of ``wall``: flooded in from the grid's border, face to face, up to it."""
    outside = np.zeros(wall.shape, bool)
    for ax in range(wall.ndim):
        if wall.shape[ax] > 1:              # a plane, (A, B, 1), has its border on its first two axes
            outside[(slice(None),) * ax + ([0, -1],)] = True
    free = ~wall
    outside &= free
    count = outside.sum()
    while True:
        for _ in range(8):
            outside = _grow(outside) & free
        n = outside.sum()
        if n == count:
            return outside
        count = n


def _section(points, tris, k, at, step):
    """Points along the surface's section by the plane where coordinate ``k`` is ``at``, no farther apart than
    ``step``, in chunks: per triangle across it, the line between the two edges that cross it."""
    d = points[:, k] - at
    over = d[tris] >= 0
    t = tris[over.any(axis=1) & ~over.all(axis=1)]
    if not len(t):
        return
    e0, e1 = t, np.roll(t, -1, axis=1)                      # its edges: (0, 1), (1, 2), (2, 0)
    d0, d1 = d[e0], d[e1]
    crosses = (d0 >= 0) != (d1 >= 0)
    w = d0 / np.where(crosses, d0 - d1, 1)
    p = points[e0] + (points[e1] - points[e0]) * w[..., None]
    rows = np.arange(len(t))
    u, v = p[rows, np.argmax(crosses, axis=1)], p[rows, 2 - np.argmax(crosses[:, ::-1], axis=1)]
    n = np.maximum(1, np.ceil(np.linalg.norm(v - u, axis=1) / step)).astype(np.int64)
    for m in np.unique(n):
        f = np.linspace(0, 1, m + 1)[None, :, None]
        yield (u[n == m][:, None] * (1 - f) + v[n == m][:, None] * f).reshape(-1, 3)


def _samples(points, tris, step):
    """Points over every triangle no farther apart than ``step``, in chunks."""
    a, b, c = (points[tris[:, i]] for i in range(3))
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=0)
    k = np.maximum(1, np.ceil(longest / step)).astype(np.int64)
    for n in np.unique(k):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        keep = i + j <= n
        u, v = (i[keep] / n)[None, :, None], (j[keep] / n)[None, :, None]
        sel = np.flatnonzero(k == n)
        per = max(1, 2_000_000 // u.shape[1])
        for s in range(0, len(sel), per):
            t = sel[s:s + per]
            yield (a[t][:, None] * (1 - u - v) + b[t][:, None] * u + c[t][:, None] * v).reshape(-1, 3)


def _grow(m):
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def _depth(solid):
    """Per cell, how many cells face to face to the outside (0 outside)."""
    d = np.where(solid, np.iinfo(np.int32).max, 0).astype(np.int64)
    while True:
        new = d.copy()
        for ax in range(3):
            for s in (1, -1):
                np.minimum(new, np.roll(d, s, axis=ax) + 1, out=new)
        new[~solid] = 0
        if (new == d).all():
            return d
        d = new


def _distance(vol, start):
    """Per solid cell, the shortest way inside the volume from cell ``start``, in cells (26 neighbours)."""
    n = len(vol.cells)
    dist = [np.inf] * n
    dist[start] = 0.0
    done = bytearray(n)
    nbs = vol.neighbours.tolist()
    steps = STEPS.tolist()
    heap = [(0.0, start)]
    while heap:
        d, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = 1
        for j, w in zip(nbs[i], steps):
            if j >= 0 and not done[j]:
                e = d + w
                if e < dist[j]:
                    dist[j] = e
                    heapq.heappush(heap, (e, j))
    return np.array(dist)


def _label(n, a, b):
    """Connected parts of ``n`` items joined by the pairs (a[i], b[i]), numbered from 0."""
    label = np.arange(n)
    while True:
        la, lb = label[a], label[b]
        m = np.minimum(la, lb)
        new = label.copy()
        np.minimum.at(new, la, m)
        np.minimum.at(new, lb, m)
        while not (new[new] == new).all():
            new = new[new]
        if (new == label).all():
            break
        label = new
    return np.unique(label, return_inverse=True)[1].ravel()


def sides(vol, cells, side, centre):
    """Per cell, the side of the mirror plane (across ``side`` through ``centre``) it lies out on: 1, -1, or 0 within
    two cells of the plane."""
    across = (vol.middles(cells) - centre) @ np.asarray(side, float)
    return np.where(across > 2 * vol.h, 1, np.where(across < -2 * vol.h, -1, 0))


def _sided(node, at, labels):
    """Per label, the side its cells lie on (ONE_SIDE of them out on it, as a limb's), else 0: across the plane."""
    u, inv = np.unique(labels, return_inverse=True)
    n = np.bincount(inv)
    plus, minus = np.bincount(inv, at == 1), np.bincount(inv, at == -1)
    return dict(zip(u.tolist(), np.where(plus >= ONE_SIDE * n, 1, np.where(minus >= ONE_SIDE * n, -1, 0)).tolist()))


def apart(vol, band, node, a, b, side, centre):
    """Twins touching below where they part are apart: a slice across the mirror plane joined toward the start
    mostly to slices each on one side of it, one either side, is the two touching (close-set legs at the knees,
    scarecrow's ankle ropes), and its cells part by side. Else the tree joining the slices breaks the loop the touch
    makes anywhere, and the legs part at the lowest touch: scarecrow's hips read at his ropes, 0.19 of his height,
    where they part at 0.41. ``a``, ``b``: the touching cells of neighbouring bands."""
    node = node.copy()
    at = sides(vol, None, side, centre)
    over = (vol.middles() - centre) @ np.asarray(side, float) >= 0
    a, b = np.where(band[a] > band[b], a, b), np.where(band[a] > band[b], b, a)      # a the farther from the start
    keep = band[a] == band[b] + 1
    a, b = a[keep], b[keep]
    order = np.argsort(band, kind='stable')
    bands = np.split(order, np.flatnonzero(np.diff(band[order])) + 1)
    above, fresh = {}, node.max() + 1
    for cells in bands:
        here = _sided(node, at[cells], node[cells])
        k = band[a] == band[cells[0]]
        x, y = node[a[k]], node[b[k]]
        for s in (s for s, w in here.items() if w == 0):
            up = np.array([above.get(t, 0) for t in y[x == s]])
            if (up == 1).any() and (up == -1).any() and (up != 0).sum() > (up == 0).sum():
                mine = cells[node[cells] == s]
                node[mine[over[mine]]] = fresh
                fresh += 1
        above = _sided(node, at[cells], node[cells])
    return np.unique(node, return_inverse=True)[1].ravel()


class Graph:
    """The slices of the volume at each distance inside it from the cell ``start`` and how they join: per node, its
    cells, its middle and how thick the volume is there; ``links``, the tree joining them. Given the mirror plane
    (``side`` through ``centre``), twins touching below where they part stay apart (apart())."""

    def __init__(self, vol, start, side=None, centre=None):
        self.vol = vol
        self.dist = _distance(vol, start)
        band = np.floor(self.dist / BAND).astype(np.int64)
        half = vol.neighbours[:, :13]                            # each pair once
        i = np.repeat(np.arange(len(vol.cells)), 13)
        j = half.ravel()
        ok = j >= 0
        i, j = i[ok], j[ok]
        same = band[i] == band[j]
        self.node = _label(len(vol.cells), i[same], j[same])     # per cell, its slice
        if side is not None:
            self.node = apart(vol, band, self.node, i[~same], j[~same], side, centre)
        n = self.node.max() + 1
        self.band = np.zeros(n, np.int64)
        self.band[self.node] = band
        mid = vol.middles()
        self.count = np.bincount(self.node, minlength=n)
        self.middle = np.stack([np.bincount(self.node, mid[:, d], minlength=n) for d in range(3)], axis=1) \
            / self.count[:, None]
        self.thick = np.zeros(n)
        np.maximum.at(self.thick, self.node, vol.depth * vol.h)
        self.start = int(self.node[start])
        # slices of neighbouring bands joined by touching cells, the tree keeping the strongest joins
        a, b = self.node[i[~same]], self.node[j[~same]]
        pairs, strength = np.unique(np.sort(np.stack([a, b], axis=1), axis=1), axis=0, return_counts=True)
        group = list(range(n))

        def find(x):
            while group[x] != x:
                group[x] = group[group[x]]
                x = group[x]
            return x
        self.links = [set() for _ in range(n)]
        for k in np.argsort(-strength, kind='stable'):
            p, q = (int(x) for x in pairs[k])
            if find(p) != find(q):
                group[find(p)] = find(q)
                self.links[p].add(q)
                self.links[q].add(p)
        self.alive = np.ones(n, bool)

    def prune(self, keep=()):
        """Take off the bumps: a branch from a fork to its tip no longer than BUMP times its own width (a limb is long
        for its width, a bump is not), shortest first, until none is left; its cells go to the fork. A branch with a
        node of ``keep`` on it (a joint the look named) is no bump, however short: what a body stands on is a limb."""
        keep = set(keep)
        while True:
            best = None
            for leaf in np.flatnonzero(self.alive):
                if len(self.links[leaf]) != 1 or leaf == self.start:
                    continue
                path = [leaf]
                while len(self.links[path[-1]]) <= 2 and len(path) <= len(self.links):
                    nxt = [x for x in self.links[path[-1]] if x not in path[-2:]]
                    if not nxt:
                        break
                    path.append(nxt[0])
                fork = path[-1]
                if len(self.links[fork]) < 3 or self.start in path[:-1] or any(x in keep for x in path[:-1]):
                    continue                                    # the whole graph is one chain, the start's, or named
                length = BAND * self.vol.h + sum(np.linalg.norm(self.middle[x] - self.middle[y])
                                                 for x, y in zip(path[:-2], path[1:-1]))
                width = 2 * self.thick[path[:-1]].max()
                if length < BUMP * width and (best is None or length < best[0]):
                    best = (length, path)
            if best is None:
                return
            path = best[1]
            fork = path[-1]
            for x in path[:-1]:
                for y in self.links[x]:
                    self.links[y].discard(x)
                self.links[x] = set()
                self.alive[x] = False
                self.node[self.node == x] = fork

    def thin(self, every=DOTS):
        """One node for ``every`` slices along each part: the slices from the start or a fork out to the next fork or
        to a tip, in turn (the start's own slice and a fork's stay as they are). A node's middle is then its cells',
        its thickness its thickest slice's."""
        parent = self.rooted(self.start)
        kids, order, _ = _tree(self, parent, self.start)
        for x in order:
            if x != self.start and len(kids[x]) < 2:
                continue                                        # a part begins at the start or at a fork
            for k in kids[x]:
                chain, c = [], k
                while len(kids[c]) < 2:
                    chain.append(c)
                    if not kids[c]:
                        break
                    c = kids[c][0]
                for i in range(0, len(chain), every):
                    run = chain[i:i + every]
                    if len(run) < 2:
                        continue
                    n = self.count[run].astype(float)
                    self.middle[run[0]] = (self.middle[run] * n[:, None]).sum(axis=0) / n.sum()
                    self.thick[run[0]] = self.thick[run].max()
                    self.count[run[0]] = self.count[run].sum()
                    self.merge(run)

    def copy(self):
        """The same slices over the same volume, its nodes and links its own: what is merged or pruned in the copy
        leaves this one as it was."""
        g = copy.copy(self)
        g.node, g.alive = self.node.copy(), self.alive.copy()
        g.middle, g.thick, g.count = self.middle.copy(), self.thick.copy(), self.count.copy()
        g.links = [set(s) for s in self.links]
        return g

    def merge(self, nodes):
        """Make ``nodes`` (joined in a chain) one node, the first: its cells, its links."""
        if len(nodes) < 2:
            return
        first, rest = nodes[0], set(nodes[1:])
        for x in rest:
            for y in self.links[x]:
                self.links[y].discard(x)
                if y != first and y not in rest:
                    self.links[y].add(first)
                    self.links[first].add(y)
            self.links[x] = set()
            self.alive[x] = False
        self.node[np.isin(self.node, list(rest))] = first

    def rooted(self, root):
        """Parent per node (-1 for ``root`` and the dead), from ``root`` out."""
        parent = np.full(len(self.links), -1)
        seen, todo = {root}, [root]
        while todo:
            x = todo.pop()
            for y in self.links[x]:
                if y not in seen:
                    seen.add(y)
                    parent[y] = x
                    todo.append(y)
        return parent


def mirror(vol, axes=np.eye(3)):
    """(axis, a point on the plane): of ``axes``, the one whose best plane across it maps the most of the volume
    onto itself, each cell counting by its depth: the trunk and the head mirror across the body's plane in any pose,
    where the limbs need not."""
    best = None
    for ax in axes:
        k = int(np.argmax(np.abs(ax)))
        c = vol.cells[:, k]
        for at in np.arange(c.min(), c.max() + 0.5, 0.5):
            m = np.round(2 * at - c).astype(np.int64)
            ok = (m >= 0) & (m < vol.solid.shape[k])
            img = vol.cells[ok].copy()
            img[:, k] = m[ok]
            score = vol.depth[ok][vol.solid[img[:, 0], img[:, 1], img[:, 2]]].sum()
            if best is None or score > best[0]:
                best = (score, np.asarray(ax, float), k, at)
    _, ax, k, at = best
    point = vol.origin.copy()
    point[k] += (at + 0.5) * vol.h
    return ax, point


def top(vol, up, side, centre):
    """The highest cell near the mirror plane (within MIDLINE of the model's width: a raised hand is off it), the
    middle one of a flat top: where the slices start. On the plane they cut both sides alike; on a body standing up
    it is the top of its head, which nothing here needs to know."""
    mid = vol.middles()
    across = (mid - centre) @ side
    near = np.flatnonzero(np.abs(across) <= MIDLINE * np.ptp(across))
    h = mid[near] @ up
    top = near[h >= h.max() - vol.h]
    return int(top[np.argmin(np.linalg.norm(mid[top] - mid[top].mean(axis=0), axis=1))])


def _up(parent, x):
    out = [x]
    while parent[out[-1]] >= 0:
        out.append(parent[out[-1]])
    return out


@dataclass
class Skeleton:
    parents: np.ndarray             # parent bone per bone, -1 for the root
    heads: np.ndarray               # (B, 3)
    tails: np.ndarray               # (B, 3)
    graph: 'Graph'
    bone_of: dict                   # the graph's node -> the bone its cells go with
    thick: np.ndarray               # per bone, how thick the volume is about it
    nodes: list = field(default_factory=list)   # per bone, its node: the bone is named for it (names())
    spine: list = field(default_factory=list)   # bones from the hips to the skull, once those are named
    arms: list = field(default_factory=list)    # each named arm's first bone, the first off the spine
    head: int = None                # the head's bone, once the skull is named
    twins: dict = field(default_factory=dict)  # the named joints of the left and the right, each the other's

    def add(self, head, tail, parent, thick):
        """A bone more; its number."""
        if np.linalg.norm(tail - head) < 0.5 * self.graph.vol.h:     # Blender drops a bone of no length
            tail = head + np.array([0., 0., 0.5 * self.graph.vol.h])
        self.heads = np.vstack([self.heads, head])
        self.tails = np.vstack([self.tails, tail])
        self.parents = np.append(self.parents, parent)
        self.thick = np.append(self.thick, thick)
        return len(self.parents) - 1

    def names(self):
        return [f'{NAME}{x}' for x in self.nodes]


def _bones(g, root, joints=None):
    """The Skeleton with a bone per node of ``g``, the tree hanging from ``root``: a bone's head at its slices' middle
    (or where ``joints`` says its joint lies), its tail at the head of the node after it down its main branch (the one
    with the most volume beyond it), or at a tip as far as its cells reach."""
    joints = joints or {}
    parent = g.rooted(root)
    kids, order, held = _tree(g, parent, root)

    def at(x):
        return joints[x] if x in joints else g.middle[x]
    skel = Skeleton(np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3)), g, {}, np.zeros(0))
    for x in order:
        tail = at(max(kids[x], key=held.get)) if kids[x] else \
            _tip(g, x, g.middle[parent[x]] if parent[x] >= 0 else None)
        skel.bone_of[x] = skel.add(at(x), tail, skel.bone_of.get(int(parent[x]), -1), g.thick[x])
        skel.nodes.append(int(x))
    return skel


def _below(kids, x):
    """Every node past ``x`` in the tree."""
    out, todo = set(), list(kids[x])
    while todo:
        y = todo.pop()
        out.add(y)
        todo.extend(kids[y])
    return out


LIMBS = {'legs': ('hip', 'knee', 'ankle', 'toe'), 'arms': ('clavicle', 'shoulder', 'elbow', 'hand')}    # from the trunk out


def _node(name, alive):
    """The node a record's joint names (``slice<n>``), one of ``alive``."""
    number = name[len(NAME):] if isinstance(name, str) and name.startswith(NAME) else ''
    if not number.isdigit() or int(number) not in alive:
        raise ValueError(f'record.json names {name!r}, which is no joint fitted inside this model')
    return int(number)


def is_fit(names):
    """Whether bones of these names are a fit's joints (read()'s): every one ``slice<n>``."""
    return bool(names) and all(re.fullmatch(NAME + r'\d+', str(n)) for n in names)


def on_limbs(graph, rec):
    """``rec`` with each limb's joints on the limb. A look names them on the tree as read() hangs it, from the top,
    where the trunk above a shoulder leads on to the arm. Hung from the hips (finish()) that trunk joint is no longer
    above the hand, and a trunk joint is no part of an arm (Big Smoke, dots-1: his right clavicle named one joint up
    the trunk from where the arm leaves, the record refused). So a limb's joint named on the spine, between the hips
    and the skull, is the limb's first joint off it on the way to the limb's end; a clavicle that is then the
    shoulder is left out. A record whose limbs lie off the spine comes back saying what it said."""
    alive = set(np.flatnonzero(graph.alive).tolist())
    parent = graph.rooted(_node(rec['hips'], alive))
    spine = {int(x) for x in _up(parent, _node(rec['skull'], alive))}
    out = dict(rec)
    for kind, parts in LIMBS.items():
        if not rec.get(kind):
            continue
        out[kind] = {}
        for side, limb in rec[kind].items():
            mine = dict(limb)
            said = [p for p in parts if mine.get(p) is not None]
            way = [int(x) for x in _up(parent, _node(mine[said[-1]], alive))] if said else []
            off = next((i for i, x in enumerate(way) if x in spine), 0)
            moved = [p for p in said if _node(mine[p], alive) in spine] if off else []
            for p in moved:
                mine[p] = f'{NAME}{way[off - 1]}'
            if moved:
                mine.pop('bones', None)         # its joints are named anew: the chain is the tree's path between them
                if mine.get('clavicle') is not None and mine['clavicle'] == mine.get('shoulder'):
                    del mine['clavicle']
            out[kind][side] = mine
    return out


def on_shoulders(graph, vol, rec, points=None, tris=None):
    """``rec`` with a neck where it says the body has none (``"neck": None``: the look saw a head sitting straight on
    the shoulders): the spine joint between the shoulders, the one nearest the middle of the arms' shoulder joints as
    the fit places them (shoulder_joints()), never before the joint the arms leave the spine from; the chest's on a
    body with no arms. The fighter's head goes on the neck joint (fill.FITTED), and a look places where a head begins
    within a joint or two: on a face that reaches the shoulders that margin left the grin on the trunk (Crash, dots-1:
    his neck named at the top of his grin, the rest of it rode Mario's chest; one joint lower, dots-3, it held).
    Whether a body has a neck is the look's to say; where its shoulders are is the fit's own. Where the arms leave the
    spine is not that: an arm lying against the side comes free of it far below the shoulder (Richard, dots-4: his
    arms leave at his waist, and all above it was his head). A chest named past the neck's joint is that joint. A
    record naming its neck, or with no such entry (the skull's, as in any record), comes back as it was."""
    if 'neck' not in rec or rec['neck'] is not None:
        return rec
    alive = set(np.flatnonzero(graph.alive).tolist())
    parent = graph.rooted(_node(rec['hips'], alive))
    spine = [int(x) for x in _up(parent, _node(rec['skull'], alive))]      # from the skull down to the hips
    leave = []
    for limb in (rec.get('arms') or {}).values():
        ends = [limb[p] for p in LIMBS['arms'] if limb.get(p) is not None]
        if ends:
            leave.append(next(spine.index(int(x)) for x in _up(parent, _node(ends[-1], alive)) if int(x) in spine))
    if not leave:
        return dict(rec, neck=rec['chest'])

    def held(node):
        out = dict(rec, neck=f'{NAME}{node}')
        chest = _node(rec['chest'], alive)
        if chest in spine and spine.index(chest) < spine.index(node):
            out['chest'] = out['neck']
        return out
    left = spine[min(leave)]
    skel = finish(graph, vol, held(left), points, tris)         # with the head where the arms leave: its shoulders
    if not skel.arms:
        return held(left)
    between = skel.heads[skel.arms].mean(axis=0)
    return held(min(spine[:min(leave) + 1], key=lambda x: float(np.linalg.norm(graph.middle[x] - between))))


def finish(graph, vol, rec, points=None, tris=None):
    """The Skeleton once a record names the body's joints (``rec``: hips, chest, neck, skull, legs and arms as
    fill.entries writes them, each a bone of read()'s by its name). ``graph`` is left as it was.
    - A limb's joints lie on the limb (on_limbs()).
    - A body with no neck has its head on its shoulders (on_shoulders()).
    - The tree hangs from the hips.
    - The head is whole: all that lies past the skull is one bone with it, as an artist binds it (hair, a hat, a
      beak), but a joint the record names there, with the way to it and what lies past it.
    - A bump is no limb (Graph.prune), unless a named joint is on it.
    - A shoulder's joint lies inside the shoulder (shoulder_joints()): each named arm's first bone, the first off the
      spine on the way to its shoulder, begins there. ``points``, ``tris``: the welded surface, which says where an
      arm of its own pieces begins.
    - The named joints of the left and the right are twins."""
    g = graph.copy()
    alive = set(np.flatnonzero(g.alive).tolist())
    rec = on_shoulders(graph, vol, on_limbs(graph, rec), points, tris)

    def at(name):
        return _node(name, alive)
    hips, skull = at(rec['hips']), at(rec['skull'])
    said = {hips, skull, at(rec['chest']), at(rec.get('neck') or rec['skull'])}
    limbs, begins = {}, {}
    for kind in ('legs', 'arms'):
        for side, limb in (rec.get(kind) or {}).items():
            limbs[kind, side] = {part: at(n) for part, n in limb.items()
                                 if part not in ('bones', 'begins') and n is not None}
            said |= set(limbs[kind, side].values()) | {at(n) for n in limb.get('bones') or []}
            if limb.get('begins') is not None:      # the spine joint a look saw the shoulder at (shoulder_joints())
                begins[kind, side] = at(limb['begins'])
    g.start = hips
    parent = g.rooted(hips)
    kids, _, _ = _tree(g, parent, hips)
    past = _below(kids, skull)
    kept = set()
    for x in past & said:
        kept |= set(_up(parent, x)) | _below(kids, x)
    g.merge([skull] + sorted(past - kept))
    g.prune(keep=said)
    parent = g.rooted(hips)
    kids, _, held = _tree(g, parent, hips)
    spine = [int(x) for x in _up(parent, skull)[::-1]]
    on = set(spine)

    def chain(top, end):
        """A limb's nodes: from its first off the spine, on the way to ``top``, through to ``end`` and on along its
        heaviest branch to its tip. None when ``top`` is on the spine itself."""
        way = [int(x) for x in _up(parent, top)]
        i = next(i for i, x in enumerate(way) if x in on)
        if i == 0:
            return None
        first = way[i - 1]
        down = [int(x) for x in _up(parent, end)]
        mid = down[:down.index(first) + 1][::-1] if first in down else [first]
        return mid + _out_along(kids, held, mid[-1])[1:]
    hung = [(chain(j['shoulder'], j['hand']), key, j) for key, j in sorted(limbs.items())
            if key[0] == 'arms' and 'shoulder' in j and 'hand' in j]
    hung = [x for x in hung if x[0]]
    arms = [c for c, _, _ in hung]
    seen = [(begins[key], j['hand']) if key in begins else None for _, key, j in hung]
    legs = [c for c in (chain(j['hip'], j.get('toe', j['ankle'])) for (kind, _), j in sorted(limbs.items())
                        if kind == 'legs' and 'hip' in j and 'ankle' in j) if c]
    joints = {}
    if arms:
        leave = min(spine.index(int(parent[c[0]])) for c in arms)
        joints = dict(zip((c[0] for c in arms),
                          shoulder_joints(g, g.middle[spine[leave]], kids, arms, legs, vol, points, tris, seen)))
    skel = _bones(g, hips, joints)
    skel.spine = [skel.bone_of[x] for x in spine]
    skel.head = skel.bone_of[skull]
    skel.arms = [skel.bone_of[c[0]] for c in arms]
    for kind in ('legs', 'arms'):
        left, right = limbs.get((kind, 'left'), {}), limbs.get((kind, 'right'), {})
        for part in set(left) & set(right):
            a, b = skel.bone_of[left[part]], skel.bone_of[right[part]]
            if a != b:
                skel.twins.update({a: b, b: a})
    return skel


def _tip(g, x, before=None):
    """How far the slice ``x`` at a tip reaches: along where its cells lie from its middle (a head's are above its
    neck), else on along the chain from ``before``, as far as they go."""
    cells = g.vol.middles(np.flatnonzero(g.node == x)) - g.middle[x]
    d = cells.mean(axis=0) if len(cells) else np.zeros(3)
    if np.linalg.norm(d) < g.vol.h and before is not None:
        d = g.middle[x] - before
    if np.linalg.norm(d) < 1e-12:
        d = np.array([0., 0., 1.])
    d = d / np.linalg.norm(d)
    return g.middle[x] + d * max((cells @ d).max() if len(cells) else 0.0, 0.5 * g.vol.h)


def _tree(g, parent, root):
    """(per node its children, the nodes from ``root`` out, per node the cells it carries: its own and all beyond)."""
    kids = {int(x): [] for x in np.flatnonzero(g.alive)}
    for x in kids:
        if parent[x] >= 0:
            kids[parent[x]].append(x)
    order = [root]
    for x in order:
        order.extend(sorted(kids[x]))
    held = dict(zip(order, np.bincount(g.node, minlength=len(g.links))[order].astype(float)))
    for x in order[:0:-1]:
        held[parent[x]] += held[x]
    return kids, order, held


def _out_along(kids, held, first):
    """A limb from its first node out along its heaviest branch to its tip."""
    chain = [first]
    while kids[chain[-1]]:
        chain.append(max(kids[chain[-1]], key=held.get))
    return chain


WELD = 1e-5         # vertices this close (of the model's size) are one: the copies a file makes along seams and cuts


def _weld(points, tris):
    """(per vertex, its welded vertex; the welded vertices; the welded triangles, each once, none of no area)."""
    size = np.ptp(points, axis=0).max()
    _, first, weld = np.unique(np.round(points / (WELD * size)).astype(np.int64), axis=0, return_index=True,
                               return_inverse=True)
    weld = weld.ravel()
    t = weld[tris]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    _, once = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    return weld, points[first], t[np.sort(once)]


def _cells(points, tris, vol):
    """Per vertex, the cell just inside it (along its normal, either way it points; else the nearest)."""
    import bpy  # noqa: F401  (mathutils comes with it)
    from mathutils.kdtree import KDTree
    n = np.zeros_like(points)
    a, b, c = (points[tris[:, i]] for i in range(3))
    f = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(n, tris[:, i], f)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cell = np.full(len(points), -1)
    for s in (-0.75, 0.75):
        ijk = np.floor((points + s * vol.h * n - vol.origin) / vol.h).astype(np.int64)
        ok = (cell < 0) & (ijk >= 0).all(axis=1) & (ijk < vol.solid.shape).all(axis=1)
        idx = np.full(len(points), -1)
        idx[ok] = vol.index[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        cell = np.where(idx >= 0, idx, cell)
    lost = np.flatnonzero(cell < 0)
    if len(lost):
        mid = vol.middles()
        tree = KDTree(len(mid))
        for i, q in enumerate(mid):
            tree.insert(q, i)
        tree.balance()
        cell[lost] = [tree.find(points[i])[1] for i in lost]
    return cell


def _owners(points, tris, vol, skel):
    """Per vertex, the bone of the slice just inside it."""
    of_node = np.full(len(skel.graph.links), -1)
    for x, b in skel.bone_of.items():
        of_node[x] = b
    return of_node[skel.graph.node[_cells(points, tris, vol)]]


def across(points):
    """How a line of points is looked at square on: (the way its steps reach least, each counted by its length, or
    None where they all run one way; the way they mostly run). A bent arm's steps lie in a plane of its own and that
    way is square to it; a leg and its foot are seen from the side. Not the plane of the points' places: a leg that
    only slants reads as flat there (Goku's thigh, pointing forward, was looked at end on)."""
    steps = np.diff(np.asarray(points, float), axis=0)
    long = np.linalg.norm(steps, axis=1)
    steps = steps[long > 1e-9] / np.sqrt(long[long > 1e-9])[:, None]
    if not len(steps):
        return None, np.array([0., 0., 1.])
    spread, axes = np.linalg.svd(steps)[1:]
    return (axes[2] if len(spread) >= 2 and spread[1] > FLAT * spread[0] else None), axes[0]


def shoulder_joints(g, chest_mid, kids, arms, legs, vol, points=None, tris=None, seen=None):
    """Per arm (its chain of slices), its shoulder joint: inside the shoulder, the arm's thickness in from its first
    slice towards ``chest_mid``, the middle of the slice the arms part from (its own slices start below the armpit).
    An arm of its own surface hanging by the trunk (a statue's arm at its side, its top under a sleeve) is fused with
    the trunk in the grid down to where it parts from it, so its slices begin low, by the elbow, and it would turn
    about there; its own surface says where it begins: the joint goes the arm's thickness in from the far end of its
    own volume from the hand (sliced inside it alone, from the hand), when that lies farther from the hand inside it
    than the slices' joint does (a glove's own surface ends at the wrist, nearer the hand; an arm hanging down joins
    the trunk above the slice it parts from, never nearer that slice's middle).
    ``seen``: per arm, None or (the spine slice a look saw its shoulder at, its hand's slice): the record's
    ``begins``. A look shown the arm square on names its shoulder at the spine joint the arm leaves when that joint
    is drawn where the shoulder is: the arm is held against the body, its own slices begin by the elbow, and the
    surface that would say where it begins stops at a sleeve's hem. The joint is then where that spine joint is seen
    from square onto the arm: its place moved into the arm's own plane (across()). Goku's right arm (arm-1,
    2026-11-10): the joint lay an arm's thickness under his sleeve's hem, the upper arm's bone was 5.7 of his height
    long beside the left one's 15.1, and Captain Falcon's upper arm stretched it 2.7 times into a slab; seen so it is
    12.7 long and stretched 1.2 times, as the left is. An arm with no plane of its own (straight) keeps the joint
    above."""
    out = _shoulder_joints(g, chest_mid, kids, arms, legs, vol, points, tris)
    for i, (c, s) in enumerate(zip(arms, seen or [])):
        if s is None:
            continue
        spine, hand = s
        line = np.array([g.middle[x] for x in (c[:c.index(hand) + 1] if hand in c else c)], float)
        square = across(line)[0]
        if square is not None:
            out[i] = g.middle[spine] - ((g.middle[spine] - line[0]) @ square) * square
    return out


def _shoulder_joints(g, chest_mid, kids, arms, legs, vol, points=None, tris=None):
    """shoulder_joints() before a look's word on where a shoulder was seen."""
    out = []
    for c in arms:
        first = g.middle[c[0]]
        reach = np.linalg.norm(first - chest_mid)
        out.append(first - (first - chest_mid) / max(reach, 1e-12) * min(g.thick[c[0]], reach / 2))
    if points is None or not arms:
        return out
    piece = segments._pieces(points, tris)
    body = np.argmax(np.bincount(piece))
    node = g.node[_cells(points, tris, vol)]

    def below(xs):
        found, todo = set(xs), list(xs)
        while todo:
            x = todo.pop()
            found.update(kids[x])
            todo.extend(kids[x])
        return found
    limbs = [below(c) for c in arms + legs]
    for i, c in enumerate(arms):
        arm = limbs[i]
        others = set().union(*(x for j, x in enumerate(limbs) if j != i))
        # the arm's own surfaces: the pieces apart from the body's a third or more on it and next to none on another
        # limb (an upper arm under a sleeve and a forearm worn into it at the elbow are one arm)
        ours = [k for k in np.unique(piece) if k != body and np.isin(node[piece == k], list(arm)).mean() >= 1 / 3
                and np.isin(node[piece == k], list(others)).mean() <= 0.1]
        if not ours:
            continue
        ids = np.flatnonzero(np.isin(piece, ours))
        remap = np.full(len(points), -1)
        remap[ids] = np.arange(len(ids))
        pts = points[ids]
        pv = Volume(pts, remap[tris[np.isin(tris, ids).all(axis=1)]],
                    cells=max(8, int(np.ceil(np.ptp(pts, axis=0).max() / vol.h))))
        mid = pv.middles()
        hand = _tip(g, c[-1], g.middle[c[-2]] if len(c) > 1 else None)
        d = _distance(pv, int(np.argmin(np.linalg.norm(mid - hand, axis=1))))
        r = g.thick[c[0]] / pv.h
        band = (d >= d.max() - r - BAND) & (d <= d.max() - r)
        at = int(np.argmin(np.linalg.norm(mid - out[i], axis=1)))      # the slices' joint, in the arm's own volume
        if band.any() and d[band].mean() > d[at]:
            out[i] = mid[band].mean(axis=0)
    return out


def slices(points, tris, vol, skel):
    """Skin weights (V, B) from the slices: a vertex rides the bone of the slice just inside it (along its normal: an
    arm's inner side lies by the trunk but inside the arm), the shoulder over an arm's joint with the arm (shoulders());
    blended with the bones near it along the skeleton (within reach of the tree from its own, so a hand resting on a
    hip never takes the hip's) by how far it lies from each, over the narrower one's thickness: joints bend smoothly,
    an end (a head) stays whole."""
    parents, heads, tails, thick = skel.parents, skel.heads, skel.tails, skel.thick
    own = _owners(points, tris, vol, skel)
    piece = segments._pieces(points, tris)
    shoulders(points, piece, own, skel, thick)
    small = worn(points, piece, own, skel, thick)
    links = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p >= 0:
            links[i].append(p)
            links[p].append(i)
    w = np.zeros((len(points), len(parents)))
    for u in np.unique(own):
        mine = np.flatnonzero(own == u)
        # the bones within reach along the tree: three of its own thickness past its own length
        reach = np.linalg.norm(tails[u] - heads[u]) + BLEND * 3 * thick[u]
        near, todo = {u: 0.0}, [u]
        while todo:
            x = todo.pop()
            for y in links[x]:
                if u == skel.head and y == parents[u]:
                    continue                        # the head is whole: the neck blends into it, not it into the neck
                d = near[x] + np.linalg.norm(heads[y] - heads[x])
                if y not in near and d <= reach:
                    near[y] = d
                    todo.append(y)
        for y in near:
            d = _to_segment(points[mine], heads[y], tails[y])
            sigma = BLEND * min(thick[u], thick[y])
            w[mine, y] = np.exp(-(d / sigma) ** 2) if y != u else 1.0
    for k, b in small.items():                     # a small piece rides its bone whole
        w[piece == k] = 0.0
        w[piece == k, b] = 1.0
    s = w.sum(axis=1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


def shoulders(points, piece, own, skel, thick):
    """The shoulders go with the arms: of the trunk's vertices from where the named arms leave it up to the head (the
    slices there run across the shoulders, from one deltoid over the neck to the other), each one on the arm's own
    surface (the piece most of its first bones' vertices are on: a sleeve's edge over an arm of its own stays with the
    shirt), nearer the arm, from its joint out along its thickness, than the spine, and within SHOULDER of its
    thickness of it, goes with that arm's first bone. Changes ``own``."""
    parents, heads, tails = skel.parents, skel.heads, skel.tails
    leave = [skel.spine.index(parents[a]) for a in skel.arms if parents[a] in skel.spine]
    if not leave:
        return
    upper = skel.spine[min(leave):skel.spine.index(skel.head)] if skel.head in skel.spine else skel.spine[min(leave):]
    mine = np.flatnonzero(np.isin(own, upper))
    if not len(mine):
        return
    p = points[mine]
    trunk = np.min([_to_segment(p, heads[b], tails[b]) for b in upper], axis=0)
    kids = [[] for _ in parents]
    for i, q in enumerate(parents):
        if q >= 0:
            kids[q].append(i)
    near = []
    for a in skel.arms:
        segs, todo, along = [a], [a], {a: 0.0}
        while todo:
            x = todo.pop()
            for y in kids[x]:
                along[y] = along[x] + np.linalg.norm(heads[y] - heads[x])
                if along[y] <= 4 * thick[a]:
                    segs.append(y)
                    todo.append(y)
        d = np.min([_to_segment(p, heads[b], tails[b]) for b in segs], axis=0)
        surface = piece[np.isin(own, segs)]
        if len(surface):
            d[piece[mine] != np.bincount(surface).argmax()] = np.inf
        near.append(d)
    near = np.array(near)
    best = np.argmin(near, axis=0)
    reach = SHOULDER * np.array([thick[a] for a in skel.arms])[best]
    d = near[best, np.arange(len(mine))]
    go = (d < trunk) & (d <= reach)
    own[mine[go]] = np.array(skel.arms)[best[go]]


def worn(points, piece, own, skel, thick):
    """A piece worn on the body (a sleeve-less arm under a shirt, a glove, a boot, goggles, a braid: a surface of its
    own, apart from the body's largest) rides one part. When a third of it or more lies on one branch off the spine
    (an arm, a leg, a tail), it is that branch's, all of it: an arm that hangs by the trunk lies in the trunk's slices
    down to where it parts from it, but it is an arm. Else it rides the limb of the bone carrying most of it, from the
    root out to it and on past it: a braid hanging down the back to the thighs lies in their slices, but it hangs
    from the head. A vertex on another part takes the nearest bone of its piece's. Changes ``own``. Returns {piece:
    its bone} for the pieces no bigger than their bone is thick (a buckle, a braid's tip, an eye), which ride it
    whole: blended between two bones, a small piece is only crushed."""
    parents, heads, tails = skel.parents, skel.heads, skel.tails
    body = np.argmax(np.bincount(piece))
    kids = [[] for _ in parents]
    for i, q in enumerate(parents):
        if q >= 0:
            kids[q].append(i)

    def below(b):
        out, todo = {b}, [b]
        while todo:
            x = todo.pop()
            out.update(kids[x])
            todo.extend(kids[x])
        return out
    on_spine = set(skel.spine)
    branch = np.array([next((x for x in _up(parents, b)[::-1] if x not in on_spine), -1) for b in range(len(parents))])
    small = {}
    for k in np.unique(piece):
        mine = np.flatnonzero(piece == k)
        if k == body:
            continue
        main = int(np.bincount(own[mine]).argmax())
        if np.linalg.norm(points[mine] - points[mine].mean(axis=0), axis=1).max() <= thick[main]:
            small[int(k)] = main
        held = branch[own[mine]]
        held = held[held >= 0]
        top = int(np.bincount(held).argmax()) if len(held) else -1
        if top >= 0 and (held == top).sum() >= len(mine) / 3:
            limb = below(top)
        else:
            limb = set(_up(parents, main)) | below(main)
        off = mine[~np.isin(own[mine], list(limb))]
        bones = sorted(limb)
        if not len(off) or not bones:
            continue
        d = np.array([_to_segment(points[off], heads[b], tails[b]) for b in bones])
        own[off] = np.array(bones)[d.argmin(axis=0)]
    return small


def _to_segment(p, a, b):
    d = b - a
    t = np.clip((p - a) @ d / max(d @ d, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def _armature(name, names, parents, heads, tails):
    """An armature object in the open scene with a bone per row, named by ``names``."""
    import bpy
    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(arm)
    for x in bpy.context.scene.objects:
        x.select_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bones = []
    for i in range(len(parents)):
        eb = data.edit_bones.new(names[i])
        eb.head, eb.tail = heads[i].tolist(), tails[i].tolist()
        bones.append(eb)
    for i, p in enumerate(parents):
        if p >= 0:
            bones[i].parent = bones[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def rig_up(meshes, counts, w, skel, mirror=None):
    """Give the open scene the Skeleton as its armature, in place of one an earlier call gave it, and bind every mesh
    to it with the weights ``w`` (V, B), its vertices stacked in mesh order. Then it reads as a rigged file
    (rig.scene), and stays one for whatever reads the scene again."""
    import bpy
    names = skel.names()
    for ob in meshes:
        for m in [m for m in ob.modifiers if m.type == 'ARMATURE' and m.name == ARMATURE]:
            ob.modifiers.remove(m)
        for vg in [vg for vg in ob.vertex_groups if re.fullmatch(NAME + r'\d+', vg.name)]:
            ob.vertex_groups.remove(vg)
    old = bpy.data.objects.get(ARMATURE)
    if old is not None and old.type == 'ARMATURE':
        data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        bpy.data.armatures.remove(data)
    arm = _armature(ARMATURE, names, skel.parents, skel.heads, skel.tails)
    # The builder's pairs are stated only once joints are named: unstated, a reader pairs by the skeleton (body.py).
    if skel.twins:
        arm['rigswap_twins'] = json.dumps({names[a]: names[b] for a, b in skel.twins.items()})
    if mirror is not None:
        arm['rigswap_mirror'] = np.asarray(mirror, float).tolist()
    start = 0
    for ob, n in zip(meshes, counts):
        if ob.data.users > 1:                   # its groups are its own
            ob.data = ob.data.copy()
        mine = w[start:start + n]
        start += n
        for b in np.flatnonzero(mine.max(axis=0) > 0):
            vg = ob.vertex_groups.new(name=names[b])
            for v in np.flatnonzero(mine[:, b] > 0):
                vg.add([int(v)], float(mine[v, b]), 'REPLACE')
        ob.modifiers.new(ARMATURE, 'ARMATURE').object = arm
    return arm


@dataclass
class Fitted:
    """What read() fitted in the open scene, kept while its joints are still to be named (named())."""
    blend: str
    meshes: list            # the mesh objects' names, their vertices stacked in this order
    counts: list            # vertices per mesh
    weld: np.ndarray        # per vertex, its welded vertex
    points: np.ndarray      # the welded surface
    tris: np.ndarray
    vol: Volume
    graph: Graph            # the slices, a node for every DOTS of them, the tree hanging from the top
    side: np.ndarray        # the mirror axis
    up: tuple               # (up, how it was settled, whether surely, the floor's way or None): run.swap's four
    done: tuple = None      # (the joints it was last finished for, that Rig)


FITTED = None       # the open scene's fit: read() sets it, run.swap drops it once the joints are named or left


def _give(f, skel, w):
    """Put ``skel`` and its skin (per welded vertex) in the open scene and read it: the Rig."""
    import bpy
    from . import rig as rigs
    rig_up([bpy.data.objects[n] for n in f.meshes], f.counts, w[f.weld], skel, f.side)
    return rigs.scene(f.blend)


def unnamed():
    """The open scene's fit with every joint offered and none named, each vertex on the slice just inside it: its
    Rig. What read() leaves, and what a body the look could not name goes back to after a try at it (named())."""
    f = FITTED
    skel = _bones(f.graph, f.graph.start)
    own = _owners(f.points, f.tris, f.vol, skel)
    w = np.zeros((len(f.points), len(skel.parents)))
    w[np.arange(len(own)), own] = 1.0
    f.done = None
    return _give(f, skel, w)


def named(rec):
    """Finish the open scene's fit now a record names its joints (finish()), skin it from its slices (slices()) and
    give the scene that armature in place of the one before. -> (its Rig, the record as the fit holds it: each
    limb's joints on the limb, on_limbs(), and a neck where it said the body has none, on_shoulders()). The same
    joints again are the same Rig.
    Raises ValueError for a record naming what is no joint of the fit (one made before 2026-11-10 names bone<n>)."""
    f = FITTED
    rec = on_shoulders(f.graph, f.vol, on_limbs(f.graph, rec), f.points, f.tris)
    key = json.dumps([rec.get(k) for k in ('hips', 'chest', 'neck', 'skull', 'legs', 'arms')], sort_keys=True)
    if f.done is None or f.done[0] != key:
        skel = finish(f.graph, f.vol, rec, f.points, f.tris)
        f.done = (key, _give(f, skel, slices(f.points, f.tris, f.vol, skel)))
    return f.done[1], rec


def read(blend, out, ask, up=None):
    """Open ``blend`` (load's output) and fit inside its one-piece model the joints a look can name (the first step
    above): the scene then reads as a rigged file (rig.scene), and FITTED keeps the fit for named(). The mirror axis
    is the volume's (Rig.mirror). The slices start from the top, so which way is up is settled here, as run.swap
    settles it for any model: a record's (``up``), else the way the file stands on its origin, else asked along the
    mirror axis (front.choose_up). FITTED.up says which."""
    import bpy
    from types import SimpleNamespace
    from . import body, front as front_reading
    global FITTED
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    meshes = segments._meshes()
    points, tris, counts = segments._surface(meshes)
    weld, welded, wtris = _weld(points, tris)
    vol = Volume(welded, wtris)
    side, centre = mirror(vol)
    vol.cut(side, centre, welded, wtris)
    if up is not None:
        settled = (np.asarray(up, float), 'record', True, True)
    else:
        floor = body.on_origin(SimpleNamespace(points=points), side)
        if floor is not None:
            settled = (floor, 'origin (the file stands on it)', True, floor)
        else:
            settled = (*front_reading.choose_up(side, out, ask), None)
    g = Graph(vol, top(vol, settled[0], side, centre), side, centre)
    g.thin()
    FITTED = Fitted(str(blend), [ob.name for ob in meshes], counts, weld, welded, wtris, vol, g, side, settled)
    return unnamed()
