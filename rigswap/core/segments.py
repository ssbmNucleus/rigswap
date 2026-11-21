"""A segmented model's own skeleton, read from its pieces: a Rig like a rigged file's, so every later step runs as is.

Dreamcast and N64 games built their characters from rigid pieces that turn about joints, with no armature and no
weights; their rips keep the pieces and lose the joints. What the geometry still says:
- a piece is a surface that holds together (its welded faces): what holds together moves together, and its author
  built apart what moves apart;
- a joint is where two pieces meet: the end of one sits in or on the other, or faces it across a small gap, so
  neither opens a hole as it turns. The contact is the surface of each, within reach of their nearest approach to the
  other or inside it; the joint is its middle, and its area says how much of the two meets;
- the more of two pieces meets, the surer the joint: a thigh's end sunk into the hips, against a forearm grazing the
  side of the chest. The tree takes the strongest contacts first;
- the skeleton is mirror-symmetric: a joint on one side is a joint on the other, so twin contacts are weighed
  together (a rip is rarely exact on both sides), and no piece is joined to one across the mirror plane (a left foot
  pressed against the right one is no joint);
- the trunk is the piece in the middle of the tree: taking it out leaves the fewest pieces in the largest part.
Each piece is a bone, its head at the joint to its parent, and rides it whole (weight 1), as its author built it.
On Sonic Adventure's Sonic (2026-10-25) this gives its author's own node tree (the DAE keeps it): every piece under
the one he hung it from, but the pupils on the face plate rather than beside it; the joints a median 2% of the
model's size from his.
But the strongest contact is not always a joint: a hand resting on a thigh meets it more broadly than the forearm
meets the upper arm at the elbow (Sofia), and every measurement tried overlaps between the two. Whether two pieces
are joined or only touch is meaning, so it is asked (touching), where it decides something: on the loops a contact
the tree leaves out closes through the trunk, a limb reaching the body two ways.

A one-piece mesh, whose trunk carries its limbs in its own surface, has no joints to read: out of scope (OnePiece).
Among the downloads so far: a Golduck rip (one welded surface; its 43 meshes are draw calls, each twice, cutting
across the body), Kevin, Cammy, Goku and Spyro. A ball body with its hands and feet as pieces would read as one too.
"""
import json
from pathlib import Path

import numpy as np

WELD = 1e-5         # vertices this close (of the model's size) are one: the split copies a file makes along seams
REACH = 0.01        # surface within this much of the model's size past the nearest approach to a piece touches it
INSIDE = 0.5        # a point the surface winds round this much (of a closed shell) is inside it
SPAN = 0.5          # a trunk spanning more than this of the model along every axis carries its limbs itself
COPY = 0.02         # a mesh within this share of a model's triangles holds a copy of it
# where a triangle's surface is sampled (corners, middles of the edges, centre) and the share of its area each holds
SAMPLES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]])
SHARES = np.array([1 / 12] * 3 + [1 / 6] * 3 + [1 / 4])

TOUCHING = (
    'The first picture shows one game character in its own colours from the front, in the pose its model file is '
    'bound in. It is built of rigid pieces, as old games built their characters, and its skeleton is read from where '
    'its pieces meet. Each picture after it (picture 1, 2...) shows the character in plain grey from the front, from '
    'its left side and from the back, with two of its pieces that meet in colours: 1 red, 2 blue; a piece and its '
    'mirror image, a left and a right, are numbered alike. Under the three views, each is drawn again on its own, '
    'close, in its own colours. First say who the character is, if you know. Then, for every picture, say what its '
    'two pieces are and where they meet (an upper arm and a forearm at the elbow, a hand and a thigh...), and whether '
    'they are "joined", one turning about the other there as a body bends at a joint (a shoulder, an elbow, a hip, a '
    'neck), or "touching", one only resting against the other where this pose puts it (a hand on a thigh, an arm '
    'against the side, a chin on the chest). Answer cant_tell only if you cannot tell. The pieces, by name: {parts}.')
TOUCHING_SCHEMA = {'type': 'object', 'properties': {
    'character': {'type': 'string'},
    'contacts': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'picture': {'type': 'integer'}, 'what': {'type': 'string'},
                                         'contact': {'type': 'string', 'enum': ['joined', 'touching', 'cant_tell']}},
        'required': ['picture', 'what', 'contact'], 'additionalProperties': False}}},
    'required': ['character', 'contacts'], 'additionalProperties': False}


class OnePiece(ValueError):
    """The model's trunk carries its limbs in its own surface: no pieces to read joints from."""


def pick(load_dir):
    """The model file of a load with no rigged one: the most triangles; among copies of one model (equal triangles:
    the same pieces, posed), the one spread out farthest, the rest pose rather than a curled-up form (Sonic, Rolling).
    A file holding the model several times over is copies of it, not more of it: one whose every mesh has as many
    triangles as another file's whole model (Toshinden's Sofia: her poses, five of her in one file) is left out.
    Returns its models.json entry, or None."""
    load_dir = Path(load_dir)
    entries = [e for e in json.loads((load_dir / 'models.json').read_text(encoding='utf-8'))
               if e.get('opened', True) and e.get('triangles')]
    import bpy

    def open_(e):
        bpy.ops.wm.open_mainfile(filepath=str(load_dir / 'models' / Path(e['blend'].replace('\\', '/')).name))

    def several(e):
        for t in {o['triangles'] for o in entries if o['triangles'] * 2 <= e['triangles'] * (1 + COPY)}:
            if abs(e['triangles'] / t - round(e['triangles'] / t)) > COPY * round(e['triangles'] / t):
                continue                  # its meshes can't each be one of these: their sum is no multiple of it
            open_(e)
            counts = []
            for ob in _meshes():
                ob.data.calc_loop_triangles()
                counts.append(len(ob.data.loop_triangles))
            if len(counts) > 1 and all(abs(c - t) <= COPY * t for c in counts):
                return True
        return False

    while entries and several(max(entries, key=lambda e: e['triangles'])):
        entries.remove(max(entries, key=lambda e: e['triangles']))
    if not entries:
        return None
    most = max(e['triangles'] for e in entries)
    copies = [e for e in entries if e['triangles'] == most]
    if len(copies) == 1:
        return copies[0]

    def spread(e):
        open_(e)
        points, tris, _ = _surface(_meshes())
        return np.linalg.norm(points - points.mean(axis=0), axis=1).mean() / np.sqrt(_area(points, tris).sum())
    return max(copies, key=spread)


LOST = 0.5          # a skeleton carrying less than this share of the mesh has lost its joints


def lost(rig):
    """Whether a rigged file's skeleton (its tree of the most bones) carries less than LOST of its mesh: the rip lost
    its joints, its vertices left unweighted (an importer binds those to a bone of its own outside the skeleton, as
    Blender's glTF importer does). Sonic Adventure 2's Amy kept two of her nodes, on 14% of the mesh; every other rig on
    the core loop's 76 carries 99% or more. Its pieces are what still say how it moves: read them (read)."""
    from .body import _ancestors
    trees = {}
    for b in range(len(rig.names)):
        trees.setdefault(_ancestors(rig.parents, b)[-1], []).append(b)
    skeleton = max(trees.values(), key=len)
    return len(rig.points) > 0 and rig.weights[:, skeleton].sum() < LOST * len(rig.points)


def read(blend, touching=()):
    """Open ``blend`` (load's output), find its pieces and joints, and give the scene an armature of them: one bone
    per piece, each piece's vertices in its bone's group at weight 1. Then it reads as a rigged file (rig.scene), and
    stays one for whatever reads the scene again. A rig the file came with is dropped first (its joints are lost:
    lost). ``touching``: the contacts (piece pairs) asked to be only touching (``touching``), which join pieces only
    where nothing else does; the contacts on loops through the trunk are kept on the armature, to ask about."""
    import bpy
    from . import rig as rigs
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    _unrig()
    meshes = _meshes()
    points, tris, counts = _surface(meshes)
    piece = _pieces(points, tris)
    parents, heads, tails, twin, axis, loops = skeleton(points, tris, piece, touching)
    _rig_up(meshes, counts, piece, parents, heads, tails, twin, axis, loops)
    return rigs.scene(blend)


def touching(rig, blend, up, front, out, ask):
    """Which contacts on loops through the trunk (read keeps them on the armature: _loops) are only touching, one
    piece resting against the other where the pose puts it (Sofia's hand on her thigh), not joined: per contact, with
    its mirror twin, the model in grey with its two pieces in two colours (labels.draw), VOTES asks, the most given.
    When any is, the skeleton is read again with those joining pieces only where nothing else does (_tree). Returns
    (the rig, the report)."""
    import bpy
    from . import labels
    arm = bpy.data.objects.get('segments')
    loops = json.loads(arm.get('rigswap_loops', '[]')) if arm is not None else []
    if not loops:
        return rig, {'contacts': []}
    index = {name: i for i, name in enumerate(rig.names)}
    pictures, parts = [], []
    for n, group in enumerate(loops, 1):
        found = [dict(roots=r, bones=[[x] for x in r]) for r in
                 (sorted({index[g[k]] for g in group}) for k in (2, 3))]
        drawn = labels.draw(rig, found, up, front, Path(out) / f'touching-{n}.png', close=True, number=n)
        pictures += drawn if n == 1 else drawn[1:]
        parts.append(f'picture {n}: 1 ' + ' and '.join(dict.fromkeys(g[2] for g in group)) + ', 2 '
                     + ' and '.join(dict.fromkeys(g[3] for g in group)))
    said = []
    for r in range(labels.VOTES):
        answer, _ = ask(TOUCHING.format(parts='; '.join(parts)) + f' (Ask {r + 1} of {labels.VOTES}.)',
                        [str(p) for p in pictures], TOUCHING_SCHEMA, {'contacts': []}, route='core touching')
        said.append(answer)
    picked = tally_touching(said, range(1, len(loops) + 1))
    touched = {tuple(sorted(g[:2])) for n, group in enumerate(loops, 1) if picked[n] for g in group}
    report = dict(character=[(a or {}).get('character') for a in said if (a or {}).get('character')],
                  contacts=[dict(picture=n, pieces=[f'{g[2]} - {g[3]}' for g in group], touching=picked[n],
                                 said=[f"{c.get('what')} ({c.get('contact')})" for a in said
                                       for c in (a or {}).get('contacts') or [] if c.get('picture') == n])
                            for n, group in enumerate(loops, 1)])
    print('touching: ' + str([c['pieces'] for c in report['contacts'] if c['touching']]))
    return (read(blend, touched) if touched else rig), report


def tally_touching(answers, numbers):
    """Per picture number, whether more than half of the answers say its two pieces only touch."""
    return {n: 2 * sum(any(c.get('picture') == n and c.get('contact') == 'touching'
                           for c in (a or {}).get('contacts') or []) for a in answers) > len(answers)
            for n in numbers}


def skeleton(points, tris, piece, touching=()):
    """(parent piece per piece, -1 for the trunk; each piece's joint to its parent, the trunk's own middle for it;
    a tail for each; the piece twins and their mirror axis; the contacts on loops through the trunk, _loops). Raises
    OnePiece when the trunk spans the model. ``touching``: contacts only touching, joining pieces only where nothing
    else does (_tree)."""
    n = piece.max() + 1
    size = np.ptp(points, axis=0).max()
    area = np.bincount(piece[tris[:, 0]], _area(points, tris), minlength=n)
    contacts = _contacts(points, tris, piece, REACH * size)
    side, twin, axis = _mirror(points, piece)
    edges = _tree(points, piece, contacts, side, twin, touching, tris)
    root = _centre(n, edges, area)
    loops = _loops(edges, contacts, side, twin, root)
    edges = _from_the_root(points, piece, contacts, edges, root, side)
    whole = np.ptp(points, axis=0)
    span = np.ptp(points[piece == root], axis=0) / np.where(whole > 0, whole, 1)
    if (span > SPAN).all():
        raise OnePiece(f'the trunk spans {np.round(span, 2).tolist()} of the model: its limbs are in its own '
                       f'surface, not pieces of their own')
    parents, heads = np.full(n, -1), np.array([points[piece == p].mean(axis=0) for p in range(n)])
    order = [root]
    for p in order:
        for (a, b), joint in edges.items():
            c = b if a == p else a if b == p else None
            if c is not None and c != root and parents[c] < 0:
                parents[c], heads[c] = p, joint
                order.append(c)
    tails = np.empty_like(heads)
    for p in range(n):
        kids = heads[parents == p]
        own = points[piece == p]
        tip = kids.mean(axis=0) if len(kids) else own[np.argmax(np.linalg.norm(own - heads[p], axis=1))]
        d = tip - heads[p]
        if np.linalg.norm(d) < 1e-3 * size:         # Blender drops a bone of no length
            d = np.array([0., 0., 1e-2 * size])
        tails[p] = heads[p] + d
    return parents, heads, tails, twin, axis, loops


def _unrig():
    """Drop the open scene's armatures: each mesh keeps the form it shows at rest, its own place in the world, and no
    binding to them."""
    import bpy
    arms = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE']
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for m in [m for m in ob.modifiers if m.type == 'ARMATURE' and m.object in arms]:
            ob.modifiers.remove(m)
        if ob.parent in arms:
            world = ob.matrix_world.copy()
            ob.parent = None
            ob.matrix_world = world
        bones = {b.name for a in arms for b in a.data.bones}
        for g in [g for g in ob.vertex_groups if g.name in bones]:
            ob.vertex_groups.remove(g)
    for a in arms:
        bpy.data.objects.remove(a)
    bpy.context.view_layer.update()


def _meshes(objects=None):
    import bpy
    # a mesh the file hides is not the form shown in play (rig.extract leaves it out too)
    return [ob for ob in (bpy.data.objects if objects is None else objects) if ob.type == 'MESH'
            and len(ob.data.polygons) and not (ob.hide_viewport or ob.hide_render)]


def _surface(meshes):
    """Every mesh's vertices in world space, stacked in order; their triangles in that numbering; each mesh's count."""
    points, tris, counts, start = [], [], [], 0
    for ob in meshes:
        me = ob.data
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get('co', co)
        m = np.array(ob.matrix_world)
        points.append(co.reshape(n, 3) @ m[:3, :3].T + m[:3, 3])
        me.calc_loop_triangles()
        t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get('vertices', t)
        tris.append(t.reshape(-1, 3) + start)
        counts.append(n)
        start += n
    return np.vstack(points), np.vstack(tris), counts


def _area(points, tris):
    a, b, c = (points[tris[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _pieces(points, tris):
    """The piece each vertex is on: the welded surfaces, numbered from 0."""
    size = np.ptp(points, axis=0).max()
    _, weld = np.unique(np.round(points / (WELD * size)).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.ravel()
    edges = np.vstack([weld[tris[:, [0, 1]]], weld[tris[:, [1, 2]]]])
    label = np.arange(weld.max() + 1)
    while True:                         # hook each edge's two labels onto the smaller, then shortcut
        a, b = label[edges[:, 0]], label[edges[:, 1]]
        new = label.copy()
        np.minimum.at(new, a, np.minimum(a, b))
        np.minimum.at(new, b, np.minimum(a, b))
        while not (new[new] == new).all():
            new = new[new]
        if (new == label).all():
            break
        label = new
    return np.unique(label, return_inverse=True)[1].ravel()[weld]


def _contacts(points, tris, piece, reach):
    """{(a, b): (joint, area)} for the pieces whose nearest approach is within ``reach``: the surface of each (sampled
    over its triangles) within ``reach`` past that approach to the other, or inside it. The joint is the middle of the
    contact's own vertices, the ends its author built (a thigh's top ring sunk in the hips, not the middle of the
    stub below it); of its surface where no vertex is in it."""
    import bpy  # noqa: F401  (mathutils comes with it)
    from mathutils.bvhtree import BVHTree
    n = piece.max() + 1
    of = piece[tris[:, 0]]
    own = [tris[of == p] for p in range(n)]
    corners = [np.flatnonzero(piece == p) for p in range(n)]
    samples = [np.einsum('sk,tkd->tsd', SAMPLES, points[t]).reshape(-1, 3) for t in own]
    shares = [np.outer(_area(points, t), SHARES).ravel() for t in own]
    lo = np.array([points[piece == p].min(axis=0) for p in range(n)]) - 2 * reach
    hi = np.array([points[piece == p].max(axis=0) for p in range(n)]) + 2 * reach
    listed = points.tolist()
    trees = [BVHTree.FromPolygons(listed, t.tolist()) for t in own]

    def near(q, y):
        """Which of the points ``q`` lie in reach of ``y``'s box, and how far each of those is from its surface (0
        inside it)."""
        idx = np.flatnonzero(((q >= lo[y]) & (q <= hi[y])).all(axis=1))
        d = np.array([_distance(trees[y].find_nearest(p, 2 * reach)) for p in q[idx].tolist()])
        d[np.abs(_winding(q[idx], points[own[y]])) > INSIDE] = 0.0
        return idx, d
    out = {}
    for a in range(n):
        for b in range(a + 1, n):
            if (lo[a] > hi[b]).any() or (lo[b] > hi[a]).any() or not len(own[a]) or not len(own[b]):
                continue
            surface = {x: near(samples[x], y) for x, y in ((a, b), (b, a))}
            gap = min((d.min() for _, d in surface.values() if len(d)), default=np.inf)
            if gap > reach:
                continue
            area = sum(shares[x][idx[d <= gap + reach]].sum() for x, (idx, d) in surface.items())
            ends = []
            for x, y in ((a, b), (b, a)):
                idx, d = near(points[corners[x]], y)
                ends.append(points[corners[x][idx[d <= gap + reach]]])
            ends = np.vstack(ends)
            if len(ends):
                joint = ends.mean(axis=0)
            else:
                at = [(samples[x][idx[d <= gap + reach]], shares[x][idx[d <= gap + reach]])
                      for x, (idx, d) in surface.items()]
                p, w = np.vstack([q for q, _ in at]), np.concatenate([s for _, s in at])
                joint = (p * w[:, None]).sum(axis=0) / w.sum()
            out[(a, b)] = (joint, area)
    return out


def _distance(found):
    return np.inf if found[3] is None else found[3]


def _winding(p, tri):
    """How many times the triangles ``tri`` (T, 3, 3) wind round each point (solid angle over 4 pi): about 1 inside a
    closed shell, 0 outside."""
    out = np.zeros(len(p))
    for s in range(0, len(p), 64):
        a, b, c = (tri[None, :, i] - p[s:s + 64, None] for i in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum('pti,pti->pt', a, np.cross(b, c))
        dot = (la * lb * lc + np.einsum('pti,pti->pt', a, b) * lc + np.einsum('pti,pti->pt', b, c) * la
               + np.einsum('pti,pti->pt', c, a) * lb)
        out[s:s + 64] = np.arctan2(det, dot).sum(axis=1) / (2 * np.pi)
    return out


def _mirror(points, piece):
    """(per piece, which side of the mirror plane it is on: 0 on the midline, its mirror image nearest itself, else
    -1 or +1; per piece, its mirror twin, itself when it has none). The plane: of the world axes (rips keep them), the
    one whose plane through the middle pairs the most vertices with a twin."""
    import bpy  # noqa: F401
    from mathutils.kdtree import KDTree
    size = np.ptp(points, axis=0).max()
    tree = KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    step = max(1, len(points) // 2000)
    best = None
    for ax in np.eye(3):
        c = np.median(points @ ax)
        mirrored = points[::step] - 2 * ((points[::step] @ ax) - c)[:, None] * ax
        twins = sum(tree.find(p)[2] <= REACH * size for p in mirrored)
        if best is None or twins > best[0]:
            best = (twins, ax, c)
    _, ax, c = best
    n = piece.max() + 1
    centres = np.array([points[piece == p].mean(axis=0) for p in range(n)])
    images = centres - 2 * ((centres @ ax) - c)[:, None] * ax
    nearest = np.linalg.norm(images[:, None] - centres[None], axis=2).argmin(axis=1)
    side = np.where(nearest == np.arange(n), 0, np.where(centres @ ax > c, 1, -1))
    twin = np.where(nearest[nearest] == np.arange(n), nearest, np.arange(n))
    return side, twin, ax


def _tree(points, piece, contacts, side, twin, touching=(), tris=None):
    """{(a, b): joint}: the strongest contacts (with their mirror twins') that join the pieces into one tree, none
    across the mirror plane; those only touching (``touching``, asked: segments.touching) after all the rest, joining
    pieces only where nothing else does. Pieces no contact reaches (an eye floating over the face) join where their
    surfaces (``tris``) come nearest, measured as contacts are (_approach): on a low-poly piece the corners stand
    apart where the faces approach (sonic32x: a leg's end over its cuff, nearer than a glove's corner beside it)."""
    n = piece.max() + 1
    group = list(range(n))

    def find(x):
        while group[x] != x:
            x = group[x]
        return x

    def strength(a, b):
        mirrored = tuple(sorted((int(twin[a]), int(twin[b]))))
        own = contacts[(a, b)][1]
        return own + (contacts[mirrored][1] if mirrored in contacts else own) if mirrored != (a, b) else 2 * own
    out, touching = {}, {tuple(sorted(map(int, k))) for k in touching}
    for a, b in sorted(contacts, key=lambda k: (k in touching, -strength(*k))):
        if find(a) != find(b) and side[a] * side[b] >= 0:
            group[find(a)] = find(b)
            out[(a, b)] = contacts[(a, b)][0]
    approach = _approach(points, tris, piece) if len({find(p) for p in range(n)}) > 1 else None
    while len({find(p) for p in range(n)}) > 1:
        # where a part not yet joined comes nearest the rest, across the plane only if need be
        parts = {}
        for p in range(n):
            parts.setdefault(find(p), []).append(p)
        main = max(parts.values(), key=lambda ps: np.isin(piece, ps).sum())
        best = (np.inf, None, None, None)
        for ps in parts.values():
            if ps is main:
                continue
            pairs = [(a, b) for a in ps for b in main]
            pairs = [(a, b) for a, b in pairs if side[a] * side[b] >= 0] or pairs
            for gap, a, b in sorted((approach.bound(a, b), a, b) for a, b in pairs):
                if gap >= best[0]:
                    break
                d, joint = approach(a, b)
                if d < best[0]:
                    best = (d, joint, a, b)
        _, joint, a, b = best
        group[find(a)] = find(b)
        out[(a, b)] = joint
    return out


class _approach:
    """approach(a, b) -> (distance, joint): where pieces a and b come nearest, surface to surface, sampled as
    _contacts samples them (a piece with no faces by its vertices); the joint the middle of that approach.
    ``bound(a, b)``: a distance they come no nearer than (their boxes')."""

    def __init__(self, points, tris, piece):
        import bpy  # noqa: F401  (mathutils comes with it)
        from mathutils.bvhtree import BVHTree
        n = piece.max() + 1
        of = piece[tris[:, 0]]
        own = [tris[of == p] for p in range(n)]
        self.points = [points[piece == p] for p in range(n)]
        self.samples = [np.einsum('sk,tkd->tsd', SAMPLES, points[t]).reshape(-1, 3) if len(t) else self.points[p]
                        for p, t in enumerate(own)]
        listed = points.tolist()
        self.trees = [BVHTree.FromPolygons(listed, t.tolist()) if len(t) else None for t in own]
        self.lo = np.array([q.min(axis=0) for q in self.points])
        self.hi = np.array([q.max(axis=0) for q in self.points])
        self.done = {}

    def bound(self, a, b):
        return float(np.linalg.norm(np.maximum(0, np.maximum(self.lo[a] - self.hi[b], self.lo[b] - self.hi[a]))))

    def __call__(self, a, b):
        if (a, b) not in self.done:
            best = (np.inf, None)
            for x, y in ((a, b), (b, a)):
                for q in self.samples[x]:
                    if self.trees[y] is not None:
                        found = self.trees[y].find_nearest(q)
                        at, d = (np.inf, None) if found[3] is None else (np.array(found[0]), found[3])
                    else:
                        k = np.argmin(np.linalg.norm(self.points[y] - q, axis=1))
                        at, d = self.points[y][k], np.linalg.norm(self.points[y][k] - q)
                    if d is not None and d < best[0]:
                        best = (d, (q + at) / 2)
            self.done[(a, b)] = self.done[(b, a)] = best
        return self.done[(a, b)]


def _loops(edges, contacts, side, twin, root):
    """The contacts on loops through the trunk (``root``), each with its mirror twin, the strongest first: per contact
    [(a, b)], or [(a, b), (twin of a, twin of b)]. A contact the tree leaves out (none across the mirror plane) closes
    a loop with the tree's path between its pieces. Most loops are a knot of neighbouring pieces at one place (a
    whip's segments, an earring against the head and the neck), where either choice hangs a piece at the same place;
    one through the trunk is a limb reaching the body two ways (Sofia's forearm, by her elbow and by her hand on her
    thigh), and which of its contacts only touch is asked (touching). A join by nearest approach is no contact."""
    links = {}
    for a, b in edges:
        links.setdefault(int(a), []).append(int(b))
        links.setdefault(int(b), []).append(int(a))

    def path(a, b):
        prev, todo = {a: None}, [a]
        for x in todo:
            for y in links.get(x, []):
                if y not in prev:
                    prev[y], _ = x, todo.append(y)
        out = [b]
        while prev[out[-1]] is not None:
            out.append(prev[out[-1]])
        return out
    tree = {tuple(sorted(map(int, e))) for e in edges}
    found = set()
    for a, b in contacts:
        if (a, b) in tree or side[a] * side[b] < 0:
            continue
        p = path(a, b)
        if root in p:
            found |= {(a, b)} | {tuple(sorted(p[i:i + 2])) for i in range(len(p) - 1)}
    found &= set(contacts)

    def mirrored(k):
        return int(twin[k[0]]), int(twin[k[1]])

    def strength(k):
        own, m = contacts[k][1], tuple(sorted(mirrored(k)))
        return own + (contacts[m][1] if m in contacts else own) if m != k else 2 * own
    out, done = [], set()
    for k in sorted(found, key=lambda k: (-strength(k), k)):
        if k in done:
            continue
        m = mirrored(k)
        out.append([k] + ([m] if tuple(sorted(m)) != k and tuple(sorted(m)) in contacts else []))
        done |= {k, tuple(sorted(m))}
    return out


def _from_the_root(points, piece, contacts, edges, root, side):
    """``edges`` with each limb hung from the trunk (``root``) where it begins. A limb can touch the trunk in more
    places than its joint: a glove resting on a flared skirt meets it more broadly than the arm's top meets the dress
    at the shoulder (Sonic Adventure 2's Amy), and the strongest contact hangs the arm from its wrist. A limb begins at
    one end, so of the places a limb (a part the tree joins to the trunk, all on one side of the mirror plane:
    ``side``, per piece) touches the trunk, it hangs from the one it reaches farthest from; its own joints stay. A part
    across the midline (the hips and both legs) is no one limb."""
    n = piece.max() + 1
    links = [[] for _ in range(n)]
    for a, b in edges:
        links[a].append(b)
        links[b].append(a)
    out = dict(edges)
    for start in links[root]:
        part, todo = {start}, [start]
        while todo:
            for y in links[todo.pop()]:
                if y != root and y not in part:
                    part.add(y)
                    todo.append(y)
        touching = {p: contacts[tuple(sorted((p, root)))][0] for p in part if tuple(sorted((p, root))) in contacts}
        if len(touching) < 2 or len({int(side[p]) for p in part}) > 1 or side[start] == 0:
            continue
        own = points[np.isin(piece, list(part))]
        begins = max(touching, key=lambda p: np.linalg.norm(own - touching[p], axis=1).max())
        if begins != start:
            out.pop(next(k for k in out if set(k) == {start, root}))
            out[(begins, root)] = touching[begins]
    return out


def _centre(n, edges, area):
    """The piece in the middle of the tree: the one whose removal leaves the fewest pieces in its largest part (the
    trunk parts the limbs, head and tail; a hand only parts its fingers from the body), the larger on a tie. Counted
    in pieces, not size: a big head is still a limb of the trunk."""
    links = [[] for _ in range(n)]
    for a, b in edges:
        links[a].append(b)
        links[b].append(a)

    def largest_part(p):
        out = 0
        for start in links[p]:
            todo, seen = [start], {p, start}
            while todo:
                for y in links[todo.pop()]:
                    if y not in seen:
                        seen.add(y)
                        todo.append(y)
            out = max(out, len(seen) - 1)
        return out
    return min(range(n), key=lambda p: (largest_part(p), -area[p]))


def _rig_up(meshes, counts, piece, parents, heads, tails, twins, mirror, loops=()):
    """Give the open scene an armature of the pieces and bind every mesh to it, each piece's vertices in its bone's
    group at weight 1. A bone is named for the mesh holding most of its piece. The contacts on loops through the trunk
    (``loops``, _loops) are kept on it, each piece by its number and its bone's name, for touching to ask about."""
    import bpy
    owner = np.repeat(np.arange(len(meshes)), counts)
    names = [meshes[int(np.bincount(owner[piece == p], minlength=len(meshes)).argmax())].name
             for p in range(len(parents))]
    data = bpy.data.armatures.new('segments')
    arm = bpy.data.objects.new('segments', data)
    bpy.context.scene.collection.objects.link(arm)
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bones = []
    for p, name in enumerate(names):
        eb = data.edit_bones.new(name)
        eb.head, eb.tail = heads[p].tolist(), tails[p].tolist()
        bones.append(eb)
    for p, q in enumerate(parents):
        if q >= 0:
            bones[p].parent = bones[q]
    names = [eb.name for eb in bones]           # as Blender made them unique
    bpy.ops.object.mode_set(mode='OBJECT')
    # Each piece is a bone; self-twins explicitly state the midline.
    arm['rigswap_twins'] = json.dumps({names[a]: names[int(b)] for a, b in enumerate(twins)})
    arm['rigswap_mirror'] = np.asarray(mirror, float).tolist()
    arm['rigswap_loops'] = json.dumps([[[int(a), int(b), names[a], names[b]] for a, b in group] for group in loops])
    start = 0
    for ob, n in zip(meshes, counts):
        if ob.data.users > 1:                   # its groups are its own
            ob.data = ob.data.copy()
        mine = piece[start:start + n]
        start += n
        for p in np.unique(mine):
            ob.vertex_groups.new(name=names[p]).add(np.flatnonzero(mine == p).tolist(), 1.0, 'REPLACE')
        ob.modifiers.new('segments', 'ARMATURE').object = arm
    return arm
