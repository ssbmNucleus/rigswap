"""How the model's hair and cloth hang: which of their surfaces can swing, and where each rests.

build.loose finds the model's hair and cloth with no partner on the fighter, each riding the bone it hangs from. That
they hang is what they are (labels.chains names every chain; the hang ask that used to settle it is gone with the
geometric gates around it). Where a surface is joined, and to what, is structure, and decides whether it can swing
(surfaces): chains whose surfaces share a corner are one surface, swung together, each from where the surface last
meets the rest of the costume (its cut): the corners it shares stay on the bone, so nothing parts. One whose surface
spans chains riding different bones cannot swing from one (SPANS); one joined to the rest all the way to its far end
has nothing free to swing (JOINED); one whose far end is its start hangs no way (LOOP).

A chain rests where it rests. Drawn as it rests (a lock swept across the face, a skirt flared, a stethoscope on the
coat), it keeps its authored shape: a crude drape undid Rosalina's hair sweep. Held out by the pose where it would not
rest (labels.chains: a feeler straight out behind the head), it rests as it hangs: from its cut, each joint in turn
turned toward straight down, as far as the rest of the costume and the floor let it pass (drape), with its surface
carried as the game skins it (skin: the nearest segment of its chains, the corners it shares with the rest held).
Either way, which segments the body or the floor props up off straight down is recorded (``propped``: what a step
toward down would carry into them): a plugin holds those where they rest whichever way the body turns, and holds the
others where gravity would leave them. What the game swings then starts at rest: nothing drops as play starts, and
nothing passes through the body to get there.

It runs after the eyes are read, so their pictures show the head as bound: draped feelers in them changed the eyes
ask's answer on Lucario.
"""
import json
from pathlib import Path

import numpy as np

DOWN = np.array([0., 0., -1.])      # the fighter's rest: z up (build.write)
ROOT_LOCK = .15                     # old importer studio/physics/constants.py: a corner this near a chain's start rides
                                    # its static root, in the costume's units
STEP = np.radians(2.)               # a drape turns a joint this far at a time, testing each step against the body

SPANS = 'its surface spans chains hanging from different bones'
JOINED = 'its surface meets the rest of the costume at its far end: nothing of it hangs free'
LOOP = 'its far end is where it starts: it hangs no way'


def _key(p):
    return tuple(np.round(p, 5))    # a corner two faces share has one position


def surfaces(chains, anchors, gathered, rest):
    """The loose chains' surfaces: ``chains`` (each its joints and far end), the bones they ride (``anchors``),
    ``gathered`` (per chain, its faces' corners) and ``rest`` (the corners of every other face of the costume). Chains
    whose faces share a corner are one surface. Returns per surface a dict: ``ids`` (its chains), ``held`` (per face,
    chain by chain, whether each corner is one the rest of the costume shares), ``cut`` (per chain, how far along it
    the surface last meets the rest) and ``why`` it cannot swing (SPANS, JOINED, LOOP), or None."""
    held = {_key(p) for p in rest}
    group, seen = list(range(len(chains))), {}

    def find(i):
        while group[i] != i:
            i = group[i]
        return i
    for i, faces in enumerate(gathered):
        for f in faces:
            for p in f:
                group[find(i)] = find(seen.setdefault(_key(p), i))
    members = {}
    for i, faces in enumerate(gathered):
        if len(faces):
            members.setdefault(find(i), []).append(i)
    out = []
    for ids in members.values():
        flags = [[_key(p) in held for p in f] for i in ids for f in gathered[i]]
        mine = [np.asarray(chains[i], float) for i in ids]
        cut = np.zeros(len(mine))
        for p in {_key(p): p for i in ids for f in gathered[i] for p in f if _key(p) in held}.values():
            c, along = _along(np.asarray(p, float), mine)
            cut[c] = max(cut[c], along)
        why = None
        if len({anchors[i] for i in ids}) > 1:
            why = SPANS
        elif any(at >= _length(c) * (1 - 1e-6) for c, at in zip(mine, cut)):
            why = JOINED
        elif any(np.linalg.norm(c[-1] - c[0]) <= 1e-6 for c in mine):
            # bones out and back to where they began (Inkling's fringe, from the brow forward and back onto it): no
            # direction to hang in, and swung about its fold it flips
            why = LOOP
        out.append(dict(ids=ids, held=flags, cut=cut.tolist(), why=why))
    return out


def _length(chain):
    return float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())


def _along(p, chains):
    """(the chain nearest ``p``, how far along it ``p`` lies)."""
    best = None
    for c, chain in enumerate(chains):
        along = 0.
        for a, b in zip(chain, chain[1:]):
            d = b - a
            n = float(d @ d)
            t = float(np.clip((p - a) @ d / n, 0, 1)) if n > 1e-18 else 0.
            error = np.linalg.norm(p - a - t * d)
            if best is None or error < best[0]:
                best = error, c, along + t * np.sqrt(n)
            along += np.sqrt(n)
    return best[1], best[2]


def cut_from(chain, at):
    """The chain from ``at`` along it on."""
    chain = np.asarray(chain, float)
    steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    done = np.concatenate([[0.], np.cumsum(steps)])
    k = min(int(np.searchsorted(done, at, side='right')) - 1, len(steps) - 1)
    if at <= 0 or steps[k] <= 0:
        return chain
    start = chain[k] + (at - done[k]) / steps[k] * (chain[k + 1] - chain[k])
    rest = chain[k + 1:]
    if np.linalg.norm(rest[0] - start) <= 1e-6 and len(rest) > 1:
        rest = rest[1:]
    return np.vstack([start, rest])


def first(chains):
    """Each chain's first bone in a swing's skeleton: 0/1 carry it, 2 is the static root, then each chain's joints in
    turn, its far end last."""
    out = [3]
    for c in chains[:-1]:
        out.append(out[-1] + len(c))
    return out


def nearest(points, chains):
    """Where each point lies on the chains: (the chain, the segment of it nearest the point, how far along that
    segment the point lies from 0 to 1, how far along the chain that is in length)."""
    result = []
    for p in np.asarray(points, float):
        best = None
        for j, chain in enumerate(chains):
            along = 0.
            for i, (a, b) in enumerate(zip(chain, chain[1:])):
                delta = b - a
                length = float(np.linalg.norm(delta))
                if length <= 1e-9:
                    continue
                t = float(np.clip((p - a) @ delta / (length * length), 0, 1))
                error = np.linalg.norm(p - a - t * delta)
                if best is None or error < best[0]:
                    best = error, (j, i, t, along + t * length)
                along += length
        if best is None:
            raise ValueError('a swinging chain has no segment with length')
        result.append(best[1])
    return result


def skin(points, chains):
    """Each point's bones in a swing's skeleton (first): along the nearest segment of the chains (nearest), the two
    joints that segment runs between by how far along it the point lies; a point within ROOT_LOCK of its chain's start
    rides the static root (old importer, studio/physics)."""
    starts, result = first(chains), []
    for j, i, t, along in nearest(points, chains):
        w = [(2, 1.)] if along <= ROOT_LOCK else [(starts[j] + i, 1 - t), (starts[j] + 1 + i, t)]
        result.append([(b, w) for b, w in w if w > 1e-8])
    return result


def _turn(axis, angle):
    """The rotation by ``angle`` about the unit ``axis``."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def drape(chains, points, skins, passes, step=STEP, turn=True):
    """The chains (each from its cut) and the surface on them (``points``, skinned by ``skins``: skin's, a held corner
    on the static root) turned to hang: joint by joint from the start of each chain, the segment from it turned toward
    straight down in steps of ``step``, carrying what comes after it, until it hangs or a step would carry a point
    through the rest of the costume (``passes(a, b)``: per point, whether moving from a to b passes into it). With
    ``turn`` False nothing turns: a part drawn as it rests is only tested, a step toward down from where it is.
    Returns (the chains, per chain whether the body props each segment up off straight down, the points, per point
    the 3x3 its neighbourhood turned by)."""
    chains = [np.asarray(c, float).copy() for c in chains]
    points = np.asarray(points, float)
    starts = first(chains)
    bones = starts[-1] + len(chains[-1])
    at = np.zeros((bones, 3))                   # each bone's joint at bind, where it now is, and how it is turned
    for s, c in zip(starts, chains):
        at[s:s + len(c)] = c
    now, turned = at.copy(), np.tile(np.eye(3), (bones, 1, 1))
    weights = np.zeros((len(points), bones))
    for i, w in enumerate(skins):
        for b, x in w:
            weights[i, b] = x

    def posed(now, turned):
        moved = np.einsum('bij,vbj->vbi', turned, points[:, None, :] - at[None]) + now[None]
        return np.einsum('vb,vbi->vi', weights, moved), np.einsum('vb,bij->vij', weights, turned)
    here, _ = posed(now, turned)
    propped = [[False] * (len(c) - 1) for c in chains]
    for k, (s, c) in enumerate(zip(starts, chains)):
        for j in range(len(c) - 1):
            carried = np.flatnonzero(weights[:, s + j:s + len(c)].sum(axis=1) > 0)
            d = now[s + j + 1] - now[s + j]
            if np.linalg.norm(d) <= 1e-9:
                continue
            d = d / np.linalg.norm(d)
            angle = float(np.arccos(np.clip(d @ DOWN, -1, 1)))
            axis = np.cross(d, DOWN)
            if angle <= 1e-6:
                continue    # hanging already
            if np.linalg.norm(axis) <= 1e-9:
                propped[k][j] = True        # straight up: no one way down, so it rests as bound
                continue
            axis /= np.linalg.norm(axis)
            done = 0.
            while done < angle - 1e-9:
                q = _turn(axis, min(step, angle - done))
                pivot = now[s + j].copy()
                n2, t2 = now.copy(), turned.copy()
                n2[s + j + 1:s + len(c)] = pivot + (now[s + j + 1:s + len(c)] - pivot) @ q.T
                t2[s + j:s + len(c)] = q @ turned[s + j:s + len(c)]
                there, _ = posed(n2, t2)
                if len(carried) and passes(here[carried], there[carried]).any():
                    propped[k][j] = True
                    break
                if not turn:
                    break
                now, turned, here = n2, t2, there
                done += min(step, angle - done)
    for s, c in zip(starts, chains):
        c[:] = now[s:s + len(c)]
    return (chains, propped, *posed(now, turned))


def off_down(chain):
    """How far the chain's far end lies off straight down from its start, in degrees."""
    d = np.asarray(chain[-1], float) - np.asarray(chain[0], float)
    n = np.linalg.norm(d)
    return round(float(np.degrees(np.arccos(np.clip(d @ DOWN / n, -1, 1)))), 1) if n > 1e-9 else None


def _scene(attribute, details=()):
    """The open build's meshes: per object, (name, world corners, its faces' corner rows, each face's chain value).
    The ``details``' objects (the eyes: eyes.json) are the body's: the game draws them as their detail, and export
    swings none of their faces."""
    import bpy
    out = []
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH' or not ob.data.polygons:
            continue
        me = ob.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get('co', co)
        mw = np.asarray(ob.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        attr = me.attributes.get(attribute)
        values = np.zeros(len(me.polygons), int)
        if attr is not None and ob.name not in details:
            attr.data.foreach_get('value', values)
        out.append((ob.name, co, [list(p.vertices) for p in me.polygons], values))
    return out


def settle(build_dir):
    """Decide which of the build's loose chains (``build_dir``/loose.json: hair and cloth, build.loose's) can swing,
    and drape those, in place: build.blend's surface and loose.json's joints at the drape, and per chain ``swings``
    (whether its surface can), ``why`` it stays, and for one draped, ``propped``: per segment from its cut, whether the
    body holds it up off straight down. A surface turns to hang only when the pose holds out every chain of it
    (``held_out``, labels.chains). Returns the report."""
    import bpy
    build_dir = Path(build_dir)
    path = build_dir / 'loose.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    rows = record['chains']
    if not rows:
        return {}
    bpy.ops.wm.open_mainfile(filepath=str(build_dir / 'build.blend'))
    seen = build_dir / 'eyes.json'
    details = {o['name'] for o in json.loads(seen.read_text(encoding='utf-8')).get('objects', [])} \
        if seen.is_file() else set()
    meshes = _scene(record['attribute'], details)
    gathered, rest = [[] for _ in rows], []
    for _, co, faces, values in meshes:
        for f, k in zip(faces, values):
            (gathered[k - 1] if 0 < k <= len(rows) else rest).append(co[f])
    found = surfaces([r['points'] for r in rows], [r['anchor'] for r in rows], gathered,
                     np.concatenate(rest) if rest else np.zeros((0, 3)))
    # a chain with no face of its own here has nothing to swing: the plugin reads its surface again after prep, which
    # can find free what the build's surface held (u-sheik's front lock, Bowser's belly flap)
    for r in rows:
        r.update(swings=False, why='it carries no face')
    for s in found:
        for i in s['ids']:
            rows[i].update(swings=s['why'] is None, why=s['why'])
    swung = [dict(s, turn=all(rows[i].get('held_out') for i in s['ids'])) for s in found if s['why'] is None]
    report = dict(surfaces=[dict(bones=[b for i in s['ids'] for b in rows[i]['bones']],
                                 kinds=sorted({rows[i].get('kind') for i in s['ids']}, key=str), why=s['why'],
                                 held_out=all(rows[i].get('held_out') for i in s['ids'])) for s in found])
    if swung:
        # the build again as it was written. Saved only when a part turned: a part drawn as it rests is only tested,
        # and saving it again would only re-encode its normals
        bpy.ops.wm.open_mainfile(filepath=str(build_dir / 'build.blend'))
        report['draped'] = _drape(record['attribute'], rows, swung, details)
        if any(s['turn'] for s in swung):
            bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
    path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    return report


def _drape(attribute, rows, swung, details=()):
    """drape each surface in ``swung`` (surfaces', each with ``turn``: whether the pose holds it out, so it turns) in
    the open build, against every face that stays in the costume (not on a chain that swings) and the floor: its
    vertices moved, its corners' normals turned with them, its chains' joints in
    ``rows`` from their cuts, and per segment from its cut whether the body props it up (``propped``). Returns per
    chain its far end's angle off straight down, before and after."""
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    swinging = {i for s in swung for i in s['ids']}
    meshes = _scene(attribute, details)
    base, verts, polys = 0, [], []
    for _, co, faces, values in meshes:
        verts.extend(map(tuple, co))
        polys.extend([base + v for v in f] for f, k in zip(faces, values) if k - 1 not in swinging)
        base += len(co)
    tree = BVHTree.FromPolygons(verts, polys) if polys else None
    size = float(np.ptp(np.asarray(verts), axis=0).max()) if verts else 1.
    # the floor the fighter stands on: the lowest of what stays (its soles, a hem): a robe bound flared down to it
    # hangs to it, not through it
    kept = np.asarray(verts)[np.unique([v for f in polys for v in f])] if polys else np.zeros((0, 3))
    floor = float(kept[:, 2].min()) if len(kept) else -np.inf

    def passes(a, b):
        """Per point, whether the move a -> b enters a face of the costume (a face it leaves from inside is no
        entry: a strand's root bound inside the scalp may come out), or goes down below the floor."""
        hit = (b[:, 2] < floor) & (b[:, 2] < a[:, 2])
        if tree is None:
            return hit
        for i, (p, q) in enumerate(zip(a, b)):
            if hit[i]:
                continue
            d = q - p
            far = float(np.linalg.norm(d))
            if far <= 1e-9:
                continue
            d = d / far
            o, gone = Vector(p), 0.
            while gone < far:
                loc, normal, _, dist = tree.ray_cast(o, Vector(d), far - gone)
                if loc is None:
                    break
                if dist > 1e-5 * size and np.dot(normal, d) < 0:
                    hit[i] = True
                    break
                gone += dist + 1e-6 * size
                o = Vector(p + d * gone)
        return hit
    out = []
    for s in swung:
        # the surface's vertices: object rows of its chains' faces; the ones a face of the rest shares are held
        mine = [(name, sorted({v for f, k in zip(faces, values) if k - 1 in s['ids'] for v in f}), co)
                for name, co, faces, values in meshes]
        held = {_key(p) for name, co, faces, values in meshes
                for f, k in zip(faces, values) if k - 1 not in s['ids'] for p in co[f]}
        where = [(name, v) for name, vs, _ in mine for v in vs]
        points = np.array([co[v] for _, vs, co in mine for v in vs]).reshape(-1, 3)
        chains = [cut_from(rows[i]['points'], at) for i, at in zip(s['ids'], s['cut'])]
        skins = [[(2, 1.)] if _key(p) in held else w for p, w in zip(points, skin(points, chains))]
        hung, propped, moved, turned = drape(chains, points, skins, passes, turn=s['turn'])
        for i, at, chain, c, up in zip(s['ids'], s['cut'], chains, hung, propped):
            rows[i]['propped'] = up
            out.append(dict(bones=rows[i]['bones'], before=off_down(chain), after=off_down(c), propped=up))
            if not s['turn']:
                continue
            # its joints before the cut as they were, then the draped chain from the cut on
            whole = np.asarray(rows[i]['points'], float)
            done = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(whole, axis=0), axis=1))])
            kept = whole[:int(np.searchsorted(done, at, side='right'))] if at > 0 else whole[:0]
            if len(kept) and np.linalg.norm(kept[-1] - c[0]) <= 1e-6:
                kept = kept[:-1]
            rows[i]['points'] = np.vstack([kept, c]).round(6).tolist()
        if not s['turn']:
            continue
        by = {}
        for (name, v), p, m in zip(where, moved, turned):
            by.setdefault(name, []).append((v, p, m))
        for name, items in by.items():
            _move(bpy.data.objects[name], items)
    return out


def _move(ob, items):
    """Set the vertices ``items`` ((index, world position, 3x3 its neighbourhood turned by)) of ``ob``, and turn the
    normals of their corners with them."""
    me = ob.data
    inverse = np.linalg.inv(np.asarray(ob.matrix_world))
    loops = len(me.loops)
    normals = np.empty(loops * 3)
    me.corner_normals.foreach_get('vector', normals)
    normals = normals.reshape(loops, 3)
    corner = np.empty(loops, dtype=np.int64)
    me.loops.foreach_get('vertex_index', corner)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    turn = {}
    for v, p, m in items:
        co[v] = inverse[:3, :3] @ p + inverse[:3, 3]
        turn[v] = m
    for l, v in enumerate(corner):
        if v in turn:
            n = turn[v] @ normals[l]
            normals[l] = n / max(np.linalg.norm(n), 1e-12)
    me.vertices.foreach_set('co', co.astype(np.float32).ravel())
    me.update()
    me.normals_split_custom_set(normals.tolist())
