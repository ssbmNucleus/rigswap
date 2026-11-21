"""The model when its download splits it across files: a body, and its hair, its clothes, its tank, in files of their
own (Splatoon's Inkling: Player00 in underwear, ten bottoms, six hairs, two tanks).

What says which files are one character is their skeletons. A game that dresses one body in parts writes each part
with the joints it rides, by the body's names: a bottom with the hips and legs (joint_root, hip, crotch_L, leg1_L...),
a tank with the chest and shoulders, a hair with the head's strands. Its joints lie where the body's do up to one turn,
scale and shift: the file's units, axes and origin (a part is written with its own root at the origin, in other units).
So a file whose named joints land on the body's by one similarity, within FIT of the body's size, belongs with it, and
that similarity places it. A file placed so carries joints of its own that place others in turn: a lower-detail body
that has the hair's strands under its head (Player00_Hlf) places the hair that has those strands. The joints that say
where a file goes are the ones its mesh rides: one it binds nothing to has the place the file anchors it at (Project
Sekai's head file keeps its Chest, its root, on its Neck), not the body's. Three of them at least, not all in a line,
or else coinciding as they are at two places or more: files written in one frame (2B's legs share only the spine with
the rest; the Sekai head its Neck and Head). One place says nothing: every file has a root at its origin. So the
Inkling's other hairs, which share only their root with anything, and Ultimate Samus's gun (its joints are its own; a
second copy of the cannon her body already wears) stay out.

What is worn and what is not, of the files placed:
- a file spanning the body's reach along each of its axes (EXTENT; its mesh and its joints) is a copy of the body, not
  worn on it: a costume (c01..c07), a level of detail, Brawl's shadow model, an effect shell around it;
- the same model in two formats (one name, as many triangles) is one file, in the format rig.pick prefers;
- a file whose every mesh the body already has, as many triangles in the same place, adds nothing (a switched prop the
  body file keeps, shipped as a file of its own too);
- a piece lies where the body has none and where no other file lies (under ELSEWHERE of its mesh on the body's, or on
  another's, or of another's on it): the head, torso or legs of a character split by body part (2B's five Unreal part
  files, Grace Ashcroft's torso and legs), the hair a bald body lacks. A piece is worn, whatever its skeleton: a whole
  one is no other body when it lies where the body has none;
- a part (not a piece) has a skeleton of its own, or a whole one lying where the body has none: clothes on the body,
  or one of several for one place (ten bottoms, two tanks). The parts riding the same joints of the body are worn on
  one place; one AI ask (VOTES times) shows each on the grey body and asks which the everyday look wears: one, layers
  worn together, or none;
- a whole skeleton lying on the body is another body: a costume, another character standing in its place (Ness's
  Poo), never worn, only a bridge;
- a file nothing places (no three named joints in common: a prop, an effect, a hand pose written at the origin, a mesh
  with no skeleton) stays out.

Which of the pieces rig.pick takes as the body matters little once the others join it: 2B's is the hair and its
headband (the most triangles), and her head, torso and legs join it, all in the one frame the files share. What
the body is measured by is its reach, its joints with its mesh: the head's file has the skeleton down to the floor.

A part's own joints (a hair's strands, a tank's) join the body's skeleton under the joint they hang from; a part's root
that isn't a body joint (the hair's head_root) is the body joint it lands on.

Of the files nothing places, one whose whole mesh rides one joint is an object the game hangs on a joint of the
character (attach): the Inkling's ink in her tank (Tnk_Gauge_Shot_First, one joint of its own name), Kamek's broom
and wand jewel, Waddle Dee's goggles and inner tube. It is written about that joint, in the units of the files it
hangs among: the ink stands up along +z from its origin as the tank's Tank_Ink does inside the glass. So it goes on
with that joint's rest frame laid on the host's. Which joint, or none (the everyday look leaves it out), and the colour
of its surfaces drawing no image (the ink's M_Ink), is meaning: one ask in words, VOTES times.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

FIT = 0.02          # a part's joints land on the body's within this share of the body's size
SPREAD = 0.05       # and span a plane: the second extent of the joints shared at least this share of the first
SHARED = 3          # joints in common, at least, for one similarity
INLIERS = 0.75      # and this share of the joints in common land
EXTENT = 0.9        # a part as long as the body's box this much along each of its axes is a copy of the body
ELSEWHERE = 1 / 3   # a file lies where the body (or another file) has none when less of its mesh than this is on it
NEAR = 0.01         # on: within about this share of the body's size
RIDES = 0.05        # a part rides a body joint carrying at least this share of its weight
VOTES = 3


def similarity(p, q):
    """The turn (no mirror), scale and shift taking the points ``p`` onto ``q`` by least squares (Umeyama). -> (4x4,
    the largest miss)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mp, mq = p.mean(axis=0), q.mean(axis=0)
    a, b = p - mp, q - mq
    u, s, vt = np.linalg.svd(b.T @ a / len(p))
    d = np.diag([1.0, 1.0, -1.0 if np.linalg.det(u @ vt) < 0 else 1.0])
    rot = u @ d @ vt
    var = (a ** 2).sum() / len(p)
    k = float((s * np.diag(d)).sum() / var) if var > 0 else 1.0
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = k * rot, mq - k * rot @ mp
    return m, float(np.linalg.norm(_apply(m, p) - q, axis=1).max())


def spread(q):
    """Whether points span a plane: their second extent is SPREAD of their first, and the first isn't nil."""
    q = np.asarray(q, float)
    if len(q) < SHARED:
        return False
    sv = np.linalg.svd(q - q.mean(axis=0), compute_uv=False)
    return bool(sv[0] > 0 and sv[1] >= SPREAD * sv[0])


def land(p, q, tol):
    """The similarity landing ``p`` on ``q`` where they agree: fitted on all, then again on the points it lands within
    ``tol`` (a joint the other file keeps elsewhere, a tank's root moved, doesn't tilt the rest). -> (4x4, which
    points land within ``tol``), or None when fewer than SHARED of them, spread over a plane, or under INLIERS of them,
    do. Files written in one frame (an Unreal character's part files, its legs sharing only the spine with the rest;
    Project Sekai's head file, only its Neck and Head) need no move at all, which the joints prove where INLIERS of
    them coincide as they are at two places or more, more than ``tol`` apart: one place says nothing, every file has a
    root at its origin."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    near = np.linalg.norm(p - q, axis=1) <= tol
    if near.sum() >= INLIERS * len(p) and near.any() and np.ptp(q[near], axis=0).max() > tol:
        return np.eye(4), near
    if len(p) < SHARED or not spread(q):
        return None
    m, _ = similarity(p, q)
    for _ in range(2):
        near = np.linalg.norm(_apply(m, p) - q, axis=1) <= tol
        if near.sum() < max(SHARED, INLIERS * len(p)) or not spread(q[near]):
            return None
        m, _ = similarity(p[near], q[near])
    return m, np.linalg.norm(_apply(m, p) - q, axis=1) <= tol


def place(skeletons, body, size):
    """Where each file sits in the body's frame: {file index: 4x4}, the body's identity. A file is placed when its
    joints land on places already known by one similarity (``land``, within FIT x ``size``): the body's joints when
    it shares enough of them, else those a placed file brought (a bridge). Only the joints its mesh rides speak for a
    file: one it binds nothing to sits where the file anchors it, not where the body has it. Its other joints are then
    known too, the first file's place for each. Passes go on until none is placed. ``skeletons``: per file {'names',
    'heads', 'rides' (optional; the joints its mesh rides, else all speak)} (heads in the file's world), None for a
    file without one; ``body``: the body's index. -> (placements, {index: the share of its shared joints that land})."""
    own = dict(zip(skeletons[body]['names'], skeletons[body]['heads']))
    known = dict(own)
    placed = {body: np.eye(4)}
    landed = {}
    while True:
        more = False
        for i, sk in enumerate(skeletons):
            if i in placed or sk is None:
                continue
            heads = np.asarray(sk['heads'])
            speak = set(sk.get('rides') or sk['names'])
            found = None
            for ref in (own, known):
                shared = [j for j, n in enumerate(sk['names']) if n in ref and n in speak]
                if not shared:
                    continue
                found = land(heads[shared], [ref[sk['names'][j]] for j in shared], FIT * size)
                landed[i] = float(found[1].mean()) if found else 0.0
                if found:
                    break
            if not found:
                continue
            placed[i], more = found[0], True
            for n, h in zip(sk['names'], _apply(found[0], heads)):
                known.setdefault(n, h)
        if not more:
            return placed, landed


def _apply(m, p):
    return np.asarray(p, float) @ m[:3, :3].T + m[:3, 3]


def skeletons(load_dir, entries):
    """Per models.json entry, its file's rest skeleton read from load's .blend (its armature with most bones): {'names',
    'parents', 'heads', 'tails', 'frames' (each joint's rest frame, 4x4)} in the file's world, and its meshes drawn:
    'points' (their vertices, in the world), 'pieces' (per mesh: its triangles, its rows of points), 'rides' (the
    joints they ride, ``_rides``) and 'materials' ({name: whether it draws an image}); None for a file with no
    armature. Opens each file in turn: call before the body is opened."""
    import bpy
    out = []
    for e in entries:
        if not (e.get('opened') and e.get('armatures')):
            out.append(None)
            continue
        bpy.ops.wm.open_mainfile(filepath=str(_blend(load_dir, e)))
        arm = max((o for o in bpy.data.objects if o.type == 'ARMATURE'), key=lambda o: len(o.data.bones))
        bones = list(arm.data.bones)
        mw = np.array(arm.matrix_world)
        index = {b.name: i for i, b in enumerate(bones)}
        out.append({'names': [b.name for b in bones],
                    'parents': [index[b.parent.name] if b.parent else -1 for b in bones],
                    'heads': _apply(mw, [b.head_local[:] for b in bones]),
                    'tails': _apply(mw, [b.tail_local[:] for b in bones]),
                    'frames': np.array([mw @ np.array(b.matrix_local) for b in bones]),
                    'rides': _rides(arm), **_meshes()})
    return out


def _drawn():
    import bpy
    return [ob for ob in bpy.data.objects
            if ob.type == 'MESH' and len(ob.data.polygons) and not ob.hide_viewport and not ob.hide_render]


def _rides(arm):
    """The joints of ``arm`` the open file's meshes drawn ride: those carrying weight of them, the one a mesh is
    parented to, the root for a mesh bound to none (as _bring binds it)."""
    bones = {b.name for b in arm.data.bones}
    root = next(b.name for b in arm.data.bones if b.parent is None)
    out = set()
    for ob in _drawn():
        if ob.parent is not None and ob.parent.type == 'ARMATURE' and ob.parent_type == 'BONE':
            out.add(ob.parent_bone)
            continue
        groups = {g.index: g.name for g in ob.vertex_groups if g.name in bones}
        out |= {groups[e.group] for v in ob.data.vertices for e in v.groups
                if e.group in groups and e.weight > 0} or {root}
    return sorted(out)


def _meshes():
    """The open file's meshes drawn (as rig.extract reads them): {'points': their vertices in its world, 'pieces':
    per mesh, (its triangles, its first row, the row after its last)}."""
    from ..load.open import triangles
    from .render import _base_color_image
    rows, pieces, n, materials = [np.zeros((0, 3))], [], 0, {}
    for ob in _drawn():
        co = np.empty(len(ob.data.vertices) * 3)
        ob.data.vertices.foreach_get('co', co)
        rows.append(_apply(np.array(ob.matrix_world), co.reshape(-1, 3)))
        pieces.append((triangles(ob.data), n, n + len(ob.data.vertices)))
        n += len(ob.data.vertices)
        materials.update({m.name: _base_color_image(m) is not None for m in ob.data.materials if m is not None})
    return {'points': np.vstack(rows), 'pieces': pieces, 'materials': materials}


def _blend(load_dir, e):
    return Path(load_dir) / 'models' / Path(e['blend'].replace('\\', '/')).name


def _stem(e):
    return Path(e['file'].replace('\\', '/')).stem


def whole(entries):
    """The entries whose skeleton is whole (half the most bones of any): bodies, never parts (rig.pick's rule)."""
    most = max((e.get('bones') or 0) for e in entries if e.get('opened'))
    return {i for i, e in enumerate(entries) if e.get('opened') and (e.get('bones') or 0) >= most / 2}


def covers(points, body):
    """Whether a part placed in the body's frame (``points``) spans the body's reach (``body``: its points and its
    joints) along each of its axes, EXTENT of it at least: a copy of the whole body (a shadow, a level of detail with
    fewer joints, a suit, a shell around it), not something worn on it. The joints count: a file holding the head of a
    character split in parts (2B's) still has the skeleton down to the floor, and its torso reaches no further."""
    if not len(points):
        return False
    return bool(np.all(np.ptp(points, axis=0) >= EXTENT * np.ptp(body, axis=0)))


def carried(points, pieces, body, body_pieces, tol):
    """Whether the body already has each of a part's meshes: one of as many triangles (within 2%) filling the same
    box (within ``tol``), the part placed in the body's frame (``points``; ``pieces`` as _meshes gives them, and the
    body's). A switched prop the body file keeps among its meshes comes as a file of its own too (Ultimate Zero Suit
    Samus's paralyzer): it adds nothing."""
    def box(pts, a, b):
        return pts[a:b].min(axis=0), pts[a:b].max(axis=0)
    if not pieces:
        return False
    for t, a, b in pieces:
        lo, hi = box(points, a, b)
        if not any(abs(t - u) <= max(1, 0.02 * max(t, u)) and np.abs(box(body, c, d)[0] - lo).max() <= tol
                   and np.abs(box(body, c, d)[1] - hi).max() <= tol for u, c, d in body_pieces):
            return False
    return True


def on(a, b, r):
    """The share of the points ``a`` lying on the points ``b``: within about ``r`` (in the same cell of side ``r`` or a
    neighbouring one)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return 0.0
    cells = set(map(tuple, np.floor(b / r).astype(np.int64)))
    steps = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    keys = np.floor(a / r).astype(np.int64)
    return float(np.mean([any((k[0] + x, k[1] + y, k[2] + z) in cells for x, y, z in steps) for k in keys.tolist()]))


def copies(entries, i, j):
    """Whether two entries are one model in two formats: one name, as many triangles."""
    a, b = entries[i], entries[j]
    return _stem(a).lower() == _stem(b).lower() and a.get('triangles') == b.get('triangles')


def survey(load_dir, entry):
    """Which files are pieces and parts of the body ``entry`` (rig.pick's). -> a namespace: the entries, their
    skeletons, the body's index, the placements, the share of each file's shared joints that land, the pieces and the
    parts (indices into the entries, in their order, one per model) and the body's size.

    Of the files placed (not the body, not a copy of it in another format, not one spanning its reach, not a mesh it
    already has; one per model): a piece lies where the body has none (under ELSEWHERE of its mesh on the body's) and
    where no other such file lies (none has ELSEWHERE of its mesh on the other's): the body's own head, torso or legs in
    a file of their own, or the hair a bald body lacks. A piece is worn, whatever its skeleton. Of the rest, a part is
    one with a skeleton of its own (not whole), or a whole one lying where the body has none: worn on the body or one
    of several for one place, the everyday look decides (choose). A whole skeleton lying on the body is another body:
    a costume, a level of detail, another character standing in its place."""
    from types import SimpleNamespace
    from .rig import PREFERRED
    entries = json.loads((Path(load_dir) / 'models.json').read_text(encoding='utf-8'))
    body = next(i for i, e in enumerate(entries) if e.get('n') == entry.get('n') and e['file'] == entry['file'])
    sks = skeletons(load_dir, entries)
    mine = sks[body]['points'] if len(sks[body]['points']) else np.asarray(sks[body]['heads'])
    size = float(np.ptp(mine, axis=0).max()) or 1.0
    reach = np.vstack([mine, sks[body]['heads']])
    placed, landed = place(sks, body, size)
    bodies = whole(entries)

    def rank(i):
        ext = Path(entries[i]['file']).suffix.lower()
        return PREFERRED.index(ext) if ext in PREFERRED else len(PREFERRED)
    there = {}
    for i in sorted(placed):
        if i == body or copies(entries, i, body):
            continue
        pts = _apply(placed[i], sks[i]['points'])
        if covers(pts, reach) or carried(pts, sks[i]['pieces'], sks[body]['points'], sks[body]['pieces'], FIT * size):
            continue
        twins = [j for j in placed if j != i and copies(entries, i, j)]
        if any((rank(j), j) < (rank(i), i) for j in twins):
            continue
        there[i] = pts
    r = NEAR * size
    elsewhere = {i: on(pts, mine, r) < ELSEWHERE for i, pts in there.items()}
    rivals = {i: [j for j in there if j != i and (on(there[i], there[j], r) >= ELSEWHERE
                                                  or on(there[j], there[i], r) >= ELSEWHERE)]
              for i in there if elsewhere[i]}
    pieces = [i for i in there if elsewhere[i] and not rivals[i]]
    parts = [i for i in there if i not in pieces and (i not in bodies or elsewhere[i])]
    # objects: files nothing places whose whole mesh rides one joint of their own, named in no other model's file
    # (attach; a hair riding only its head_root, which every hair has, is none), one per model
    def own(i):
        return not any(sks[i]['rides'][0] in sk['names'] for j, sk in enumerate(sks)
                       if j != i and sk is not None and not copies(entries, i, j))
    alone = [i for i, sk in enumerate(sks) if sk is not None and i not in placed and len(sk['rides']) == 1 and own(i)]
    objects = [i for i in alone
               if not any((rank(j), j) < (rank(i), i) for j in alone if j != i and copies(entries, i, j))]
    return SimpleNamespace(load_dir=Path(load_dir), entries=entries, skeletons=sks, body=body, placed=placed,
                           landed=landed, pieces=pieces, parts=parts, objects=objects, size=size, joined=[])


def read(load_dir, entry, blend):
    """The body's Rig (rig.read of ``blend``) with every part the files place joined to it (``join``); the scene stays
    open. -> (rig, the survey with the parts joined, or None when there are none and no objects)."""
    from . import rig as rigs
    found = survey(load_dir, entry)
    rig = rigs.read(blend)
    if not found.pieces and not found.parts:
        return rig, found if found.objects else None
    found.joined = join(found)
    return rigs.scene(blend), found


def _key(found, i):
    """A part's name: its file's, with the entry's number where two parts' files share one (Smash 4's model.DAE)."""
    stems = [_stem(found.entries[j]) for j in found.pieces + found.parts]
    stem = _stem(found.entries[i])
    return stem if stems.count(stem) == 1 else f"{stem}_{found.entries[i]['n']}"


def join(found):
    """Bring each part's meshes into the open scene, placed in the body's frame and skinned to the body's armature.
    The joints it shares with the body by name are the body's; its own join the body's skeleton under the joint they
    hang from; its root, where the body has no joint of that name, is the body joint it lands on (else it hangs from
    the nearest). A joint it shares by name but keeps elsewhere, or names as another part's kept elsewhere, is its own,
    renamed. -> [{'key', 'file', 'index', 'meshes', 'bones' (the joints added for it), 'names' (its joints renamed:
    {its name: the scene's})}]."""
    import bpy
    arm = max((o for o in bpy.data.objects if o.type == 'ARMATURE'), key=lambda o: len(o.data.bones))
    mw = np.array(arm.matrix_world)
    body = {b.name: _apply(mw, [b.head_local[:]])[0] for b in arm.data.bones}
    names = list(body)
    heads = np.array([body[n] for n in names])
    tol = FIT * found.size
    added = {}                                  # bone name: (head, tail, parent name), in the body's frame
    out = []
    for i in sorted(found.pieces + found.parts):
        sk, m, key = found.skeletons[i], found.placed[i], _key(found, i)
        h, t = _apply(m, sk['heads']), _apply(m, sk['tails'])
        rename = {}
        for j, n in enumerate(sk['names']):
            if n in body:
                if np.linalg.norm(h[j] - body[n]) > tol:
                    rename[n] = f'{n}.{key}'
            elif sk['parents'][j] < 0:
                d = np.linalg.norm(heads - h[j], axis=1)
                if d.min() <= tol:
                    rename[n] = names[int(d.argmin())]
            elif n in added and np.linalg.norm(added[n][0] - h[j]) > tol:
                rename[n] = f'{n}.{key}'
        mine = []
        for j, n in enumerate(sk['names']):
            name = rename.get(n, n)
            if name in body or name in added:
                continue
            p = sk['parents'][j]
            parent = (rename.get(sk['names'][p], sk['names'][p]) if p >= 0
                      else names[int(np.linalg.norm(heads - h[j], axis=1).argmin())])
            added[name] = (h[j], t[j], parent)
            mine.append(name)
        meshes = _bring(_blend(found.load_dir, found.entries[i]), key, m, rename, arm)
        out.append({'key': key, 'file': found.entries[i]['file'], 'index': i, 'meshes': meshes, 'bones': mine,
                    'names': rename, 'piece': i in found.pieces})
    _add_bones(arm, added, found.size)
    return out


def _bring(path, key, m, rename, arm):
    """Append the meshes drawn of the .blend at ``path`` to the open scene, moved by ``m`` and bound to ``arm``, their
    vertex groups renamed by ``rename``; each named for the part (``key``) unless its name starts so. Whatever else
    came with them (the part's armature) goes. -> their names."""
    import bpy
    from mathutils import Matrix
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.objects = list(src.objects)
    new = [ob for ob in dst.objects if ob is not None]
    for ob in new:                  # an object's place in the world is worked out once it is in the scene
        bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    worlds = {ob.name: np.array(ob.matrix_world) for ob in new}
    arms = [ob for ob in new if ob.type == 'ARMATURE']
    root = max(arms, key=lambda ob: len(ob.data.bones)) if arms else None
    out = []
    for ob in new:
        if ob.type != 'MESH' or not len(ob.data.polygons) or ob.hide_viewport or ob.hide_render:
            continue
        world = m @ worlds[ob.name]
        if ob.parent is not None and ob.parent.type == 'ARMATURE' and ob.parent_type == 'BONE':
            g = ob.vertex_groups.get(ob.parent_bone) or ob.vertex_groups.new(name=ob.parent_bone)
            g.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
        elif root is not None and not any(md.type == 'ARMATURE' for md in ob.modifiers) and not (
                ob.parent is not None and ob.parent.type == 'ARMATURE'):
            # a mesh in the part's file bound to nothing rides the part's root
            top = next(b.name for b in root.data.bones if b.parent is None)
            ob.vertex_groups.new(name=top).add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
        for old, name in rename.items():
            _regroup(ob, old, name)
        ob.parent = None
        ob.matrix_world = Matrix(world.tolist())
        for md in [md for md in ob.modifiers if md.type == 'ARMATURE']:
            ob.modifiers.remove(md)
        md = ob.modifiers.new('Armature', 'ARMATURE')
        md.object, md.use_vertex_groups = arm, True
        if not ob.name.startswith(key):
            ob.name = f'{key}_{ob.name}'
        out.append(ob.name)
    for ob in new:
        if ob.name not in out:
            bpy.data.objects.remove(ob, do_unlink=True)
    return out


def _regroup(ob, old, new):
    """Rename ``ob``'s vertex group ``old`` to ``new``, its weights added into ``new``'s where that exists."""
    g = ob.vertex_groups.get(old)
    if g is None or old == new:
        return
    into = ob.vertex_groups.get(new)
    if into is None:
        g.name = new
        return
    for v in ob.data.vertices:
        for e in v.groups:
            if e.group == g.index and e.weight > 0:
                into.add([v.index], e.weight, 'ADD')
    ob.vertex_groups.remove(g)


def _add_bones(arm, added, size):
    """Add ``added`` ({name: (head, tail, parent name)}, in the world) to the armature ``arm``."""
    import bpy
    if not added:
        return
    inv = np.linalg.inv(np.array(arm.matrix_world))
    least = 1e-3 * size * np.cbrt(abs(np.linalg.det(inv[:3, :3])))     # in the armature's own units
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bones, made = arm.data.edit_bones, {}
    for name, (head, tail, _) in added.items():
        eb = bones.new(name)
        h, t = _apply(inv, [head, tail])
        if np.linalg.norm(t - h) < least:                  # a bone of no length would be dropped
            t = h + np.array([0.0, 0.0, 10 * least])
        eb.head, eb.tail, eb.use_deform = h.tolist(), t.tolist(), True
        made[name] = eb
    for name, (_, _, parent) in added.items():
        p = made.get(parent) or bones.get(parent)
        if p is not None and p is not made[name]:
            made[name].parent = p
    bpy.ops.object.mode_set(mode='OBJECT')


QUESTION = ('The pictures show one game character with parts that came in files of their own, each drawn on its body '
            'in its own colours, the rest of the body in plain grey. The first, tile 0, draws them all together; each '
            'picture after it is a group, lettered at the top right of its tiles: the parts of a group are worn on the '
            'same place of the body, and each numbered tile draws one of them alone. Their files, the joint each '
            'rides most and its group: {names}. A download can hold everything a character may wear where it wears '
            'one at a time: several hairstyles, tops, bottoms, shoes, a pack or a tank in a few versions. The grey '
            'body is drawn as its own file has it, which a download that dresses it in parts leaves bare where they '
            'go (in underwear, bald): it is not dressed already. The everyday look is how the character stands idle, '
            'ready to fight, in its usual outfit, each piece worn once. For each group, list the numbers of its parts the everyday look wears: one when they are versions '
            'of one piece; more when they are layers worn together; none when the group holds only what the '
            'everyday look leaves out. Set cant_tell only if you cannot tell.')


def choose(rig, found, up, front, out, ask):
    """Of the parts ``read`` joined, keep those the everyday look wears (the pieces are worn, unasked): the parts riding
    the same joints of the body are one place (``places``); one ask, VOTES times, with a picture of each on the grey
    body (the pieces in it); a part stays when most answers keep it. The others' meshes leave the scene, and the joints
    added for them with them. -> (the rig read again, the report)."""
    from . import rig as rigs, variants
    rows, start = _rows(rig)
    report = {'files': {p['key']: p['file'] for p in found.joined},
              'pieces': [p['key'] for p in found.joined if p['piece']]}
    joined = [p for p in found.joined if len(rows(p['meshes'])) and not p['piece']]
    if not joined:
        return rig, report
    worn = {m for p in joined for m in p['meshes']}
    body = variants.Unit('body', rig, [m for m in rig.meshes if m not in worn],
                         rows([m for m in rig.meshes if m not in worn]))
    units = []
    for p in joined:
        u = variants.Unit(p['key'], rig, [m for m in p['meshes'] if m in start], rows(p['meshes']))
        u.switched = False
        units.append(u)
    own = {b for p in found.joined for b in p['bones']}
    groups = places(rig, units, own)
    size = float(np.ptp(rig.points, axis=0).max())
    path = Path(out) / 'parts.png'
    pictures = [str(p) for p in variants.draw([body] + units, groups, sorted(worn), up, front,
                                              rig.points[body.rows] if len(body.rows) else rig.points, size, path)]
    members = [u for g in groups for u in g]
    letter = {u.key: variants._letter(g) for g, row in enumerate(groups) for u in row}
    names = '; '.join(f'{j + 1} {u.key} ({rig.names[u.bone]}, {letter[u.key]})' for j, u in enumerate(members))
    answers, info = [], None
    for r in range(VOTES):
        answer, info = ask(QUESTION.format(names=names) + f' (Ask {r + 1} of {VOTES}.)', pictures,
                           variants.schema(groups), {'cant_tell': True}, route='core parts')
        answers.append(answer or {'cant_tell': True})
    kept = [members[j - 1].key for j in tally(answers, [len(g) for g in groups])]
    for p in joined:
        if p['key'] not in kept:
            _remove(p['meshes'])
    rig = rigs.scene(rig.blend)
    # a joint a worn part shares with an unworn one that brought it stays (a tank's Tank_Ink, carrying nothing)
    used = {p['names'].get(n, n) for p in found.joined if p['piece'] or p['key'] in kept
            for n in found.skeletons[p['index']]['names']}
    _drop_bones(rig, {b for p in joined if p['key'] not in kept for b in p['bones']} - used)
    report.update(groups=[[u.key for u in g] for g in groups], kept=kept, answers=answers,
                  source=(info or {}).get('source'))
    return rigs.scene(rig.blend), report


def tally(answers, sizes):
    """The members kept (numbered from 1 across the groups, ``sizes`` long each) by the ``answers`` (one per ask: per
    group's letter, the numbers it keeps): a member most answers keep; and a group most answers dress, though they
    name different versions of its piece (2 of 3 asks dress the Inkling's hips, one in shorts and one in a skirt),
    keeps the one named most, the first named of those tied."""
    from .variants import _letter
    out, start = [], 0
    for g, n in enumerate(sizes):
        mine = range(start + 1, start + n + 1)
        named = [[j for j in a.get(_letter(g)) or [] if j in mine] for a in answers]
        votes = {j: sum(j in ns for ns in named) for j in mine}
        kept = [j for j in mine if votes[j] * 2 > len(answers)]
        if not kept and sum(bool(ns) for ns in named) * 2 > len(answers):
            order = [j for ns in named for j in ns]
            kept = [max(mine, key=lambda j: (votes[j], -order.index(j) if j in order else -len(order)))]
        out += kept
        start += n
    return out


def places(rig, units, own):
    """The parts grouped by the place they are worn: the body joints each rides (those carrying RIDES of its weight; a
    joint of a part's own counts as the body joint it hangs from). Two parts sharing half the joints of the one riding
    fewer are worn on one place."""
    index = {n: i for i, n in enumerate(rig.names)}
    mine = {index[n] for n in own if n in index}

    def body_joint(b):
        while b >= 0 and b in mine:
            b = rig.parents[b]
        return b
    rides = []
    for u in units:
        w = rig.weights[u.rows].sum(axis=0)
        share = {}
        for b in np.flatnonzero(w):
            k = body_joint(int(b))
            share[k] = share.get(k, 0.0) + w[b] / max(w.sum(), 1e-12)
        rides.append({k for k, v in share.items() if v >= RIDES})
    parent = list(range(len(units)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = rides[i], rides[j]
            if a and b and len(a & b) * 2 >= min(len(a), len(b)):
                parent[find(i)] = find(j)
    out = {}
    for i, u in enumerate(units):
        out.setdefault(find(i), []).append(u)
    return sorted(out.values(), key=lambda g: units.index(g[0]))


def _drop_bones(rig, unworn):
    """Remove from the open scene's armature the joints added for parts not worn (``unworn``) that carry nothing,
    nor anything below them, with what hangs below them."""
    import bpy
    carried = rig.carried()
    kids = rig.children()

    def below(b):
        out = [b]
        for k in kids[b]:
            out += below(k)
        return out
    gone = set()
    for b, n in enumerate(rig.names):
        if n in unworn and not carried[below(b)].any():
            gone.update(rig.names[x] for x in below(b))
    if not gone:
        return
    arm = max((o for o in bpy.data.objects if o.type == 'ARMATURE'), key=lambda o: len(o.data.bones))
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for n in gone:
        eb = arm.data.edit_bones.get(n)
        if eb is not None:
            arm.data.edit_bones.remove(eb)
    bpy.ops.object.mode_set(mode='OBJECT')


ATTACH = ('{first}{sheets} show {who}, whose model came in several files, in plain grey but for objects that came in'
          ' files of their own: each a mesh written about one joint of its own, which the game hangs on a joint of '
          'the character, or which its everyday look does not show. Each of those pictures is one object, lettered '
          'at the top right of its tiles; each numbered tile draws it in its own colours hung on one of the joints '
          'of the character that carry no mesh, made for something to hang on, and again over what covers it, so '
          'that one inside or behind another part is seen. The objects: {objects}. The files on its body: {worn}. '
          'The joints of the tiles: {tiles}. The everyday look is how the character stands idle, ready to fight, in '
          'its usual outfit. For each lettered object, give the number of the tile showing it as its everyday look '
          'has it, or 0 when that look does not show it{bare}; and the colour its surfaces drawing no image show, as'
          ' #rrggbb (where its game paints a colour of its own on them in play, a team or ink colour, the one of its'
          ' best-known look), or none when it has no such surface.')


def attach(rig, found, up, front, out, ask, character=None, picture=None):
    """Hang the objects ``survey`` found where the everyday look hangs them. Each is drawn hung on each joint carrying
    nothing (the joint its mesh rides laid on that one at rest, ``hosts``; joints at one frame drawn once), a copy per
    joint, one picture per object, after ``picture`` (the character in its own colours as its files draw it, none
    of them on: what 0 answers; without it, grey tiles alone hung Waddle Dee's swim goggles, fitting his face, at-6);
    one ask, VOTES times (``hung``). The copy every answer names stays, its surfaces drawing no image in the colour
    named; the others go. -> (the rig read again, the report)."""
    import bpy
    from . import rig as rigs, variants
    frames = hosts(found, rig.names)
    empty = []
    for n, w in zip(rig.names, rig.carried()):
        if w <= 0 and n in frames and not any(_same(frames[n], frames[e], FIT * found.size) for e in empty):
            empty.append(n)
    letters = [variants._letter(k) for k in range(len(found.objects))]
    report = {'objects': dict(zip(letters, (found.entries[i]['file'] for i in found.objects))), 'hung': {}}
    if not empty:
        return rig, dict(report, skipped='no joint carries nothing')
    whose = dict.fromkeys(found.skeletons[found.body]['names'], found.entries[found.body]['file'])
    for p in found.joined:
        for n in found.skeletons[p['index']]['names']:
            whose.setdefault(p['names'].get(n, n), p['file'])
    arm = max((o for o in bpy.data.objects if o.type == 'ARMATURE'), key=lambda o: len(o.data.bones))
    copies = []                                 # (its tile's number, letter, the object's entry, joint, meshes)
    for letter, i in zip(letters, found.objects):
        sk = found.skeletons[i]
        own = sk['frames'][sk['names'].index(sk['rides'][0])]
        for joint in empty:
            key = f'{letter}{len(copies) + 1}'
            meshes = _bring(_blend(found.load_dir, found.entries[i]), key, frames[joint] @ np.linalg.inv(own),
                            {b: joint for b in sk['names']}, arm)
            copies.append((len(copies) + 1, letter, i, joint, meshes))
    rig = rigs.scene(rig.blend)
    rows, start = _rows(rig)
    mine = {m for c in copies for m in c[4]}
    body = variants.Unit('body', rig, [m for m in rig.meshes if m not in mine],
                         rows([m for m in rig.meshes if m not in mine]))
    groups = [[variants.Unit(f'{c[1]}{c[0]}', rig, [m for m in c[4] if m in start], rows(c[4]))
               for c in copies if c[1] == letter] for letter in letters]
    size = float(np.ptp(rig.points, axis=0).max())
    pictures = [str(p) for p in variants.draw([body] + [u for g in groups for u in g], groups, sorted(mine), up,
                                              front, rig.points[body.rows], size, Path(out) / 'attach.png',
                                              through=True)][1:]
    first = picture is not None and Path(picture).is_file()
    pictures = ([str(picture)] if first else []) + pictures
    objects = '; '.join(f"{a} {found.entries[i]['file']} (materials: " + ', '.join(
        f"{m} ({'an image' if drawn else 'no image'})" for m, drawn in found.skeletons[i]['materials'].items())
        + ')' for a, i in zip(letters, found.objects))
    tiles = '; '.join(f'{c[0]} {c[3]} (of {whose[c[3]]})' if c[3] in whose else f'{c[0]} {c[3]}' for c in copies)
    worn = [found.entries[found.body]['file']] + [p['file'] for p in found.joined
                                                  if any(m in bpy.data.objects for m in p['meshes'])]
    question = ATTACH.format(
        first=f"Picture 1 shows {character or 'a game character'} in its own colours, as its files draw it, none of "
              "the objects on. " if first else '', sheets='The pictures after it' if first else 'The pictures',
        who='it' if first else character or 'a game character', objects=objects, worn=', '.join(worn), tiles=tiles,
        bare=', as in picture 1' if first else '')
    row = {'type': 'object', 'properties': {'row': {'type': 'string', 'enum': letters},
                                            'tile': {'type': 'integer', 'enum': list(range(len(copies) + 1))},
                                            'colour': {'type': 'string'}},
           'required': ['row', 'tile', 'colour'], 'additionalProperties': False}
    schema = {'type': 'object', 'properties': {'rows': {'type': 'array', 'items': row}},
              'required': ['rows'], 'additionalProperties': False}
    answers, info = [], None
    for r in range(VOTES):
        answer, info = ask(question + f' (Ask {r + 1} of {VOTES}.)', pictures, schema, {'rows': []},
                           route='core attach')
        answers.append(answer or {'rows': []})
    kept = hung(answers, {letter: [c[0] for c in copies if c[1] == letter] for letter in letters})
    for number, letter, i, joint, meshes in copies:
        if kept.get(letter, (None,))[0] != number:
            _remove(meshes)
            continue
        colour = kept[letter][1]
        if colour:
            _colour(meshes, colour)
        report['hung'][letter] = {'joint': joint, 'tile': number,
                                  'colour': '#%02x%02x%02x' % colour if colour else None, 'meshes': meshes}
    report.update(tiles={c[0]: [c[1], c[3]] for c in copies}, answers=answers, source=(info or {}).get('source'))
    return rigs.scene(rig.blend), report


def hung(answers, rows):
    """Per lettered object (``rows``: {letter: its tiles' numbers}), the tile every one of the ``answers`` names (0
    names none): an object the files leave off goes on only so, as paint's dark side (three answers split two to one
    are a coin: Waddle Dee's swim goggles, at-5); and of the colours they give it, the one nearest the rest
    (paint.nearest; None when they give none). -> {letter: (tile, (r, g, b) or None)}, the objects hung."""
    from .paint import _rgb, nearest
    out = {}
    for letter, numbers in rows.items():
        said = [next((r for r in (a or {}).get('rows') or [] if r.get('row') == letter), None) for a in answers]
        said = [r for r in said if r and r.get('tile') in numbers]
        tile, n = Counter(r['tile'] for r in said).most_common(1)[0] if said else (None, 0)
        if not answers or n < len(answers):
            continue
        colours = [c for c in (_rgb(r.get('colour')) for r in said if r['tile'] == tile) if c]
        out[letter] = (tile, nearest(colours) if colours else None)
    return out


def _same(a, b, tol):
    """Whether two rest frames (4x4) are one: their places within ``tol``, their axes within 1%."""
    def axes(f):
        return f[:3, :3] / (np.cbrt(abs(np.linalg.det(f[:3, :3]))) or 1.0)
    return bool(np.linalg.norm(a[:3, 3] - b[:3, 3]) <= tol and np.abs(axes(a) - axes(b)).max() <= 0.01)


def _rows(rig):
    """-> (rows(meshes): the rig's rows of those meshes, {mesh: its first row})."""
    import bpy
    counts = {n: len(bpy.data.objects[n].data.vertices) for n in rig.meshes}
    start = dict(zip(rig.meshes, np.cumsum([0] + [counts[n] for n in rig.meshes])))

    def rows(meshes):
        meshes = [m for m in meshes if m in start]
        return np.concatenate([np.arange(start[m], start[m] + counts[m]) for m in meshes]).astype(int) if meshes \
            else np.zeros(0, int)
    return rows, start


def _remove(meshes):
    """Take the meshes out of the open scene, with their data where nothing else uses it."""
    import bpy
    for name in meshes:
        ob = bpy.data.objects[name]
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data.users == 0:
            bpy.data.meshes.remove(data)


def hosts(found, names):
    """The rest frame (4x4, in the body's world) of each joint of the open scene's skeleton (``names``): the body's
    own, or that of the part that brought it, placed."""
    sk = found.skeletons[found.body]
    frames = dict(zip(sk['names'], sk['frames']))
    for p in found.joined:
        sk, m = found.skeletons[p['index']], found.placed[p['index']]
        for n, f in zip(sk['names'], sk['frames']):
            frames.setdefault(p['names'].get(n, n), m @ f)
    return {n: frames[n] for n in names if n in frames}


def _colour(meshes, rgb):
    """Give the materials of ``meshes`` drawing no image the colour ``rgb`` (sRGB, 0 to 255)."""
    import bpy
    from .render import _base_color_image
    c = np.asarray(rgb, float) / 255
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    for name in meshes:
        for mat in bpy.data.objects[name].data.materials:
            if mat is None or _base_color_image(mat) is not None:
                continue
            mat.diffuse_color = (*linear, 1.0)
            for node in (mat.node_tree.nodes if mat.node_tree else []):
                if node.type == 'BSDF_PRINCIPLED' and not node.inputs['Base Color'].is_linked:
                    node.inputs['Base Color'].default_value = (*linear, 1.0)
