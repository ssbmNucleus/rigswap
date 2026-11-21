"""One m-ex accessory per chosen hanging surface, within the game's dynamics limit.

The new middle's build says which of the model's hair and cloth have no partner
on the fighter (loose.json, rigswap.core.build), which faces each carries, and
how each rests (rigswap.core.hang); loose_groups says which of those swing, and
together, on prep's final surface, and springs holds each where it rests. Prep supplies final triangles and textures. Body and rigid pieces never become accessories.
Body and accessories partition those triangles. The costume's
body skeleton stays vanilla. Each accessory gets an independent local skeleton,
one dynamics descriptor per chain from melee.library, and high/low/metal lookup tables.
The old importer's physics/accessories.py and shared/modellab/costume_extensions.py
are the source for the native --accessory call and lossless DAT graft respectively.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

from ....core import hang
from ... import Failed
from ..dat import Dat, dat_joints, joint_dobjs, model_scale
from ..library import PHYSICS, dynamics
from .write import hsdraw, write_culling

# Blender z-up to the game's y-up. The same frame as export/write.py.
TO_GAME = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
ROOT_LOCK = hang.ROOT_LOCK     # old importer studio/physics/constants.py, game model units


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=1), encoding='utf-8')


def skin_weights(objects):
    """The build's surface influences, excluding loose vertices and zero groups."""
    for ob in objects:
        if ob.type != 'MESH':
            continue
        names = {g.index: g.name for g in ob.vertex_groups}
        used = {i for face in ob.data.polygons for i in face.vertices}
        for i in sorted(used):
            yield {names[g.group]: g.weight for g in ob.data.vertices[i].groups if g.weight > 0}


def merge(body, donors, out):
    """Append complete donor blocks, rebase every relocation, preserve existing overrides."""
    d = Dat.read(body)
    if d.counts[1]:
        raise Failed('accessory graft requires a DAT without external references')
    root = d.root('mexCostume')
    old = []
    if root is not None:
        array = d.ptr(root + 12)
        old = [d.ptr(array + 4 * i) for i in range(d.u32(root + 8))]
    else:
        root = d.add(bytes(16))
        d.roots.append(('mexCostume', root))
    pointers = list(old)
    for path in donors:
        src = Dat.read(path)
        ext = src.root('mexCostume')
        if src.counts[1] or ext is None or src.u32(ext + 8) != 1:
            raise Failed(f'{path}: expected exactly one native accessory without external references')
        base = d.add(src.data, align=32)
        for field in src.relocations:
            d.point(base + field, base + src.u32(field))
        pointers.append(base + src.ptr(src.ptr(ext + 12)))
    array = d.add(bytes(4 * len(pointers)))
    for i, pointer in enumerate(pointers):
        d.point(array + 4 * i, pointer)
    struct.pack_into('>I', d.data, root + 8, len(pointers))
    d.point(root + 12, array if pointers else None)
    strings, roots = bytearray(), bytearray()
    for name, offset in d.roots:
        roots.extend(struct.pack('>II', offset, len(strings)))
        strings.extend(name.encode('ascii') + b'\0')
    d.tail, d.counts = bytes(roots + strings), (len(d.roots), 0)
    d.save(out)
    return len(old)


def definitions(path):
    """Read emitted m-ex parameters from the DAT bytes, not from authoring JSON."""
    d = Dat.read(path)
    root = d.root('mexCostume')
    if root is None:
        return []
    out, array = [], d.ptr(root + 12)
    for i in range(d.u32(root + 8)):
        acc = d.ptr(array + i * 4)
        descs = []
        for k in range(d.u32(acc + 8)):
            desc = d.ptr(d.ptr(acc + 12) + k * 4)
            params, count = d.ptr(desc + 4), d.u32(desc + 8)
            p1, p2, p3 = struct.unpack_from('>3f', d.data, desc + 12)
            descs.append(dict(bone=d.u32(desc), PARAM1=p1, PARAM2=p2, PARAM3=p3,
                              joints=[list(struct.unpack_from('>15f', d.data, params + j * 60))
                                      for j in range(count)]))
        out.append(dict(index=i, attach_bone=d.u32(acc + 4), dynamics=descs,
                        root_joint=d.ptr(acc), lookup_count=d.u32(acc + 16),
                        collision_shapes=d.u32(acc + 24)))
    return out


def _weights(points, chains):
    """Skin along the nearest segment of the accessory's chains, with the old importer's static root lock
    (hang.skin, as the build's drape carries the surface too). Chain j's joints are the local bones from
    _first(chains)[j] on."""
    try:
        return hang.skin(points, chains)
    except ValueError:
        raise Failed('physics chain has no segment with length')


def _first(chains):
    """Each chain's first local bone (hang.first): 0/1 are identity carriers, 2 the static root, then each chain's
    joints in turn, its tip last."""
    return hang.first(chains)


def chain_of(row, triangles):
    """The chain an accessory swings on, in the game's frame: the row's points, a terminal tail an importer put on its
    head extended along the last real segment, or to the surface's farthest point from the chain's start."""
    chain = np.asarray(row['points'], float) @ TO_GAME.T
    if np.linalg.norm(chain[-1] - chain[-2]) <= 1e-8:
        if len(chain) > 2 and np.linalg.norm(chain[-2] - chain[-3]) > 1e-8:
            chain[-1] = chain[-2] + (chain[-2] - chain[-3])
        else:
            surface = np.concatenate([t[1] for t in triangles]) @ TO_GAME.T
            chain[-1] = surface[np.linalg.norm(surface - chain[0], axis=1).argmax()]
    return chain


def chains_of(row, triangles):
    """The chains an accessory swings on, in the game's frame: the row's ``chains`` (several chains sharing one
    surface), else chain_of's one."""
    if row.get('chains'):
        return [np.asarray(c, float) @ TO_GAME.T for c in row['chains']]
    return [chain_of(row, triangles)]


def springs(kind, chain, scale=1., propped=None):
    """Per moving joint of ``chain`` (the game's frame, at rest), the spring that holds it where it rests: each frame
    gravity turns a link toward straight down by PARAM3 / its length in the game's units (the costume's times the
    fighter's model ``scale``: lb_80011710 reads it at the joints' world places) times the sine of its angle off down
    (lb_8001044C), and the spring turns it back as far. A link hanging straight down needs none. One the body props
    up (``propped``, per link: hang.drape's) gets the whole of that turn, which the sine never passes: it stays where
    it rests on the body whichever way the body turns (a stethoscope on a coat leaning forward), where a spring for
    its angle at rest only would let it sink into the body as the body leans over it. So does a link resting above
    the horizontal: there the spring for its angle balances gravity only as an inverted pendulum does, and a nudge
    down lets gravity's turn outgrow it until the link falls through to the mirror angle below (Inkling's fringe,
    drawn from the brow back up onto it, flipped out from 120 degrees off down to 60)."""
    gravity = PHYSICS.get(kind, PHYSICS['other'])['gravity']
    held = propped if propped is not None and len(propped) == len(chain) - 1 else [False] * (len(chain) - 1)
    out = []
    for a, b, up in zip(chain, chain[1:], held):
        d = np.asarray(b, float) - np.asarray(a, float)
        n = float(np.linalg.norm(d))
        out.append(0. if n <= 1e-9 else gravity / (n * scale) * (1. if up or d[1] > 0 else
                                                                 float(np.sqrt(max(0., 1 - (-d[1] / n) ** 2)))))
    return out


def jointed(triangles, chains, depth=10):
    """``triangles`` with each one that reaches past a joint of the chain it swings on cut until none does: a
    triangle is flat, so one whose corners lie more than a joint apart along its chain cannot bend where the chain
    does and spans the bend as a sheet. Diddy's tail on Fox (tail-1, tail-2, 2026-11-10): 92 of its 1057 triangles
    reached past a joint, 35 of them past two, the longest 6.7 of the tail's 15.5 from its root to its middle; where
    the tail curled they stood across the curl, a second strand beside the tail. Its edge that reaches farthest
    along the chain is halved, the new corner between its ends in place, normal, UV and build weights (held to the
    costume only where both ends are), and the two halves are looked at again; an edge two triangles share is halved
    for both, at the same point, so no gap opens. A triangle whose corners lie on different chains is left whole.

    A corner is counted where it lies along the chain (hang.nearest), also one the costume holds still: counted at
    the chain's start, a held corner far down the chain kept every piece between it and its swinging neighbours
    reaching as far as before however often it was halved (Goku's sash, most of it held: 52 triangles became 845).
    And an edge shorter than the chain's shortest joint is never halved: it cannot lie past a joint by its length,
    so where its ends still do, the nearest stretch of the chain jumps between them (a curl, a wide piece between two
    stretches of its chain) and no cut mends a jump. Whether an edge is halved is so read from the edge alone, and
    the two triangles it lies between agree. ``depth``: how many times one triangle may be halved along one line,
    should neither stop it."""
    try:
        short = [min(float(n) for n in np.linalg.norm(np.diff(c, axis=0), axis=1) if n > 1e-9) for c in chains]
    except ValueError:
        raise Failed('physics chain has no segment with length')

    def along(pts):
        """(the chain the corners lie on, or None for more than one; each corner's place along it, in joints)."""
        where = hang.nearest(np.asarray(pts) @ TO_GAME.T, chains)
        on = {j for j, *_ in where}
        return (on.pop(), [i + t for _, i, t, _ in where]) if len(on) == 1 else (None, [])
    out, todo = [], [(t, depth) for t in triangles][::-1]
    while todo:
        tri, left = todo.pop()
        material, pts, normals, uv, *more = tri
        held = list(more[1]) if len(more) > 1 else [False] * 3
        chain, at = along(pts)
        pts, normals, uv = np.asarray(pts, float), np.asarray(normals, float), np.asarray(uv, float)
        reach = [abs(at[e] - at[(e + 1) % 3]) for e in range(3)] if chain is not None else []
        long = [e for e, r in enumerate(reach) if r > 1 + 1e-6
                and np.linalg.norm((pts[(e + 1) % 3] - pts[e]) @ TO_GAME.T) >= short[chain]]
        if left <= 0 or not long:
            out.append(tri)
            continue
        i = max(long, key=lambda e: reach[e])
        j, k = (i + 1) % 3, (i + 2) % 3
        mid = (pts[i] + pts[j]) / 2, normals[i] + normals[j], (uv[i] + uv[j]) / 2
        mid = mid[0], mid[1] / (np.linalg.norm(mid[1]) or 1.0), mid[2]
        extra = []
        if more:        # the build's weights, then whether the costume holds the corner
            a, b = more[0][i], more[0][j]
            between = {bone: (a.get(bone, 0.) + b.get(bone, 0.)) / 2 for bone in {*a, *b}}
            extra = [(more[0], between)] + ([(held, held[i] and held[j])] if len(more) > 1 else [])
        for first, second in ((None, j), (i, None)):     # (i, mid, k) then (mid, j, k), the winding kept: the
                                                         # second is put by first, to be looked at after it
            order = [first if first is not None else 'mid', second if second is not None else 'mid', k]
            pick = lambda values, new: [new if o == 'mid' else values[o] for o in order]    # noqa: E731
            piece = (material, np.array(pick(pts, mid[0])), np.array(pick(normals, mid[1])), np.array(pick(uv, mid[2])),
                     *[pick(values, new) for values, new in extra], *more[2:])
            todo.append((piece, left - 1))
    return out


def write_accessory(row, triangles, world, folder, scale=1.):
    """SMD in anchor-local coordinates, textures sidecar, and native dynamics JSON.

    triangles: (material, positions, normals, UVs). 0/1 are identity carriers,
    2 the static root, then each chain (chains_of) off it: its moving joints and
    its terminal tip (one chain of n: 3..n+2 moving, n+3 the tip). A loose.json
    chain gets the springs that hold it at rest (springs: ``scale`` is the fighter's model scale).
    """
    folder.mkdir(parents=True, exist_ok=True)
    inverse = np.linalg.inv(world)
    chains = chains_of(row, triangles)
    triangles = jointed(triangles, chains)      # none reaches past a joint: the surface bends where the chain does
    local = lambda pts: np.asarray(pts) @ inverse[:3, :3].T + inverse[:3, 3]
    heads = [local(c) for c in chains]
    positions, parents = [np.zeros(3), np.zeros(3), heads[0][0]], [-1, 0, 1]
    for h in heads:         # each chain off the static root, its tip last
        positions += [h[0] - heads[0][0], *np.diff(h, axis=0)]
        parents += [2, *range(len(parents), len(parents) + len(h) - 1)]
    materials = list(dict.fromkeys(t[0] for t in triangles))
    nodes = {m: len(positions) + i for i, m in enumerate(materials)}
    lines = ['version 1', 'nodes']
    lines += [f'{i} "JOBJ_{i}" {parent}' for i, parent in enumerate(parents)]
    lines += [f'{node} "Joint_0_Object_{i}" -1' for i, node in enumerate(nodes.values())]
    lines += ['end', 'skeleton', 'time 0']
    lines += [f'{i} ' + ' '.join(f'{v:.9g}' for v in p) + ' 0 0 0' for i, p in enumerate(positions)]
    lines += [f'{node} 0 0 0 0 0 0' for node in nodes.values()]
    lines += ['end', 'triangles']
    for material, pts, normals, uv, *more in triangles:
        game = np.asarray(pts) @ TO_GAME.T
        pos = local(game)
        norms = np.asarray(normals) @ TO_GAME.T @ world[:3, :3]
        norms /= np.maximum(np.linalg.norm(norms, axis=1)[:, None], 1e-12)
        lines.append(material)
        skins = _weights(game, chains)
        if len(more) > 1:       # a corner the rest of the costume shares (loose_groups) stays on the static root
            skins = [[(2, 1.)] if held else w for w, held in zip(skins, more[1])]
        for p, n, tex, weights in zip(pos, norms, uv, skins):
            values = ' '.join(f'{v:.9g}' for v in [*p, *n, *tex])
            skin = ' '.join(f'{b} {w:.9g}' for b, w in weights)
            lines.append(f'{nodes[material]} {values} {len(weights)} {skin}')
    lines.append('end')
    smd, dyn = folder / 'accessory.smd', folder / 'dynamics.json'
    smd.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    held = row.get('attachment_from') == LOOSE
    propped = row.get('propped') or [None] * len(chains)
    definition = [dynamics(row['kind'], len(c) - 1, bone=b,
                           springs=springs(row['kind'], c, scale, up) if held else None)
                  for c, b, up in zip(chains, _first(chains), propped)]
    _json(dyn, definition)
    row.update(local_bones=[b + i for c, b in zip(chains, _first(chains)) for i in range(len(c) - 1)],
               dynamics=definition,
               parameter_source=PHYSICS.get(row['kind'], PHYSICS['other'])['source'],
               smd=str(smd), dynamics_file=str(dyn), triangles=len(triangles))
    return smd, dyn


LOOSE = 'loose.json'    # the new middle's chains with no partner on the fighter (rigswap.core.build)
def split(build, work, f):
    """Make an export-only body and accessories from prep's surface and the core's loose.json."""
    import bpy
    from ..targets.physics import record as target_physics, classify
    from .physics_choice import select
    from .. import FILES_VAR, _folder
    # Diagnose the prepared skin before the accessory partition changes its weights.
    # Count fresh target bytes: a prepared sidecar cannot hide a changed target.
    files = work / 'vanilla-files'
    if not (files / f'Pl{f.code}.dat').is_file():
        files = _folder(None, FILES_VAR, 'vanilla game files')
    bpy.ops.wm.open_mainfile(filepath=str(build / 'build.blend'))
    target_record = classify(target_physics(f, files), skin_weights(bpy.context.scene.objects))
    target_record['weight_source'] = str(build / 'build.blend')
    target_path = build / 'target.json'
    target_data = json.loads(target_path.read_text(encoding='utf-8')) if target_path.is_file() else {'name': f.name}
    target_data['physics'] = target_record
    _json(target_path, target_data)
    work.joinpath('physics').mkdir(exist_ok=True)
    _json(work / 'physics/selection.json', select([], target_record))
    if (build / LOOSE).is_file():
        return split_loose(build, work, f, target_record, model_scale(f, files))
    bpy.ops.wm.read_homefile(use_empty=True)
    return build / 'build.blend', [], []


def split_loose(build, work, f, target_record, scale=1.):
    """split for the new middle's build: its loose chains (``loose.json``: the model's hair and cloth with no partner on
    the fighter, each riding the bone it hangs from, with its joints at the fighter's rest and what it is: build.loose
    and hang.settle's), each carrying the faces its face attribute names. Which of them swing, and together,
    is loose_groups'; which fit in the game's room, the biggest first (physics_choice.select). Only the faces of the
    chains that swing leave the body; the others stay where the build put them. ``scale``: the fighter's model scale,
    for the springs that hold each at rest."""
    import bpy
    record = json.loads((build / LOOSE).read_text(encoding='utf-8'))
    rows = [dict(kind=c.get('kind', 'other'), source_bones=c['bones'], anchor=c['anchor'], points=c['points'],
                 pieces=[], attachment_from=LOOSE, **{k: c[k] for k in ('swings', 'why', 'propped') if k in c})
            for c in record['chains']]
    gathered, _, materials, faces = _take(build, len(rows), record['attribute'],
                                          lambda k, pts: (k - 1, None) if 0 < k <= len(rows) else None)
    empty = [dict(r, why='no retained surface is carried by this chain') for r, t in zip(rows, gathered) if not t]
    taken = {f for fs in faces for f in fs}
    rest = [_corners(ob, {i for name, i in taken if name == ob.name})
            for ob in bpy.context.scene.objects if ob.type == 'MESH' and ob.data.polygons]
    candidates, standing, leaving = [], [], []
    for row, tris, why, ids in loose_groups(rows, gathered, np.concatenate(rest) if rest else np.zeros((0, 3))):
        if why:
            standing.append((row, tris, why))
        else:
            row['area'] = round(float(sum(0.5 * np.linalg.norm(np.cross(t[1][1] - t[1][0], t[1][2] - t[1][0]))
                                          for t in tris)), 6)
            candidates.append((row, tris))
            leaving.append([f for i in ids for f in faces[i]])
    twins([r for r, _ in candidates])
    body, emitted = _partition(work, f, candidates, standing, materials, target_record, leaving, scale)
    return body or build / 'build.blend', emitted, empty


def twins(rows):
    """Mark the accessories that are mirror images across the fighter's middle (x = 0 at its rest), each ``twin`` the
    other's index: each is the nearest to the other's mirror image, and mirroring brings it nearer than it was (a
    left and a right feeler; not two strands on one side). physics_choice.select keeps twins together."""
    def along(row):
        c = np.asarray((row.get('chains') or [row['points']])[0], float)
        steps = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        t = np.linspace(0, steps[-1], 8)
        return np.column_stack([np.interp(t, steps, c[:, k]) for k in range(3)])
    shapes = [along(r) for r in rows]
    flip = np.array([-1., 1, 1])

    def near(a, b):
        return float(np.linalg.norm(a - b, axis=1).mean())
    best = [min((j for j in range(len(rows)) if j != i), key=lambda j: near(shapes[i] * flip, shapes[j]), default=None)
            for i in range(len(rows))]
    for i, j in enumerate(best):
        if j is not None and best[j] == i and near(shapes[i] * flip, shapes[j]) < near(shapes[i], shapes[j]):
            rows[i]['twin'] = j


def loose_groups(rows, gathered, rest):
    """Which loose chains swing, and with which. ``gathered``: each chain's triangles; ``rest``: the corners of every
    face left in the costume. Chains whose surfaces share a corner are one surface: they swing together, one accessory
    of several chains, from one bone (hang.surfaces). A chain is hair or cloth (loose.json: labels.chains named it),
    and hangs because of what it is: it swings from where its surface last meets the rest of the costume (its chain is
    cut there, and the corners it shares with the rest stay on the bone it hangs from, so nothing parts), unless its
    surface can't (hang.surfaces' why: it spans bones, it is joined to the end, it loops). Returns [(row, triangles
    with each corner's held flag, why it stays in the body or None, its chains)]."""
    out = []
    for s in hang.surfaces([r['points'] for r in rows], [r['anchor'] for r in rows],
                           [[t[1] for t in g] for g in gathered], rest):
        ids, flags = s['ids'], iter(s['held'])
        tris = [(*t[:5], next(flags)) for i in ids for t in gathered[i]]
        chains = [np.asarray(rows[i]['points'], float) for i in ids]
        row = dict(rows[ids[0]], source_bones=[b for i in ids for b in rows[i]['source_bones']],
                   held_along=[round(at / (hang._length(c) or 1.), 3) for c, at in zip(chains, s['cut'])],
                   propped=[rows[i].get('propped') for i in ids])
        swing = [hang.cut_from(c, at) for c, at in zip(chains, s['cut'])]
        why = s['why']
        if not why:
            row.update(points=swing[0].round(6).tolist(), chains=[c.round(6).tolist() for c in swing])
        out.append((row, tris, why, ids))
    return out


def _corners(ob, without=()):
    """The positions of the corners of ``ob``'s faces, but for the faces ``without`` names."""
    me = ob.data
    used = np.array([v for p in me.polygons if p.index not in without for v in p.vertices], int)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    mw = np.asarray(ob.matrix_world)
    return co.reshape(-1, 3)[np.unique(used)] @ mw[:3, :3].T + mw[:3, 3]


def _take(build, n, attribute, pick):
    """Open build.blend, triangulate its meshes and read every triangle ``pick`` (its face's ``attribute`` value, its
    corners) gives one of ``n`` chains, as (chain, piece or None). Returns each chain's triangles (material key,
    corners, normals, UVs, each corner's {bone: weight}), pieces and faces ((object name, face): _remove's), and the
    materials by key; the scene stays open."""
    import bpy
    import bmesh
    from ..prep import detail_objects
    bpy.ops.wm.open_mainfile(filepath=str(build / 'build.blend'))
    gathered, pieces, faces, materials = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], {}
    details = set(detail_objects(build))
    for ob in list(bpy.context.scene.objects):
        if ob.type != 'MESH' or not ob.data.polygons or ob.name in details:
            continue
        me = ob.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(me)
        bm.free()
        attr = me.attributes.get(attribute)
        if attr is None:
            continue  # details generated after fit, such as eyelids
        me.calc_loop_triangles()
        mw = np.asarray(ob.matrix_world)
        names = {g.index: g.name for g in ob.vertex_groups}
        for tri in me.loop_triangles:
            pts = np.array([me.vertices[i].co[:] for i in tri.vertices]) @ mw[:3, :3].T + mw[:3, 3]
            took = pick(attr.data[tri.polygon_index].value, pts)
            if took is None:
                continue
            index, piece = took
            normals = np.array([me.corner_normals[i].vector[:] for i in tri.loops]) @ np.linalg.inv(mw[:3, :3])
            uv = [me.uv_layers.active.data[i].uv[:] for i in tri.loops] if me.uv_layers.active else [(0, 0)] * 3
            mat = ob.material_slots[tri.material_index].material
            skin = [{names[g.group]: g.weight for g in me.vertices[i].groups if g.weight > 0} for i in tri.vertices]
            key = next((k for k, m in materials.items() if m == mat), f'material_{len(materials)}')
            materials[key] = mat
            gathered[index].append((key, pts, normals, uv, skin))
            if piece is not None and piece not in pieces[index]:
                pieces[index].append(piece)
            faces[index].append((ob.name, tri.polygon_index))
    return gathered, pieces, materials, faces


def _remove(faces):
    """Delete ``faces`` ((object name, face index), as _take read them) from the open scene's meshes; a mesh left with
    no face goes too."""
    import bpy
    import bmesh
    by = {}
    for name, i in faces:
        by.setdefault(name, set()).add(i)
    for name, remove in by.items():
        ob = bpy.data.objects[name]
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in remove], context='FACES')
        bm.to_mesh(ob.data)
        bm.free()
        if not ob.data.polygons:
            bpy.data.objects.remove(ob, do_unlink=True)


def _partition(work, f, candidates, standing, materials, target_record, leaving, scale=1.):
    """The chains that swing, within the game's room (physics_choice.select), as accessories; the rest (``standing``:
    (row, triangles, why)) in the body as the build skinned them. Saves the body as ``work``/body.blend and returns
    it. ``leaving``: per candidate, its faces in the open scene, taken out only when it swings (the body is None when
    none does).
    ``scale``: the fighter's model scale (write_accessory)."""
    import bpy
    from ..prep.textures import _base_color_image
    from .physics_choice import select
    choice = select([r for r, _ in candidates], target_record)
    work.joinpath('physics').mkdir(exist_ok=True)
    _json(work / 'physics/selection.json', choice)
    world = dat_joints(work / 'vanilla-files' / f'Pl{f.code}Nr.dat', f.symbol)
    emitted, rigid, out = [], [], []
    for row, tris, why in standing:
        rigid.append(dict(row, why=why, triangles=len(tris)))
    for i, (row, tris) in enumerate(candidates):
        if i not in choice['selected']:
            rigid.append(dict(row, why='not chosen within the game dynamics limit', triangles=len(tris)))
            continue
        out += leaving[i]
        folder = work / 'physics' / f'chain-{len(emitted)}'
        smd, _ = write_accessory(row, tris, world[row['anchor']], folder, scale)
        tex = {}
        for key in dict.fromkeys(t[0] for t in tris):
            mat = materials[key]
            img = _base_color_image(mat.node_tree) if mat and mat.node_tree else None
            path = folder / f'{key}.png'
            if img:
                img.save(filepath=str(path))
            else:
                from PIL import Image
                rgba = tuple(round(float(v) * 255) for v in mat.diffuse_color) if mat else (180, 180, 180, 255)
                Image.new('RGBA', (4, 4), rgba).save(path)
            tex[key] = path.name
        _json(str(smd) + '.textures.json', tex)
        row['two_sided'] = [not materials[key].use_backface_culling if materials[key] else False for key in tex]
        emitted.append(row)
    choice['rigid_chains'] = rigid
    _json(work / 'physics/selection.json', choice)
    _remove(out)
    body = work / 'body.blend' if out else None    # None: the build is the body, untouched
    if body:
        bpy.ops.wm.save_as_mainfile(filepath=str(body))
    bpy.ops.wm.read_homefile(use_empty=True)
    return body, emitted


def emit(out, rows, empty, work):
    """Author independent donors on an empty archive, graft, and verify the serialized definitions."""
    report = dict(required_runtime='m-ex', accessories=rows, empty_chains=empty, count=len(rows))
    selection = work / 'physics/selection.json'
    if selection.is_file():
        report['selection'] = json.loads(selection.read_text(encoding='utf-8'))
    if not rows:
        if selection.is_file():
            _json(work / 'physics/record.json', report)
        return report
    blank = work / 'physics/empty.dat'
    blank.write_bytes(struct.pack('>5I', 32, 0, 0, 0, 0) + bytes(12))
    donors = []
    for i, row in enumerate(rows):
        donor = work / 'physics' / f'native-{i}.dat'
        hsdraw('--accessory', blank, row['smd'], int(row['anchor'].split('_')[1]), row['dynamics_file'], donor)
        if not donor.is_file():
            raise Failed(f'native importer wrote no accessory {i}')
        d = Dat.read(donor)
        root = definitions(donor)[0]['root_joint']
        objects = []
        def walk(j):
            for joint in d.chain(j, 12):
                objects.extend(joint_dobjs(d, joint))
                walk(d.ptr(joint + 8))
        walk(root)
        row['sidedness'] = write_culling(d, objects, row['two_sided'])
        d.save(donor)
        donors.append(donor)
    merged = work / 'physics/merged.dat'
    existing = merge(out, donors, merged)
    read = definitions(merged)[existing:]
    if len(read) != len(rows):
        raise Failed('accessory count changed during graft')
    for row, actual in zip(rows, read):
        expected = row['dynamics']
        if actual['attach_bone'] != int(row['anchor'].split('_')[1]) or len(actual['dynamics']) != len(expected):
            raise Failed('accessory attachment or descriptor count changed during graft')
        if any(got['bone'] != want['bone'] or not all(np.allclose(got[k], want[k], rtol=1e-6, atol=1e-7)
                                                      for k in ('PARAM1', 'PARAM2', 'PARAM3', 'joints'))
               for got, want in zip(actual['dynamics'], expected)):
            raise Failed('accessory dynamics changed during graft')
        row['readback'] = actual
        row['dat_path'] = f"mexCostume/Accessories/Array_{actual['index']}/RootJoint"
    report.update(existing_accessories=existing, bytes_added=merged.stat().st_size - out.stat().st_size,
                  definitions_checked=True)
    out.write_bytes(merged.read_bytes())
    _json(work / 'physics/record.json', report)
    return report
