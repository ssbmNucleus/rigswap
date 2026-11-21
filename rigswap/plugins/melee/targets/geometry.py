"""Measure a target's mesh, toes and weapon boxes."""
from pathlib import Path
import numpy as np
from ... import Failed
from ....library.body_types import FRONT, of, part_ends
from ....vocabulary import BODY_TYPES
from .skeleton import part_starts


MOVES_MOST = 0.5        # the head bones carry a vertex when they have more than this share of its weight


def _above(body_type, part):
    """The parts above ``part`` in its body type's chains, nearest first."""
    out, p = [], part
    while True:
        parent = next(((ps[ps.index(p) - 1] if ps.index(p) else up) for up, ps in of(body_type).chains if p in ps),
                      None)
        if parent is None or parent in out:
            return out
        out.append(parent)
        p = parent


FOOT_FLOOR = 0.1        # a foot whose mesh reaches less than this share of the model's shin ahead of its ankle has


def foot_fronts(bones, body_type, points, weights, skip=None):
    """For a side with bones (a model with no bones: its temporary armature's): {foot: {'tip': point, 'name': the
    foot's end joint (a biped's toe), 'from': how it was found}} for the feet (the body type's) whose mesh reaches in
    front of the ankle: the far end of the foot's mesh on the ground, the vertex farthest from the ankle (the head of
    the foot's first bone) across the ground (the middle of those equally far: a square toe's corners), as a hand's
    fingertip is the far end of its mesh beyond the wrist: the toe of a foot turned out in a stance as of one that
    points ahead, never a heel the ankle stands over. The foot's mesh is what the foot's bones and the bones under
    them that are no part of the body (a toe answered none) carry, more than MOVES_MOST of each vertex's weight. A
    foot whose mesh reaches less than FOOT_FLOOR of the model's shin (the longest part its feet hang from, from its
    start to the ankle) ahead of the ankle, the way the model faces (FRONT: both sides face -y), has no toe: there is
    no foot there to measure (no_toes says why). Arguments as hand_meshes'."""
    return {foot: m['toe'] for foot, m in _foot_reaches(bones, body_type, points, weights, skip).items()
            if m['toe'] is not None}


def _foot_reaches(bones, body_type, points, weights, skip=None):
    """Each foot's carried vertex indices, toe (or None), and why no joint-based toe was found."""
    starts = part_starts(bones, body_type)
    by_name = {b['name']: b for b in bones}
    body = set(BODY_TYPES[body_type])
    pts = np.asarray(points if points is not None else np.zeros((0, 3)), float).reshape(-1, 3)
    w = np.asarray(weights if weights is not None else np.zeros((0, len(bones))), float).reshape(-1, len(bones))
    total = w.sum(1)
    ahead = np.asarray(FRONT, float)
    feet = [f for f in of(body_type).feet if f in starts]

    def hangs_from(foot):
        """The length of the part a foot hangs from: from the start of the nearest part above it with one."""
        up = next((p for p in _above(body_type, foot) if p in starts), None)
        return 0.0 if up is None else float(np.linalg.norm(np.subtract(starts[foot]['start'], starts[up]['start'])))
    shin = max((hangs_from(f) for f in feet), default=0.0)
    out = {}
    for foot in feet:
        if not len(pts):
            continue
        mine = set(starts[foot]['bones'])

        def under(b):
            while b['parent'] is not None and b['parent'] in by_name:
                b = by_name[b['parent']]
                if b['name'] in mine:
                    return True
                if body & set(b.get('parts') or [b['role']]):
                    return False
            return False
        toes = [b['name'] for b in bones if b['name'] not in mine and not body & set(b.get('parts') or [b['role']])
                and under(b)]
        cols = [i for i, b in enumerate(bones) if b['name'] in mine or b['name'] in toes]
        carried = w[:, cols].sum(1) / np.where(total > 0, total, 1.0) > MOVES_MOST
        if skip is not None:
            carried &= ~np.asarray(skip, bool)
        idx = np.flatnonzero(carried)
        if not len(idx):
            continue
        ankle = np.asarray(starts[foot]['start'], float)
        reach = (pts[idx] - ankle) @ ahead
        k = int(np.argmax(reach))
        mesh = (f"of the {len(idx)} vertices the foot's {len(mine) + len(toes)} bones carry, the one farthest ahead "
                f"of the ankle (the head of {starts[foot]['bone']}) the way the model faces, {float(reach[k]):.4g} "
                f"ahead")
        if reach[k] <= 0 or reach[k] < FOOT_FLOOR * shin:
            out[foot] = {'vertices': idx, 'toe': None,
                         'why': (f"no foot to measure: {mesh}, under {FOOT_FLOOR:g} of the model's shin "
                                              f"({shin:.4g}, the longest part its feet hang from)")}
            continue
        # the far end across the ground; vertices equally far (a square toe's two corners) meet at their middle
        across = np.linalg.norm((pts[idx] - ankle)[:, :2], axis=1)
        far = across >= across.max() * (1 - 1e-6)
        out[foot] = {'vertices': idx, 'toe': {'tip': [float(v) for v in pts[idx[far]].mean(0)],
                                              'name': part_ends(body_type)[foot][1],
                             'from': (f"the far end of the foot's mesh on the ground: of the {len(idx)} vertices the "
                                      f"foot's {len(mine) + len(toes)} bones carry, the one farthest from the ankle "
                                      f"(the head of {starts[foot]['bone']}) across the ground, "
                                      f"{float(across.max()):.4g} from it")}, 'why': None}
    return out


def _evaluated_points(objects):
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for ob in objects:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get('co', co)
            mw = np.array(ev.matrix_world)
            chunks.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        finally:
            ev.to_mesh_clear()
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def _source_weights(ob, armatures, bone_index, deform):
    """(n, bones) weights of one mesh object on the source's bones: from its vertex groups when an Armature modifier
    binds it to a source armature, or all on one bone when it is parented to that bone."""
    w = np.zeros((len(ob.data.vertices), len(bone_index)))
    if ob.parent is not None and ob.parent.name in armatures and ob.parent_type == 'BONE':
        if ob.parent_bone in bone_index:
            w[:, bone_index[ob.parent_bone]] = 1.0
        return w
    if not any(m.type == 'ARMATURE' and m.object is not None and m.object.name in armatures for m in ob.modifiers):
        return w
    col = {g.index: bone_index[g.name] for g in ob.vertex_groups if g.name in bone_index and deform.get(g.name, True)}
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in col:
                w[v.index, col[g.group]] += g.weight
    return w


def mesh_weights(meshes, arm, bones, glb):
    """(points, weights) of a target's original as the fit measures it: the vertices of ``meshes``, evaluated, (n, 3),
    and their weights (n, bones) on ``bones`` (a bone graph's list, by name) through armature ``arm``."""
    each = [_evaluated_points([ob]) for ob in meshes]
    index = {b['name']: i for i, b in enumerate(bones)}
    weights = []
    for ob, pts in zip(meshes, each):
        if len(pts) != len(ob.data.vertices):
            raise Failed(f"{Path(glb).name}: {ob.name} has {len(pts)} vertices evaluated and "
                         f"{len(ob.data.vertices)} in its mesh; its weights can't be read against its points")
        weights.append(_source_weights(ob, {arm.name}, index, {}))
    return (np.concatenate(each) if each else np.zeros((0, 3)),
            np.concatenate(weights) if weights else np.zeros((0, len(bones))))


def target_mesh(glb, bones):
    """A target's .glb measured as the fit measures it (mesh_weights), imported into an empty scene, which is left
    empty: (points, weights on ``bones``). For a plugin writing a target's description (its toes: foot_fronts)."""
    import bpy
    bpy.ops.wm.read_homefile(use_empty=True)
    arm, other = _import_target(glb)
    out = mesh_weights([ob for ob in other if ob.type == 'MESH' and len(ob.data.polygons)], arm, bones, glb)
    bpy.ops.wm.read_homefile(use_empty=True)
    return out


def _import_target(glb):
    """Import the target's .glb. Returns (its armature object, its other objects)."""
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb), disable_bone_shape=True)
    new = [ob for ob in bpy.data.objects if ob not in before]
    arms = [ob for ob in new if ob.type == 'ARMATURE']
    if len(arms) != 1:
        raise Failed(f'{Path(glb).name}: expected one armature, found {len(arms)}')
    return arms[0], [ob for ob in new if ob is not arms[0]]


def item_box(points):
    """The item's own box: (its middle, its axes (3 rows, the longest extent first), its extents). The axes are the
    item's principal directions."""
    pts = np.asarray(points, float).reshape(-1, 3)
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=True)
    proj = (pts - mean) @ vt.T
    lo, hi = proj.min(0), proj.max(0)
    order = np.argsort(-(hi - lo), kind='stable')
    axes, lo, hi = vt[order], lo[order], hi[order]
    return mean + ((lo + hi) / 2) @ axes, axes, hi - lo
