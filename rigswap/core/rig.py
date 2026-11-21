"""The model as its author rigged it: the rest skeleton and the skin, in world space, read from load's .blend.

A rigged download carries its own answer to "how does this mesh deform": bones at rest and a weight per vertex per
bone. A mesh parented to one bone rides that bone whole. Nothing here guesses; it only reads.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Formats that keep a rig best first: on the same model in several files, the first of these wins.
PREFERRED = ('.glb', '.gltf', '.fbx', '.dae', '.obj')
LAYERS = '_uvsplit'         # a 3DS rip's model with each UV layer a mesh of its own (pick)


@dataclass
class Rig:
    blend: str                      # load's .blend this was read from
    names: list                     # bone names, index = bone number
    parents: np.ndarray             # parent index per bone, -1 for a root
    heads: np.ndarray               # (B, 3) world rest heads
    tails: np.ndarray               # (B, 3) world rest tails (made up by importers for leaf bones)
    meshes: list = field(default_factory=list)   # mesh object names, in the order their vertices are stacked
    points: np.ndarray = None       # (V, 3) world rest positions of every mesh vertex
    weights: np.ndarray = None      # (V, B) skin weights, rows summing to 1 (0 rows: unskinned vertices)
    edges: np.ndarray = None        # (E, 2) the meshes' edges, as rows of points
    mirror: np.ndarray = None       # the axis it mirrors across when its shape says it apart from the skeleton (a
                                    # one-piece model's, onepiece.py); else read from the skeleton (body.py)
    twins: dict = None              # bone -> twin stated by the skeleton's builder; None means unstated
    counts: list = None             # vertices per mesh, in the order of meshes (None: not read from a scene)

    def children(self):
        kids = [[] for _ in self.names]
        for b, p in enumerate(self.parents):
            if p >= 0:
                kids[p].append(b)
        return kids

    def carried(self):
        """Weight each bone carries in all (its column sum)."""
        return self.weights.sum(axis=0)


def objects(rig):
    """The open scene's mesh objects ``rig``'s vertices come from."""
    import bpy
    return [bpy.data.objects[n] for n in rig.meshes]


def pick(load_dir):
    """The model file to use among the rigged ones: one with a whole skeleton (at least half the most bones any file
    has). Among those, the body's variants (LODs, bakes, posed copies) extend the base's stem (Rosetta:
    Rosetta_bake, RosettaLow, RosettaReading); then the most triangles, then the preferred format (the same model in
    several files). A 3DS rip comes as pc03_m4 and pc03_m4_uvsplit: the first keeps its first UV layer alone, with
    that layer's texture (an eye decal over the whole body: drawn, a see-through mannequin); the second has every
    layer as a mesh of its own with its texture, the body's colours among them. They are one model, not a variant of
    it, and the split one, with more triangles, is it as drawn. Returns its models.json entry."""
    entries = json.loads((Path(load_dir) / 'models.json').read_text(encoding='utf-8'))
    rigged = [e for e in entries if e.get('armatures') and e.get('skinned')]
    if not rigged:
        return None
    stems = [Path(e['file'].replace('\\', '/')).stem.lower().removesuffix(LAYERS) for e in rigged]

    def rank(i):
        e, ext = rigged[i], Path(rigged[i]['file']).suffix.lower()
        extended = sum(1 for j, s in enumerate(stems) if j != i and s != stems[i] and s.startswith(stems[i]))
        return (-extended, -(e.get('triangles') or 0), PREFERRED.index(ext) if ext in PREFERRED else len(PREFERRED))
    # a body carries a whole skeleton: a prop, a vehicle or an effect has far fewer bones (a grapple beam is one long
    # chain, so it can have many bones but almost no mesh: the triangles decide among skeletons)
    most = max(e.get('bones') or 0 for e in rigged)
    bodies = [i for i, e in enumerate(rigged) if (e.get('bones') or 0) >= most / 2]
    return rigged[min(bodies, key=rank)]


def read(blend):
    """Open ``blend`` (load's output) and read its one armature's rest skeleton and its meshes' skin. The scene is
    left open for the caller (the meshes are what gets carried over)."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    _rest_as_shown()
    return scene(blend)


TORN = 2.0              # a rest whose surface is this many times as long as the file shows it is torn apart


def _rest_as_shown():
    """Make the form the file shows the open scene's rest, when its rest is torn apart. A glTF's rest is its bind pose
    (the inverse bind matrices); a file can bind parts far from where it shows them (Tung Tung Sahur's body is bound
    700 units from its limbs, and whole only as posed). Posing moves a skin without stretching its surface much (on
    the core loop's 76 rigs, 2026-10-26, the rest's edges are 0.999 to 1.18 times as long as posed), so a rest whose
    edges are TORN times as long as posed or more is no form of the model: each mesh takes its posed shape and the
    armature its pose as the rest, and every later reading (the pictures, the skin, the eyes) sees the model whole."""
    import bpy
    arms = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE']
    if not arms:
        return
    arm = max(arms, key=lambda ob: len(ob.data.bones))
    if all(np.allclose(np.array(pb.matrix_basis), np.eye(4), atol=1e-6) for pb in arm.pose.bones):
        return
    bound = [ob for ob in bpy.data.objects if ob.type == 'MESH' and len(ob.data.edges)
             and not (ob.hide_viewport or ob.hide_render)
             and any(m.type == 'ARMATURE' and m.object is arm for m in ob.modifiers)]
    if not bound:
        return
    # the armature alone deforms while the two shapes are read
    shown = {(ob.name, m.name): m.show_viewport for ob in bound for m in ob.modifiers}
    for ob in bound:
        for m in ob.modifiers:
            m.show_viewport = m.type == 'ARMATURE' and m.object is arm
    position, shapes = arm.data.pose_position, {}
    for state in ('REST', 'POSE'):
        arm.data.pose_position = state
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        for ob in bound:
            ev = ob.evaluated_get(graph)
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get('co', co)
            shapes[state, ob.name] = co.reshape(-1, 3)
            ev.to_mesh_clear()
    arm.data.pose_position = position
    for ob in bound:
        for m in ob.modifiers:
            m.show_viewport = shown[ob.name, m.name]

    def surface(state):
        total = 0.0
        for ob in bound:
            e = np.empty(len(ob.data.edges) * 2, dtype=np.int64)
            ob.data.edges.foreach_get('vertices', e)
            e, co = e.reshape(-1, 2), shapes[state, ob.name]
            if len(co) == len(ob.data.vertices):
                total += np.linalg.norm(co[e[:, 0]] - co[e[:, 1]], axis=1).sum()
        return total
    if surface('REST') < TORN * surface('POSE'):
        return
    for ob in bound:
        rest, posed = shapes['REST', ob.name], shapes['POSE', ob.name]
        if len(rest) != len(ob.data.vertices) or len(posed) != len(rest):
            continue
        moved = posed - rest                    # the pose's move of each vertex, its shape keys moved with it
        if ob.data.users > 1:
            ob.data = ob.data.copy()
        keys = ob.data.shape_keys.key_blocks if ob.data.shape_keys else []
        for coords in [ob.data.vertices] + [kb.data for kb in keys]:
            co = np.empty(len(rest) * 3)
            coords.foreach_get('co', co)
            coords.foreach_set('co', (co.reshape(-1, 3) + moved).ravel())
        ob.data.update()
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def scene(blend):
    """The open scene read as ``read`` reads it: again, after meshes left it (variants.drop)."""
    import bpy
    arms = [ob for ob in bpy.data.objects if ob.type == 'ARMATURE']
    if not arms:
        raise ValueError(f'{Path(blend).name}: no armature')
    arm = max(arms, key=lambda ob: len(ob.data.bones))
    return extract(str(blend), arm, [ob for ob in bpy.data.objects if ob.type == 'MESH'])


def import_fighter(glb):
    """Import a fighter's .glb into the open scene and read it like a model: (its Rig, its armature object, its own
    mesh objects)."""
    import bpy
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb), disable_bone_shape=True)
    new = [ob for ob in bpy.data.objects if ob not in before]
    arm = next(ob for ob in new if ob.type == 'ARMATURE')
    arm.data.pose_position = 'REST'
    meshes = [ob for ob in new if ob.type == 'MESH']
    return extract(str(glb), arm, meshes), arm, meshes


def extract(source, arm, mesh_objects):
    """``arm``'s rest skeleton in world space, and the skin of ``mesh_objects``."""
    bones = list(arm.data.bones)
    index = {b.name: i for i, b in enumerate(bones)}
    world = np.array(arm.matrix_world)
    heads = _apply(world, np.array([b.head_local[:] for b in bones]))
    tails = _apply(world, np.array([b.tail_local[:] for b in bones]))
    parents = np.array([index[b.parent.name] if b.parent else -1 for b in bones])
    rig = Rig(source, [b.name for b in bones], parents, heads, tails)
    # Builder statements are bone names in the armature's own coordinates, so
    # scene rereads, bone ordering and object transforms keep their meaning.
    if 'rigswap_twins' in arm:
        rig.twins = {index[a]: index[b] for a, b in json.loads(arm['rigswap_twins']).items()
                     if a in index and b in index}
    if 'rigswap_mirror' in arm:
        axis = world[:3, :3] @ np.asarray(arm['rigswap_mirror'], float)
        rig.mirror = axis / np.linalg.norm(axis)
    deform = {b.name for b in bones if b.use_deform}
    points, rows, names, edges, base = [], [], [], [], 0
    for ob in mesh_objects:
        # a mesh the file hides is not the form shown in play
        if not len(ob.data.polygons) or ob.hide_viewport or ob.hide_render:
            continue
        n = len(ob.data.vertices)
        co = np.empty(n * 3)
        ob.data.vertices.foreach_get('co', co)
        co = co.reshape(n, 3)
        keys = ob.data.shape_keys
        if keys and keys.reference_key:                 # the shape the file shows: its keys' mix at rest
            basis = co.copy()
            for kb in keys.key_blocks:
                if kb is keys.reference_key or kb.mute or kb.value == 0:
                    continue
                k = np.empty(n * 3)
                kb.data.foreach_get('co', k)
                co += kb.value * (k.reshape(n, 3) - basis)
        points.append(_apply(np.array(ob.matrix_world), co))
        e = np.empty(2 * len(ob.data.edges), dtype=np.int64)
        ob.data.edges.foreach_get('vertices', e)
        edges.append(e.reshape(-1, 2) + base)
        base += n
        w = np.zeros((n, len(bones)))
        binds = (ob.parent is arm and ob.parent_type == 'ARMATURE') or any(
            m.type == 'ARMATURE' and m.object is arm and m.use_vertex_groups for m in ob.modifiers)
        if ob.parent is arm and ob.parent_type == 'BONE' and ob.parent_bone in index:
            w[:, index[ob.parent_bone]] = 1.0
        elif not binds and not any(m.type == 'ARMATURE' and m.object for m in ob.modifiers) and not (
                ob.parent is not None and ob.parent.type == 'ARMATURE'):
            # a mesh bound to nothing (Pink's staff) rides the root, as a part's does (assemble._bring)
            w[:, int(np.flatnonzero(parents < 0)[0])] = 1.0
        elif binds:
            # a vertex group skins only through a deforming bone of the armature that binds the mesh
            groups = {g.index: index[g.name] for g in ob.vertex_groups if g.name in index and g.name in deform}
            for v in ob.data.vertices:
                for g in v.groups:
                    if g.group in groups and g.weight > 0:
                        w[v.index, groups[g.group]] += g.weight
        s = w.sum(axis=1, keepdims=True)
        w = np.divide(w, s, out=np.zeros_like(w), where=s > 0)
        rows.append(w)
        names.append(ob.name)
    rig.meshes, rig.points, rig.weights = names, np.vstack(points), np.vstack(rows)
    rig.counts = [len(w) for w in rows]
    rig.edges = np.vstack(edges) if edges else np.zeros((0, 2), np.int64)
    return rig


def _apply(m, p):
    return p @ m[:3, :3].T + m[:3, 3]
