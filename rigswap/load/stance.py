"""load.stance: a file carrying its own idle says how the character stands and what it shows. Its bind pose only says
how it was rigged. So the idle's pose becomes the rest, and a surface the idle collapses goes (settle). A file with no
idle is left exactly as it is."""
from collections import defaultdict
import math
import re

import numpy as np

ROUNDOFF = float(np.finfo(np.float32).eps)      # how finely a float32 holds a number, relative to its size


def _bindings():
    """{action: [(owner, slot)]}: the IDs an imported action's NLA strips bind. An idle moves more than the armature:
    shape keys and shader inputs too."""
    import bpy
    found = defaultdict(list)
    for owner in (*bpy.data.objects, *bpy.data.shape_keys, *(m.node_tree for m in bpy.data.materials if m.node_tree)):
        ad = owner.animation_data
        if ad is None:
            continue
        if ad.action:
            found[ad.action].append((owner, ad.action_slot))
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.action and (owner, strip.action_slot) not in found[strip.action]:
                    found[strip.action].append((owner, strip.action_slot))
    return found


def _joints(arm):
    """{bone name: (place, turn, lengths)} at the current frame, in armature space.

    A joint is a place, a turn and a length along each of its own axes. It has no shear: the game keeps a parent's scale
    on the same-named axis of its child, whichever way the child turns (HSD_JObjMakeMatrix). A TRS file can't say that.
    Its matrix product shears a turned child of an uneven parent, and a scale channel can't hold a shear: all it holds
    is how long each of the child's axes measures in the parent's axes. So a child's length is that measure, over what
    one unit along the same axis measures in the parent. With an even parent, or axes that line up, this is the product.
    Where a parent is folded flat, as the game hides a joint, the channels read 0.0006 then 600: the product throws the
    child's surface out by hundreds of times the body, and this leaves it flat. An axis reaching a parent with no length
    has none either.
    """
    joints = {}
    for pb in sorted(arm.pose.bones, key=lambda b: len(b.parent_recursive)):
        offset = np.array(pb.bone.matrix_local)
        place, turn, lengths = np.zeros(3), np.eye(3), np.ones(3)
        if pb.parent:
            place, turn, lengths = joints[pb.parent.name]
            offset = np.linalg.inv(np.array(pb.parent.bone.matrix_local)) @ offset
        rotation = pb.rotation_quaternion.normalized() if pb.rotation_mode == 'QUATERNION' else pb.rotation_euler
        axes = offset[:3, :3] @ np.array(rotation.to_matrix())
        with np.errstate(divide='ignore', invalid='ignore'):
            unit = np.linalg.norm(np.where(axes == 0, 0, axes / lengths[:, None]), axis=0)
            own = np.array(pb.scale) / unit
        joints[pb.name] = (place + turn @ (lengths * (offset[:3, 3] + offset[:3, :3] @ np.array(pb.location))),
                           turn @ axes, own)
    return joints


def _stand(arms, frame):
    """Hold the rig at a frame of the idle, its joints as _joints reads them. -> [_joints of each armature].

    Blender's Aligned scale inheritance is the game's rule, so a bone's scale is its lengths over its parent's. What
    the idle doesn't key sits at rest, so nothing written here carries over to the next frame.
    """
    import bpy
    for arm in arms:
        for pb in arm.pose.bones:
            pb.scale = (1, 1, 1)
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    held = [_joints(arm) for arm in arms]
    for arm, joints in zip(arms, held):
        for pb in arm.pose.bones:
            lengths = joints[pb.name][2]
            over = joints[pb.parent.name][2] if pb.parent else np.ones(3)
            pb.scale = np.divide(lengths, over, out=lengths.copy(), where=over != 0)
    return held


def _area(ob, mesh):
    """Surface area of ``mesh`` where ``ob`` stands in the world."""
    mesh.calc_loop_triangles()
    points = np.array([ob.matrix_world @ v.co for v in mesh.vertices]).reshape(-1, 3)
    tris = np.array([t.vertices[:] for t in mesh.loop_triangles], int).reshape(-1, 3)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2


def _posed_area(ob, graph):
    ev = ob.evaluated_get(graph)
    area = _area(ev, ev.to_mesh())
    ev.to_mesh_clear()
    return area


def _shared_skin(meshes):
    """Copies of a surface point move as one. A material split repeats a point, a rounding away from its copy, and a
    rigid skin can hang each copy on a joint of its own, so posing would tear the surface there. Each copy takes the
    copies' mean skin before the pose. A copy is a float32 rounding away (a coordinate through a few matrix products),
    never a weld radius. Normals and UV corners stay apart, and meshes on different armatures share nothing."""
    from mathutils.kdtree import KDTree
    rigs = defaultdict(list)
    for ob in meshes:
        arm = ob.find_armature()
        if arm is None:
            continue
        groups = {g.index: g.name for g in ob.vertex_groups if g.name in arm.data.bones}
        for v in ob.data.vertices:
            skin = {groups[g.group]: g.weight for g in v.groups if g.group in groups and g.weight > 0}
            if skin:
                rigs[arm].append((ob, v.index, ob.matrix_world @ v.co, skin))
    for rows in rigs.values():
        noise = 8 * ROUNDOFF * max(max(abs(c) for c in row[2]) for row in rows)
        tree = KDTree(len(rows))
        for i, row in enumerate(rows):
            tree.insert(row[2], i)
        tree.balance()
        done = set()
        for i, row in enumerate(rows):
            if i in done:
                continue
            copies = [j for _, j, _ in tree.find_range(row[2], noise) if j not in done]
            done.update(copies)
            if all(rows[j][3] == row[3] for j in copies):
                continue
            mean = defaultdict(float)
            for j in copies:
                for name, weight in rows[j][3].items():
                    mean[name] += weight / len(copies)
            for j in copies:
                ob, vertex, _, skin = rows[j]
                for name in skin:
                    ob.vertex_groups[name].remove([vertex])
                for name, weight in mean.items():
                    (ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)).add([vertex], weight, 'REPLACE')
    for ob in meshes:
        ob.data.update()


def _bake(meshes, arms, held, owners):
    """The held pose becomes the mesh, custom normals included, and each joint's place and turn become its rest. A rest
    has no scale, so a joint at no length keeps its bone at its own length (Blender deletes a bone of none)."""
    import bpy
    from mathutils import Matrix
    graph = bpy.context.evaluated_depsgraph_get()
    baked = [(ob, ob.evaluated_get(graph).matrix_world.copy(),
              bpy.data.meshes.new_from_object(ob.evaluated_get(graph), preserve_all_data_layers=True,
                                              depsgraph=graph)) for ob in meshes]
    for owner in owners:                 # the idle's objects, shape keys and shader inputs would pose the bake again
        owner.animation_data_clear()
    for ob, _, mesh in baked:
        ob.data = mesh
        for modifier in list(ob.modifiers):
            if modifier.type != 'ARMATURE':
                ob.modifiers.remove(modifier)
    for arm, joints in zip(arms, held):
        bpy.ops.object.select_all(action='DESELECT')
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode='EDIT')
        for bone in arm.data.edit_bones:
            place, turn, _ = joints[bone.name]
            matrix = Matrix(turn.tolist()).to_4x4()
            matrix.translation = place
            bone.use_connect = False
            bone.matrix = matrix
        bpy.ops.object.mode_set(mode='OBJECT')
        for pb in arm.pose.bones:
            pb.matrix_basis.identity()
        for bone in arm.data.bones:
            bone.inherit_scale = 'FULL'
    bpy.context.view_layer.update()
    for ob, world, _ in baked:
        ob.matrix_world = world
    bpy.context.view_layer.update()


def settle():
    """Stand the model in the idle the file names. -> {'action', 'frame', 'hidden'}, or None if it names none, or more
    than one (then the file doesn't say which look is its own).

    The stance is the idle's own middle, the frame nearest the mean of its joints' places, so a sway or a breath isn't
    baked in. What shows is what the idle shows: a surface goes when, through every frame, its area is below what a
    float32 can tell from none, against its area at rest. One that appears for a moment (a blink) stays.
    """
    import bpy
    bindings = _bindings()
    idles = [a for a in bindings if re.search(r'(?<![a-z])idle(?![a-z])', a.name, re.I)]
    arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(idles) != 1 or not arms:
        return None
    action = idles[0]
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    area = {ob: _area(ob, ob.data) for ob in meshes}
    owners = {owner for bound in bindings.values() for owner, _ in bound}
    for owner in owners:
        for track in owner.animation_data.nla_tracks:
            track.mute = True
        owner.animation_data.action = None
    for owner, slot in bindings[action]:
        owner.animation_data.action = action
        if slot:
            owner.animation_data.action_slot = slot
    for arm in arms:
        for bone in arm.data.bones:
            bone.inherit_scale = 'ALIGNED'
    frames = np.arange(action.frame_range[0], action.frame_range[1] + 1).tolist()
    places = np.array([[j[0] for joints in _stand(arms, f) for j in joints.values()] for f in frames])
    frame = frames[int(np.argmin(((places - places.mean(axis=0)) ** 2).sum(axis=(1, 2))))]
    hidden = [ob for ob in meshes if area[ob] > 0]
    for at in (frame, *frames):
        if not hidden:
            break
        _stand(arms, at)
        graph = bpy.context.evaluated_depsgraph_get()
        hidden = [ob for ob in hidden if _posed_area(ob, graph) <= area[ob] * ROUNDOFF]
    record = {'action': action.name, 'frame': frame, 'hidden': [ob.name for ob in hidden]}
    for ob in hidden:
        meshes.remove(ob)
        bpy.data.objects.remove(ob, do_unlink=True)
    _shared_skin(meshes)
    _bake(meshes, arms, _stand(arms, frame), owners)
    return record
