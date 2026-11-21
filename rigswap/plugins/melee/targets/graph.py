"""Read a target armature and its skin into a bone graph."""
import json
from math import atan2, cos, dist, pi, radians, sin
from pathlib import Path


def _numbering():
    """Every bone of every armature in the scene, numbered from 1 depth-first from each root (armatures by name, bones
    in the file's order), so a chain's bones get numbers in a row. Returns [(armature, bone)] in number order."""
    import bpy
    order = []
    for arm in sorted((o for o in bpy.context.scene.objects if o.type == 'ARMATURE'), key=lambda o: o.name):
        def visit(bone):
            order.append((arm, bone))
            for child in bone.children:
                visit(child)
        for bone in arm.data.bones:
            if bone.parent is None:
                visit(bone)
    return order


def _skins(ob, names):
    """The armatures (by name) whose bones deform mesh ``ob`` through its vertex groups."""
    out = {md.object.name for md in ob.modifiers
           if md.type == 'ARMATURE' and md.object is not None and md.object.name in names and md.use_vertex_groups}
    if ob.parent is not None and ob.parent.name in names and ob.parent_type == 'ARMATURE':
        out.add(ob.parent.name)
    return sorted(out)


def _weights(numbers):
    """For each mesh in the scene, every (vertex, bone number, weight) that moves it: through the vertex groups of the
    deforming bones of an armature that skins it, and weight 1 for every vertex of a mesh parented to one bone.
    Returns {mesh name: (how, [(vertex, n, weight)])}."""
    import bpy
    names = {arm for arm, _ in numbers}
    out = {}
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        me = ob.data
        if (ob.parent is not None and ob.parent_type == 'BONE' and (ob.parent.name, ob.parent_bone) in numbers):
            n = numbers[(ob.parent.name, ob.parent_bone)]
            out[ob.name] = ('parented', [(v, n, 1.0) for v in range(len(me.vertices))])
            continue
        triples = []
        for arm_name in _skins(ob, names):
            bones = bpy.data.objects[arm_name].data.bones
            group_n = {g.index: numbers[(arm_name, g.name)] for g in ob.vertex_groups
                       if g.name in bones and bones[g.name].use_deform}
            if not group_n:
                continue
            for v in me.vertices:
                for g in v.groups:
                    n = group_n.get(g.group)
                    if n is not None and g.weight > 0:
                        triples.append((v.index, n, g.weight))
        if triples:
            out[ob.name] = ('weights', triples)
    return out


def graph():
    """Every bone in the open scene, numbered (see _numbering), at rest, in world space. Returns (bones, weights):
    bones a list of dicts in number order; weights as _weights returns them."""
    order = _numbering()
    numbers = {(arm.name, bone.name): i for i, (arm, bone) in enumerate(order, 1)}
    weights = _weights(numbers)
    moves = {}                                   # n -> {mesh: [vertices, weight sum, how]}
    for mesh, (how, triples) in weights.items():
        for v, n, w in triples:
            entry = moves.setdefault(n, {}).setdefault(mesh, [[], 0.0, how])
            entry[0].append(v)
            entry[1] += w
    bones = []
    for n, (arm, bone) in enumerate(order, 1):
        mw = arm.matrix_world
        head, tail = mw @ bone.head_local, mw @ bone.tail_local
        by_mesh = moves.get(n, {})
        bones.append({
            'n': n, 'name': bone.name, 'armature': arm.name,
            'parent': numbers[(arm.name, bone.parent.name)] if bone.parent else None,
            'children': [numbers[(arm.name, c.name)] for c in bone.children],
            'head': [round(v, 6) for v in head], 'tail': [round(v, 6) for v in tail],
            'length': round((tail - head).length, 6),
            'deform': bool(bone.use_deform),
            'vertices': sum(len(m[0]) for m in by_mesh.values()),
            'weight': round(sum(m[1] for m in by_mesh.values()), 4),
            'moves': [{'mesh': mesh, 'vertices': m[0], 'weight': round(m[1], 4), 'how': m[2]}
                      for mesh, m in sorted(by_mesh.items())],
        })
    return bones, weights


def write_graph(path, bones, **about):
    """``bones.json``: what ``about`` says (up, front, frame...) and the bones, one per line so it reads and diffs."""
    head = json.dumps(about, indent=1)[:-2]       # the object without its closing brace
    lines = ',\n'.join('  ' + json.dumps(b) for b in bones)
    Path(path).write_text(f'{head},\n "bones": [\n{lines}\n ]\n}}\n', encoding='utf-8')
