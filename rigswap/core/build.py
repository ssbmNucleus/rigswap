"""Hand the result to the plumbing: build.blend as export reads it (contracts.md section 2).

The fighter's armature at rest, every model mesh moved into the fighter's rest with vertex groups named for the
fighter's JOBJs and an Armature modifier on that armature, and target.json; then the plugin's prep. And loose.json:
the model's hair and cloth with no partner on the fighter (correspond.Match.loose), each riding the bone it hangs from,
for the plugin's export to make them swing where its game can; the face attribute LOOSE says which faces each one
carries.
"""
import json
from pathlib import Path

import numpy as np

from . import refine

LOOSE = 'rigswap_loose'     # INT face attribute: 1 + the face's chain in loose.json, 0 for the body


NEAR = 8                    # a loose chain's joint moves as this many of its surface's vertices nearest it do


def loose(rig, match, T, frig, w, fighter_bones):
    """The loose chains as export reads them: the model bones each is, the fighter bone it hangs from, its joints and
    far end at the fighter's rest, what it is (``kind``, hair or cloth) and whether the bind pose holds it out where it
    would not rest (``held_out``: labels.chains), and per vertex, the weight it carries (V, chains). A chain riding the
    bone it hangs
    from moves with that bone's transform, as its mesh does; one whose bones ride other bones (a strand on a leg) lies
    in a surface that blends them, so each of its points moves as the vertices of its surface nearest it do (``w``,
    the build's weights on ``fighter_bones``): where its surface meets the rest of the costume stays where it is on
    the chain."""
    rows, carried = [], np.zeros((len(rig.points), len(match.loose)))
    for i, (attach, chain, bones) in enumerate(match.loose):
        fb = match.carrier(attach, chain.joints[0])
        pts = np.vstack([chain.joints, chain.end])
        mine = np.flatnonzero(rig.weights[:, bones].sum(axis=1) > 0)
        if all(match.owner.get(b, fb) == fb for b in bones) or not len(mine):
            points = T[fb](pts)
        else:
            near = np.argsort(np.linalg.norm(pts[:, None] - rig.points[mine][None], axis=2), axis=1)[:, :NEAR]
            share = w[mine][near].mean(axis=1)
            share /= np.maximum(share.sum(axis=1, keepdims=True), 1e-12)
            points = sum(share[:, [j]] * T[f](pts) for j, f in enumerate(fighter_bones)
                         if share[:, j].any())
        rows.append(dict(bones=[rig.names[b] for b in chain.bones], anchor=frig.names[fb],
                         points=np.asarray(points).round(6).tolist(), kind=chain.kind, held_out=chain.held_out))
        carried[:, i] = rig.weights[:, bones].sum(axis=1)
    return rows, carried


def write(build_dir, rig, points, w, fighter_bones, frig, farm, fighter_meshes, fighter_name, chains=(), carried=None):
    """Write ``build_dir``/build.blend from the open scene (the model's file with the fighter imported into it).
    ``chains``, ``carried``: loose(); a face is the loose chains' when they carry most of its corners' weight, and then
    the one of them that carries the most of it. Shading is read from the moved surface (refine.shade), keeping apart
    the corners the file's own normals kept apart."""
    import bpy
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    for ob in list(fighter_meshes):
        bpy.data.objects.remove(ob, do_unlink=True)
    keep = set(rig.meshes)
    for ob in list(bpy.data.objects):
        if ob is farm:
            continue
        if ob.type != 'MESH' or ob.name not in keep:
            bpy.data.objects.remove(ob, do_unlink=True)
    names = [frig.names[b] for b in fighter_bones]
    faces = [0] * len(chains)
    start = 0
    files = []
    for name in rig.meshes:
        ob = bpy.data.objects[name]
        me = ob.data
        n = len(me.vertices)
        rows = slice(start, start + n)
        start += n
        if me.users > 1:
            ob.data = me = me.copy()
        files.append(refine.stored(ob))
        if me.shape_keys:
            ob.shape_key_clear()
        ob.parent = None
        ob.matrix_world = np.eye(4).tolist()
        ob.modifiers.clear()
        ob.vertex_groups.clear()
        me.vertices.foreach_set('co', points[rows].astype(np.float32).ravel())
        me.update()
        sub = w[rows]
        for i, gname in enumerate(names):
            idx = np.flatnonzero(sub[:, i] > 0)
            if not len(idx):
                continue
            g = ob.vertex_groups.new(name=gname)
            for v in idx:
                g.add([int(v)], float(sub[v, i]), 'REPLACE')
        if len(chains):
            on = carried[rows]
            values = []
            for poly in me.polygons:
                share = on[list(poly.vertices)].mean(axis=0)
                k = int(np.argmax(share))
                values.append(k + 1 if share.sum() >= 0.5 else 0)
                if values[-1]:
                    faces[k] += 1
            me.attributes.new(LOOSE, 'INT', 'FACE').data.foreach_set('value', values)
        mod = ob.modifiers.new('Armature', 'ARMATURE')
        mod.object = farm
        me.update()
    refine.shade([bpy.data.objects[name] for name in rig.meshes], files)
    bpy.ops.wm.save_as_mainfile(filepath=str(build_dir / 'build.blend'))
    (build_dir / 'target.json').write_text(json.dumps({'name': fighter_name}), encoding='utf-8')
    (build_dir / 'loose.json').write_text(json.dumps({'attribute': LOOSE, 'chains': [
        dict(c, faces=n) for c, n in zip(chains, faces)]}, indent=1), encoding='utf-8')
    return build_dir / 'build.blend'
