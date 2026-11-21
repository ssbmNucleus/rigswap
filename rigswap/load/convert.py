"""convert: a .dae becomes a .glb, through an installed Blender 4.x.

Blender 5 dropped the Collada importer, and rips ship DAE often, so this one format goes through a separate Blender 4
in a subprocess, which runs this file as its script (``__main__`` below). Everything else is imported by the bpy inside
our Python. Where Blender 4 lives: ``RIGSWAP_BLENDER4``, else the usual install folder.

The normals come in as the model has them, in the frame the file wrote them in (read_normals). Collada keeps a mesh's
normals in the mesh's own space, beside its positions. A rip of a rigidly skinned game model may keep each vertex's
normal in the frame of the one joint it rides, as the game does, while it writes the positions out in the mesh's space:
read as stored, those normals lie across their faces and the model shades faceted and patchy. The file's skin says
which joint each vertex rides and that joint's bind (bind_rotations), so each such normal is turned into the mesh's
space by its joint's bind. Which frame the file used is read from the file: the one in which the normals of its
rigidly skinned vertices lie nearer their own faces (normals_frame).

Every joint a skin names stays a bone (restore_joints). Blender 4's Collada importer gives some joints no length and
then deletes them as zero sized ("removed zero sized bone" in its debug output), hanging their children on their
parents: Wind Waker's Link lost his head, his right shoulder, arm and hand and his cap's joints this way, and the 804
vertices they carried came out of the glTF on a stray root at the origin. The file still says where each joint is
bound (its inverse bind matrix) and what it hangs from, so each is put back there."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

CANDIDATES = [
    os.environ.get('RIGSWAP_BLENDER4', ''),
    r'C:\Program Files\Blender Foundation\Blender 4.3\blender.exe',
    r'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe',
    r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe',
    r'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe',
]
RECORD = 'rigswap-normals: '       # the line of Blender 4's output that carries read_normals' record back
MATERIAL_RECORD = 'rigswap-materials: '
JOINTS_RECORD = 'rigswap-joints: '
LEAF = 0.02                        # a joint given back with no child elsewhere reaches this share of the skeleton


def blender4():
    for c in CANDIDATES:
        if c and Path(c).is_file():
            return c
    raise RuntimeError('no Blender 4.x found for .dae conversion; set RIGSWAP_BLENDER4')


def dae_to_glb(dae, glb):
    """Convert ``dae`` to ``glb`` in Blender 4 (the file as it reads it: repair). -> the imported normals and
    material-colour records."""
    dae, glb = Path(dae), Path(glb)
    src = repair(dae)
    r = subprocess.run([blender4(), '--background', '--python', str(Path(__file__).resolve()), '--', str(src), str(glb)],
                       capture_output=True, text=True, timeout=600)
    if not glb.is_file():
        out = (r.stdout + r.stderr).strip().splitlines()
        tail = '\n'.join([line for line in out if 'Schema validation' in line][:3] + out[-8:])
        raise RuntimeError(f'{dae.name}: Blender 4 could not convert it\n{tail}')
    facts = {'repaired': True} if src != dae else {}
    for key, prefix in (('normals', RECORD), ('material_colours', MATERIAL_RECORD),
                        ('joints_restored', JOINTS_RECORD)):
        lines = [line[len(prefix):] for line in r.stdout.splitlines() if line.startswith(prefix)]
        if lines:
            facts[key] = json.loads(lines[-1])
    return facts


def repair(dae):
    """The .dae as Blender 4 reads it: the file itself, or a copy beside it (its images where they were) with what the
    file leaves out filled in where the file itself says what it must be. Blender 4's importer refuses a whole document
    over one required attribute missing. A ``<texture>`` without its ``texcoord`` (Deltarune's Pink and Maddie, written
    so) draws with texture coordinates, and the file's meshes have one set of them: it names that one. Where they have
    several, which is meant is not in the file, and the file goes as it is (refused). -> the path to give Blender 4."""
    import re
    dae = Path(dae)
    text = dae.read_text(encoding='utf-8', errors='surrogateescape')
    bare = re.compile(r'<texture\b(?![^>]*\btexcoord=)')
    if not bare.search(text):
        return dae
    sets = set()
    for m in re.finditer(r'<input\b[^>]*\bsemantic="TEXCOORD"[^>]*>', text):
        s = re.search(r'\bset="(\d+)"', m.group(0))
        sets.add(s.group(1) if s else '0')
    if len(sets) != 1:
        return dae
    out = dae.with_name(dae.stem + '.repaired.dae')
    out.write_text(bare.sub(f'<texture texcoord="UVSET{sets.pop()}"', text), encoding='utf-8', errors='surrogateescape')
    return out



def bind_rotations(dae):
    """{geometry name: {joint name: 3x3}} for each skinned geometry of the .dae: the matrix that turns a normal written
    in the joint's own frame into the mesh's space. A point goes from the mesh's space into the joint's by the inverse
    bind matrix after the bind shape matrix (A); a normal comes back by A's linear part transposed."""
    import xml.etree.ElementTree as ET
    import numpy as np
    root = ET.parse(dae).getroot()
    ns = root.tag[:root.tag.index('}') + 1] if root.tag.startswith('{') else ''
    sources = {s.get('id'): s for s in root.iter(ns + 'source')}
    geometries = {g.get('id'): g.get('name') or g.get('id') for g in root.iter(ns + 'geometry')}
    out = {}
    for skin in root.iter(ns + 'skin'):
        joints = skin.find(ns + 'joints')
        geometry = geometries.get(skin.get('source', '').lstrip('#'))
        if joints is None or geometry is None:
            continue
        inputs = {i.get('semantic'): sources.get(i.get('source', '').lstrip('#')) for i in joints.findall(ns + 'input')}
        if inputs.get('JOINT') is None or inputs.get('INV_BIND_MATRIX') is None:
            continue
        names = next((a.text.split() for tag in ('Name_array', 'IDREF_array')
                      if (a := inputs['JOINT'].find(ns + tag)) is not None and a.text), [])
        ibm = np.array(inputs['INV_BIND_MATRIX'].find(ns + 'float_array').text.split(), float).reshape(-1, 4, 4)
        shape = skin.find(ns + 'bind_shape_matrix')
        bsm = np.array(shape.text.split(), float).reshape(4, 4) if shape is not None else np.eye(4)
        out[geometry] = {j: (m @ bsm)[:3, :3].T for j, m in zip(names, ibm)}
    return out


def normals_frame(stored, turned, faces):
    """'joints' when the normals turned out of their joints' frames lie nearer their own faces than the normals as
    stored, else 'mesh'. Rows are corners, all unit: the stored normal, the turned one and the face's normal, for the
    corners that can be turned (their vertex rides one joint). Nearer: the median cosine between a corner's normal and
    its face's is higher."""
    import numpy as np
    if not len(stored):
        return 'mesh'
    return 'joints' if np.median((turned * faces).sum(1)) > np.median((stored * faces).sum(1)) else 'mesh'


def _unit(rows):
    import numpy as np
    length = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, length, out=np.zeros_like(rows), where=length > 0)


def _corners(o, rotations):
    """A skinned mesh's corner normals as stored, and of the corners that can be turned (their vertex rides one joint
    the skin names; their face has an area): the indices, those normals turned by the joint's bind (bind_rotations),
    and their faces' normals."""
    import numpy as np
    me = o.data
    n = len(me.loops)
    stored = np.empty(n * 3)
    me.corner_normals.foreach_get('vector', stored)
    stored = _unit(stored.reshape(-1, 3))
    vertex = np.empty(n, dtype=np.int64)
    me.loops.foreach_get('vertex_index', vertex)
    face = np.empty(n, dtype=np.int64)
    for p in me.polygons:
        face[p.loop_start:p.loop_start + p.loop_total] = p.index
    face_normals = np.empty(len(me.polygons) * 3)
    me.polygons.foreach_get('normal', face_normals)
    faces = _unit(face_normals.reshape(-1, 3))[face]
    by_group = {g.index: rotations.get(g.name) for g in o.vertex_groups}
    rotation = [None] * len(me.vertices)
    for v in me.vertices:
        riding = [g.group for g in v.groups if g.weight > 0]
        if len(riding) == 1:
            rotation[v.index] = by_group.get(riding[0])
    index = np.array([i for i in range(n) if rotation[vertex[i]] is not None and faces[i].any()], dtype=np.int64)
    turned = _unit(np.array([rotation[vertex[i]] @ stored[i] for i in index]).reshape(-1, 3))
    return stored, index, turned, faces[index]


def read_normals(dae):
    """In Blender 4, once the .dae is imported: read the skinned meshes' normals in the frame the file wrote them in
    (normals_frame, over all their rigidly skinned corners), and set them so when that is the joints'.
    -> {'frame': 'mesh' | 'joints', 'meshes': {mesh: the median degrees from the faces, as stored and turned}}"""
    import bpy
    import numpy as np
    rotations = bind_rotations(dae)
    meshes = [(o, _corners(o, rotations[o.data.name])) for o in bpy.data.objects
              if o.type == 'MESH' and o.data.name in rotations and o.data.has_custom_normals]
    meshes = [(o, c) for o, c in meshes if len(c[1])]
    stored = np.concatenate([s[i] for _, (s, i, _, _) in meshes]) if meshes else np.empty((0, 3))
    turned = np.concatenate([t for _, (_, _, t, _) in meshes]) if meshes else np.empty((0, 3))
    faces = np.concatenate([f for _, (_, _, _, f) in meshes]) if meshes else np.empty((0, 3))
    frame = normals_frame(stored, turned, faces)

    def degrees(normals, faces):
        return round(float(np.median(np.degrees(np.arccos(np.clip((normals * faces).sum(1), -1, 1))))), 1)

    record = {'frame': frame, 'meshes': {}}
    for o, (normals, index, turned, faces) in meshes:
        record['meshes'][o.name] = {'stored': degrees(normals[index], faces), 'turned': degrees(turned, faces)}
        if frame == 'joints':
            normals[index] = turned
            o.data.normals_split_custom_set(normals.tolist())
            o.data.update()
    return record


def joint_binds(dae):
    """The .dae's joints: ({joint: the joint it hangs from, or None}, {joint: its 4x4 bind in the file's world, from a
    skin's inverse bind matrix}). Joints by node name (Blender's), a skin's references by sid, id or name."""
    import xml.etree.ElementTree as ET
    import numpy as np
    root = ET.parse(dae).getroot()
    ns = root.tag[:root.tag.index('}') + 1] if root.tag.startswith('{') else ''
    parents, lookup = {}, {}

    def walk(node, parent):
        joint = node.get('type') == 'JOINT'
        name = node.get('name') or node.get('id')
        if joint:
            parents.setdefault(name, parent)
            for k in (node.get('sid'), node.get('id'), node.get('name')):
                if k:
                    lookup.setdefault(k, name)
        for child in node.findall(ns + 'node'):
            walk(child, name if joint else parent)
    for scene in root.iter(ns + 'visual_scene'):
        for node in scene.findall(ns + 'node'):
            walk(node, None)
    sources = {s.get('id'): s for s in root.iter(ns + 'source')}
    binds = {}
    for skin in root.iter(ns + 'skin'):
        joints = skin.find(ns + 'joints')
        if joints is None:
            continue
        inputs = {i.get('semantic'): sources.get(i.get('source', '').lstrip('#')) for i in joints.findall(ns + 'input')}
        if inputs.get('JOINT') is None or inputs.get('INV_BIND_MATRIX') is None:
            continue
        refs = next((a.text.split() for tag in ('Name_array', 'IDREF_array')
                     if (a := inputs['JOINT'].find(ns + tag)) is not None and a.text), [])
        ibm = np.array(inputs['INV_BIND_MATRIX'].find(ns + 'float_array').text.split(), float).reshape(-1, 4, 4)
        for ref, m in zip(refs, ibm):
            name = lookup.get(ref)
            if name is not None and name not in binds and abs(np.linalg.det(m)) > 1e-12:
                binds[name] = np.linalg.inv(m)
    return parents, binds


def _similarity(p, q):
    """The turn, scale and shift taking the points ``p`` onto ``q`` (least squares, no mirror): (4x4, largest miss)."""
    import numpy as np
    mp, mq = p.mean(axis=0), q.mean(axis=0)
    a, b = p - mp, q - mq
    u, s, vt = np.linalg.svd(b.T @ a / len(p))
    d = np.diag([1.0, 1.0, -1.0 if np.linalg.det(u @ vt) < 0 else 1.0])
    rot = u @ d @ vt
    var = (a ** 2).sum() / len(p)
    k = float((s * np.diag(d)).sum() / var) if var > 0 else 1.0
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = k * rot, mq - k * rot @ mp
    return m, float(np.linalg.norm(p @ m[:3, :3].T + m[:3, 3] - q, axis=1).max())


def restore_joints(dae):
    """In Blender 4, once the .dae is imported: a bone again for every joint a skin of the file names that the importer
    left without one, where the file binds it, hanging from the joint the file hangs it from (and the joints that hung
    from it hang from it again). Where the file's frame lies against the armature's is read from the joints that came
    through: one similarity, which must land them within a thousandth of the skeleton's size. A bone reaches to its one
    child where that sits elsewhere, else along its joint's own y axis, LEAF of the skeleton's size (the importer's own
    bones can be far shorter: it is their length that dropped these). -> the joints
    given back."""
    import bpy
    import numpy as np
    parents, binds = joint_binds(dae)
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not arms or not binds:
        return []
    arm = max(arms, key=lambda o: len(o.data.bones))
    have = {b.name: np.array(b.head_local[:]) for b in arm.data.bones}
    missing = [n for n in binds if n not in have]
    common = [n for n in binds if n in have]
    if not missing or len(common) < 3:
        return []
    p = np.array([binds[n][:3, 3] for n in common])
    q = np.array([have[n] for n in common])
    m, miss = _similarity(p, q)
    size = float(np.ptp(q, axis=0).max()) or 1.0
    if miss > 1e-3 * size:
        return []
    leaf = LEAF * size
    kids = {}
    for n, par in parents.items():
        kids.setdefault(par, []).append(n)

    def at(n):
        return m[:3, :3] @ binds[n][:3, 3] + m[:3, 3] if n in binds else have.get(n)
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    bones = arm.data.edit_bones
    made = []
    for n in missing:
        head = at(n)
        ends = [e for e in (at(c) for c in kids.get(n, [])) if e is not None]
        if len(ends) == 1 and np.linalg.norm(ends[0] - head) > 1e-3 * size:
            tail = ends[0]
        else:
            y = m[:3, :3] @ binds[n][:3, 1]
            tail = head + leaf * y / (np.linalg.norm(y) or 1.0)
        eb = bones.new(n)
        eb.head, eb.tail, eb.use_deform = head.tolist(), tail.tolist(), True
        made.append(n)
    for n in list(parents):
        par = parents[n]
        while par is not None and bones.get(par) is None:
            par = parents.get(par)
        eb = bones.get(n)
        if eb is not None and par is not None and (n in made or par in made):
            eb.use_connect = False
            eb.parent = bones[par]
    bpy.ops.object.mode_set(mode='OBJECT')
    return [n for n in made if n in arm.data.bones]


if __name__ == '__main__':      # run by Blender 4: blender --background --python convert.py -- <dae> <glb>
    import bpy
    import runpy
    src, dst = sys.argv[sys.argv.index('--') + 1:][:2]
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.collada_import(filepath=src)
    restored = restore_joints(src)
    if restored:
        print(JOINTS_RECORD + json.dumps(restored), flush=True)
    print(RECORD + json.dumps(read_normals(src)), flush=True)
    materials = runpy.run_path(str(Path(__file__).with_name('materials.py')))
    print(MATERIAL_RECORD + json.dumps(materials['convert']()), flush=True)
    bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', export_image_format='AUTO')
