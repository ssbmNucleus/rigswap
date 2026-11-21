"""A fighter made into a target from its vanilla costume (``prepare``): the DObjs the game draws in play exported by
HSDRawViewer (export/write.py: export_dae), the .dae made a .glb with every joint through Blender 4 (``dae_to_glb``),
each material's texture as read_dae reads it, a texture applied after lighting taken from the export's texture map
beside the .dae (``drawn_after_lighting``), the game camera written beside the .glb (camera.py), and the fighter's
game frames (frames.py). See the package's docstring for what ``prepare`` writes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from ... import Failed
from .. import FILES_VAR, TARGETS_VAR, _folder
from ..dae import NS, _image_files, read_dae, texture_map
from ..export.checks import compare_joints, tree_differs
from ..export.write import _blender4, export_dae, glb_textured, vanilla_copies
from ..facts.fighters import fighter, in_scope
from ..facts.memory import read_record, write_record
from .camera import camera
from .frames import write as write_frames


def prepare(files_dir=None, out_dir=None, names=None, refresh_memory=False):
    """Make every in-scope fighter's target from the vanilla costume: export the DObjs the game draws in play with
    HSDRawViewer, convert them to a .glb with every joint, and write the game camera beside it. Skips a fighter whose
    .glb, camera and memory record are already there. ``refresh_memory`` regenerates importer measurements even for
    existing targets. Returns {name: 'made' | 'already there' | why it failed}, and writes the same to
    ``<out_dir>/prepare.json``."""
    files = _folder(files_dir, FILES_VAR, 'vanilla game files')
    out_dir = out_dir or os.environ.get(TARGETS_VAR)
    if not out_dir:
        raise Failed(f'where do the targets go? pass the folder or set {TARGETS_VAR}')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in (fighter(n) for n in names) if names else in_scope():
        from .physics import write as write_physics
        try:
            write_physics(f, files, out)
            write_frames(f, files, out)
        except (OSError, ValueError, Failed, subprocess.TimeoutExpired) as e:
            report[f.name] = f'failed: {e}'
            continue
        glb, cam = out / f'{f.name}.glb', out / f'{f.name}.camera.json'
        memory_ready = False
        if not refresh_memory:
            try:
                read_record(f, out)
                memory_ready = True
            except Failed:
                pass
        if glb.is_file() and cam.is_file() and memory_ready:
            report[f.name] = 'already there'
            continue
        try:
            dat = files / f'Pl{f.code}Nr.dat'
            for needed in (dat, files / f'Pl{f.code}.dat'):
                if not needed.is_file():
                    raise Failed(f'{needed.name} is not in {files}')
            if not (glb.is_file() and cam.is_file()):
                dae = out / 'vanilla' / f.name / f'{f.name}.dae'
                if not dae.is_file():
                    with tempfile.TemporaryDirectory() as tmp:
                        export_dae(vanilla_copies(tmp, dat, files / f'Pl{f.code}.dat')[0], f.symbol, dae, drawn_by=f)
                model = read_dae(dae)
                if not model.triangles:
                    raise Failed(f'{dae.name} has no triangles')
                dae_to_glb(dae, glb)
                extent = float(np.ptp(model.points, axis=0).max())
                cam.write_text(json.dumps(camera(f.scale, extent), indent=1), encoding='utf-8')
            if not memory_ready:
                write_record(f, files, out)
            report[f.name] = 'made'
        except (Failed, RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            report[f.name] = f'failed: {e}'
        print(f'{f.name}: {report[f.name]}', flush=True)
    (out / 'prepare.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def drawn_after_lighting(dae):
    """{material name: image file} for each material of ``dae`` whose effect samples no image and that
    ``<dae>.textures.json`` names one for: the texture HSDRawViewer's exporter leaves out of the .dae because the
    material applies it after lighting (see read_dae)."""
    dae = Path(dae)
    named = texture_map(dae)
    root = ET.parse(dae).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    out = {}
    for m in root.iter(NS + 'material'):
        name = m.get('name', m.get('id'))
        if name in named and not _image_files(ids, m.get('id'), dae.parent):
            out[name] = str(named[name])
    return out


def uncoloured(dae):
    """The names of ``dae``'s materials whose effect samples an image but none in its diffuse, so nothing gives the
    image to base colour: HSDRawViewer writes a specular stage's texture to <specular> (Ganondorf's Material_DCDB4560
    samples Texture_20 there over a plain diffuse), and Blender's Collada importer keeps it off base colour."""
    root = ET.parse(dae).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    out = set()
    for m in root.iter(NS + 'material'):
        instance = m.find(NS + 'instance_effect')
        effect = ids.get(instance.get('url', '').lstrip('#')) if instance is not None else None
        if effect is not None and effect.find('.//' + NS + 'texture') is not None and                 not any(d.find(NS + 'texture') is not None for d in effect.iter(NS + 'diffuse')):
            out.add(m.get('name', m.get('id')))
    return out


# ---- a .glb's skeleton, read from the file itself -----------------------------------------------------------------

def glb_joints(path):
    """{joint name: 4x4 world matrix at rest} for every joint of every skin in a .glb, read from the file itself, in
    tree order (depth first, children in the file's order)."""
    return _glb_skeleton(path)[0]


def glb_parents(path):
    """{joint name: the joint it hangs from, or None} for a .glb."""
    return _glb_skeleton(path)[1]


def _glb_skeleton(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    nodes = gltf.get('nodes', [])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    is_joint = {j for skin in gltf.get('skins', []) for j in skin['joints']}

    def local(n):
        if 'matrix' in n:
            return np.array(n['matrix'], float).reshape(4, 4).T
        t, (x, y, z, w), s = n.get('translation', [0, 0, 0]), n.get('rotation', [0, 0, 0, 1]), n.get('scale', [1, 1, 1])
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = r * np.array(s, float)
        m[:3, 3] = t
        return m

    def name(i):
        return nodes[i].get('name', str(i))

    worlds, parents = {}, {}

    def walk(i, above, joint):
        w = above @ local(nodes[i])
        if i in is_joint:
            worlds[name(i)], parents[name(i)] = w, joint
            joint = name(i)
        for c in nodes[i].get('children', []):
            walk(c, w, joint)
    for i in range(len(nodes)):
        if i not in parent:
            walk(i, np.eye(4), None)
    return worlds, parents


# ---- Blender 4 for Collada (Blender 5 has none) -------------------------------------------------------------------

# Blender's Collada importer makes bones only for joints a skin uses and their parents; a fighter's other joints
# (fingers on a hand drawn as one piece, item and effect anchors) are dropped. It also turns the bones it makes and
# puts the ones it keeps out of order. The costume needs every joint, in the costume's order (a joint is known by its
# place in the tree), and the game's animations turn each joint about its own axes. So the bones are made again, one
# per joint in the .dae's order, each at its joint's rest matrix and under its parent; the meshes keep their weights,
# which go by bone name. The importer keeps the armature's space the .dae's (y up; the turn to z up is on the armature
# object): checked on every bone it made. A bone's length is never written to the .glb; BONE_LENGTH only keeps Blender
# from dropping it. A material whose texture HSDRawViewer's exporter left out of the .dae, as one applied after lighting
# (drawn_after_lighting, which read_dae takes from the same texture map), gets that image linked to its Base Color, as
# the importer links the others'.
DAE_TO_GLB = r'''
import bpy, json, sys
from mathutils import Matrix, Vector
src, joints_json, dst, after_json = sys.argv[sys.argv.index('--') + 1:][:4]
BONE_LENGTH = 0.1
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.wm.collada_import(filepath=src)
for name, path in json.load(open(after_json, encoding='utf-8')).items():
    m = bpy.data.materials.get(name)
    bsdf = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if m and m.node_tree else None
    if bsdf is None:
        raise SystemExit(f'{name}: the importer made no material with a Principled BSDF for it')
    image = m.node_tree.nodes.new('ShaderNodeTexImage')
    image.image = bpy.data.images.load(path)
    m.node_tree.links.new(image.outputs['Color'], bsdf.inputs['Base Color'])
    print('LINKED', name, path)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if len(arms) != 1:
    raise SystemExit(f'expected one armature, found {len(arms)}')
arm = arms[0]
joints = json.load(open(joints_json, encoding='utf-8'))


def rest(m):
    loc, rot, _ = Matrix(m).decompose()
    return Matrix.LocRotScale(loc, rot, None)


for j in joints:
    b = arm.data.bones.get(j['name'])
    if b is not None and (b.head_local - Vector([row[3] for row in j['matrix'][:3]])).length > 1e-3:
        raise SystemExit(f"{j['name']}: the importer put it at {tuple(b.head_local)}, not in the .dae's space")
for o in bpy.context.view_layer.objects:
    o.select_set(o == arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
bones = arm.data.edit_bones
added = len(joints) - len(bones)
for b in list(bones):
    bones.remove(b)
for j in joints:
    b = bones.new(j['name'])
    b.head, b.tail, b.roll = (0, 0, 0), (0, BONE_LENGTH, 0), 0
    b.matrix = rest(j['matrix'])
    b.parent = bones[j['parent']] if j['parent'] else None
bpy.ops.object.mode_set(mode='OBJECT')
print('JOINTS ADDED', added)
bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', export_image_format='AUTO', export_animations=False,
                          export_all_influences=True)
'''


def dae_to_glb(dae, glb):
    """A fighter's .dae to a .glb with every joint the .dae has (Blender's own importer drops unskinned ones) and every
    material's texture as read_dae reads it (a texture applied after lighting from HSDRawViewer's texture map), then
    checked: the .glb has the same joints, each where the .dae puts it, and a base colour texture on every material
    the .dae colours with a texture (not ``uncoloured``). Blender is given whole paths: it finds a .dae's images from
    the .dae's own."""
    dae, glb = Path(dae).resolve(), Path(glb).resolve()
    model = read_dae(dae)
    joints_json = glb.with_suffix('.joints.json')
    joints_json.write_text(json.dumps([{'name': n, 'parent': model.parents[n], 'matrix': m.tolist()}
                                       for n, m in model.joints.items()]), encoding='utf-8')
    after_json = glb.with_suffix('.after-lighting.json')
    after_json.write_text(json.dumps(drawn_after_lighting(dae)), encoding='utf-8')
    try:
        code, log = _blender4(DAE_TO_GLB, dae, joints_json, glb, after_json)
    finally:
        joints_json.unlink(missing_ok=True)
        after_json.unlink(missing_ok=True)
    if code != 0 or not glb.is_file():
        tail = '\n'.join(log.strip().splitlines()[-8:])
        raise Failed(f'{dae.name}: Blender 4 could not convert it\n{tail}')
    ids = {e.get('id'): e for e in ET.parse(dae).getroot().iter() if e.get('id')}
    textured = {ids[m].get('name', m) if m in ids else m for m, images in model.material_images.items() if images}
    lost = sorted(textured - uncoloured(dae) - glb_textured(glb))
    if lost:
        raise Failed(f'{glb.name} has no base colour texture on the materials {lost}, which {dae.name} draws with one')
    written, parents = _glb_skeleton(glb)
    check = compare_joints(model.positions(), written)
    if check.moved():
        raise Failed(f'{glb.name} does not hold the .dae\'s skeleton: {check}')
    differs = tree_differs(model.joints, model.parents, written, parents)
    if differs:
        raise Failed(f'{glb.name}: its joint tree is not the .dae\'s: {differs}')
    turned = {n: float(np.abs(_axes(m) - _axes(written[n])).max()) for n, m in model.joints.items()}
    worst = max(turned, key=turned.get)
    if turned[worst] > 1e-3:
        raise Failed(f'{glb.name}: joint {worst} is turned from its rest in the .dae ({turned[worst]:.3g})')
    return glb


def _axes(m):
    """A joint's axes, without scale."""
    r = np.asarray(m, float)[:3, :3]
    return r / np.linalg.norm(r, axis=0)
