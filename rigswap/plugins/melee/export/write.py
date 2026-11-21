"""The costume written: HSDRawViewer run on copies of the vanilla files (hsdraw, vanilla_copies, export_dae,
import_dae); the build made a .glb (blend_to_glb, its meshes first joined to fit the vanilla places: prep/join.py), and
the .glb a .dae HSDRawViewer imports, through Blender 4 (glb_to_dae); the texture formats and alpha modes the import
is told (_formats_by_dae_id, plan_alpha_modes); and the vanilla DObj list put back around the build's DObjs
(fill_vanilla_dobjs), the vanilla parts the costume keeps copied in (copy_parts).

HSDRawViewer's import makes one DObj for each mesh and material of the .dae, on the root joint (the .dae's mesh nodes
sit beside the root joint, not under one) or, for a mesh named ..SINGLE.. and bound wholly to one joint, on that joint
(HSDRawViewer ModelImportSettings.cs); it keeps the vanilla file's material animation tree. So the export imports
first and then puts the vanilla DObj list back around the build's DObjs (``fill_vanilla_dobjs``): the build's DObjs
that look the same at one high-detail place, and the same MObj and PObjs again at the low-detail places, so the costume
casts its shadow and shows in the magnifier. Every other form of the build's object then draws what it stands for
(``fill_forms``).
"""
from __future__ import annotations

from bisect import bisect_right
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from ....load import convert
from ... import Failed
from ..dae import NS, _floats, _image_files, _local
from ..dat import JOINT_NO_DOBJ, Dat, joints, joint_dobjs
from ..facts import parts as melee_parts
from ..facts.alpha import choose, draws_nothing, gate as alpha_gate
from . import dobjs as melee_dobjs
from .checks import check_parts


HSDRAW_VAR = 'RIGSWAP_HSDRAW'
# Set RIGSWAP_HSDRAW to the ssbmNucleus HSDLib fork's HSDRawViewer (it has --dobjs, --fighter, --show and --names;
# an older build ignores them and exports every DObj, which export_dae refuses). There is no default install path.
HSDRAW_DEFAULT = ''


def hsdraw(*args, timeout=600):
    """Run HSDRawViewer with ``args``; its output. Failed with the last lines when it exits non-zero."""
    exe = os.environ.get(HSDRAW_VAR) or HSDRAW_DEFAULT
    if not exe or not Path(exe).is_file():
        raise Failed(f'HSDRawViewer not found{" at " + exe if exe else ""}; set {HSDRAW_VAR}')
    exe = Path(exe)
    r = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, errors='replace',
                       timeout=timeout)
    out = (r.stdout or '') + (r.stderr or '')
    if r.returncode != 0 or 'ERROR:' in out:
        tail = '\n'.join(line for line in out.strip().splitlines()[-6:])
        raise Failed(f'HSDRawViewer {" ".join(map(str, args[:2]))} failed (exit {r.returncode}):\n{tail}')
    return out


def vanilla_copies(folder, *paths):
    """Copies of the vanilla files ``paths``, under their own names in ``folder``/vanilla-files, for HSDRawViewer to
    read. It opens a file it reads for reading and writing and lets others only read it while it runs (HSDLib
    HSDRaw/HSDRawFile.cs Open: ``new FileStream(filePath, FileMode.Open)``), so two of its runs on one vanilla file at
    once fail; each run that reads the game's files reads its own copies, and two runs of one fighter can go side by
    side. The names stay: HSDRawViewer finds the fighter's Pl<code>.dat beside its costume (``export_dae``)."""
    import shutil
    out = Path(folder) / 'vanilla-files'
    out.mkdir(parents=True, exist_ok=True)
    copies = []
    for path in map(Path, paths):
        copies.append(out / path.name)
        shutil.copyfile(path, copies[-1])
    return copies


def export_dae(dat, symbol, dae, drawn_by=None, by_place=False):
    """The model under ``symbol`` in ``dat`` as a .dae, with its textures and ``<dae>.textures.json`` beside it. Every
    DObj, or with ``drawn_by`` (a Fighter) only the ones the game draws in play when costume 0 spawns: high detail,
    with option 0 of each model in ``drawn_by.shown`` (the fighter's Pl<code>.dat beside ``dat`` has the tables). With
    ``drawn_by``, or ``by_place`` for every DObj, each mesh is named Joint_<joint>_Object_<its place in the costume's
    DObj list>."""
    dae = Path(dae)
    dae.parent.mkdir(parents=True, exist_ok=True)
    only = ['--names', 'place'] if by_place else []
    if drawn_by is not None:
        only = ['--names', 'place', '--dobjs', 'high', '--fighter', Path(dat).with_name(f'Pl{drawn_by.code}.dat'),
                '--show', ','.join(f'{m}:0' for m in drawn_by.shown)]
    log = hsdraw('--model', 'export', dat, symbol, dae, *only)
    if only and 'DObjs: ' not in log:
        raise Failed(f'HSDRawViewer ({os.environ.get(HSDRAW_VAR) or HSDRAW_DEFAULT}) does not know --dobjs: it '
                     'would export every DObj; use the build of the HSDLib fork from branch fighter-export-complete')
    if not dae.is_file():
        raise Failed(f'HSDRawViewer wrote no {dae.name}')
    return dae


def import_dae(dat, symbol, dae, out, texture_formats=None, alpha_modes=None):
    """``dat`` with the model under ``symbol`` replaced by ``dae``, written to ``out``. When the .dae's joint tree has
    the vanilla shape, HSDRawViewer keeps the vanilla joints and binds the mesh to them. ``texture_formats`` is a
    {material id in the .dae: "FORMAT[:PALETTE]"} .json file (see _formats_by_dae_id): each material's texture is
    written in that GX format instead of the import's own CMPR, or RGB5A3 for partial alpha."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    formats = ('--texture-formats', texture_formats) if texture_formats else ()
    modes = ('--alpha-modes', alpha_modes) if alpha_modes else ()
    log = hsdraw('--model', 'import', dat, symbol, dae, out, '--alpha-mode', 'Test', *formats, *modes)
    if not out.is_file():
        raise Failed(f'HSDRawViewer wrote no {out.name}')
    if 'preserving original bones' not in log:
        raise Failed(f'{Path(dae).name}: its joint tree does not have the vanilla shape, so the costume would get new '
                     'joints and the game\'s animations would not fit it')
    return out


# ---- Blender 4 for Collada (Blender 5 has none) -------------------------------------------------------------------

# The other way, for the costume: a .glb to a .dae HSDRawViewer imports, laid out the way HSDRawViewer writes its own:
#   - Only the armature and the meshes it deforms. HSDRawViewer reads the scene's nodes as its joint tree, so anything
#     else (an empty, a mesh with no faces) could be taken for a joint; a mesh that isn't skinned has no joint to ride
#     and fails.
#   - Y up, in the game's model space: Blender holds the scene z up, so the armature and meshes are turned back to y up
#     in their data, every object left at identity. The file then gets Y_UP and the root joint at the top of the scene
#     (see _lay_out_as_hsdraw).
#   - Textures as PNG files beside the .dae, which is where the importer reads them, named texture_<n>.png. The
#     importer takes a texture's format from any word of its file name that names one (HSDRawViewer
#     Extensions/TOBJExtensions.cs FormatFromString), over --texture-formats and over its own rule for partial alpha;
#     the last such word wins, and the palettes are format names too, so a vanilla Texture_25_CI8_RGB565 comes back
#     RGB565. Blender's Collada export names the file after the image, so the images are renamed first: the formats
#     come from prep's textures.json alone.
#   - Each material's image linked straight to its Principled BSDF's Base Color. HSDRawViewer takes a material's
#     texture from the .dae's diffuse texture, and Blender's Collada export writes one only for an image node linked
#     straight to Base Color. The glTF importer puts a Mix node between them that multiplies in the vertex colours
#     when the mesh has any, so the export would write the material with no texture, and the costume would get an
#     MObj with no TObj, which stops the game when the match loads (see export). The image is the one nearest
#     upstream of Base Color, the one prep sized and gave a format (prep/textures.py: _base_color_image); the
#     vertex colours are left out of the material, as the import leaves them out of the costume. glb_to_dae fails
#     when a material the .glb draws with a texture has none in the .dae (textures_lost).
GLB_TO_DAE = r'''
import bpy, math, os, re, sys
from mathutils import Matrix
def bake_worlds(arm, meshes):
    bpy.context.view_layer.update()
    worlds = {o.name: o.matrix_world.copy() for o in [arm] + meshes}
    # Empty parents can carry part of the armature or mesh transform. Capture all worlds before deleting any.
    for o in [arm] + meshes:
        o.parent = None
        o.matrix_parent_inverse = Matrix.Identity(4)
    for o in list(bpy.data.objects):
        if o != arm and o not in meshes:
            bpy.data.objects.remove(o)
    y_up = Matrix.Rotation(-math.pi / 2, 4, 'X')
    for o in [arm] + meshes:
        if o.data.users > 1:
            o.data = o.data.copy()
        baked = y_up @ worlds[o.name]
        o.data.transform(baked)
        if o.type == 'MESH' and baked.determinant() < 0:
            o.data.flip_normals()
        o.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for o in meshes:
        o.parent = arm
        o.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()

src, dst = sys.argv[sys.argv.index('--') + 1:][:2]
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, disable_bone_shape=True)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if len(arms) != 1:
    raise SystemExit(f'expected one armature, found {len(arms)}')
arm = arms[0]
meshes = [o for o in bpy.data.objects if o.type == 'MESH' and len(o.data.polygons)]
loose = [o.name for o in meshes if not any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
if loose:
    raise SystemExit(f'meshes not skinned to the armature: {loose}')
bake_worlds(arm, meshes)
arm.name = arm.data.name = 'Armature'
# Each mesh becomes a DObj of the costume for each of its materials, in the order Blender writes them, which is the
# order they were linked into the scene; export then gives each DObj a vanilla DObj's place (dobjs.py). They are
# linked again in name order, numbers taken as numbers: Object_2 before Object_10, so the order is the same each time.
def by_number(o):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', o.name)]
for o in sorted(meshes, key=by_number):
    for c in list(o.users_collection):
        c.objects.unlink(o)
    bpy.context.scene.collection.objects.link(o)
def upstream_image(socket):
    todo, seen = [link.from_node for link in socket.links], set()
    while todo:
        node = todo.pop(0)
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node
        todo += [link.from_node for i in node.inputs for link in i.links]
    return None
for m in bpy.data.materials:
    for n in (m.node_tree.nodes if m.node_tree else []):
        if n.type != 'BSDF_PRINCIPLED':
            continue
        base = n.inputs['Base Color']
        image = upstream_image(base)
        if image is not None and not any(link.from_node == image for link in base.links):
            m.node_tree.links.new(image.outputs['Color'], base)
            print('LINKED', m.name, image.image.name)
folder = os.path.dirname(os.path.abspath(dst))
for n, img in enumerate(list(bpy.data.images)):
    if not img.has_data and not img.packed_file:
        continue
    img.name = f'texture_{n:03d}'
    path = os.path.join(folder, img.name + '.png')
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    img.filepath = path
bpy.ops.wm.collada_export(filepath=dst, apply_modifiers=False, include_armatures=True, deform_bones_only=False,
                          include_children=False, selected=False, include_animations=False, triangulate=True,
                          use_texture_copies=False, use_blender_profile=False)
'''


def _blender4(script, *args, timeout=900):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script)
        path = f.name
    try:
        r = subprocess.run([convert.blender4(), '--background', '--factory-startup', '--python', path, '--',
                            *map(str, args)], capture_output=True, text=True, errors='replace', timeout=timeout)
    finally:
        os.unlink(path)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def glb_to_dae(glb, dae):
    """A .glb to a .dae for HSDRawViewer, through Blender 4: y up, textures as PNG files beside it."""
    glb, dae = Path(glb), Path(dae)
    _emission_to_diffuse(glb)
    dae.parent.mkdir(parents=True, exist_ok=True)
    code, log = _blender4(GLB_TO_DAE, glb, dae)
    if code != 0 or not dae.is_file():
        tail = '\n'.join(log.strip().splitlines()[-8:])
        raise Failed(f'{glb.name}: Blender 4 could not write a .dae\n{tail}')
    lost = textures_lost(glb, dae)
    if lost:
        raise Failed(f'{dae.name} lost the texture of the materials {lost}, which {glb.name} draws with one: the '
                     'costume would draw them untextured')
    _lay_out_as_hsdraw(dae)
    return dae


# The record of the corner normals repair_normals gave a direction, beside the .dae; none when it gave none.
NORMALS_REPAIRED = 'normals-repaired.json'


def repair_normals(root):
    """Give every corner normal of the .dae's meshes (``root``, its parsed tree) a direction. A normal of no length
    (ZERO_NORMAL or shorter, or NaN) stops HSDRawViewer's import; prep repairs the build's (prep/normals.py), and a
    corner of a sliver can still come out of Blender 4's Collada export with none. Such a corner's face draws nothing,
    so any direction serves: its mesh's mean normal, or up. -> {normals source id: corners repaired}"""
    from ..prep.normals import ZERO_NORMAL
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    sources = {i.get('source', '').lstrip('#') for i in root.iter(NS + 'input') if i.get('semantic') == 'NORMAL'}
    repaired = {}
    for key in sorted(s for s in sources if s in ids and ids[s].tag == NS + 'source'):
        array = ids[key].find(NS + 'float_array')
        values = _floats(array).reshape(-1, 3) if array is not None else np.zeros((0, 3))
        length = np.linalg.norm(values, axis=1)
        bad = ~(length > ZERO_NORMAL)
        if not bad.any():
            continue
        mean = values[~bad].sum(axis=0) if (~bad).any() else np.zeros(3)
        values[bad] = mean / np.linalg.norm(mean) if np.linalg.norm(mean) > ZERO_NORMAL else (0., 1., 0.)
        array.text = ' '.join(f'{v:.7g}' for v in values.ravel())
        repaired[key] = int(bad.sum())
    return repaired


def glb_textured(glb):
    """The names of the materials ``glb`` gives a base colour texture."""
    gltf = _gltf(glb)
    return {m.get('name') for m in gltf.get('materials', [])
            if (m.get('pbrMetallicRoughness') or {}).get('baseColorTexture') is not None}


def _gltf(glb):
    raw = Path(glb).read_bytes()
    length, kind = struct.unpack_from('<2I', raw, 12)          # the first chunk of a .glb is its JSON
    if kind != 0x4E4F534A:
        raise Failed(f'{Path(glb).name}: its first chunk is not JSON')
    return json.loads(raw[20:20 + length])


def _emission_to_diffuse(glb):
    """Carry an emissive-only colour image through the importer's diffuse-only texture channel.

    Normalize the intermediate GLB once, before conversion and its checks: a black base with no base image
    contributes no colour, so its active emission image supplies the colour texture. Keep the whole texture info
    (UV set and transform included), the base alpha and emission metadata used by plan_alpha_modes. All later
    consumers, including sampler preservation and the missing-texture check, then see the same base image.
    Other GLB chunks stay byte for byte; materials with a base image or nonblack base keep their authored channels.
    """
    glb = Path(glb)
    doc, changed = _gltf(glb), False
    for material in doc.get('materials', []):
        base = material.get('pbrMetallicRoughness', {})
        colour = base.get('baseColorFactor', [1, 1, 1, 1])
        emission = material.get('emissiveTexture')
        factor = material.get('emissiveFactor', [0, 0, 0])
        strength = material.get('extensions', {}).get('KHR_materials_emissive_strength', {}).get('emissiveStrength', 1)
        if (base.get('baseColorTexture') is not None or any(colour[:3]) or emission is None
                or not any(factor) or strength == 0):
            continue
        base['baseColorTexture'] = emission
        base['baseColorFactor'] = [*factor, colour[3]]
        changed = True
    if changed:
        raw = glb.read_bytes()
        tail = raw[20 + struct.unpack_from('<I', raw, 12)[0]:]
        data = json.dumps(doc).encode('utf-8')
        data += b' ' * (-len(data) % 4)
        glb.write_bytes(struct.pack('<3I', 0x46546C67, 2, 20 + len(data) + len(tail))
                        + struct.pack('<2I', len(data), 0x4E4F534A) + data + tail)


def preserve_samplers(glb, dae, imported, symbol):
    """Carry the GLB's per-material wrap modes across Collada, which drops them.

    Apply before placement, using the import's DObj order (eyes.dae_dobjs). HSDRaw also caches TObjs by image path,
    so setting Collada samplers alone would still conflate materials sharing an image with different wrap modes.
    GX's clamp/repeat/mirror are 0/1/2 (HSD_TOBJ +0x34/+0x38); glTF defaults to repeat on each omitted axis.
    UVs must stay outside [0, 1]: clamping vertices changes interpolation across the whole triangle.
    """
    from .eyes import dae_dobjs
    gltf = _gltf(glb)
    modes = {33071: 0, 10497: 1, 33648: 2}
    wanted = {}
    for material in gltf.get('materials', []):
        base = material.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if base is None:
            continue
        texture = gltf['textures'][base['index']]
        sampler = gltf.get('samplers', [])[texture['sampler']] if 'sampler' in texture else {}
        wanted[material['name']] = tuple(modes[sampler.get(axis, 10497)] for axis in ('wrapS', 'wrapT'))
    names = {m.get('id'): m.get('name') for m in ET.parse(dae).getroot().iter(NS + 'material')}
    source = dae_dobjs(dae)
    d = Dat.read(imported)
    objects = [o for j in joints(d, symbol) for o in joint_dobjs(d, j)]
    if len(source) != len(objects):
        raise Failed(f'texture samplers: {len(source)} DAE groups but {len(objects)} imported DObjs')
    rows = []
    for place, ((_, material), obj) in enumerate(zip(source, objects)):
        name = names.get(material)
        if name not in wanted:
            continue
        mobj = d.ptr(obj + 8)
        textures = d.chain(d.ptr(mobj + 8), 4) if mobj is not None else []
        if len(textures) != 1:
            raise Failed(f'texture samplers: {name} has {len(textures)} imported textures, expected one')
        tobj = textures[0]
        before = (d.u32(tobj + 0x34), d.u32(tobj + 0x38))
        struct.pack_into('>2I', d.data, tobj + 0x34, *wanted[name])
        rows.append({'place': place, 'material': name, 'id': material, 'before': list(before),
                     'written': list(wanted[name])})
    d.save(imported)
    return {'materials': rows, 'modes': {'clamp': 0, 'repeat': 1, 'mirror': 2}}


def textures_lost(glb, dae):
    """The materials ``glb`` gives a base colour texture that have no texture in ``dae``, by name."""
    textured = glb_textured(glb)
    root = ET.parse(dae).getroot()
    every = {e.get('id'): e for e in root.iter() if e.get('id')}
    kept = {m.get('name') for m in root.iter(NS + 'material') if _image_files(every, m.get('id'), Path(dae).parent)}
    return sorted(textured - kept)


def write_culling(d, objects, two_sided):
    """Write PObj culling for material-separated DObjs; retain all other flags and all geometry bytes.

    HSD_POBJ.Flags at +0x0c: CULLBACK=1<<14, CULLFRONT=1<<15. The import's clockwise triangles use CULLFRONT.
    Placement and low-detail copies retain these flags. A mixed material was split before the slot planner ran.
    """
    if len(objects) != len(two_sided):
        raise Failed(f'sidedness: {len(two_sided)} material groups but {len(objects)} imported DObjs')
    rows = []
    for obj, both in zip(objects, two_sided):
        polys = d.chain(d.ptr(obj + 12), 4)
        for p in polys:
            flags = struct.unpack_from('>H', d.data, p + 12)[0]
            struct.pack_into('>H', d.data, p + 12, (flags & ~0xc000) | (0 if both else 0x8000))
        rows.append({'two_sided': bool(both), 'pobjs': len(polys)})
    return rows


def preserve_sidedness(glb, dae, imported, symbol):
    """Carry prep's topology decision through GLB material.doubleSided into the imported PObjs."""
    from .eyes import dae_dobjs
    wanted = {m['name']: m.get('doubleSided', False) for m in _gltf(glb).get('materials', [])}
    names = {m.get('id'): m.get('name') for m in ET.parse(dae).getroot().iter(NS + 'material')}
    order = dae_dobjs(dae)
    missing = [material for _, material in order if names.get(material) not in wanted]
    if missing:
        raise Failed(f'sidedness: DAE materials absent from GLB: {missing}')
    d = Dat.read(imported)
    rows = write_culling(d, [o for j in joints(d, symbol) for o in joint_dobjs(d, j)],
                         [wanted[names[material]] for _, material in order])
    d.save(imported)
    return rows


def _lay_out_as_hsdraw(dae):
    """Blender writes the armature as a node holding the root joint and the mesh nodes; HSDRawViewer reads every node
    under it as a joint, so the root would have siblings the vanilla root doesn't. HSDRawViewer's own files have the
    root joint and the mesh nodes side by side at the top. The armature's node is at identity (checked), so taking its
    children up a level moves nothing. The file also says Y_UP, which the data now is. A corner normal with no
    direction is given one (repair_normals; NORMALS_REPAIRED beside the .dae says which)."""
    ET.register_namespace('', NS[1:-1])
    tree = ET.parse(dae)
    root = tree.getroot()
    axis = root.find(f'{NS}asset/{NS}up_axis')
    if axis is None:
        axis = ET.SubElement(root.find(NS + 'asset'), NS + 'up_axis')
    axis.text = 'Y_UP'
    for scene in root.iter(NS + 'visual_scene'):
        for node in list(scene.findall(NS + 'node')):
            children = node.findall(NS + 'node')
            if node.get('type') == 'JOINT' or not any(c.get('type') == 'JOINT' for c in children):
                continue
            if not np.allclose(_local(node), np.eye(4), atol=1e-6):
                raise Failed(f'{Path(dae).name}: the armature node {node.get("name")} is not at identity')
            at = list(scene).index(node)
            scene.remove(node)
            for k, child in enumerate(children):
                scene.insert(at + k, child)
    repaired = repair_normals(root)
    marker = Path(dae).with_name(NORMALS_REPAIRED)
    marker.unlink(missing_ok=True)
    if repaired:
        marker.write_text(json.dumps(repaired, indent=1), encoding='utf-8')
    tree.write(dae, encoding='utf-8', xml_declaration=True)


# ---- the build, for the import -------------------------------------------------------------------------------------

def mapped(objects):
    """Give every mesh with faces and no texture map one, each corner at the image's origin: where Blender draws a
    face that has none. Returns the names of the meshes given one.
    Blender's glTF export writes the material of a mesh with no map a second time under the same name (its texture's
    texCoord -1); Blender 4 reads the second as <name>.002; and no material of that name was ever planned. Kanade on
    Peach, an eight-point lock of hair with no map among 66 with one: "alpha modes: DAE material
    'mtl_chr_hair_00.002' has no GLB source", every build of hers from 2026-11-09."""
    given = []
    for ob in objects:
        if ob.type != 'MESH' or not len(ob.data.polygons) or len(ob.data.uv_layers):
            continue
        layer = ob.data.uv_layers.new(name='UVMap', do_init=False)
        zeros = [0.0] * (2 * len(ob.data.loops))
        if hasattr(layer, 'uv'):
            layer.uv.foreach_set('vector', zeros)
        else:
            layer.data.foreach_set('uv', zeros)
        given.append(ob.name)
    return given


def blend_to_glb(blend, glb, vanilla_dat=None, fighter_dat=None, symbol=None, kept=None, build_dir=None):
    """A .blend to a .glb with the bpy inside our Python, for Blender 4 to turn into Collada. With the vanilla files,
    its looks are first fit to the vanilla places left open by ``kept``, the vanilla places the costume keeps (default:
    the fighter's, parts.for_symbol): prep/join.py."""
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    if vanilla_dat is not None:
        from ..prep import detail_objects, drawn_meshes
        from ..prep.join import join_for_places
        from .eyes import plan as eye_plan
        from .flat import hull_of
        build_dir = Path(build_dir) if build_dir else Path(blend).parent
        # the objects each detail's build record lists stay their own DObjs (the eyes', for the blink)
        keep = detail_objects(build_dir)
        kept = melee_parts.for_symbol(symbol) if kept is None else kept
        planned = eye_plan(build_dir, vanilla_dat, fighter_dat, symbol, Path(glb).parent / 'vanilla' / 'vanilla.dae')
        texture_path = build_dir / 'textures.json'
        formats = json.loads(texture_path.read_text(encoding='utf-8')) if texture_path.is_file() else {}
        joins = join_for_places(drawn_meshes(), melee_dobjs.layout(vanilla_dat, symbol),
                                melee_parts.reserve(melee_dobjs.fighter_tables(fighter_dat), kept), keep=keep,
                                deferred=[*planned.get('destinations', {}), *hull_of(build_dir)], formats=formats)
        joins['eye_plan'] = planned
        Path(glb).with_name('joins.json').write_text(json.dumps(joins, indent=1), encoding='utf-8')
    mapped(bpy.data.objects)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB', export_animations=False, export_all_influences=True)
    bpy.ops.wm.read_homefile(use_empty=True)
    if not Path(glb).is_file():
        raise Failed(f'{Path(blend).name}: no .glb written')
    return Path(glb)


def _formats_by_dae_id(textures_json, dae):
    """prep's textures.json, {material name: format}, keyed instead by each material's id in ``dae``: HSDRawViewer
    reads the .dae with IONET, which knows a material by its id (Blender writes "<name>-material"), not its name.
    Written beside the .dae as texture-formats.json; None when prep wrote no textures.json. Missing prep names fail
    explicitly, including names which are no longer on a face."""
    textures_json = Path(textures_json)
    if not textures_json.is_file():
        return None
    by_name = json.loads(textures_json.read_text(encoding='utf-8'))
    ids = {m.get('name'): m.get('id') for m in ET.parse(dae).getroot().iter(NS + 'material')}
    missing = sorted(set(by_name) - set(ids))
    if missing:
        raise Failed(f'{Path(dae).name}: prep texture formats name missing materials: {missing}')
    out = Path(dae).with_name('texture-formats.json')
    out.write_text(json.dumps({ids[n]: v for n, v in by_name.items()}, indent=1), encoding='utf-8')
    return out


def plan_alpha_modes(glb, dae, model, texture_formats=None, variants=None):
    """Write the import map using DAE IDs; retain the source material and reason (facts/alpha.py: the rules). GLB
    alphaMode and emissive texture metadata carry the source's intent across the Blender/Collada boundary.
    The mode reads the encoded alpha where the triangles draw it. For encodings we cannot mirror (alpha palettes),
    use source alpha for that image and name the exception in the row; alpha_mode_check reads the written file.

    Each row also has the import's gate (facts/alpha.py: gate): the format the import writes the material's texture in
    (``texture_formats``, the --texture-formats map by DAE ID, as _formats_by_dae_id writes it; None: the import's
    own), the share of its texels below 255 as the PNG has them and as the import writes them, and whether the import
    gives the material an alpha operation at all (``transparent``), which alpha_mode_check holds the file to.

    A material the build draws opaque whose texture, as the import writes it, lets no texel through its alpha test
    (facts/alpha.py: draws_nothing) would draw nothing in game: its row's ``fully_transparent`` says so, and its image
    is written opaque, as the build draws it (``_drawn_opaque``), before its row is planned again.

    ``variants`` ({a material's name: the first's it shares a place with}, the join's ``one_place``): two materials
    differing only in their culling are one material of the file (fill_vanilla_dobjs), so they are planned on the alpha
    where either draws."""
    gltf = _gltf(glb)
    materials = {m['name']: m for m in gltf.get('materials', [])}
    asked = json.loads(Path(texture_formats).read_text(encoding='utf-8')) if texture_formats else {}
    rows, meanings, clear = {}, {}, {}

    def read(key):
        images = []
        for path in model.material_images.get(key, []):
            with Image.open(path) as image:
                images.append(np.asarray(image.convert('RGBA')))
        return images

    def plan(key, images):
        from ..facts.alpha import encoded_alpha, written_format
        alphas, unmirrored = [], []
        for image in images:
            fmt = written_format(asked.get(key), image[..., 3])
            alpha = encoded_alpha(image, fmt)
            if alpha is None:
                unmirrored.append(fmt)
                alpha = image[..., 3]
            alphas.append(alpha)
        family = [k for k in meanings if origin(k) == origin(key)]
        row = choose(meanings[key], np.concatenate([_drawn_alpha(model, k, alphas) for k in family]))
        row.update(alpha_basis='encoded' if not unmirrored else 'source for unmirrored encodings')
        if unmirrored:
            row['unmirrored_formats'] = sorted(set(unmirrored))
        return dict(row, **alpha_gate(images, asked.get(key)), source=meanings[key]['name'])
    drawn = set(model.face_materials)
    for material in ET.parse(dae).getroot().iter(NS + 'material'):
        if material.get('id') in drawn:
            meanings[material.get('id')] = {'name': material.get('name')}
    origin = lambda key: (variants or {}).get(meanings[key]['name'], meanings[key]['name'])
    for key, meaning in meanings.items():
        name = meaning['name']
        if name not in materials:
            raise Failed(f'alpha modes: DAE material {name!r} has no GLB source')
        source = materials[name]
        meaning['alpha_mode'] = source.get('alphaMode', 'OPAQUE')
        # The old semantic "glow" is an emissive-image card, not every lit material.
        extras = source.get('extras', {})
        if extras.get('kind') == 'glow' or (source.get('emissiveTexture') and
                any(source.get('emissiveFactor', (0, 0, 0)))):
            meaning['kind'] = 'glow'
        images = read(key)
        rows[key] = plan(key, images)
        if meaning['alpha_mode'] == 'OPAQUE' and draws_nothing(images, asked.get(key), rows[key]['mode']):
            clear[key] = _base_colour_image_name(gltf, source)
    for key, image_name in clear.items():
        rows[key]['fully_transparent'] = _drawn_opaque(key, image_name, model, clear, meanings)
    for key in clear:
        if rows[key]['fully_transparent']['written'] == 'opaque':
            rows[key] = dict(plan(key, read(key)), fully_transparent=rows[key]['fully_transparent'])
    destination = Path(dae).with_name('alpha-modes.json')
    destination.write_text(json.dumps({key: r['mode'] for key, r in rows.items()}, indent=1), encoding='utf-8')
    return destination, rows


def _drawn_alpha(model, key, alphas):
    """The encoded alpha planes ``alphas`` where material ``key``'s triangles draw them: at each triangle's centre and its
    corners pulled a third of the way in. An atlas's alpha elsewhere (another surface's fringe, a mask the source game
    reads for something else) says nothing of how this material draws: a body sampling solid texels draws solid
    whatever its file's alpha holds. The whole images' alpha when the material has no UVs."""
    if not alphas:
        return np.array([], dtype=np.uint8)
    rows = [i for i, m in enumerate(model.face_materials) if m == key]
    if not rows or len(model.face_uvs) != len(model.face_materials):
        return np.concatenate([a.ravel() for a in alphas])
    uv = np.asarray(model.face_uvs, float).reshape(-1, 3, 2)[rows]
    centre = uv.mean(axis=1, keepdims=True)
    at = np.concatenate([centre, uv + (centre - uv) / 3], axis=1).reshape(-1, 2)
    out = []
    for alpha in alphas:
        h, w = alpha.shape
        x = (np.mod(at[:, 0], 1.0) * w).astype(int) % w
        y = ((1.0 - np.mod(at[:, 1], 1.0)) * h).astype(int) % h
        out.append(alpha[y, x])
    return np.concatenate(out)


def _base_colour_image_name(gltf, material):
    """The name of the image a glTF material draws as its base colour, or None."""
    base = (material.get('pbrMetallicRoughness') or {}).get('baseColorTexture')
    try:
        image = gltf['images'][gltf['textures'][base['index']]['source']]
    except (KeyError, IndexError, TypeError):
        return None
    return image.get('name') or Path(image.get('uri', '')).name or None


def _drawn_opaque(key, image_name, model, clear, meanings):
    """A material the build draws opaque (its source alpha mode OPAQUE) whose texture, as the import writes it, has no
    texel its alpha test lets through (``clear``: every such material by DAE ID): the game would draw nothing of it,
    where the build and its pictures show the texture's colours (Blender draws an opaque material's texture without
    its alpha). goku-budokai-Zelda (runs/pilot6): the download's hair.png is black with every texel's alpha 0, its
    material opaque in the .mtl (d 1.0), and Goku came out bald. A download fault, written as the build draws it: its
    image files beside the .dae get alpha 255 in every texel, so the import's gate gives the material no alpha
    operation; unless another material draws the same file with its alpha, which keeps it as it is. -> the record
    the export's check and notes say it from"""
    paths = model.material_images.get(key, [])
    shared = sorted(meanings.get(other, {}).get('name', other) for other, images in model.material_images.items()
                    if other not in clear and other in meanings and set(images) & set(paths))
    alpha = []
    for path in paths:
        with Image.open(path) as image:
            alpha.append(int(np.asarray(image.convert('RGBA'))[..., 3].max()))
    name, texture = meanings[key]['name'], image_name or ', '.join(Path(p).name for p in paths)
    record = {'texture': texture, 'files': [Path(p).name for p in paths], 'alpha_max': max(alpha, default=0)}
    if shared:
        record.update(written='as it is', said=f'texture {texture} is fully transparent and its material {name} '
                      f'draws it opaque; materials {", ".join(shared)} draw the same image with its alpha, so it is '
                      f'written as it is and the game draws nothing of {name}')
        return record
    for path in paths:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert('RGBA')).copy()
        if (rgba[..., 3] != 255).any():
            rgba[..., 3] = 255
            Image.fromarray(rgba, 'RGBA').save(path)
    record.update(written='opaque', said=f'texture {texture} is fully transparent (no texel\'s alpha above '
                  f'{record["alpha_max"]}) and its material {name} draws it opaque: written opaque, in its own '
                  f'colours, as the build draws it; with its alpha the game would draw nothing of {name}')
    return record


# The wrap modes the import gives a TObj before preserve_samplers sets the source's: repeat on both axes (its
# record's ``before``).
IMPORT_WRAP = (1, 1)


def wraps_written(samplers):
    """{DAE material ID: the wrap modes preserve_samplers wrote into its TObj} (``samplers``: its record)."""
    return {r['id']: tuple(r['written']) for r in (samplers or {}).get('materials', []) if 'id' in r}


def written_alike(model, plan, wraps=None):
    """{DAE material ID: what the import writes for it}, which checks.material_check counts the source's materials by.
    The import makes a material from its diffuse image alone, in the format it writes it (plan's ``format``), with
    the alpha operation its gate gives it (the mode where ``transparent``), and the .dae's colours left out
    (HSDRawViewer ModelImporter.cs GenerateMaterial and ConfigureAlpha, ImportMaterialInfo off; an image file already
    read is cloned, pathToTObj); then its TObj gets the source's wrap modes (``wraps``: wraps_written; IMPORT_WRAP
    where none was written). The read-back names a material by its bytes (ModelExporter.cs: Material_<FNV hash of
    the MObj and everything it points at>). So two .dae materials drawing the same pixels in the same format with the
    same alpha operation and wrap modes are one material in the file: a body and its hair outline both drawing one
    image read back as one; an eye decal clamped and the same image repeated read back as two. Every untextured
    material is the import's defaults."""
    import hashlib
    pixels = {}

    def digest(path):
        if path not in pixels:
            with Image.open(path) as image:
                rgba = np.asarray(image.convert('RGBA'))
            pixels[path] = hashlib.sha1(repr(rgba.shape).encode() + rgba.tobytes()).hexdigest()
        return pixels[path]
    out = {}
    for key in sorted(set(model.face_materials)):
        images = model.material_images.get(key, [])
        if not images:
            out[key] = ('untextured',)
            continue
        row = plan.get(key, {})
        alpha = None if row.get('transparent') is False else row.get('mode')
        out[key] = (tuple(digest(p) for p in images), str(row.get('format')), alpha,
                    (wraps or {}).get(key, IMPORT_WRAP))
    return out


# ---- the vanilla DObjs around the build's ---------------------------------------------------------------------------

def fill_vanilla_dobjs(imported, vanilla_dat, fighter_dat, symbol, out, costume=0, keep=None, deferred=(),
                       high_only=()):
    """Rewrite ``imported`` (the build as HSDRawViewer imports it into the vanilla costume: one DObj for each mesh and
    material, and the vanilla material animation tree) so its DObjs are the vanilla costume's in number and order, and
    write it to ``out``, by the rules of dobjs.py.

    A DObj of the import whose chain holds more PObjs than the game can load in one (dobjs.MAX_POBJS) is first cut into
    several with the same material (``split_long_chains``). A place draws one material and a chain of PObjs, and
    culling is each PObj's own (pobj.c, the switch on POBJ_CULLFRONT and POBJ_CULLBACK; write_culling), so the DObjs of
    one joint that look the same (_look) take one vanilla place between them, up to the chain limit (``choose_places``):
    a body's culled and two-sided faces, or any meshes of one material. ``deferred`` lists imported DObj indices whose
    final animated places the detail exporter
    already planned: retain their descriptors for that exporter without taking temporary body places. They still
    draw in low detail, but for those in ``high_only`` (an outline hull: export/flat.py), and obey the chain limit. The
    low-detail places draw all of the build's other DObjs again (``choose_low_places``): a low-detail DObj points at the same MObj and PObjs as the build's DObj, so set 1
    draws the build itself, in shadows and the magnifier, and the file grows only by the pointers to the PObjs and the
    PObjs copied to join two DObjs' geometry under one place (0x18 bytes each; a PObj's next is its own field, so a
    PObj is in one chain only), never by geometry. The kept places (``keep``, default the fighter table's,
    facts/parts.py: for_symbol; a build that replaced a weapon keeps fewer, kept_places) retain vanilla
    geometry and materials (copy_parts). Every other vanilla DObj keeps its place with no geometry and a
    blank material with vanilla's number of TObjs and their texture map ids. The DObj list, the visibility tables and
    the TObj places then come out as vanilla's, and the vanilla material animation gives the animated TObjs their
    texture animations: the eyes keep vanilla's animation and its images, on DObjs that draw nothing. Nothing is placed
    for the metal set (3). -> what was done"""
    d = Dat.read(imported)
    imported_data_bytes, imported_relocations = len(d.data), len(d.relocations)
    vanilla = melee_dobjs.layout(vanilla_dat, symbol)
    keep = melee_parts.for_symbol(symbol) if keep is None else keep
    tables = melee_parts.reserve(melee_dobjs.fighter_tables(fighter_dat, costume), keep)
    if len(vanilla) > melee_dobjs.MAX_DOBJS:
        raise Failed(f'the vanilla costume has {len(vanilla)} DObjs; the game takes {melee_dobjs.MAX_DOBJS}')
    tree = joints(d, symbol)
    vanilla_joints = len(joints(Dat.read(vanilla_dat), symbol))
    if len(tree) != vanilla_joints:
        raise Failed(f'the imported costume has {len(tree)} joints; the vanilla one has {vanilla_joints}')
    imported_dobjs = [(k, dobj) for k, j in enumerate(tree) for dobj in joint_dobjs(d, j)]
    deferred = set(deferred)
    if not deferred <= set(range(len(imported_dobjs))):
        raise Failed('a deferred DObj is outside the imported list')
    build, cut = melee_dobjs.split_long_chains(d, imported_dobjs)
    if deferred.intersection(cut):
        raise Failed('an animated DObj exceeds the PObj chain limit')
    origin = [n for n in range(len(imported_dobjs)) for _ in cut.get(n, [None])]
    needs, looks, chains = [], [], []
    for k, dobj in build:
        mobj = d.ptr(dobj + 0x08)
        needs.append((k, len(d.chain(d.ptr(mobj + 0x08), 0x04)) if mobj is not None else 0))
        looks.append(_look(d, dobj))
        chains.append(len(d.chain(d.ptr(dobj + 0x0C), 0x04)))
    alike = {}
    for u, n in enumerate(origin):
        if n not in deferred:
            alike.setdefault((build[u][0], looks[u]), []).append(u)
    runs = [run for group in alike.values()
            for run in melee_dobjs._runs(group, [chains[u] for u in group], melee_dobjs.MAX_POBJS)]
    allocated = melee_dobjs.choose_places(vanilla, tables, [needs[run[0]] for run in runs])
    places = {u: p for run, p in zip(runs, allocated) for u in run}
    taken = {p: [build[u][1] for u in run] for run, p in zip(runs, allocated)}
    lowered = [u for u, n in enumerate(origin) if n not in high_only]
    low, left_out = melee_dobjs.choose_low_places(vanilla, tables, [(*needs[u], _draws(d, build[u][1]), looks[u],
                                                                     chains[u]) for u in lowered])
    low = {p: [lowered[n] for n in ns] for p, ns in low.items()}
    left_out = [lowered[n] for n in left_out]
    counts = melee_dobjs.texture_counts(vanilla, tables, {
        **{p: needs[n][1] for n, p in places.items()}, **{p: needs[ns[0]][1] for p, ns in low.items()}})

    image_data = d.add(bytes(32), melee_dobjs.IMAGE_ALIGN)                       # 8x8 at 4 bits a pixel
    image = d.add(struct.pack('>I2H2I2f', 0, 8, 8, melee_dobjs.GX_I4, 0, 0., 0.))
    d.point(image, image_data)
    material = d.add(struct.pack('>4B4B4B2f', 128, 128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1., 50.))
    cache = {}
    retained, retained_record = copy_parts(Dat.read(vanilla_dat), d, symbol, keep)

    made = [None] * len(vanilla)
    low_bytes = 0           # what the low-detail DObjs add over blank ones: copied PObjs and pointers to PObjs
    padded, padding = {}, {}

    def material_at(i, mobj):       # the drawn material, padded where its place keeps more TObjs (texture_counts)
        own = len(d.chain(d.ptr(mobj + 0x08), 0x04)) if mobj is not None else 0
        if mobj is None or counts[i] <= own:
            return mobj
        padded[i] = [own, counts[i]]
        return melee_dobjs.padded_material(d, mobj, counts[i], image, padding)
    for i in reversed(range(len(vanilla))):
        dobj = d.add(bytes(0x10))
        if i in retained:
            d.point(dobj + 0x08, retained[i][0])
            d.point(dobj + 0x0C, retained[i][1])
        elif i in taken:
            d.point(dobj + 0x08, material_at(i, d.ptr(taken[i][0] + 0x08)))
            d.point(dobj + 0x0C, melee_dobjs._joined_pobjs(d, taken[i])[0])
        elif i in low:
            drawn = [build[n][1] for n in low[i]]
            d.point(dobj + 0x08, material_at(i, d.ptr(drawn[0] + 0x08)))
            head, added = melee_dobjs._joined_pobjs(d, drawn)
            d.point(dobj + 0x0C, head)
            low_bytes += added + 4
        else:
            textures = vanilla[i].textures if counts[i] == len(vanilla[i].textures) else tuple(range(counts[i]))
            d.point(dobj + 0x08, melee_dobjs._blank_material(d, textures, image, material, cache))
        nxt = i + 1
        if nxt < len(vanilla) and vanilla[nxt].joint == vanilla[i].joint:
            d.point(dobj + 0x04, made[nxt])
        made[i] = dobj
    longest = max((len(d.chain(d.ptr(dobj + 0x0C), 0x04)) for dobj in made), default=0)
    if longest > melee_dobjs.MAX_POBJS:
        raise Failed(f'a DObj of the costume draws {longest} PObjs; the game loads at most {melee_dobjs.MAX_POBJS} '
                     'in one')
    firsts = {s.joint: made[i] for i, s in reversed(list(enumerate(vanilla)))}
    for k, j in enumerate(tree):
        if k in firsts or joint_dobjs(d, j):
            if d.u32(j + 4) & JOINT_NO_DOBJ:
                raise Failed(f'joint {k} holds DObjs in the vanilla costume and is a particle or spline joint here')
            d.point(j + 0x10, firsts.get(k))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    d.save(out)
    # by the import's DObjs (the .dae's order): each one's place, a cut one's first chain's; the other chains' places
    # are under chains_cut
    firsts_of = {}
    for unit, n in enumerate(origin):
        firsts_of.setdefault(n, []).append(places.get(unit))
    return {'dobjs': len(vanilla), 'build_dobjs': len(imported_dobjs),
            'imported_data_bytes_kept': imported_data_bytes,
            'data_bytes_appended': len(d.data) - imported_data_bytes,
            'relocation_bytes_added': 4 * (len(d.relocations) - imported_relocations),
            'places': [firsts_of[n][0] for n in range(len(imported_dobjs))],
            'deferred': {n: imported_dobjs[n] for n in sorted(deferred)},
            'chains_cut': {n: {'pobjs': parts, 'places': firsts_of[n]} for n, parts in sorted(cut.items())},
            'longest_chain': longest, 'max_pobjs': melee_dobjs.MAX_POBJS,
            'blank': len(vanilla) - len(taken) - len(low) - len(retained), 'animated_textures': list(tables.animated),
            **({'padded_textures': {i: padded[i] for i in sorted(padded)}} if padded else {}),
            'blank_texture_slots': {i: counts[i] for i in range(len(vanilla))
                                   if i not in taken and i not in low and counts[i] != len(vanilla[i].textures)},
            'kept_parts': check_parts(vanilla_dat, out, symbol, retained_record),
            'animated_at': [list(p) if p else None for p in melee_dobjs.texture_places(vanilla, tables.animated)],
            'low_detail': {'how': "the build's DObjs again: the same MObjs and PObjs, the geometry not copied",
                           'places': {i: [origin[u] for u in low[i]] for i in sorted(low)},
                           'left_out': sorted({origin[u] for u in left_out}),
                           'bytes_added': low_bytes}}


def fill_forms(costume, tables, symbol):
    """The forms of the build's objects (dobjs.forms), in ``costume`` as export has laid it out, the details' places
    filled: each form with nothing of its own draws what it stands for, in the same MObj and PObjs (dobjs.py: Loading),
    so a form adds no bytes. Each keeps its own material animation, so a form of a blinking eye shows the eye as its
    material has it. A form of a place that draws nothing stays blank: its object is drawn with the build's body.
    -> {form: place}"""
    d = Dat.read(costume)
    listed = [dobj for j in joints(d, symbol) for dobj in joint_dobjs(d, j)]
    filled = {}
    for form, place in melee_dobjs.forms(tables).items():
        if d.ptr(listed[form] + 0x0C) is None and d.ptr(listed[place] + 0x0C) is not None:
            d.point(listed[form] + 0x08, d.ptr(listed[place] + 0x08))
            d.point(listed[form] + 0x0C, d.ptr(listed[place] + 0x0C))
            filled[form] = place
    d.save(costume)
    return filled


def _draws(d, dobj):
    """How much a DObj draws: the bytes of its PObjs' display lists (+0x0E, in 32-byte units)."""
    return 32 * sum(d.u16(p + 0x0E) for p in d.chain(d.ptr(dobj + 0x0C), 0x04))


def _look(d, dobj):
    """What a DObj's material draws with: its render mode, colours, pixel engine settings, and each TObj's texture map
    id, wrap modes and image data. The import writes a material once for each mesh that uses it; HSDRaw writes identical
    image data once, so two DObjs from one material look the same."""
    mobj = d.ptr(dobj + 0x08)
    if mobj is None:
        return None
    colours, engine = d.ptr(mobj + 0x0C), d.ptr(mobj + 0x14)
    textures = []
    for tobj in d.chain(d.ptr(mobj + 0x08), 0x04):
        image = d.ptr(tobj + 0x4C)
        textures.append((d.u32(tobj + 0x08), d.u32(tobj + 0x34), d.u32(tobj + 0x38),
                         d.ptr(image) if image is not None else None))
    return (d.u32(mobj + 0x04), bytes(d.data[colours:colours + 0x14]) if colours is not None else None,
            bytes(d.data[engine:engine + 0x0C]) if engine is not None else None, tuple(textures))


def copy_parts(vanilla, destination, symbol, places):
    """Copy reachable DAT blocks once, remapping bone pointers to the new skeleton.

    Relocation targets delimit HSD archive blocks. Keep textures 32-byte aligned;
    preserve native samplers, display lists, envelopes and material flags verbatim.
    Neither the vanilla skeleton nor its other drawables are traversed.
    """
    source_tree, target_tree = joints(vanilla, symbol), joints(destination, symbol)
    if len(source_tree) != len(target_tree):
        raise Failed('kept parts: skeleton sizes differ')
    mapped = dict(zip(source_tree, target_tree))
    starts = sorted({0, len(vanilla.data), *[vanilla.u32(p) for p in vanilla.relocations],
                     *[p for _, p in vanilla.roots]})
    relocations = sorted(vanilla.relocations)
    blocks = []
    before, pointers = len(destination.data), len(destination.relocations)

    def copy(at):
        if at is None:
            return None
        if at in mapped:
            return mapped[at]
        end = starts[bisect_right(starts, at)]
        new = destination.add(vanilla.data[at:end], 32)
        mapped[at] = new
        blocks.append((at, new, end - at))
        for field in relocations:
            if at <= field < end:
                destination.point(new + field - at, copy(vanilla.u32(field)))
        return new

    objects = [o for j in source_tree for o in joint_dobjs(vanilla, j)]
    kept = {}
    for place in places:
        if not 0 <= place < len(objects):
            raise Failed(f'kept place {place} is outside the vanilla DObj list')
        obj = objects[place]
        kept[place] = (copy(vanilla.ptr(obj + 8)), copy(vanilla.ptr(obj + 0x0c)))
    return kept, {'places': list(places), 'blocks': blocks, 'pointers': mapped,
                  'bytes': len(destination.data) - before + 4 * (len(destination.relocations) - pointers)}
