"""load.open: unpack the download, let Blender import every model file in it, match textures. Nothing is judged, and
nothing is dropped but what the file itself declares: surfaces its idle collapses (stance) and coarser levels of detail
(finest_levels: meshes named as one base name's LOD0, LOD1, ...; only the finest of each stays, so the game never draws
a coarse copy over the detailed one). Each file's objects are linked again in one order the file gives (settle_order:
by name, then mesh name, then triangles), whatever order the importer linked them in, so the pieces' numbers and the
pictures are the same each run.

Writes ``<out>/models/<n>-<name>.blend`` per model file and ``<out>/models.json`` listing them with counts.
``<out>/load.json`` records files seen with no importer, with the reason they cannot open."""
from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath
import shutil
import zipfile

from . import binds, convert, materials, stance, texture

UNSUPPORTED = {'.smd': 'no importer for .smd in Blender 5.2'}
# The kinds of file a model file refers to beside it: glTF buffers, OBJ materials, textures. A bare model file is
# copied with the files of these kinds in its folder.
REFERENCED_SUFFIXES = {'.bin', '.mtl', '.png', '.jpg', '.jpeg', '.tga', '.bmp', '.dds', '.tif', '.tiff', '.webp', '.gif',
                       '.ktx2'}
NESTED_ZIP_DEPTH = 3       # zips inside the download are opened this many levels down


def _import_gltf(p):
    # Left to itself, the glTF importer adds a mesh to draw bones with: an "Icosphere" in a hidden collection,
    # "glTF_not_exported". It isn't the download's, so it would be counted and carried as if it were. Not made at all.
    # A joint no vertex is bound to holds what its exporter liked as its bind; Blender would bind the bones under it by
    # that (load.binds).
    import bpy
    bpy.ops.import_scene.gltf(filepath=str(binds.bound(p)), disable_bone_shape=True)


def _import_dae(p):
    # Blender 5 has no Collada importer: convert once through Blender 4, then import the .glb. The conversion reads the
    # normals in the frame the file wrote them in and converts empty colour textures before the intermediate export
    # loses the material's own colour. The listing keeps both records.
    glb = Path(p).with_suffix('.converted.glb')
    facts = convert.dae_to_glb(p, glb)
    _import_gltf(glb)
    return facts


def _import_fbx(p):
    import importlib
    import bpy
    parse_fbx = importlib.import_module('io_scene_fbx.parse_fbx')
    parse, layered = parse_fbx.parse, []

    def base_layers_only(*args, **kwargs):
        root, version = parse(*args, **kwargs)
        layered.extend(base_layers(root))
        return root, version
    parse_fbx.parse = base_layers_only      # the importer reads the file through this module's parse
    try:
        bpy.ops.import_scene.fbx(filepath=p)
    finally:
        parse_fbx.parse = parse
    facts = {}
    moved = _meant_textures(p)
    if moved:
        facts['textures_meant'] = moved
    if layered:
        facts['base_layers'] = layered
    return facts or None


# An FBX can connect several textures to one property of a material (its DiffuseColor): they are layers, the first
# the base and each after it laid over it, as the FBX SDK reads a property's textures (the first is its texture) and as
# a writer lays out a stack of them (Blender 2.7x wrote its texture slots so, in slot order: a burlap knit tiled three
# times, then the dirt map multiplied over it). The game draws one colour image per material, and the base is that
# colour; a layer over it shades or tints it (dirt, ambient occlusion, a reflection map). Blender's importer gives each
# texture a property's slot in turn, so it kept the last layer: LittleBigPlanet's Sackboy came out the grey of his dirt
# map, not the brown of his knit. So the importer is handed the file with each material property's first texture
# only, and reads that one's image, tiling and alpha as it would a lone texture's. Over the 184 FBX of the corpus, 4
# connect several textures to a diffuse colour: Sackboy, Brawl Zelda (her jewel's texture, then Ref.png, its
# reflection map with no UV set), and Inkling's and Odyssey Bowser's, several of one file.
def base_layers(root):
    """In a parsed FBX (``root``: parse_fbx's element tree), keep only the first texture connected to each property of
    each material, removing the later connections in place. -> [{'material', 'property', 'kept', 'dropped'}], the
    properties that had several, each texture by its name (and file, where it names one)."""
    kinds, names = {}, {}
    for objects in (e for e in root.elems if e.id == b'Objects'):
        for e in objects.elems:
            if len(e.props) < 2:
                continue
            kinds[e.props[0]] = e.id
            name = e.props[1].split(b'\x00')[0].decode('utf-8', 'replace') if isinstance(e.props[1], bytes) else ''
            file = next((c.props[0].decode('utf-8', 'replace') for c in e.elems
                         if c.id in (b'RelativeFilename', b'FileName') and c.props), '')
            names[e.props[0]] = name + (f' ({PureWindowsPath(file).name})' if file else '')
    record, first = [], {}
    for connections in (e for e in root.elems if e.id == b'Connections'):
        kept = []
        for c in connections.elems:
            layer = (len(c.props) > 3 and c.props[0] == b'OP' and kinds.get(c.props[1]) == b'Texture'
                     and kinds.get(c.props[2]) == b'Material')
            key = (c.props[2], c.props[3]) if layer else None
            if key is None or key not in first:
                if key is not None:
                    first[key] = {'material': names[c.props[2]], 'property': c.props[3].decode('utf-8', 'replace'),
                                  'kept': names[c.props[1]], 'dropped': []}
                kept.append(c)
                continue
            first[key]['dropped'].append(names[c.props[1]])
        connections.elems[:] = kept
    return [r for r in first.values() if r['dropped']]


# An FBX names a texture twice: by the path it had when the file was written (the ripper's own C:\Users\...\
# Hair00_00_Nin_NX_NVN_szs\M_TeamColor_Alb.png) and by a relative one, often the bare file name. Neither is on this
# machine, so Blender's importer looks for the name under the file's folder and takes the first file of that name it
# walks to. Where the download has several (a Splatoon rip ships each hair's M_TeamColor_Alb.png in images/Hair00_0N/,
# beside the body's in images/), the file meant is the one in the folder the recorded path names: every word of its
# name is a word of the folder the texture was in when the file was written (Hair00_00_Nin_NX_NVN_szs:
# images/Hair00_00/), and of several so named, the one with most words (Tnk_Simple_Shot_First_Nin_NX_NVN_szs:
# images/Tnk_Shot_First/, not images/Tnk/). Where no file's folder is named, or two are equally, the search's pick
# stands.
def meant(found, recorded, same_name):
    """The file a texture means: ``found``, the one Blender's search took; ``recorded``, the paths the model file
    records for it; ``same_name``, every file of its name under the model file's folder."""
    folders = [w for w in (_words(PureWindowsPath(r).parent.name) for r in recorded) if w]

    def named(q):
        mine = _words(Path(q).parent.name)
        return max((len(mine) for f in folders if mine and mine <= f), default=0)
    most = max((named(q) for q in same_name), default=0)
    best = [q for q in same_name if most and named(q) == most]
    return best[0] if len(best) == 1 else found


def _words(name):
    return set(re.findall('[a-z0-9]+', name.lower()))


def _meant_textures(p):
    """Each image the import loaded from a file, moved to the file meant (meant). -> [{image, found, meant}]."""
    import importlib
    import os
    import bpy
    parse_fbx = importlib.import_module('io_scene_fbx.parse_fbx')
    root, _ = parse_fbx.parse(str(p))
    recorded = {}
    for objects in (e for e in root.elems if e.id == b'Objects'):
        for tex in (e for e in objects.elems if e.id in (b'Texture', b'Video')):
            for e in tex.elems:
                if e.id in (b'FileName', b'Filename', b'RelativeFilename') and e.props:
                    path = e.props[0].decode('utf-8', 'replace')
                    recorded.setdefault(PureWindowsPath(path).name.lower(), set()).add(path)
    files = {}
    for q in Path(p).parent.rglob('*'):
        if q.is_file():
            files.setdefault(q.name.lower(), []).append(str(q))
    moved = []
    for img in bpy.data.images:
        found = bpy.path.abspath(img.filepath) if img.source == 'FILE' else ''
        same_name = sorted(files.get(Path(found).name.lower(), []))
        if len(same_name) < 2 or not os.path.isfile(found):
            continue
        choice = meant(found, recorded.get(Path(found).name.lower(), ()), same_name)
        if os.path.normcase(os.path.abspath(choice)) != os.path.normcase(os.path.abspath(found)):
            img.filepath = choice
            img.reload()
            moved.append({'image': img.name, 'found': found, 'meant': choice})
    return moved


def _import_obj(p):
    import bpy
    bpy.ops.wm.obj_import(filepath=p)


# Each importer brings one file into the empty scene and returns what it adds to the file's listing, if anything.
IMPORTERS = {
    '.fbx': _import_fbx,
    '.dae': _import_dae,
    '.obj': _import_obj,
    '.glb': _import_gltf,
    '.gltf': _import_gltf,
}
MODEL_SUFFIXES = set(IMPORTERS)


def unpack(download, into):
    """A zip is extracted into ``into``, and each zip inside it into a folder of its own beside it, down to
    NESTED_ZIP_DEPTH levels. A bare model file is copied there with the files in its folder it may refer to
    (REFERENCED_SUFFIXES), so a .gltf keeps its .bin and an .obj its .mtl. Returns the folder."""
    download, into = Path(download), Path(into)
    into.mkdir(parents=True, exist_ok=True)
    if download.suffix.lower() == '.zip':
        _extract(download, into, 0)
    else:
        shutil.copy2(download, into / download.name)
        for f in download.parent.iterdir():
            if f.is_file() and f != download and f.suffix.lower() in REFERENCED_SUFFIXES:
                shutil.copy2(f, into / f.name)
    return into


_DRIVE = re.compile(r'[A-Za-z]:')


def entry_name(name):
    """A zip entry's name with its folders split by ``/``, or None for one that would land outside the folder it is
    extracted to (a ``..``, an absolute path, a drive letter). A zip made on Windows (PowerShell's Compress-Archive)
    can separate folders with ``\\``, and Python's zipfile turns that into ``/`` only where ``os.sep`` is ``\\``: on
    Linux and macOS ``Clank\\Clank.obj`` was extracted as one file of that name, and its ``mtllib Clank.mtl`` found
    nothing beside it (SSBM Nucleus 0.9.0 build 3, Linux)."""
    parts = name.replace('\\', '/').split('/')
    if parts[0] == '' or _DRIVE.match(parts[0]) or '..' in parts:
        return None
    kept = [p for p in parts if p not in ('', '.')]
    return '/'.join(kept) + ('/' if parts[-1] == '' else '') if kept else None


def _extract(zip_path, into, depth):
    """Extract one zip into ``into``, keeping every entry inside it (entry_name). Then each zip it held goes into
    ``<name>/`` beside it (``<name> (2)/`` if that is taken), one level deeper, until NESTED_ZIP_DEPTH. A nested zip
    that won't open, or lies deeper, stays as it is."""
    nested = []
    with zipfile.ZipFile(zip_path) as z:
        into.mkdir(parents=True, exist_ok=True)
        root = into.resolve()
        for m in z.infolist():
            name = entry_name(m.filename)
            if name is None:
                continue
            dest = (into / name).resolve()
            if root not in dest.parents:
                continue
            m.filename = name       # what extract writes; the entry is still read by the name in its own header
            path = Path(z.extract(m, into))
            if not m.is_dir() and path.suffix.lower() == '.zip':
                nested.append(path)
    if depth >= NESTED_ZIP_DEPTH:
        return
    for inner in nested:
        folder, k = inner.with_suffix(''), 2
        while folder.exists():
            folder, k = inner.with_name(f'{inner.stem} ({k})'), k + 1
        try:
            _extract(inner, folder, depth + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError):   # not a zip, encrypted, unknown compression
            continue


def model_files(folder):
    return sorted(p for p in Path(folder).rglob('*')
                  if p.suffix.lower() in MODEL_SUFFIXES
                  and not p.name.endswith(('.converted.glb', '.bound.glb', '.bound.gltf')))


def triangles(mesh):
    """How many triangles the mesh's faces make: a face of n corners is n - 2. Unlike the vertex count, this doesn't
    change with how an importer splits vertices at UV seams and hard edges, so the same mesh counts the same from any
    format."""
    import numpy as np
    corners = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.polygons.foreach_get('loop_total', corners)
    return int((corners - 2).sum())


_BLENDER_SUFFIX = re.compile(r'\.\d{3}$')     # the ".001" Blender adds to a name already taken


def _order_key(ob):
    """Where ``ob`` goes in the settled order: what the file gives, never the order it was linked in. Its name, then
    its mesh's name (each without the ".001" Blender adds when a name is taken, which follows the link order), then
    its triangles and vertices, then the centre of its box in the world, then its whole name."""
    from mathutils import Vector
    if ob.type != 'MESH':
        return (_BLENDER_SUFFIX.sub('', ob.name), '', 0, 0, (), ob.name)
    me = ob.data
    centre = ob.matrix_world @ (sum((Vector(c) for c in ob.bound_box), Vector()) / 8)
    return (_BLENDER_SUFFIX.sub('', ob.name), _BLENDER_SUFFIX.sub('', me.name), triangles(me), len(me.vertices),
            tuple(round(float(v), 6) for v in centre), ob.name)


def settle_order():
    """Link the objects of every collection of the scene again in one order the file gives (_order_key), so whatever
    reads the scene's objects in turn reads them the same way each run: the pieces' numbers, the pictures, every build
    step. Blender's FBX importer links a skinned file's objects in an order that changes from one run to the next (it
    goes through a set of its helper nodes, ordered by where they sit in memory): one model drew
    its outline as piece 1 in one run and its body in another. Other importers follow the file; they are settled the
    same way, so one rule covers every format. Returns the order, by name."""
    import bpy

    def settle(coll):
        objects = sorted(coll.objects, key=_order_key)
        for ob in objects:
            coll.objects.unlink(ob)
        for ob in objects:
            coll.objects.link(ob)
        for child in coll.children:
            settle(child)

    settle(bpy.context.scene.collection)
    return [ob.name for ob in bpy.context.scene.objects]


# A level of detail in a mesh's name: "LOD" in any case (or "Lod" after a lower-case letter, as in "BodyLod1"), not
# inside a word, then the level, maybe after one separator: Body_LOD0, body_lod_1, BodyLOD2, Body LOD-3.
_LEVEL = re.compile(r'(?:(?<![A-Za-z])[Ll][Oo][Dd]|(?<=[a-z])(?:LOD|Lod))[ _.-]?(\d+)(?!\d)')
_DIGITS = re.compile(r'\d+')
_SEPARATORS = re.compile(r'[^a-z]+')


def level_of_detail(name):
    """(base name, level) that ``name`` declares, or None when it names no level. The base name is the rest of the name
    in lower case with its digits and separators taken out, so an exporter's running numbers ("_000_Body_LOD0",
    "_003_Body_LOD0" and "_004_Body_LOD1" are all base "body") and Blender's ".001" don't split one base in two."""
    found = list(_LEVEL.finditer(_BLENDER_SUFFIX.sub('', name)))
    if not found:
        return None
    m = found[-1]
    rest = (m.string[:m.start()] + ' ' + m.string[m.end():]).lower()
    return _SEPARATORS.sub(' ', _DIGITS.sub(' ', rest)).strip(), int(m.group(1))


def finest_levels(names):
    """Of ``names`` (a file's meshes), the ones a coarser level of detail the file declares: {name: (base, its level,
    the level kept)}. Meshes that name the same base at two or more levels are one group, and only its lowest level
    stays (LOD0, or the lowest number when there is no LOD0). A mesh that names no level, or whose base no other level
    shares, is left alone."""
    groups = {}
    for n in names:
        found = level_of_detail(n)
        if found is not None:
            groups.setdefault(found[0], []).append((n, found[1]))
    out = {}
    for base, members in groups.items():
        levels = {level for _, level in members}
        if len(levels) > 1:
            out.update({n: (base, level, min(levels)) for n, level in members if level != min(levels)})
    return out


def drop_coarser_levels():
    """Remove from the scene every mesh finest_levels names a coarser level of detail (by its object's name, else its
    mesh's), with its mesh data. -> [{'mesh', 'base', 'level', 'kept_level', 'triangles'}], the meshes removed."""
    import bpy
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    named = {o.name: (o.name if level_of_detail(o.name) else o.data.name) for o in meshes}
    coarser = finest_levels(list(named.values()))
    removed = []
    for o in meshes:
        if named[o.name] in coarser:
            base, level, kept = coarser[named[o.name]]
            removed.append({'mesh': o.name, 'base': base, 'level': level, 'kept_level': kept,
                            'triangles': triangles(o.data)})
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
    return removed


def colours():
    """A colour set is a face's colour where the face draws no image (STATE, the mechanisms, 6). Over a face whose
    material draws an image a game multiplies its set in on its own scale, which no file states (Smash Ultimate's
    sets are half, its neutral, over every face): the image is drawn as it is, the set made white there (a vertex
    faces of both kinds share keeps its colour). A layer then white everywhere carries no colour and goes, as does the
    white COLOR_0 Blender's glTF exporter makes up when no material draws the file's own colours (it writes those as
    COLOR_1, and the importer draws COLOR_0: Final Fantasy VII's Cloud, through our own Blender 4 step, came out white
    so). The first layer left is the one drawn, and the materials of the faces drawing no image draw it (drawn_by):
    glTF's importer wires a set into its material itself, Blender's OBJ and FBX importers keep it on the mesh only,
    and the costume's export carries only a set a material draws. -> {mesh: {'white_over_images': layers,
    'dropped': layers, 'drawn_by': materials}}"""
    import bpy
    import numpy as np
    out = {}
    for me in sorted({o.data for o in bpy.data.objects if o.type == 'MESH'}, key=lambda m: m.name):
        layers = me.color_attributes
        if not len(layers):
            continue
        n = len(me.polygons)
        index, totals = np.empty(n, np.int64), np.empty(n, np.int64)
        me.polygons.foreach_get('material_index', index)
        me.polygons.foreach_get('loop_total', totals)
        imaged = np.array([_draws_image(m) for m in me.materials] or [False])
        over = np.repeat(imaged[np.minimum(index, len(imaged) - 1)], totals)        # per corner
        corner_vertex = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', corner_vertex)
        rec = {'white_over_images': [], 'dropped': []}
        for a in list(layers):
            v = np.empty(len(a.data) * 4)
            a.data.foreach_get('color', v)
            v = v.reshape(-1, 4)
            if a.domain == 'CORNER':
                cover = over
            else:       # a vertex only faces drawing an image use
                cover = np.ones(len(v), bool)
                np.logical_and.at(cover, corner_vertex, over)
                cover &= np.bincount(corner_vertex, minlength=len(v)) > 0
            if cover.any() and not np.allclose(v[cover], 1, atol=1 / 255):
                v[cover] = 1
                a.data.foreach_set('color', v.ravel())
                rec['white_over_images'].append(a.name)
            if np.allclose(v, 1, atol=1 / 255):
                rec['dropped'].append(a.name)
                layers.remove(a)
        if rec['dropped'] and len(layers):
            layers.active_color_index = layers.render_color_index = 0
        if len(layers):
            used = set(np.unique(index).tolist())
            rec['drawn_by'] = [m.name for k, m in enumerate(me.materials)
                               if k in used and m is not None and not imaged[k] and _draw_set(m)]
        if rec['white_over_images'] or rec['dropped'] or rec.get('drawn_by'):
            out[me.name] = {k: v for k, v in rec.items() if v or k != 'drawn_by'}
    return out


def _draw_set(material):
    """Wire the mesh's drawn colour set into ``material``, a material drawing no image, as glTF's importer wires a
    vertex colour: the set times the material's own colour, into its shader's Base Color. A material whose Base Color
    is already fed (a set, a colour graph) is left as it is. -> whether it was wired."""
    tree = material.node_tree
    shader = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if tree else None
    if shader is None or shader.inputs['Base Color'].is_linked:
        return False
    base = shader.inputs['Base Color']
    attribute = tree.nodes.new('ShaderNodeVertexColor')        # its layer left empty: the mesh's drawn set
    mix = tree.nodes.new('ShaderNodeMix')
    mix.data_type, mix.blend_type = 'RGBA', 'MULTIPLY'
    mix.inputs['Factor'].default_value = 1.0
    a, b = (next(s for s in mix.inputs if s.identifier == k) for k in ('A_Color', 'B_Color'))
    b.default_value = base.default_value
    tree.links.new(attribute.outputs['Color'], a)
    tree.links.new(next(s for s in mix.outputs if s.identifier == 'Result_Color'), base)
    return True


def _draws_image(material):
    """Whether ``material``'s surface colour comes from an image: one upstream of its shader's Base Color."""
    tree = material.node_tree if material is not None else None
    if tree is None:
        return False
    todo = [l.from_node for n in tree.nodes if n.type == 'BSDF_PRINCIPLED' for l in n.inputs['Base Color'].links]
    seen = set()
    while todo:
        node = todo.pop()
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return True
        todo.extend(l.from_node for s in node.inputs if s.name not in ('Normal', 'Tangent') for l in s.links)
    return False


def _pack_images():
    """Pack every image that loaded into the .blend; return the names of those that didn't. A texture the file names
    but doesn't ship is a fact about the download, not a failure."""
    import bpy
    missing = []
    for img in bpy.data.images:
        if img.source != 'FILE':
            continue
        try:
            img.pack()
        except RuntimeError:
            missing.append(img.name)
    return missing


def run(download, out):
    import bpy
    out = Path(out)
    src = unpack(download, out / 'source')
    unsupported = [{'file': str(p.relative_to(src)), 'why': UNSUPPORTED[p.suffix.lower()]}
                   for p in sorted(src.rglob('*')) if p.is_file() and p.suffix.lower() in UNSUPPORTED]
    (out / 'load.json').write_text(json.dumps({'unsupported': unsupported}, indent=1), encoding='utf-8')
    files = model_files(src)
    if not files:
        why = '; '.join(f"{f['file']}: {f['why']}" for f in unsupported)
        raise RuntimeError(f'nothing to open in {Path(download).name}: ' +
                           (why or 'no FBX, DAE, OBJ, GLB or glTF file'))
    models_dir = out / 'models'
    models_dir.mkdir(exist_ok=True)
    listing = []
    for n, f in enumerate(files, 1):
        bpy.ops.wm.read_homefile(use_empty=True)
        importer = IMPORTERS[f.suffix.lower()]
        try:
            facts = importer(str(f)) or {}
            standing = stance.settle()
            if standing:
                facts['stance'] = standing
            mapped = texture.bake()
            if mapped:
                facts['texture_transforms'] = mapped
            drawn = colours()
            if drawn:
                facts['colours'] = drawn
            if 'material_colours' not in facts:
                facts['material_colours'] = materials.convert()
        except Exception as e:   # Blender's importers raise RuntimeError with the reason in the message
            listing.append({'n': n, 'file': str(f.relative_to(src)), 'opened': False, 'why': str(e)[:300]})
            continue
        coarser = drop_coarser_levels()
        if coarser:
            facts['coarser_levels_dropped'] = coarser
        settle_order()
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']
        arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
        blend = models_dir / f'{n}-{f.stem}.blend'
        missing = _pack_images()
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        listing.append({
            'n': n, 'file': str(f.relative_to(src)), 'opened': True, 'blend': blend.name,
            'meshes': len(meshes), 'vertices': sum(len(o.data.vertices) for o in meshes),
            'triangles': sum(triangles(o.data) for o in meshes),
            'materials': len(bpy.data.materials), 'images': len(bpy.data.images) - len(missing),
            'armatures': len(arms), 'bones': sum(len(a.data.bones) for a in arms),
            'skinned': any(o.vertex_groups and o.find_armature() for o in meshes),
            'missing_images': missing, **facts,
        })
    (out / 'models.json').write_text(json.dumps(listing, indent=1), encoding='utf-8')
    if not any(m['opened'] for m in listing):
        raise RuntimeError('nothing opens: ' + '; '.join(f"{m['file']}: {m['why']}" for m in listing))
    return listing
