"""Collada, read the same way whoever wrote it (read_dae, into a Model): HSDRawViewer's export of a costume,
Blender's of a build, a target's vanilla .dae; what a .dae holds, as compared before and after a trip through the
game's file. And the name HSDRawViewer gives a mesh when it exports by place (PLACE)."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import numpy as np


NS = '{http://www.collada.org/2005/11/COLLADASchema}'


@dataclass
class Model:
    """What a .dae holds, as compared before and after a trip through the game's file."""
    joints: dict = field(default_factory=dict)      # joint name -> 4x4 world matrix at rest, in file order
    parents: dict = field(default_factory=dict)     # joint name -> parent joint name, or None
    triangles: int = 0                              # drawn: three different corners
    degenerate: int = 0                             # two corners the same vertex (strip joins); never drawn
    materials: int = 0                              # materials on drawn triangles
    textures: dict = field(default_factory=dict)    # image file -> hash of its pixels file, for drawn materials
    points: np.ndarray = None                       # (N, 3) the drawn vertices at rest, world space
    weights: list = field(default_factory=list)     # per point: {joint name: weight}
    faces: list = field(default_factory=list)       # triangles indexing points, retaining their winding
    face_materials: list = field(default_factory=list)
    face_objects: list = field(default_factory=list)  # mesh node name; --names place identifies the written DObj
    face_uvs: list = field(default_factory=list)  # one UV pair per triangle corner, for the exported comparison sheet
    material_images: dict = field(default_factory=dict)

    def positions(self):
        return {name: m[:3, 3] for name, m in self.joints.items()}


def _floats(e):
    return np.array((e.text or '').split(), float)


def _local(node):
    """A node's transform: its <matrix>, <translate>, <rotate> and <scale> elements, in order."""
    m = np.eye(4)
    for e in node:
        tag = e.tag[len(NS):]
        if tag == 'matrix':
            m = m @ _floats(e).reshape(4, 4)
        elif tag == 'translate':
            t = np.eye(4)
            t[:3, 3] = _floats(e)
            m = m @ t
        elif tag == 'scale':
            m = m @ np.diag([*_floats(e), 1.])
        elif tag == 'rotate':
            x, y, z, deg = _floats(e)
            axis = np.array([x, y, z]) / (np.linalg.norm([x, y, z]) or 1.)
            a = np.radians(deg)
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            r = np.eye(4)
            r[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
            m = m @ r
    return m


def _image_files(ids, material_id, folder):
    """The image files a material's effect samples."""
    material = ids.get(material_id)
    if material is None:
        return []
    effect = ids.get(material.find(NS + 'instance_effect').get('url').lstrip('#'))
    files = []
    for init in effect.iter(NS + 'init_from') if effect is not None else ():
        image = ids.get((init.text or '').strip())
        if image is None or image.tag != NS + 'image':
            continue
        source = image.find(NS + 'init_from')
        ref = source.find(NS + 'ref') if source is not None else None
        text = (ref.text if ref is not None else source.text if source is not None else '') or ''
        text = unquote(text.strip())
        if text.startswith('file:'):
            text = text[5:]
            text = text.lstrip('/') if re.match(r'^/*[A-Za-z]:', text) else '/' + text.lstrip('/')
        path = Path(text)
        files.append(path if path.is_absolute() else folder / path)
    return files


def texture_map(dae):
    """{material name: image file}, from ``<dae>.textures.json`` beside the .dae: the image HSDRawViewer's exporter
    names for each material (see read_dae); {} when the .dae's writer left no such file."""
    dae = Path(dae)
    path = dae.with_name(dae.name + '.textures.json')
    if not path.is_file():
        return {}
    return {name: dae.parent / file for name, file in json.loads(path.read_text(encoding='utf-8')).items()}


def read_dae(path):
    """A Model from a .dae: every JOINT node at rest, and the drawn triangles with their vertices, weights, materials
    and textures. Skinned vertices are placed by the skin's bind shape; unskinned ones by their node, and weighted
    wholly to the joint above that node.

    A material whose effect samples no image takes the one ``<path>.textures.json`` names for it, when the .dae's
    writer left that file beside it. HSDRawViewer's exporter (ModelExporter.cs) writes every TObj's image and names each
    material's first in that file, but samples it in the .dae only when the TObj is in the diffuse or specular stage: a
    texture applied after lighting (LIGHTMAP_EXT, where the import's alpha Test puts a texture with alpha) is in the
    file and not in the .dae."""
    path = Path(path)
    root = ET.parse(path).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    named = texture_map(path)

    def images(material):
        files = _image_files(ids, material, path.parent)
        name = ids[material].get('name', material) if material in ids else material
        return files or ([named[name]] if name in named else [])
    model = Model()
    sid_to_name = {}
    instances = []   # (node world, nearest joint above, instance element, mesh node name)

    def walk(node, world, joint):
        for child in node.findall(NS + 'node'):
            w = world @ _local(child)
            j = joint
            if child.get('type') == 'JOINT':
                name = child.get('name') or child.get('sid') or child.get('id')
                model.joints[name] = w
                model.parents[name] = joint
                for k in (child.get('sid'), child.get('id'), name):
                    if k:
                        sid_to_name[k] = name
                j = name
            for inst in child:
                if inst.tag in (NS + 'instance_controller', NS + 'instance_geometry'):
                    instances.append((w, j, inst, child.get('name') or child.get('id')))
            walk(child, w, j)

    for scene in root.iter(NS + 'visual_scene'):
        walk(scene, np.eye(4), None)

    used_materials = set()
    texture_files = set()
    points, weights = [], []
    for world, joint, inst, object_name in instances:
        target = ids.get(inst.get('url').lstrip('#'))
        skin = target.find(NS + 'skin') if inst.tag == NS + 'instance_controller' else None
        geometry = ids.get(skin.get('source').lstrip('#')) if skin is not None else target
        mesh = geometry.find(NS + 'mesh') if geometry is not None else None
        if mesh is None:
            continue
        symbols = {m.get('symbol'): m.get('target').lstrip('#') for m in inst.iter(NS + 'instance_material')}
        vertices = mesh.find(NS + 'vertices')
        source = next(i.get('source').lstrip('#') for i in vertices.findall(NS + 'input')
                      if i.get('semantic') == 'POSITION')
        positions = _floats(ids[source].find(NS + 'float_array')).reshape(-1, 3)
        corners, face_materials = [], []
        for prims in mesh:
            kind = prims.tag[len(NS):]
            if kind not in ('triangles', 'polylist') or prims.find(NS + 'p') is None:
                continue
            inputs = prims.findall(NS + 'input')
            stride = max(int(i.get('offset')) for i in inputs) + 1
            at = next(int(i.get('offset')) for i in inputs if i.get('semantic') == 'VERTEX')
            all_index = _floats(prims.find(NS + 'p')).astype(int).reshape(-1, stride)
            index = all_index[:, at]
            tex_input = next((i for i in inputs if i.get('semantic') == 'TEXCOORD'), None)
            uv = np.zeros((len(index), 2))
            if tex_input is not None:
                src = ids[tex_input.get('source').lstrip('#')]
                width = int(src.find('.//' + NS + 'accessor').get('stride', '2'))
                values = _floats(src.find(NS + 'float_array')).reshape(-1, width)
                uv = values[all_index[:, int(tex_input.get('offset'))], :2]
            if kind == 'polylist':
                counts = _floats(prims.find(NS + 'vcount')).astype(int)
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
                tris = [(index[s], index[s + i], index[s + i + 1]) for s, n in zip(starts, counts)
                        for i in range(1, n - 1)]
                tris = np.array(tris, int).reshape(-1, 3)
                uv = np.array([(uv[s], uv[s + i], uv[s + i + 1]) for s, n in zip(starts, counts)
                               for i in range(1, n - 1)]).reshape(-1, 3, 2)
            else:
                tris = index.reshape(-1, 3)
                uv = uv.reshape(-1, 3, 2)
            same = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
            model.degenerate += int(same.sum())
            drawn = tris[~same]
            if len(drawn):
                model.triangles += len(drawn)
                model.face_uvs.extend(uv[~same].tolist())
                material = symbols.get(prims.get('material'), prims.get('material'))
                model.material_images[material] = [str(p) for p in images(material)]
                face_materials.extend([material] * len(drawn))
                if material not in used_materials:
                    used_materials.add(material)
                    texture_files.update(images(material))
                corners.append(drawn.ravel())
        if not corners:
            continue
        drawn_vertices = np.unique(np.concatenate(corners))
        point_ids = {v: len(points) + i for i, v in enumerate(drawn_vertices)}
        model.faces.extend([[point_ids[v] for v in tri] for tri in np.concatenate(corners).reshape(-1, 3)])
        model.face_materials.extend(face_materials)
        model.face_objects.extend([object_name] * len(face_materials))
        if skin is not None:
            # At rest a skinned vertex sits at sum(weight x joint's world x joint's inverse bind) x bind shape x v,
            # whatever space the writer kept the mesh in.
            bind = _floats(skin.find(NS + 'bind_shape_matrix')).reshape(4, 4) \
                if skin.find(NS + 'bind_shape_matrix') is not None else np.eye(4)
            shaped = (bind @ np.c_[positions, np.ones(len(positions))].T).T
            sources = {i.get('semantic'): ids[i.get('source').lstrip('#')]
                       for i in skin.find(NS + 'joints').findall(NS + 'input')}
            names = (sources['JOINT'].find(NS + 'Name_array').text or '').split()
            names = [sid_to_name.get(n, n) for n in names]
            inverse = _floats(sources['INV_BIND_MATRIX'].find(NS + 'float_array')).reshape(-1, 4, 4)
            to_rest = [model.joints[n] @ inverse[i] if n in model.joints else np.eye(4) for i, n in enumerate(names)]
            vw = skin.find(NS + 'vertex_weights')
            inputs = vw.findall(NS + 'input')
            stride = max(int(i.get('offset')) for i in inputs) + 1
            ja = int(next(i for i in inputs if i.get('semantic') == 'JOINT').get('offset'))
            weight_input = next(i for i in inputs if i.get('semantic') == 'WEIGHT')
            wa = int(weight_input.get('offset'))
            values = _floats(ids[weight_input.get('source').lstrip('#')].find(NS + 'float_array'))
            counts = _floats(vw.find(NS + 'vcount')).astype(int)
            pairs = _floats(vw.find(NS + 'v')).astype(int).reshape(-1, stride)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            for v in drawn_vertices:
                w, m = {}, np.zeros((4, 4))
                if v < len(counts):
                    for row in pairs[starts[v]:starts[v] + counts[v]]:
                        if row[ja] >= 0:
                            w[names[row[ja]]] = w.get(names[row[ja]], 0.) + float(values[row[wa]])
                            m += float(values[row[wa]]) * to_rest[row[ja]]
                points.append((m @ shaped[v])[:3] if w else shaped[v][:3])
                weights.append(w)
        else:
            placed = (world @ np.c_[positions, np.ones(len(positions))].T).T[:, :3]
            for v in drawn_vertices:
                points.append(placed[v])
                weights.append({joint: 1.} if joint else {})
    model.materials = len(used_materials)
    for f in sorted(texture_files):
        model.textures[str(f)] = hashlib.md5(f.read_bytes()).hexdigest() if f.is_file() else None
    model.points = np.array(points, float).reshape(-1, 3)
    model.weights = weights
    return model


# A mesh of a .dae HSDRawViewer exported with --names place: Joint_<joint>_Object_<its place in the costume's DObj
# list>, _SINGLE when the DObj is bound wholly to one joint.
PLACE = re.compile(r'Joint_\d+_Object_(\d+)(?:_SINGLE)?')
